import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# ========== 기존 PerStepEncoder 및 RNN 셀들 ==========
class PerStepEncoder(nn.Module):
    def __init__(self, in_dim=3+1+7, hid=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid),
            nn.ReLU(),
            nn.Linear(hid, hid),
            nn.ReLU(),
        )
    def forward(self, x):
        return self.net(x)

class LSTMCell(nn.Module):
    def __init__(self, in_dim, hid):
        super().__init__()
        self.cell = nn.LSTMCell(in_dim, hid)
        self.hid = hid
    def forward(self, x, state):
        return self.cell(x, state)

class GRUCell(nn.Module):
    def __init__(self, in_dim, hid):
        super().__init__()
        self.cell = nn.GRUCell(in_dim, hid)
        self.hid = hid
    def forward(self, x, state):
        h_prev, c_prev = state
        h_t = self.cell(x, h_prev)
        return h_t, c_prev

class SRULSTMCell(nn.Module):
    def __init__(self, in_dim, hid, refined=False):
        super().__init__()
        self.hid = hid
        self.refined = refined
        self.i = nn.Linear(in_dim + hid, hid)
        self.f = nn.Linear(in_dim + hid, hid)
        self.o = nn.Linear(in_dim + hid, hid)
        self.xs = nn.Linear(in_dim, hid)
        self.xg = nn.Linear(in_dim, hid)
        self.hg = nn.Linear(hid, hid)
    def forward(self, x, state):
        h_prev, c_prev = state
        concat = torch.cat([x, h_prev], dim=-1)
        i_t = torch.sigmoid(self.i(concat))
        f_t = torch.sigmoid(self.f(concat))
        o_t = torch.sigmoid(self.o(concat))
        s_t = self.xs(x)
        cand_lin = self.xg(x) + self.hg(h_prev)
        g_t = torch.tanh(s_t * cand_lin)

        if self.refined:
            # r_t = i_t ⊙ (1 - (1 - f_t)^2) + (1 - i_t) ⊙ (f_t^2)
            r_t = i_t * (1 - (1 - f_t) ** 2) + (1 - i_t) * (f_t ** 2)
            c_t = r_t * c_prev + (1 - r_t) * g_t
        else:
            c_t = f_t * c_prev + i_t * g_t

        h_t = o_t * torch.tanh(c_t)
        return h_t, c_t

# ========== Delta Positional Encoding (기존과 동일) ==========
class DeltaPositionalEncoding(nn.Module):
    def __init__(self, d_model, use_sinus=True):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(7, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.use_sinus = use_sinus

    def sinusoid(self, T, d_model, device):
        position = torch.arange(T, device=device).float().unsqueeze(1)
        i = torch.arange(0, d_model, 2, device=device).float()
        div_term = torch.exp(-math.log(10000.0) * i / d_model)
        pe = torch.zeros(T, d_model, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, motion_seq, d_model):
        B, T, _ = motion_seq.shape
        m = self.proj(motion_seq)
        if self.use_sinus:
            pe_sin = self.sinusoid(T, d_model, motion_seq.device).unsqueeze(0)
            return m + pe_sin
        return m

# ========== Delta Transformer (기존) ==========
class DeltaTransformerAggregator(nn.Module):
    def __init__(self, d_model=128, nhead=4, num_layers=4, dim_feedforward=256, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.motion_pe = DeltaPositionalEncoding(d_model=d_model, use_sinus=True)
        self.d_model = d_model

    def forward(self, z_seq, motion_seq):
        pe = self.motion_pe(motion_seq, self.d_model)
        x = z_seq + pe
        mem = self.encoder(x)
        h_T = mem[:, -1, :]
        return h_T

# ========== 논문식 DeltaNet ==========
class DeltaNetLayer(nn.Module):
    def __init__(self, d_model, beta_hidden=64, layernorm=True):
        super().__init__()
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_beta = nn.Sequential(
            nn.Linear(d_model, beta_hidden),
            nn.ReLU(),
            nn.Linear(beta_hidden, 1),
        )
        self.layernorm = nn.LayerNorm(d_model) if layernorm else nn.Identity()

    def forward(self, x, S=None):
        """
        x: [B, T, d_model]
        """
        B, T, D = x.shape
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)
        beta = torch.sigmoid(self.W_beta(x))  # [B,T,1]

        if( S is None):
            S = torch.zeros(B, D, D, device=x.device, dtype=x.dtype)
        outs = []

        for t in range(T):
            k_t = k[:, t, :].unsqueeze(2)
            v_t = v[:, t, :].unsqueeze(2)
            q_t = q[:, t, :].unsqueeze(2)
            b_t = beta[:, t, :].view(B, 1, 1)

            # Delta rule update
            o_t, S = delta_rule_recurrent_step(q_t, k_t, v_t, b_t, S)
            outs.append(o_t)

        out = torch.stack(outs, dim=1)
        return self.layernorm(out), S

# ========== Head Decoder ==========
class HeadDecoder(nn.Module):
    def __init__(self, hid, T):
        super().__init__()
        self.T = T
        self.coord_head = nn.Sequential(
            nn.Linear(hid, hid),
            nn.ReLU(),
            nn.Linear(hid, 3*T)
        )
        self.label_head = nn.Sequential(
            nn.Linear(hid, hid//2),
            nn.ReLU(),
            nn.Linear(hid//2, T)
        )
    def forward(self, h_T):
        coords = self.coord_head(h_T)
        labels = self.label_head(h_T)
        return coords, labels

# ========== 통합 SpatialMemoryNet ==========
class SpatialMemoryNet(nn.Module):
    def __init__(self, steps=15, enc_hid=128, rec_hid=128, cell_type='lstm',
                 tf_heads=4, tf_layers=4, tf_ff=256, tf_dropout=0.1):
        super().__init__()
        self.steps = steps
        self.cell_type = cell_type
        self.encoder = PerStepEncoder(in_dim=3+1+7, hid=enc_hid)

        if cell_type in ['lstm', 'gru', 'sru_lstm', 'sru_lstm_refined']:
            if cell_type == 'lstm':
                self.cell = LSTMCell(enc_hid, rec_hid)
            elif cell_type == 'gru':
                self.cell = GRUCell(enc_hid, rec_hid)
            elif cell_type == 'sru_lstm_refined':
                self.cell = SRULSTMCell(enc_hid, rec_hid, refined=True)
            else:
                self.cell = SRULSTMCell(enc_hid, rec_hid)
            self.hid = rec_hid
            self.decoder = HeadDecoder(rec_hid, steps)

        elif cell_type == 'delta_transformer':
            self.aggregator = DeltaTransformerAggregator(
                d_model=enc_hid, nhead=tf_heads, num_layers=tf_layers,
                dim_feedforward=tf_ff, dropout=tf_dropout)
            self.hid = enc_hid
            self.decoder = HeadDecoder(enc_hid, steps)

        elif cell_type == 'deltanet':
            self.aggregator = DeltaNetLayer(enc_hid)
            self.hid = enc_hid
            self.decoder = HeadDecoder(enc_hid, steps)

        elif cell_type == "deltanet_transformer":
            self.aggregator = DeltaNetTransformer(
                d_model=enc_hid, num_layers=tf_layers
            )
            self.hid = enc_hid
            self.decoder = HeadDecoder(enc_hid, steps)

        elif cell_type == "gated_deltanet":
            self.aggregator = GatedDeltaNetAggregator(
                d_model=enc_hid,
                num_heads=tf_heads,
                bidirectional=False,     # causal seq
                min_len_for_chunk=65     # ensures chunk mode during training
            )
            self.hid = enc_hid
            self.decoder = HeadDecoder(enc_hid, steps)

        else:
            raise ValueError("cell_type must be one of: lstm, gru, sru_lstm, delta_transformer, deltanet")

    def forward(self, obs_l, obs_c, obs_m):
        B, T, _ = obs_l.shape
        x = torch.cat([obs_l, obs_c, obs_m], dim=-1)
        z = self.encoder(x)

        if self.cell_type in ['lstm', 'gru', 'sru_lstm', 'sru_lstm_refined']:
            h = torch.zeros(B, self.hid, device=obs_l.device)
            c = torch.zeros_like(h)
            for t in range(T):
                h, c = self.cell(z[:, t, :], (h, c))
            h_T = h

        elif self.cell_type == 'delta_transformer':
            h_T = self.aggregator(z, obs_m)

        elif self.cell_type == 'deltanet':
            S = None
            for t in range(T):
                h_t, S = self.aggregator.forward(z[:, t, :].unsqueeze(1), S)
            h_T = h_t.squeeze(1)

        elif self.cell_type == 'deltanet_transformer':
            h_seq = self.aggregator(z)
            h_T = h_seq[:, -1, :]

        elif self.cell_type == 'gated_deltanet':
            h_T = self.aggregator(z)

        coords_flat, label_logits = self.decoder(h_T)
        coords = coords_flat.view(B, self.steps, 3)
        return coords, label_logits


import torch
import torch.nn as nn
import torch.nn.functional as F

from fla.layers import GatedDeltaNet

class GatedDeltaNetAggregator(nn.Module):
    """
    Safe wrapper so SpatialMemoryNet can use FLA GatedDeltaNet
    even when seq_len <= 64 in training.

    - If training & T <= 64 → automatically pads to >= 65
      to force FLA to use chunk mode (the only mode stable in training).
    - Output h_T is always the *true* last-step state (index T-1).
    """
    def __init__(self, d_model=128, num_heads=4, bidirectional=False, min_len_for_chunk=65):
        super().__init__()
        self.min_len_for_chunk = min_len_for_chunk

        # the official Gated DeltaNet layer
        self.delta = GatedDeltaNet(
            hidden_size=d_model,
            num_heads=num_heads,
            bidirectional=bidirectional,
            mode="chunk",          # must be chunk mode for training
            chunk_size=None        # let FLA auto-select optimal chunk
        )

    def forward(self, z_seq, motion_seq=None):
        """
        z_seq: (B, T, d)
        return: (B, d)  -- final hidden state
        """
        B, T, D = z_seq.shape

        # ========== TRAINING CASE (T <= 64 requires padding) ==========
        if self.training and T <= 64:
            pad_len = self.min_len_for_chunk - T
            if pad_len < 0:
                pad_len = 0

            if pad_len > 0:
                pad = torch.zeros(B, pad_len, D,
                                  device=z_seq.device,
                                  dtype=z_seq.dtype)
                z_in = torch.cat([z_seq, pad], dim=1)  # (B, T+pad_len, D)
            else:
                z_in = z_seq

            # Forward through FLA (returns y, kv, gate)
            y, _, _ = self.delta(z_in)

            # IMPORTANT:
            # We must return the *true* last state of the real sequence
            return y[:, T - 1, :]

        # ========== EVAL OR LONG SEQ ==========
        else:
            y, _, _ = self.delta(z_seq)
            return y[:, -1, :]

# ============================================================
# RMSNorm
# ============================================================
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.norm(2, dim=-1, keepdim=True)
        rms = norm / (x.shape[-1] ** 0.5)
        return x / (rms + self.eps) * self.weight


# ============================================================
# SwiGLU FFN
# ============================================================
class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim * 2)
        self.w2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        a, b = self.w1(x).chunk(2, dim=-1)
        return self.w2(F.silu(a) * b)


# ============================================================
# Depthwise Conv (short conv)
# ============================================================
def delta_rule_recurrent_step(q_t, k_t, v_t, beta_t, S_prev):
    S_new = S_prev @ (torch.eye(S_prev.size(-1), device=S_prev.device) - beta_t * (k_t @ k_t.transpose(1,2))) + beta_t * (v_t @ k_t.transpose(1,2))
    o_t = (S_new @ q_t).squeeze(2)
    return o_t, S_new

# ============================================================
# DeltaNet Transformer Block
# ============================================================
class DeltaNetBlock(nn.Module):
    def __init__(self, dim, neg_eigen=False):
        super().__init__()
        self.norm1 = RMSNorm(dim)

        # q/k/v short convolution
        self.W_q = nn.Linear(dim, dim)
        self.W_k = nn.Linear(dim, dim)
        self.W_v = nn.Linear(dim, dim)
        self.W_beta = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 1),
        )


        self.norm2 = RMSNorm(dim)
        self.ff = SwiGLU(dim, dim)
        self.layernorm = nn.LayerNorm(dim)

    def forward(self, x, S=None):
        h = x
        B, T, D = x.shape

        # conv before projections
        q = self.W_q(h)
        k = self.W_k(h)
        v = self.W_v(h)
        beta = torch.sigmoid(self.W_beta(h))  # [B,T,1]
        # temporarily replace input for DeltaBlock
        # use three input streams by concatenation trick
        # but simplest is: do separate projections INSIDE DeltaBlock
        # → we simply replace x with a mix containing q/k/v signals
        # easiest method: feed h_q + h_k + h_v

        if( S is None):
            S = torch.zeros(B, D, D, device=x.device, dtype=x.dtype)
        outs = []

        for t in range(T):
            k_t = k[:, t, :].unsqueeze(2)
            v_t = v[:, t, :].unsqueeze(2)
            q_t = q[:, t, :].unsqueeze(2)
            b_t = beta[:, t, :].view(B, 1, 1)

            o_t, S = delta_rule_recurrent_step(q_t, k_t, v_t, b_t, S)
            outs.append(o_t)

        delta_out = torch.stack(outs, dim=1)

        return self.layernorm(delta_out), S

        # delta_ff = x + delta_out
        # # ---- FFN ----
        # ff_out = self.ff(self.norm2(delta_ff))
        # out = ff_out + delta_ff
        # return out, S

# ============================================================
# Full DeltaNet Transformer
# ============================================================
class DeltaNetTransformer(nn.Module):
    def __init__(self, d_model=128, num_layers=2, neg_eigen=False):
        super().__init__()


        self.layers = nn.ModuleList([
            DeltaNetBlock(
                dim=d_model,
                neg_eigen=neg_eigen
            )
            for _ in range(num_layers)
        ])

        self.norm_out = RMSNorm(d_model)
        self.proj_out = nn.Linear(d_model, d_model)

    def forward(self, x):
        S = None
        for layer in self.layers:
            x, S = layer(x, S)
        x = self.norm_out(x)
        return self.proj_out(x)