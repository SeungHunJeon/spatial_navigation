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

    def forward(self, x):
        """
        x: [B, T, d_model]
        """
        B, T, D = x.shape
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)
        beta = torch.sigmoid(self.W_beta(x))  # [B,T,1]

        S = torch.zeros(B, D, D, device=x.device, dtype=x.dtype)
        outs = []
        eye = torch.eye(D, device=x.device).unsqueeze(0)  # [1,D,D]

        for t in range(T):
            k_t = k[:, t, :].unsqueeze(2)
            v_t = v[:, t, :].unsqueeze(2)
            q_t = q[:, t, :].unsqueeze(2)
            b_t = beta[:, t, :].view(B, 1, 1)

            # Delta rule update
            S = S @ (eye - b_t * (k_t @ k_t.transpose(1,2))) + b_t * (v_t @ k_t.transpose(1,2))
            o_t = (S @ q_t).squeeze(2)
            outs.append(o_t)

        out = torch.stack(outs, dim=1)
        return self.layernorm(out)

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
                d_model=enc_hid, d_ff=tf_ff, num_layers=tf_layers, dropout=tf_dropout
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
            h_seq = self.aggregator(z)
            h_T = h_seq[:, -1, :]

        elif self.cell_type == 'deltanet_transformer':
            h_seq = self.aggregator(z)
            h_T = h_seq[:, -1, :]

        coords_flat, label_logits = self.decoder(h_T)
        coords = coords_flat.view(B, self.steps, 3)
        return coords, label_logits


import torch
import torch.nn as nn
import torch.nn.functional as F


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
class DepthwiseConv1d(nn.Module):
    def __init__(self, dim, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(dim, dim, kernel_size, padding=padding, groups=dim)

    def forward(self, x):
        # x: (B, L, D)
        return self.conv(x.transpose(1, 2)).transpose(1, 2)


# ============================================================
# Delta Rule core (병렬/순차 둘 다 가능)
# ============================================================
# (➡️ 이미 완성된 DeltaBlock 은 그대로 복사)
# ------------------------------------------------------------
# 🔥 여기에는 너가 직전에 만든 DeltaBlock 코드가 통째로 들어간다고 보면 됨
# ------------------------------------------------------------

# === 👇 그대로 붙여넣음 ====================================

def chunk_batched_delta_rule_forward(Q, K, V, beta, C: int):
    B, L, d = Q.shape
    assert L % C == 0
    device = Q.device
    dtype = Q.dtype

    Q, K, V = map(lambda x: x.reshape(B, -1, C, d), (Q, K, V))
    beta = beta.reshape(B, -1, C)

    K_beta = K * beta.unsqueeze(-1)
    V_beta = V * beta.unsqueeze(-1)

    mask_upper = torch.triu(torch.ones(C, C, device=device, dtype=torch.bool), 0)

    K_t = K.transpose(2, 3)
    T = -(K_beta @ K_t)
    T = T.masked_fill(mask_upper, 0)

    eye = torch.eye(C, device=device, dtype=dtype).unsqueeze(0)

    # -------- SAFE FORWARD SUBSTITUTION (no inplace) --------
    for k in range(L // C):
        for i in range(1, C):
            T_slice = T[:, k, i, :i]                          # (B, i)
            upd = (T[:, k, i, :, None] * T[:, k, :, :i]).sum(-2)
            T = T.clone()
            T[:, k, i, :i] = T_slice + upd

        T = T.clone()
        T[:, k] = T[:, k] + eye

    # --------------------------------------------------------
    W = T @ K_beta
    U = T @ V_beta

    S = torch.zeros(B, d, d, device=device, dtype=dtype)
    O = torch.empty_like(V)

    mask_strict_upper = torch.triu(torch.ones(C, C, device=device, dtype=torch.bool), 1)

    for i in range(L // C):
        q_i = Q[:, i]
        k_i = K[:, i]
        w_i = W[:, i]
        u_i = U[:, i] - w_i @ S

        o_inter = q_i @ S
        A_i = (q_i @ k_i.transpose(1, 2)).masked_fill(mask_strict_upper, 0)
        o_intra = A_i @ u_i

        S = S + k_i.transpose(1, 2) @ u_i
        O[:, i] = o_intra + o_inter

    return O.reshape(B, L, d)
def delta_rule_recurrent_step(q_t, k_t, v_t, beta_t, S_prev):
    v_old = S_prev @ k_t
    v_new = beta_t * v_t + (1 - beta_t) * v_old
    S_new = S_prev - torch.outer(v_old, k_t) + torch.outer(v_new, k_t)
    o_t = S_new @ q_t
    return o_t, S_new


class DeltaBlock(nn.Module):
    def __init__(self, d, expand=1, neg_eigen=False):
        super().__init__()
        self.d = d
        self.expand = expand
        self.Wq = nn.Linear(d, d * expand)
        self.Wk = nn.Linear(d, d * expand)
        self.Wv = nn.Linear(d, d * expand)

        self.beta = nn.Linear(d, 1)
        self.sigma = nn.Sigmoid()
        self.alpha = 2 if neg_eigen else 1

        self.proj_out = nn.Linear(d * expand, d)

    def forward(self, X, chunk=1):
        B, L, d = X.shape
        if chunk == 1:
            chunk = L

        Q = self.Wq(X)
        K = self.Wk(X)
        V = self.Wv(X) / self.alpha
        beta = self.alpha * self.sigma(self.beta(X))

        O = chunk_batched_delta_rule_forward(Q, K, V, beta, chunk)
        return self.proj_out(O)

    def step(self, X, S=None):
        if S is None:
            D = self.d * self.expand
            S = torch.zeros(D, D, device=X.device, dtype=X.dtype)

        q = self.Wq(X)
        k = self.Wk(X)
        v = self.Wv(X) / self.alpha
        beta_t = self.alpha * self.sigma(self.beta(X))

        y_fast, S_new = delta_rule_recurrent_step(q, k, v, beta_t, S)
        return self.proj_out(y_fast), S_new

# ============================================================
# DeltaNet Transformer Block
# ============================================================
class DeltaNetBlock(nn.Module):
    def __init__(self, dim, expand=1, ff_mult=4, neg_eigen=False, kernel_size=3, chunk=64):
        super().__init__()
        self.chunk = chunk

        self.norm1 = RMSNorm(dim)
        self.delta = DeltaBlock(dim, expand=expand, neg_eigen=neg_eigen)

        # q/k/v short convolution
        self.conv_q = DepthwiseConv1d(dim)
        self.conv_k = DepthwiseConv1d(dim)
        self.conv_v = DepthwiseConv1d(dim)

        self.norm2 = RMSNorm(dim)
        self.ff = SwiGLU(dim, dim * ff_mult)

    def forward(self, x):
        # ---- DeltaNet attention part ----
        h = self.norm1(x)

        # conv before projections
        h_q = self.conv_q(h)
        h_k = self.conv_k(h)
        h_v = self.conv_v(h)

        # temporarily replace input for DeltaBlock
        # use three input streams by concatenation trick
        # but simplest is: do separate projections INSIDE DeltaBlock
        # → we simply replace x with a mix containing q/k/v signals
        # easiest method: feed h_q + h_k + h_v
        h_delta_in = h_q + h_k + h_v

        attn_out = self.delta(h_delta_in, chunk=self.chunk)
        x = x + attn_out

        # ---- FFN ----
        ff_out = self.ff(self.norm2(x))
        x = x + ff_out
        return x


# ============================================================
# Full DeltaNet Transformer
# ============================================================
class DeltaNetTransformer(nn.Module):
    def __init__(self, d_model=128, d_ff=256, num_layers=6, dropout=0.1, chunk=1, neg_eigen=False):
        super().__init__()

        ff_mult = d_ff // d_model

        self.layers = nn.ModuleList([
            DeltaNetBlock(
                dim=d_model,
                expand=1,
                ff_mult=ff_mult,
                neg_eigen=neg_eigen,
                kernel_size=3,
                chunk=chunk
            )
            for _ in range(num_layers)
        ])

        self.norm_out = RMSNorm(d_model)
        self.proj_out = nn.Linear(d_model, d_model)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.norm_out(x)
        return self.proj_out(x)