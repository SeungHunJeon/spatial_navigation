
# spatial_mapping_figure2_se3_delta.py
# PyTorch reimplementation of "Figure 2 spatial mapping" toy task in 3D (SE(3))
# Adds a Delta-Transformer aggregator (motion-aware transformer) in addition to LSTM/GRU/SRU-LSTM
# Requires: torch, numpy, matplotlib

import math, random, argparse, os
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ---------------------------
# 1) Geometry helpers (SE(3))
# ---------------------------

def euler_to_R(roll, pitch, yaw):
    """XYZ intrinsic rotations (roll=x, pitch=y, yaw=z)."""
    cx, sx = math.cos(roll), math.sin(roll)
    cy, sy = math.cos(pitch), math.sin(pitch)
    cz, sz = math.cos(yaw), math.sin(yaw)
    Rx = np.array([[1, 0, 0],
                   [0, cx, -sx],
                   [0, sx, cx]], dtype=np.float32)
    Ry = np.array([[cy, 0, sy],
                   [0, 1, 0],
                   [-sy, 0, cy]], dtype=np.float32)
    Rz = np.array([[cz, -sz, 0],
                   [sz,  cz, 0],
                   [0,   0,  1]], dtype=np.float32)
    R = Rz @ Ry @ Rx
    return R

def se3_from_pose(x, y, z, roll, pitch, yaw):
    R = euler_to_R(roll, pitch, yaw)
    T = np.eye(4, dtype=np.float32)
    T[:3,:3] = R
    T[:3, 3] = np.array([x, y, z], dtype=np.float32)
    return T

def se3_inv(T):
    R = T[:3,:3]; t = T[:3,3]
    R_inv = R.T
    t_inv = -R_inv @ t
    Tinv = np.eye(4, dtype=np.float32)
    Tinv[:3,:3] = R_inv
    Tinv[:3, 3] = t_inv
    return Tinv

def se3_mul(A, B):
    return (A @ B).astype(np.float32)

def se3_apply(T, p_xyz):
    p = np.array([p_xyz[0], p_xyz[1], p_xyz[2], 1.0], dtype=np.float32)
    q = T @ p
    return q[:3]

def mat_to_quat(R):
    """Convert 3x3 rotation matrix to (w,x,y,z) quaternion."""
    m00, m01, m02 = R[0,0], R[0,1], R[0,2]
    m10, m11, m12 = R[1,0], R[1,1], R[1,2]
    m20, m21, m22 = R[2,0], R[2,1], R[2,2]
    tr = m00 + m11 + m22
    if tr > 0:
        S = math.sqrt(tr + 1.0) * 2
        w = 0.25 * S
        x = (m21 - m12) / S
        y = (m02 - m20) / S
        z = (m10 - m01) / S
    elif (m00 > m11) and (m00 > m22):
        S = math.sqrt(1.0 + m00 - m11 - m22) * 2
        w = (m21 - m12) / S
        x = 0.25 * S
        y = (m01 + m10) / S
        z = (m02 + m20) / S
    elif m11 > m22:
        S = math.sqrt(1.0 + m11 - m00 - m22) * 2
        w = (m02 - m20) / S
        x = (m01 + m10) / S
        y = 0.25 * S
        z = (m12 + m21) / S
    else:
        S = math.sqrt(1.0 + m22 - m00 - m11) * 2
        w = (m10 - m01) / S
        x = (m02 + m20) / S
        y = (m12 + m21) / S
        z = 0.25 * S
    q = np.array([w, x, y, z], dtype=np.float32)
    q = q / (np.linalg.norm(q) + 1e-8)
    return q

def se3_to_vec(T):
    """Encode pose delta as quaternion (w,x,y,z) + translation (tx,ty,tz) -> 7D vector."""
    R = T[:3,:3]
    t = T[:3,3]
    q = mat_to_quat(R)
    return np.concatenate([q, t.astype(np.float32)], axis=0).astype(np.float32)  # [7]

# ------------------------------------
# 2) Synthetic data: 3D spiral/helix
# ------------------------------------

@dataclass
class TaskConfig:
    steps: int = 15           # T
    spiral_scale: float = 0.25
    yaw_per_step: float = 0.35
    pitch_per_step: float = 0.02
    z_per_step: float = 0.10
    noise_landmark: float = 0.0  # optional obs noise in robot frame

def make_spiral_poses(cfg: TaskConfig):
    """
    Generate T robot poses along a 3D helix (spiral) in world frame.
    The robot ascends in z while yawing; small pitch changes add richness.
    """
    poses = []
    x = y = z = 0.0
    roll = 0.0
    pitch = 0.0
    yaw = 0.0
    for t in range(cfg.steps):
        r = cfg.spiral_scale * (t+1)
        x = r * math.cos(0.5 * (t+1))
        y = r * math.sin(0.5 * (t+1))
        z += cfg.z_per_step
        yaw += cfg.yaw_per_step
        pitch += cfg.pitch_per_step  # gentle nose-up
        poses.append(se3_from_pose(x, y, z, roll, pitch, yaw))
    return poses  # list of 4x4

def sample_landmarks_world(cfg: TaskConfig, poses):
    """One landmark per step; fixed in world; at step t we 'observe' the t-th landmark."""
    T = cfg.steps
    landmarks_w = []
    rng = np.random.default_rng()
    for t, Tw in enumerate(poses):
        # place landmark a bit ahead (robot x-axis) and slightly offset in +z
        forward = Tw[:3,0]  # robot x-axis in world (3D)
        base = Tw[:3,3] + 0.8 * forward + np.array([0,0,0.1], dtype=np.float32)
        jitter = rng.normal(0, 0.15, size=3).astype(np.float32)
        p = (base + jitter).astype(np.float32)
        landmarks_w.append(p)
    labels = np.array([random.randint(0,1) for _ in range(T)], dtype=np.float32)
    return np.stack(landmarks_w, axis=0), labels  # [T,3], [T]

def make_episode(cfg: TaskConfig):
    """Return per-step inputs and training targets for one sequence (3D)."""
    poses = make_spiral_poses(cfg)
    T = cfg.steps

    # Ego-motion deltas M_{t-1}^t (robot-local transform from t-1 to t)
    ego_deltas = []
    for t in range(T):
        if t == 0:
            ego_deltas.append(np.eye(4, dtype=np.float32))
        else:
            T_w_tm1 = poses[t-1]
            T_w_t   = poses[t]
            T_tm1_t = se3_mul(se3_inv(T_w_tm1), T_w_t)
            ego_deltas.append(T_tm1_t)

    lm_w, labels = sample_landmarks_world(cfg, poses)

    # Observations: landmark in current robot frame (t), and delta transform vector (quat+trans)
    l_t = []
    M_vec = []
    for t in range(T):
        T_w_t = poses[t]
        T_t_w = se3_inv(T_w_t)
        p_t = se3_apply(T_t_w, lm_w[t])
        if cfg.noise_landmark > 0:
            p_t = p_t + np.random.normal(0, cfg.noise_landmark, size=3).astype(np.float32)
        l_t.append(p_t.astype(np.float32))
        M_vec.append(se3_to_vec(ego_deltas[t]))  # [7]

    l_t   = np.stack(l_t, axis=0)        # [T,3]
    M_vec = np.stack(M_vec, axis=0)      # [T,7]
    c_t   = labels[:,None]               # [T,1]

    # Targets at final frame T: transform every landmark into frame T
    T_w_T = poses[-1]
    T_T_w = se3_inv(T_w_T)
    l_T_targets = []
    for i in range(T):
        pT = se3_apply(T_T_w, lm_w[i])
        l_T_targets.append(pT.astype(np.float32))
    l_T_targets = np.stack(l_T_targets, axis=0)  # [T,3]

    return {
        "obs_l": l_t,
        "obs_label": c_t,
        "obs_motion": M_vec,
        "target_coords_T": l_T_targets,
        "target_labels": labels.astype(np.float32),
        "poses": poses,
        "lm_world": lm_w
    }

# --------------------------------
# 3) Model: encoder + aggregator + heads
# --------------------------------

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
    def __init__(self, in_dim, hid):
        super().__init__()
        self.hid = hid
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
        c_t = f_t * c_prev + i_t * g_t
        h_t = o_t * torch.tanh(c_t)
        return h_t, c_t

# ---- Delta-Transformer aggregator ----
class DeltaPositionalEncoding(nn.Module):
    """
    Motion-aware positional encoding.
    Map per-step SE(3) delta vector (7D: quat+trans) -> d_model and add to token.
    Optionally add standard index-based sinusoidal encoding.
    """
    def __init__(self, d_model, use_sinus=True):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(7, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.use_sinus = use_sinus

    def sinusoid(self, T, d_model, device):
        position = torch.arange(T, device=device).float().unsqueeze(1)  # [T,1]
        i = torch.arange(0, d_model, 2, device=device).float()          # [d/2]
        div_term = torch.exp(-math.log(10000.0) * i / d_model)
        pe = torch.zeros(T, d_model, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe  # [T,d]

    def forward(self, motion_seq, d_model):
        """
        motion_seq: [B,T,7]
        returns pe: [B,T,d_model]
        """
        B, T, _ = motion_seq.shape
        m = self.proj(motion_seq)  # [B,T,d]
        if self.use_sinus:
            pe_sin = self.sinusoid(T, d_model, motion_seq.device).unsqueeze(0)  # [1,T,d]
            return m + pe_sin
        return m


# ===== DeltaNet (논문식 Delta Rule Attention) =====
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
            k_t = k[:, t, :].unsqueeze(2)       # [B,D,1]
            v_t = v[:, t, :].unsqueeze(2)       # [B,D,1]
            q_t = q[:, t, :].unsqueeze(2)       # [B,D,1]
            b_t = beta[:, t, :].view(B, 1, 1)   # [B,1,1]

            # Delta rule update: S_t = S_{t-1}(I - βk kᵀ) + βv kᵀ
            S = S @ (eye - b_t * (k_t @ k_t.transpose(1,2))) + b_t * (v_t @ k_t.transpose(1,2))

            o_t = (S @ q_t).squeeze(2)
            outs.append(o_t)

        out = torch.stack(outs, dim=1)
        return self.layernorm(out)


# ===== DeltaNet + DeltaPositionalEncoding =====
class DeltaNetWithDeltaPE(nn.Module):
    def __init__(self, d_model=128, num_layers=2, beta_hidden=64):
        super().__init__()
        self.layers = nn.ModuleList([DeltaNetLayer(d_model, beta_hidden) for _ in range(num_layers)])
        self.delta_pe = DeltaPositionalEncoding(d_model)
        self.norm_out = nn.LayerNorm(d_model)

    def forward(self, x, motion_seq):
        """
        x: [B,T,d_model], motion_seq: [B,T,7]
        """
        x = x + self.delta_pe(motion_seq, x.size(-1))
        for layer in self.layers:
            x = layer(x)
        return self.norm_out(x)

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
        """
        z_seq: [B,T,d_model]  (per-step encoded tokens)
        motion_seq: [B,T,7]   (SE3 deltas)
        returns h_T: [B,d_model] pooled (use last token)
        """
        pe = self.motion_pe(motion_seq, self.d_model)  # [B,T,d]
        x = z_seq + pe
        mem = self.encoder(x)  # [B,T,d]
        h_T = mem[:, -1, :]    # take last token as summary
        return h_T

class HeadDecoder(nn.Module):
    def __init__(self, hid, T):
        super().__init__()
        self.T = T
        self.coord_head = nn.Sequential(
            nn.Linear(hid, hid),
            nn.ReLU(),
            nn.Linear(hid, 3*T)   # all l_i^T flattened (3D)
        )
        self.label_head = nn.Sequential(
            nn.Linear(hid, hid//2),
            nn.ReLU(),
            nn.Linear(hid//2, T)  # logits for all c_i
        )
    def forward(self, h_T):
        coords = self.coord_head(h_T)      # [B, 3T]
        labels = self.label_head(h_T)      # [B, T]
        return coords, labels

class SpatialMemoryNet(nn.Module):
    def __init__(self, steps=15, enc_hid=128, rec_hid=128, cell_type='lstm',
                 tf_heads=4, tf_layers=4, tf_ff=256, tf_dropout=0.1):
        super().__init__()
        self.steps = steps
        self.cell_type = cell_type
        self.encoder = PerStepEncoder(in_dim=3+1+7, hid=enc_hid)

        if cell_type in ['lstm', 'gru', 'sru_lstm']:
            if cell_type == 'lstm':
                self.cell = LSTMCell(enc_hid, rec_hid)
            elif cell_type == 'gru':
                self.cell = GRUCell(enc_hid, rec_hid)
            else:
                self.cell = SRULSTMCell(enc_hid, rec_hid)
            self.hid = rec_hid
            self.decoder = HeadDecoder(rec_hid, steps)

        elif cell_type == 'delta_transformer':
            self.aggregator = DeltaTransformerAggregator(
                d_model=enc_hid, nhead=tf_heads,
                num_layers=tf_layers, dim_feedforward=tf_ff,
                dropout=tf_dropout)
            self.hid = enc_hid
            self.decoder = HeadDecoder(enc_hid, steps)

        elif cell_type == 'deltanet':
            self.aggregator = DeltaNetLayer(enc_hid)
            self.hid = enc_hid
            self.decoder = HeadDecoder(enc_hid, steps)

        elif cell_type == 'deltanet_delta_pe':
            self.aggregator = DeltaNetWithDeltaPE(d_model=enc_hid)
            self.hid = enc_hid
            self.decoder = HeadDecoder(enc_hid, steps)

        else:
            raise ValueError("cell_type must be one of: lstm, gru, sru_lstm, delta_transformer, deltanet, deltanet_delta_pe")

    def forward(self, obs_l, obs_c, obs_m):
        """
        obs_l : [B,T,3]
        obs_c : [B,T,1]
        obs_m : [B,T,7]
        """
        B, T, _ = obs_l.shape
        x = torch.cat([obs_l, obs_c, obs_m], dim=-1)   # [B,T,11]
        z = self.encoder(x)                            # [B,T,enc_hid]

        if self.cell_type in ['lstm', 'gru', 'sru_lstm']:
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

        elif self.cell_type == 'deltanet_delta_pe':
            h_seq = self.aggregator(z, obs_m)
            h_T = h_seq[:, -1, :]

        coords_flat, label_logits = self.decoder(h_T)
        coords = coords_flat.view(B, self.steps, 3)
        return coords, label_logits
# ---------------------------
# 4) Training / Evaluation
# ---------------------------

def batchify(episodes):
    obs_l = np.stack([e["obs_l"] for e in episodes], 0)
    obs_c = np.stack([e["obs_label"] for e in episodes], 0)
    obs_m = np.stack([e["obs_motion"] for e in episodes], 0)
    y_coords = np.stack([e["target_coords_T"] for e in episodes], 0)
    y_labels = np.stack([e["target_labels"] for e in episodes], 0)
    return (
        torch.tensor(obs_l, dtype=torch.float32),
        torch.tensor(obs_c, dtype=torch.float32),
        torch.tensor(obs_m, dtype=torch.float32),
        torch.tensor(y_coords, dtype=torch.float32),
        torch.tensor(y_labels, dtype=torch.float32),
    )

def train(model, cfg, epochs=1000, batch_size=64, lr1=2e-3, lr2=4e-4, lr_milestone=800, device="cpu"):
    opt = optim.Adam(model.parameters(), lr=lr1, betas=(0.9, 0.999), eps=1e-8)
    model.train()
    for ep in range(1, epochs+1):
        episodes = [ make_episode(cfg) for _ in range(batch_size) ]
        obs_l, obs_c, obs_m, y_coords, y_labels = batchify(episodes)
        obs_l, obs_c, obs_m = obs_l.to(device), obs_c.to(device), obs_m.to(device)
        y_coords, y_labels = y_coords.to(device), y_labels.to(device)

        pred_coords, pred_logits = model(obs_l, obs_c, obs_m)
        loss_spatial = F.mse_loss(pred_coords, y_coords)
        loss_temporal = F.binary_cross_entropy_with_logits(pred_logits, y_labels)
        loss = loss_spatial + loss_temporal

        opt.zero_grad()
        loss.backward()
        opt.step()

        if ep == lr_milestone:
            for g in opt.param_groups:
                g['lr'] = lr2

        if ep % 100 == 0:
            with torch.no_grad():
                acc = ((torch.sigmoid(pred_logits) > 0.5) == (y_labels>0.5)).float().mean().item()
                print(f"[{ep:04d}] loss={loss.item():.4f}  spatial={loss_spatial.item():.4f}  temporal={loss_temporal.item():.4f}  label_acc={acc:.3f}")

def evaluate_spatial_error_curve(model, cfg, n_trials=200, device="cpu"):
    """Return mean Euclidean error per observation index (ordered final→initial)."""
    model.eval()
    T = cfg.steps
    errs = np.zeros((n_trials, T), dtype=np.float32)
    with torch.no_grad():
        for k in range(n_trials):
            ep = make_episode(cfg)
            obs_l = torch.tensor(ep["obs_l"][None,...], dtype=torch.float32, device=device)
            obs_c = torch.tensor(ep["obs_label"][None,...], dtype=torch.float32, device=device)
            obs_m = torch.tensor(ep["obs_motion"][None,...], dtype=torch.float32, device=device)
            y_coords = torch.tensor(ep["target_coords_T"][None,...], dtype=torch.float32, device=device)

            pred_coords, _ = model(obs_l, obs_c, obs_m)
            diff = (pred_coords - y_coords).cpu().numpy()[0]  # [T,3]
            e = np.linalg.norm(diff, axis=1)                  # [T]
            errs[k,:] = e[::-1]
    return errs.mean(0), errs.std(0)

def plot_mapping_example(model, cfg, args, device="cpu", title=""):
    """Plot one episode in 3D: predicted vs GT in final frame."""
    os.makedirs(args.out, exist_ok=True)
    ep = make_episode(cfg)
    obs_l = torch.tensor(ep["obs_l"][None,...], dtype=torch.float32, device=device)
    obs_c = torch.tensor(ep["obs_label"][None,...], dtype=torch.float32, device=device)
    obs_m = torch.tensor(ep["obs_motion"][None,...], dtype=torch.float32, device=device)
    y_coords = torch.tensor(ep["target_coords_T"], dtype=torch.float32).cpu().numpy()  # [T,3]

    model.eval()
    with torch.no_grad():
        pred_coords, _ = model(obs_l, obs_c, obs_m)
    pred = pred_coords.cpu().numpy()[0]  # [T,3]

    T = cfg.steps
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(y_coords[:,0], y_coords[:,1], y_coords[:,2], marker='o', label='GT l_i^T')
    ax.scatter(pred[:,0], pred[:,1], pred[:,2], marker='x', label='Pred l_i^T')
    for i in range(T):
        ax.text(y_coords[i,0], y_coords[i,1], y_coords[i,2], f"{i+1}", fontsize=8)
    ax.set_title(title or "Spatial mapping in final frame (3D)")
    ax.legend()
    fig.savefig(os.path.join(args.out, "mapping_example_3d_" + args.cell + ".png"), dpi=160)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", type=str, default="lstm", choices=["lstm","gru","sru_lstm","delta_transformer", "deltanet", "deltanet_delta_pe"])
    ap.add_argument("--steps", type=int, default=15)
    ap.add_argument("--hidden", type=int, default=128, help="RNN hidden size or Transformer d_model")
    ap.add_argument("--enc", type=int, default=128, help="Per-step encoder hidden (also d_model for transformer)")
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--out", type=str, default="runs_figure2_se3")
    # Transformer-specific
    ap.add_argument("--tf_heads", type=int, default=4)
    ap.add_argument("--tf_layers", type=int, default=4)
    ap.add_argument("--tf_ff", type=int, default=256)
    ap.add_argument("--tf_dropout", type=float, default=0.1)
    args = ap.parse_args()

    device = torch.device(args.device)
    cfg = TaskConfig(steps=args.steps)

    model = SpatialMemoryNet(steps=cfg.steps,
                             enc_hid=args.enc,
                             rec_hid=args.hidden,
                             cell_type=args.cell,
                             tf_heads=args.tf_heads,
                             tf_layers=args.tf_layers,
                             tf_ff=args.tf_ff,
                             tf_dropout=args.tf_dropout).to(device)
    train(model, cfg, epochs=args.epochs, device=device)

    # Figure 2(c)-style error curve (3D)
    mean_err, std_err = evaluate_spatial_error_curve(model, cfg, device=device)
    os.makedirs(args.out, exist_ok=True)
    x = np.arange(cfg.steps) + 1  # 1..T (final to initial)
    fig, ax = plt.subplots()
    ax.plot(x, mean_err)
    ax.fill_between(x, mean_err-std_err, mean_err+std_err, alpha=0.2)
    ax.set_yscale("log")
    ax.set_xlabel("Observation index (final→initial)")
    ax.set_ylabel("Mean spatial error (log, 3D Euclidean)")
    ax.set_title(f"Spatial memory error 3D ({args.cell})")
    fig.savefig(os.path.join(args.out, f"spatial_error3d_{args.cell}.png"), dpi=160)
    plt.close(fig)

    # One qualitative mapping plot (final frame, 3D)
    plot_mapping_example(model, cfg, args, device, title=f"Mapping example (3D, {args.cell})")

if __name__ == "__main__":
    main()