# spatial_mapping_figure2.py
# PyTorch reimplementation of the "Figure 2 spatial mapping" toy task
# Requires: torch, numpy, matplotlib

import math, random, argparse, os
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ---------------------------
# 1) Geometry helpers (SE(2))
# ---------------------------

def se2_from_pose(x, y, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    T = np.array([[c, -s, x],
                  [s,  c, y],
                  [0,  0, 1]], dtype=np.float32)
    return T

def se2_inv(T):
    R = T[:2,:2]; t = T[:2,2]
    R_inv = R.T
    t_inv = -R_inv @ t
    Tinv = np.eye(3, dtype=np.float32)
    Tinv[:2,:2] = R_inv
    Tinv[:2,2]  = t_inv
    return Tinv

def se2_mul(A, B):
    return (A @ B).astype(np.float32)

def se2_apply(T, p_xy):
    p = np.array([p_xy[0], p_xy[1], 1.0], dtype=np.float32)
    q = T @ p
    return q[:2]

def se2_to_vec(T):
    # encode pose delta as [cos θ, sin θ, tx, ty]
    c, s = T[0,0], T[1,0]
    tx, ty = T[0,2], T[1,2]
    return np.array([c, s, tx, ty], dtype=np.float32)

# ------------------------------------
# 2) Synthetic data: spiral trajectory
# ------------------------------------

@dataclass
class TaskConfig:
    steps: int = 15          # T
    spiral_scale: float = 0.2
    yaw_per_step: float = 0.35
    noise_landmark: float = 0.0  # optional obs noise in robot frame

def make_spiral_poses(cfg: TaskConfig):
    """Generate T robot poses along a 2D spiral in world frame."""
    poses = []
    x = y = yaw = 0.0
    for t in range(cfg.steps):
        r = cfg.spiral_scale * (t+1)
        # move forward along a spiral arm
        x = r * math.cos(0.5 * (t+1))
        y = r * math.sin(0.5 * (t+1))
        yaw += cfg.yaw_per_step
        poses.append(se2_from_pose(x, y, yaw))
    return poses  # list of 3x3

def sample_landmarks_world(cfg: TaskConfig, poses):
    """One landmark per step; fix them in world, then 'observe' from each robot frame when visited."""
    T = cfg.steps
    # Place landmarks roughly near the path (with small random spread)
    landmarks_w = []
    rng = np.random.default_rng()
    for t, Tw in enumerate(poses):
        # put landmark a bit ahead of the robot in its heading dir
        forward = Tw[:2,0]  # robot x-axis in world
        base = Tw[:2,2] + 0.7 * forward
        jitter = rng.normal(0, 0.15, size=2)
        p = (base + jitter).astype(np.float32)
        landmarks_w.append(p)
    # Binary labels tied to order (temporal memory)
    labels = np.array([random.randint(0,1) for _ in range(T)], dtype=np.float32)
    return np.stack(landmarks_w, axis=0), labels  # [T,2], [T]

def make_episode(cfg: TaskConfig):
    """Return per-step inputs and training targets for one sequence."""
    poses = make_spiral_poses(cfg)
    T = cfg.steps

    # Ego-motion deltas M_{t-1}^t (robot-local transform from t-1 to t)
    ego_deltas = []
    for t in range(T):
        if t == 0:
            ego_deltas.append(np.eye(3, dtype=np.float32))
        else:
            T_w_tm1 = poses[t-1]
            T_w_t   = poses[t]
            T_tm1_t = se2_mul(se2_inv(T_w_tm1), T_w_t)
            ego_deltas.append(T_tm1_t)

    lm_w, labels = sample_landmarks_world(cfg, poses)

    # Observations: landmark in current robot frame (t), and delta transform vector
    l_t = []
    M_vec = []
    for t in range(T):
        T_w_t = poses[t]
        T_t_w = se2_inv(T_w_t)
        p_t = se2_apply(T_t_w, lm_w[t])
        if cfg.noise_landmark > 0:
            p_t = p_t + np.random.normal(0, cfg.noise_landmark, size=2).astype(np.float32)
        l_t.append(p_t.astype(np.float32))
        M_vec.append(se2_to_vec(ego_deltas[t]))

    l_t   = np.stack(l_t, axis=0)        # [T,2]
    M_vec = np.stack(M_vec, axis=0)      # [T,4]
    c_t   = labels[:,None]               # [T,1] — label is observed at each step (temporal memory input)

    # Targets at final frame T: transform every landmark into frame T
    T_w_T = poses[-1]
    T_T_w = se2_inv(T_w_T)
    l_T_targets = []
    for i in range(T):
        pT = se2_apply(T_T_w, lm_w[i])
        l_T_targets.append(pT.astype(np.float32))
    l_T_targets = np.stack(l_T_targets, axis=0)  # [T,2]

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
# 3) Model: encoder + recurrent + heads
# --------------------------------

class PerStepEncoder(nn.Module):
    def __init__(self, in_dim=2+1+4, hid=64):
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
    def forward(self, x, state):
        return self.cell(x, state)

class GRUCell(nn.Module):
    def __init__(self, in_dim, hid):
        super().__init__()
        self.cell = nn.GRUCell(in_dim, hid)
    def forward(self, x, state):
        # GRUCell returns h_t; wrap to look like LSTMCell interface
        h_prev, c_prev = state
        h_t = self.cell(x, h_prev)
        return h_t, c_prev

class SRULSTMCell(nn.Module):
    """
    SRU-LSTM (paper Sec. 4.4):
      s_t = W_xs x_t + b_s
      g_t = tanh( s_t ⊙ (W_xg x_t + W_hg h_{t-1} + b_g) )
      ... then use standard LSTM gates (i,f,o) but replace candidate with g_t
    """
    def __init__(self, in_dim, hid):
        super().__init__()
        self.hid = hid
        self.i = nn.Linear(in_dim + hid, hid)
        self.f = nn.Linear(in_dim + hid, hid)
        self.o = nn.Linear(in_dim + hid, hid)
        self.xs = nn.Linear(in_dim, hid)     # for s_t
        self.xg = nn.Linear(in_dim, hid)     # for candidate part
        self.hg = nn.Linear(hid, hid)

    def forward(self, x, state):
        h_prev, c_prev = state
        concat = torch.cat([x, h_prev], dim=-1)
        i_t = torch.sigmoid(self.i(concat))
        f_t = torch.sigmoid(self.f(concat))
        o_t = torch.sigmoid(self.o(concat))
        s_t = self.xs(x)                      # [B,H]
        cand_lin = self.xg(x) + self.hg(h_prev)
        g_t = torch.tanh(s_t * cand_lin)      # ⊙ elementwise
        c_t = f_t * c_prev + i_t * g_t
        h_t = o_t * torch.tanh(c_t)
        return h_t, c_t

# (선택) SRU-Ours의 refined gate는 논문 식(추가 정제 게이팅) 참고해 아래처럼 바꿔볼 수 있습니다:
# rt = i_t ⊙ (1 - (1 - f_t)^2) + (1 - i_t) ⊙ (f_t^2);  c_t = rt ⊙ c_{t-1} + (1-rt) ⊙ g_t
# 구현 시 위 두 줄로 c_t 업데이트만 교체하면 됩니다. (본문 수식 참조)  :contentReference[oaicite:3]{index=3}

class HeadDecoder(nn.Module):
    def __init__(self, hid, T):
        super().__init__()
        self.T = T
        self.coord_head = nn.Sequential(
            nn.Linear(hid, hid),
            nn.ReLU(),
            nn.Linear(hid, 2*T)   # all l_i^T flattened
        )
        self.label_head = nn.Sequential(
            nn.Linear(hid, hid//2),
            nn.ReLU(),
            nn.Linear(hid//2, T)  # logits for all c_i
        )
    def forward(self, h_T):
        coords = self.coord_head(h_T)      # [B, 2T]
        labels = self.label_head(h_T)      # [B, T]
        return coords, labels

class SpatialMemoryNet(nn.Module):
    def __init__(self, steps=15, enc_hid=64, rec_hid=128, cell_type='lstm'):
        super().__init__()
        self.steps = steps
        self.encoder = PerStepEncoder(in_dim=2+1+4, hid=enc_hid)

        if cell_type == 'lstm':
            self.cell = LSTMCell(enc_hid, rec_hid)
        elif cell_type == 'gru':
            self.cell = GRUCell(enc_hid, rec_hid)
        elif cell_type == 'sru_lstm':
            self.cell = SRULSTMCell(enc_hid, rec_hid)
        else:
            raise ValueError("cell_type must be one of: lstm, gru, sru_lstm")

        self.decoder = HeadDecoder(rec_hid, steps)

    def forward(self, obs_l, obs_c, obs_m):
        """
        obs_l : [B,T,2]
        obs_c : [B,T,1]
        obs_m : [B,T,4]
        """
        B, T, _ = obs_l.shape
        h = torch.zeros(B, self.cell.hid if hasattr(self.cell,'hid') else 128, device=obs_l.device)
        c = torch.zeros_like(h)
        for t in range(T):
            x_t = torch.cat([obs_l[:,t,:], obs_c[:,t,:], obs_m[:,t,:]], dim=-1)  # [B,7]
            z_t = self.encoder(x_t)                                              # [B,enc_hid]
            h, c = self.cell(z_t, (h, c))

        coords_flat, label_logits = self.decoder(h)       # at final step
        coords = coords_flat.view(B, self.steps, 2)
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
    opt = optim.Adam(model.parameters(), lr=lr1, betas=(0.9, 0.999), eps=1e-8)  # Nesterov Adam 변형 대신 Adam 사용
    model.train()
    for ep in range(1, epochs+1):
        # make a fresh batch every iter (randomization)
        episodes = [ make_episode(cfg) for _ in range(batch_size) ]
        obs_l, obs_c, obs_m, y_coords, y_labels = batchify(episodes)
        obs_l, obs_c, obs_m = obs_l.to(device), obs_c.to(device), obs_m.to(device)
        y_coords, y_labels = y_coords.to(device), y_labels.to(device)

        pred_coords, pred_logits = model(obs_l, obs_c, obs_m)
        loss_spatial = F.mse_loss(pred_coords, y_coords)                    # MSE (spatial)  :contentReference[oaicite:4]{index=4}
        loss_temporal = F.binary_cross_entropy_with_logits(pred_logits, y_labels)  # BCE (temporal)  :contentReference[oaicite:5]{index=5}
        loss = loss_spatial + loss_temporal

        opt.zero_grad()
        loss.backward()
        opt.step()

        if ep == lr_milestone:
            for g in opt.param_groups:
                g['lr'] = lr2  # schedule as in appendix

        if ep % 100 == 0:
            with torch.no_grad():
                acc = ((torch.sigmoid(pred_logits) > 0.5) == (y_labels>0.5)).float().mean().item()
                print(f"[{ep:04d}] loss={loss.item():.4f}  spatial={loss_spatial.item():.4f}  temporal={loss_temporal.item():.4f}  label_acc={acc:.3f}")

def evaluate_spatial_error_curve(model, cfg, n_trials=200, device="cpu"):
    """Return mean Euclidean error per observation index (ordered final→initial like Fig.2c)."""
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
            diff = (pred_coords - y_coords).cpu().numpy()[0]  # [T,2]
            e = np.linalg.norm(diff, axis=1)                  # [T]
            # order from final (T) to initial (1)
            errs[k,:] = e[::-1]
    return errs.mean(0), errs.std(0)

def plot_mapping_example(model, cfg, save_dir, device="cpu", title=""):
    """Plot one episode: world traj, landmarks, and predicted vs GT in final frame, akin to Fig.2(a)(b)."""
    os.makedirs(save_dir, exist_ok=True)
    ep = make_episode(cfg)
    obs_l = torch.tensor(ep["obs_l"][None,...], dtype=torch.float32, device=device)
    obs_c = torch.tensor(ep["obs_label"][None,...], dtype=torch.float32, device=device)
    obs_m = torch.tensor(ep["obs_motion"][None,...], dtype=torch.float32, device=device)
    y_coords = torch.tensor(ep["target_coords_T"], dtype=torch.float32).cpu().numpy()  # [T,2]

    model.eval()
    with torch.no_grad():
        pred_coords, _ = model(obs_l, obs_c, obs_m)
    pred = pred_coords.cpu().numpy()[0]  # [T,2]

    # Plot in final frame
    T = cfg.steps
    fig, ax = plt.subplots()
    ax.set_aspect('equal', 'box')
    ax.scatter(y_coords[:,0], y_coords[:,1], label='GT l_i^T', marker='o')
    ax.scatter(pred[:,0], pred[:,1], label='Pred l_i^T', marker='x')
    for i in range(T):
        ax.text(y_coords[i,0], y_coords[i,1], f"{i+1}", fontsize=8)
    ax.legend(); ax.set_title(title or "Spatial mapping in final frame")
    fig.savefig(os.path.join(save_dir, "mapping_example.png"), dpi=160)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", type=str, default="lstm", choices=["lstm","gru","sru_lstm"])
    ap.add_argument("--steps", type=int, default=15)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--enc", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--out", type=str, default="runs_figure2")
    args = ap.parse_args()

    device = torch.device(args.device)
    cfg = TaskConfig(steps=args.steps)

    model = SpatialMemoryNet(steps=cfg.steps, enc_hid=args.enc, rec_hid=args.hidden, cell_type=args.cell).to(device)
    train(model, cfg, epochs=args.epochs, device=device)

    # Figure 2(c)-style error curve
    mean_err, std_err = evaluate_spatial_error_curve(model, cfg, device=device)
    os.makedirs(args.out, exist_ok=True)
    x = np.arange(cfg.steps) + 1  # 1..T (final to initial)
    fig, ax = plt.subplots()
    ax.plot(x, mean_err)
    ax.fill_between(x, mean_err-std_err, mean_err+std_err, alpha=0.2)
    ax.set_yscale("log")
    ax.set_xlabel("Observation index (final→initial)")
    ax.set_ylabel("Mean spatial error (log)")
    ax.set_title(f"Spatial memory error ({args.cell})")
    fig.savefig(os.path.join(args.out, f"spatial_error_{args.cell}.png"), dpi=160)
    plt.close(fig)

    # One qualitative mapping plot (final frame)
    plot_mapping_example(model, cfg, args.out, device, title=f"Mapping example ({args.cell})")

if __name__ == "__main__":
    main()
