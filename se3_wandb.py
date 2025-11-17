
# spatial_mapping_figure2_se3_wandb.py
# 3D (SE3) spatial mapping task with RNNs + Delta-Transformer and Weights & Biases logging.
# Usage examples:
#   python spatial_mapping_figure2_se3_wandb.py --cell all --epochs 800 --steps 15 --wandb 1 --wandb_project myproj
#   python spatial_mapping_figure2_se3_wandb.py --cell delta_transformer --enc 128 --tf_layers 6 --wandb 1

import math, random, argparse, os, sys
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from model import *

# ---------------------------
# 0) Optional WANDB
# ---------------------------
try:
    import wandb
    _WANDB_AVAILABLE = True
except Exception as e:
    _WANDB_AVAILABLE = False
    wandb = None

# ---------------------------
# 1) Geometry helpers (SE(3))
# ---------------------------

def euler_to_R(roll, pitch, yaw):
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
    R = T[:3,:3]
    t = T[:3,3]
    q = mat_to_quat(R)
    return np.concatenate([q, t.astype(np.float32)], axis=0).astype(np.float32)  # [7]

# ------------------------------------
# 2) Synthetic data: 3D spiral/helix
# ------------------------------------

@dataclass
class TaskConfig:
    steps: int = 15
    spiral_scale: float = 0.25
    yaw_per_step: float = 0.35
    pitch_per_step: float = 0.02
    z_per_step: float = 0.10
    noise_landmark: float = 0.0

def make_spiral_poses(cfg: TaskConfig):
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
        pitch += cfg.pitch_per_step
        poses.append(se3_from_pose(x, y, z, roll, pitch, yaw))
    return poses

def sample_landmarks_world(cfg: TaskConfig, poses):
    T = cfg.steps
    landmarks_w = []
    rng = np.random.default_rng()
    for t, Tw in enumerate(poses):
        forward = Tw[:3,0]
        base = Tw[:3,3] + 0.8 * forward + np.array([0,0,0.1], dtype=np.float32)
        jitter = rng.normal(0, 0.15, size=3).astype(np.float32)
        p = (base + jitter).astype(np.float32)
        landmarks_w.append(p)
    labels = np.array([random.randint(0,1) for _ in range(T)], dtype=np.float32)
    return np.stack(landmarks_w, axis=0), labels  # [T,3], [T]

def make_episode(cfg: TaskConfig):
    poses = make_spiral_poses(cfg)
    T = cfg.steps
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

def train(model, cfg, epochs=1000, batch_size=64, lr1=2e-3, lr2=4e-4, lr_milestone=800, device="cpu",
          log_every=100, wb=None, run_name=None):
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
        nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        opt.step()

        if ep == lr_milestone:
            for g in opt.param_groups:
                g['lr'] = lr2

        if ep % log_every == 0:
            with torch.no_grad():
                acc = ((torch.sigmoid(pred_logits) > 0.5) == (y_labels>0.5)).float().mean().item()
            if wb is not None:
                wb.log({"loss/total": loss.item(),
                        "loss/spatial": loss_spatial.item(),
                        "loss/temporal": loss_temporal.item(),
                        "metrics/label_acc": acc,
                        "step": ep})
            print(f"[{ep:04d}] loss={loss.item():.4f}  spatial={loss_spatial.item():.4f}  temporal={loss_temporal.item():.4f}  label_acc={acc:.3f}")

def evaluate_spatial_error_curve(model, cfg, n_trials=200, device="cpu"):
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
            diff = (pred_coords - y_coords).cpu().numpy()[0]
            e = np.linalg.norm(diff, axis=1)
            errs[k,:] = e[::-1]
    return errs.mean(0), errs.std(0)

def plot_mapping_example(model, cfg, out_dir, device="cpu", title=""):
    os.makedirs(out_dir, exist_ok=True)
    ep = make_episode(cfg)
    obs_l = torch.tensor(ep["obs_l"][None,...], dtype=torch.float32, device=device)
    obs_c = torch.tensor(ep["obs_label"][None,...], dtype=torch.float32, device=device)
    obs_m = torch.tensor(ep["obs_motion"][None,...], dtype=torch.float32, device=device)
    y_coords = torch.tensor(ep["target_coords_T"], dtype=torch.float32).cpu().numpy()

    model.eval()
    with torch.no_grad():
        pred_coords, _ = model(obs_l, obs_c, obs_m)
    pred = pred_coords.cpu().numpy()[0]

    T = cfg.steps
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(y_coords[:,0], y_coords[:,1], y_coords[:,2], marker='o', label='GT l_i^T')
    ax.scatter(pred[:,0], pred[:,1], pred[:,2], marker='x', label='Pred l_i^T')
    for i in range(T):
        ax.text(y_coords[i,0], y_coords[i,1], y_coords[i,2], f"{i+1}", fontsize=8)
    ax.set_title(title or "Spatial mapping in final frame (3D)")
    ax.legend()
    path = os.path.join(out_dir, "mapping_example_3d.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path

def run_one(args, cell_type, device, group=None):
    cfg = TaskConfig(steps=args.steps,
                     spiral_scale=args.spiral_scale,
                     pitch_per_step=args.pitch_per_step,
                     z_per_step=args.z_per_step,
                     noise_landmark=args.noise_landmark)
    # Build model
    model = SpatialMemoryNet(steps=cfg.steps,
                             enc_hid=args.enc,
                             rec_hid=args.hidden,
                             cell_type=cell_type,
                             tf_heads=args.tf_heads,
                             tf_layers=args.tf_layers,
                             tf_ff=args.tf_ff,
                             tf_dropout=args.tf_dropout).to(device)

    # WANDB init per model
    wb = None
    run = None
    run_name = f"{cell_type}_T{args.steps}_enc{args.enc}_hid{args.hidden}"
    if args.wandb and _WANDB_AVAILABLE:
        run = wandb.init(project=args.wandb_project or "spatial-mapping",
                         entity=args.wandb_entity or None,
                         name=run_name,
                         group=group or args.wandb_group or None,
                         config={
                             "cell": cell_type, "steps": args.steps, "epochs": args.epochs,
                             "enc": args.enc, "hidden": args.hidden,
                             "tf_heads": args.tf_heads, "tf_layers": args.tf_layers,
                             "tf_ff": args.tf_ff, "tf_dropout": args.tf_dropout,
                             "batch_size": args.batch_size, "lr1": args.lr1, "lr2": args.lr2,
                         })
        wb = wandb

    # Train
    train(model, cfg, epochs=args.epochs, batch_size=args.batch_size, lr1=args.lr1, lr2=args.lr2,
          lr_milestone=args.lr_milestone, device=device, log_every=args.log_every, wb=wb, run_name=run_name)

    # Evaluate + plot
    mean_err, std_err = evaluate_spatial_error_curve(model, cfg, device=device)
    os.makedirs(args.out, exist_ok=True)
    x = np.arange(cfg.steps) + 1
    fig, ax = plt.subplots()
    ax.plot(x, mean_err)
    ax.fill_between(x, mean_err-std_err, mean_err+std_err, alpha=0.2)
    ax.set_yscale("log")
    ax.set_xlabel("Observation index (final→initial)")
    ax.set_ylabel("Mean spatial error (log, 3D Euclidean)")
    ax.set_title(f"Spatial memory error 3D ({cell_type})")
    curve_path = os.path.join(args.out, f"spatial_error3d_{cell_type}.png")
    fig.savefig(curve_path, dpi=160)
    plt.close(fig)

    map_path = plot_mapping_example(model, cfg, args.out, device, title=f"Mapping example (3D, {cell_type})")

    if wb is not None:
        wb.log({
            "eval/mean_error_last": float(mean_err[0]),  # index 1..T reversed (0 is most recent)
        })
        wb.log({
            "plots/error_curve": wandb.Image(curve_path),
            "plots/mapping_example": wandb.Image(map_path),
        })
        # Attach as artifacts
        art = wandb.Artifact(f"figs_{cell_type}", type="plots")
        art.add_file(curve_path)
        art.add_file(map_path)
        wb.log_artifact(art)

    if run is not None:
        run.finish()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", type=str, default="lstm",
                    choices=["lstm","gru","sru_lstm", "sru_lstm_refined", "delta_transformer","deltanet","deltanet_transformer", "gated_deltanet", "all"])
    ap.add_argument("--steps", type=int, default=15)
    ap.add_argument("--spiral_scale", type=float, default=1.0)
    ap.add_argument("--pitch_per_step", type=float, default=0.15)
    ap.add_argument("--z_per_step", type=float, default=0.5)
    ap.add_argument("--noise_landmark", type=float, default=0.1)

    ap.add_argument("--hidden", type=int, default=128, help="RNN hidden size or Transformer d_model")
    ap.add_argument("--enc", type=int, default=128, help="Per-step encoder hidden (also d_model for transformer)")
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr1", type=float, default=2e-3)
    ap.add_argument("--lr2", type=float, default=4e-4)
    ap.add_argument("--lr_milestone", type=int, default=800)
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--out", type=str, default="runs_figure2_se3")
    # Transformer-specific
    ap.add_argument("--tf_heads", type=int, default=2)
    ap.add_argument("--tf_layers", type=int, default=1)
    ap.add_argument("--tf_ff", type=int, default=256)
    ap.add_argument("--tf_dropout", type=float, default=0.1)
    # WANDB
    ap.add_argument('--wandb', action='store_true', help='wandb on/off')
    ap.add_argument("--wandb_project", type=str, default=None)
    ap.add_argument("--wandb_entity", type=str, default=None)
    ap.add_argument("--wandb_group", type=str, default=None)
    args = ap.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.out, exist_ok=True)

    # If all, loop over all cells under a common WANDB group
    if args.cell == "all":
        group = args.wandb_group or "model-sweep"
        for cell in ["lstm","gru","sru_lstm", "sru_lstm_refined", "delta_transformer", "deltanet", "deltanet_transformer", "gated_deltanet"]:
            run_one(args, cell, device, group=group)
    else:
        run_one(args, args.cell, device, group=args.wandb_group)

if __name__ == "__main__":
    main()