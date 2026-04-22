# TAR · Terrain-Adaptive RL for Thunder Quadruped

[![Paper: TAR (IROS 2025)](https://img.shields.io/badge/Paper-arXiv:2503.20839-9cf.svg)](https://arxiv.org/abs/2503.20839)
[![Paper: TerAdapt (RA-L 2026)](https://img.shields.io/badge/Paper-IEEE%20RA--L%202026-blue.svg)](#)
[![Robot: Thunder 16DOF](https://img.shields.io/badge/Robot-Thunder%2016DOF-orange.svg)](#)
[![Isaac Lab 0.46](https://img.shields.io/badge/IsaacLab-0.46.2-lightgrey.svg)](https://github.com/isaac-sim/IsaacLab)

Training & comparison of **proprioception-only terrain adaptation** for **Thunder** wheeled-legged quadruped (16 DOF = 12 leg + 4 wheel). Deployed on Horizon S100P board.

## Architectures

| ID | Task ID | Entry | Key Idea |
|---|---|---|---|
| **HIM** | `RobotLab-Isaac-Velocity-Rough-Thunder-Hist-v0` | `train_him.py` | SwAV contrastive on proprio history (Long et al. 2024) |
| **TAR** | `GaitTarRough` | `train_tar.py` | Paper-accurate port of `ammousa/TARLoco` — contrastive hinge + encoder_critic + num_hist_short=4 |
| **TerAdapt** | `GaitTerRough` | `train_teradapt.py` | VQ-VAE 256-code terrain codebook + dual-horizon proprio (Short MLP + Long 1D CNN) |

TAR uses the official `ammousa/TARLoco` architecture line-by-line; only Thunder's 16-DOF observation slicing is adapted. Full reference source kept in `reference/TARLoco/`.

## Paper Alignment (TAR)

### Observation layout
Follows TARLoco's `TarMlpGo1LocomotionVelocityRoughEnvCfg`:
- **Policy** (history=10, `flatten_history_dim=False` when stacking):
  ```
  base_ang_vel(3) + projected_gravity(3) + velocity_commands(3)
  + joint_pos(16) + joint_vel(16) + last_action(16) = 57 dims/frame
  ```
  Base linear velocity and height_scan explicitly **excluded** (proprio only).
- **Critic** (history=1, single frame, full privileged):
  ```
  [0:3]   base_lin_vel            ← extract_critic takes vel from here
  [3:60]  proprio (57 dims)       ← extract_critic takes proprio from here
  [60:]   height_scan + contacts + friction + mass
  ```

### Training
- **Single stage end-to-end PPO** (no teacher pretraining). TAR contrastive loss applied concurrently.
- **max_iterations = 200** (smoke-test current default; TARLoco official Go1 uses 1500)
- **num_steps_per_env=24, num_learning_epochs=5, num_mini_batches=4** (TARLoco Go1RoughPpoRunnerCfg)
- **γ=0.99, λ=0.95, entropy_coef=0.01, desired_kl=0.01** (TARLoco `tar_algo_cfg`)
- **empirical_normalization=True** (obs whitening)
- Adam optimizer, adaptive KL schedule, lr_max=1e-3

### Terrain
Thunder uses its native rough terrain inherited from `thunder_gait`. TARLoco paper used `ROUGH_TERRAINS_CFG` which includes `random_tracks` (railway tracks 50%) — this terrain type doesn't translate well to wheeled-legged robots (wheels roll over rails very differently than legs step over them), so we keep Thunder's original curriculum. Domain randomization matches paper: friction [0.1, 3.0], payload [-2, 10] kg.

## Directory Layout

```
tar/
├── docs/
├── reference/
│   └── TARLoco/                      ← ammousa/TARLoco source (CC-BY-NC-SA 4.0)
├── scripts/reinforcement_learning/rsl_rl/
│   ├── modules/
│   │   ├── tar_actor_critic.py              (paper-accurate TARLoco port)
│   │   ├── teradapt_tca.py
│   │   └── teradapt_actor_critic.py
│   ├── storage/
│   │   ├── tar_rollout_storage.py
│   │   └── teradapt_rollout_storage.py
│   ├── algorithms/
│   │   ├── tar_ppo.py
│   │   └── teradapt_ppo.py
│   ├── runners/
│   │   ├── tar_on_policy_runner.py
│   │   └── teradapt_on_policy_runner.py
│   ├── train_tar.py
│   └── train_teradapt.py
└── source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/thunder_him_gait/
    ├── __init__.py                    ← registers 3 gym tasks
    ├── him_rough_env_cfg.py
    ├── tar_rough_env_cfg.py
    ├── teradapt_rough_env_cfg.py
    └── agents/
        ├── rsl_rl_ppo_cfg.py          (HIM)
        ├── rsl_rl_tar_cfg.py
        └── rsl_rl_teradapt_cfg.py
```

## Training

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate thunder
cd robot_lab

# HIM baseline (existing)
CUDA_VISIBLE_DEVICES=0 python scripts/reinforcement_learning/rsl_rl/train_him.py \
    --task RobotLab-Isaac-Velocity-Rough-Thunder-Hist-v0 --num_envs 4096 --headless

# TAR (paper-accurate)
CUDA_VISIBLE_DEVICES=1 python scripts/reinforcement_learning/rsl_rl/train_tar.py \
    --task GaitTarRough --num_envs 4096 --headless

# TerAdapt (VQ-VAE)
CUDA_VISIBLE_DEVICES=2 python scripts/reinforcement_learning/rsl_rl/train_teradapt.py \
    --task GaitTerRough --num_envs 4096 --headless
```

Each training takes ~8-12 hours on RTX 3090 (4096 envs; TAR 200-1500 iter for smoke/full, HIM/TerAdapt 20000 iter baseline).

## Architecture Details

### HIM — Baseline
10-frame proprio history encoder + SwAV contrastive learning (Long et al. 2024). Uses existing Thunder HIM framework.

### TAR — Paper-accurate TARLoco port
```
encoder_actor:   proprio_history[10×57=570] → MLP[256,128,64] → z_a[45]
encoder_critic:  critic_obs              → MLP[256,128,64] → z_c[45]
trans:           [z_a, action]           → MLP[64]         → z_a_next[45]
vel_estimator:   cat(z_a, hist_short[4×57=228])[273] → MLP[64,32] → v̂[3]
Actor:  cat(prop[57], z_a.detach()[45], v̂.detach()[3])[105] → MLP[512,256,128] → action[16]
Critic: cat(prop[57], z_c[45], vel_priv[3])[105]            → MLP[512,256,128] → value

Losses:
  L_tar = ‖next_z_c − trans(z_a, a)‖²_sum.mean()                       # positive
        + max(0, 1 − ‖next_z_c − next_neg_z‖²_sum).mean()              # hinge negative
  L_vel = MSE(v̂, vel_priv)
# Negatives sampled from batch; excludes same-env indices (mod num_envs).
```

### TerAdapt — VQ-VAE Codebook
```
TCA (teacher):
  height_scan[187] → MLP[64,32] → z_t[16]
  VQ Codebook (256 × 16, EMA decay=0.99) → quantize → indices[0..255], z_q
  MLP[32,64] → ĥ[187]

Student (dual horizon):
  short_hist[5×57]  → MLP[128,64]                → h_short[16]
  long_hist[50×57]  → 1D CNN[32,32,32, k=8,5,5]  → h_long[16]
  cat(h_short, h_long)[32] → MLP[64,32] → l_tilde[16]
  cat(h_short, h_long)[32] → MLP[64,32] → v_hat[3]
  l_tilde[16]       → MLP[64,128]        → logits[256]

Actor: cat(obs[57], h_short[16], h_long[16], l_tilde[16], v_hat[3])[109] → MLP[512,256,128] → action[16]

L_vel  = MSE(v_hat, vel_gt)
L_tok  = CE(logits, indices.detach())
L_vq   = MSE(ĥ, h) + 0.25 · MSE(z, sg(z_q))
```

## Environment
- Isaac Sim 4.5 + Isaac Lab 0.46.2
- Python 3.10 · PyTorch 2.x · CUDA 12.x
- RSL-RL PPO base

## References
- [TAR paper (arXiv:2503.20839, IROS 2025)](https://arxiv.org/abs/2503.20839) — official repo: [ammousa/TARLoco](https://github.com/ammousa/TARLoco)
- [TerAdapt (IEEE RA-L, June 2026)](#)
- [HIMLoco (Long et al. 2024)](https://arxiv.org/abs/2312.11460)
