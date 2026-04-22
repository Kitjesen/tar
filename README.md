# TAR · Terrain-Adaptive RL for Thunder Quadruped

[![Paper: TAR (IROS 2025)](https://img.shields.io/badge/Paper-arXiv:2503.20839-9cf.svg)](https://arxiv.org/abs/2503.20839)
[![Paper: TerAdapt (RA-L 2026)](https://img.shields.io/badge/Paper-IEEE%20RA--L%202026-blue.svg)](#)
[![Robot: Thunder (NOVA Dog)](https://img.shields.io/badge/Robot-Thunder%2016DOF-orange.svg)](#)
[![Isaac Lab 0.46](https://img.shields.io/badge/IsaacLab-0.46.2-lightgrey.svg)](https://github.com/isaac-sim/IsaacLab)

Training & comparison of **4 proprioception-only terrain adaptation architectures** for **Thunder** wheeled-legged quadruped (16 DOF = 12 leg joints + 4 wheel joints, deployed on Horizon S100P).

## Architectures

| ID | Task ID | Entry | Key Idea |
|---|---|---|---|
| **HIM** | `RobotLab-Isaac-Velocity-Rough-Thunder-Hist-v0` | `train_him.py` | SwAV contrastive on proprio-history latent (Long et al. 2024) |
| **TAR (simplified)** | `GaitTarRough` | `train_tar.py` | Classic triplet `clamp(d_pos - d_neg + margin)`, 45-dim latent. Our quick take on TAR paper abstract. |
| **TAR (official)** | `GaitTarOfficialRough` | `train_tar_official.py` | Paper-accurate port of `ammousa/TARLoco`. Contrastive-hinge loss, encoder_critic, 20-dim latent. |
| **TerAdapt** | `GaitTerRough` | `train_teradapt.py` | VQ-VAE 256-code terrain codebook, dual-horizon proprio (Short MLP + Long 1D CNN). |

> TAR ships two variants intentionally. "Simplified" is our from-scratch implementation written from the paper abstract — quick to iterate, classic triplet loss. "Official" is a line-by-line port of `ammousa/TARLoco` adapted to Thunder's 16 DOF. Running both and comparing against HIM and TerAdapt gives a 4-way architecture ablation. Full architectural diff: [`docs/TAR_MIGRATION.md`](docs/TAR_MIGRATION.md).

## Directory Layout

```
tar/
├── docs/
│   ├── TAR_MIGRATION.md              ← simplified-vs-official full diff
│   └── TAR_vs_PAPER.md               ← simplified-vs-paper diff
├── reference/
│   └── TARLoco/                      ← ammousa/TARLoco source (CC-BY-NC-SA 4.0)
├── scripts/reinforcement_learning/rsl_rl/
│   ├── modules/
│   │   ├── tar_actor_critic.py            (simplified)
│   │   ├── tar_official_actor_critic.py   (paper-accurate)
│   │   ├── teradapt_tca.py
│   │   └── teradapt_actor_critic.py
│   ├── storage/
│   │   ├── tar_rollout_storage.py
│   │   ├── tar_official_rollout_storage.py
│   │   └── teradapt_rollout_storage.py
│   ├── algorithms/
│   │   ├── tar_ppo.py
│   │   ├── tar_official_ppo.py
│   │   └── teradapt_ppo.py
│   ├── runners/
│   │   ├── tar_on_policy_runner.py
│   │   ├── tar_official_on_policy_runner.py
│   │   └── teradapt_on_policy_runner.py
│   ├── train_tar.py
│   ├── train_tar_official.py
│   └── train_teradapt.py
└── source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/thunder_him_gait/
    ├── __init__.py                    ← registers 4 gym tasks
    ├── him_rough_env_cfg.py
    ├── tar_rough_env_cfg.py
    ├── tar_official_rough_env_cfg.py
    ├── teradapt_rough_env_cfg.py
    └── agents/
        ├── rsl_rl_ppo_cfg.py          (HIM)
        ├── rsl_rl_tar_cfg.py
        ├── rsl_rl_tar_official_cfg.py
        └── rsl_rl_teradapt_cfg.py
```

## Training

All commands assume `thunder` conda env and `robot_lab` cwd.

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate thunder
cd robot_lab

# HIM baseline (existing framework)
CUDA_VISIBLE_DEVICES=0 python scripts/reinforcement_learning/rsl_rl/train_him.py \
    --task RobotLab-Isaac-Velocity-Rough-Thunder-Hist-v0 --num_envs 4096 --headless

# TAR simplified (our take)
CUDA_VISIBLE_DEVICES=1 python scripts/reinforcement_learning/rsl_rl/train_tar.py \
    --task GaitTarRough --num_envs 4096 --headless

# TAR official (paper-accurate)
CUDA_VISIBLE_DEVICES=2 python scripts/reinforcement_learning/rsl_rl/train_tar_official.py \
    --task GaitTarOfficialRough --num_envs 4096 --headless

# TerAdapt (VQ-VAE codebook)
CUDA_VISIBLE_DEVICES=3 python scripts/reinforcement_learning/rsl_rl/train_teradapt.py \
    --task GaitTerRough --num_envs 4096 --headless
```

**Common hyperparameters**: `num_envs=4096`, `num_steps_per_env=48`, `max_iterations=20000`, PPO with adaptive KL schedule (`desired_kl=0.01`), Adam optimizer. Each training takes ~8-12 hours on RTX 3090.

## Architecture Details

### HIM — Baseline
5-10 frame proprio history encoder + SwAV contrastive learning (Long et al. 2024). Uses existing Thunder HIM framework.

### TAR (simplified)
```
Teacher encoder: privileged_obs → z_T[45]
Student encoder: proprio_history[15×57=855] → z_S[45]
Dynamics model:  [z_S, action[16]] → z_S_next[45]
Vel estimator:   z_S → [vel(3), height(1), com_xy(2)] = 6 dims
Actor:  cat(current_obs[57], z_S.detach(), vel_est.detach())[108] → MLP[512,256,128] → action[16]
Critic: privileged_obs → MLP[512,256,128] → value

L_tar_simple = clamp(‖z_T − z_S_next‖² − ‖z_T − z_T_neg‖² + 0.5, 0).mean()
L_est = MSE(vel_est, gt_6dim)
```

### TAR (official, paper-accurate)
```
encoder_actor:   proprio_history[10×57=570] → MLP[256,128,64] → z_a[20]
encoder_critic:  critic_obs → MLP[256,128,64] → z_c[20]
trans:           [z_a, action] → MLP[32] → z_a_next[20]
vel_estimator:   cat(z_a, hist_short[4×57=228])[248] → MLP[64,32] → v̂[3]
Actor:  cat(prop_current[57], z_a.detach()[20], v̂.detach()[3])[80] → MLP[256,128,128] → action[16]
Critic: cat(prop_current[57], z_c[20], vel_priv[3])[80] → MLP[512,256,256] → value

L_tar_official = ‖next_z_c - trans(z_a, action)‖².sum().mean()         # pos
               + max(0, 1 - ‖next_z_c - next_neg_z‖²).mean()            # hinge neg
L_vel_mse = MSE(v̂, vel_priv)
# Negatives drawn from batch excluding same-env indices (mod num_envs).
```

### TerAdapt — VQ-VAE Codebook
```
TCA (teacher):
  height_scan[187] → MLP[64,32] → z_t[16]
  VQ Codebook (256 × 16, EMA decay=0.99) → quantize → indices[0..255], z_q
  MLP[32,64] → ĥ[187]                        # reconstruction

Student (dual horizon):
  short_hist[5×57]  → MLP[128,64]                → h_short[16]
  long_hist[50×57]  → 1D CNN[32,32,32, k=8,5,5]  → h_long[16]
  cat(h_short, h_long)[32] → MLP[64,32] → l_tilde[16]
  cat(h_short, h_long)[32] → MLP[64,32] → v_hat[3]
  l_tilde[16]       → MLP[64,128]        → logits[256]

Actor:  cat(obs[57], h_short[16], h_long[16], l_tilde[16], v_hat[3])[109] → MLP[512,256,128] → action[16]

L_vel  = MSE(v_hat, vel_gt)
L_tok  = CE(logits, indices.detach())
L_vq   = MSE(ĥ, h) + 0.25 · MSE(z, sg(z_q))
```

## Environment
- Isaac Sim 4.5 + Isaac Lab 0.46.2
- Python 3.10 · PyTorch 2.x · CUDA 12.x
- RSL-RL PPO base

## License
- Our code: **Apache 2.0** (Inovxio / 穹沛科技, 2024-2026)
- `reference/TARLoco/`: CC-BY-NC-SA 4.0 by Amr Mousa (University of Manchester)

## References
- [TAR paper (arXiv:2503.20839, IROS 2025)](https://arxiv.org/abs/2503.20839) — official repo: [ammousa/TARLoco](https://github.com/ammousa/TARLoco)
- [TerAdapt (IEEE RA-L, June 2026)](#)
- [HIMLoco (Long et al. 2024)](https://arxiv.org/abs/2312.11460)
