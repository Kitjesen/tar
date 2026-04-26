# TAR · Terrain-Adaptive RL for Thunder Quadruped

[![Paper: HIM (RSS 2024)](https://img.shields.io/badge/Paper-arXiv:2312.11460-9cf.svg)](https://arxiv.org/abs/2312.11460)
[![Paper: TAR (IROS 2025)](https://img.shields.io/badge/Paper-arXiv:2503.20839-9cf.svg)](https://arxiv.org/abs/2503.20839)
[![Paper: TerAdapt (RA-L 2026)](https://img.shields.io/badge/Paper-IEEE%20RA--L%202026-blue.svg)](#)
[![Robot: Thunder 16DOF](https://img.shields.io/badge/Robot-Thunder%2016DOF-orange.svg)](#)
[![Isaac Lab 0.46](https://img.shields.io/badge/IsaacLab-0.46.2-lightgrey.svg)](https://github.com/isaac-sim/IsaacLab)

Training and comparison of proprioception-only terrain adaptation methods for the Thunder wheeled-legged quadruped (16 DOF = 12 leg + 4 wheel).

## Architectures

| ID | Task ID | Entry | Key Idea |
|---|---|---|---|
| **HIM** | `GaitHimRough` | `train_him.py` | SwAV contrastive on proprio history (Long et al. 2024) |
| **TAR** | `GaitTarRough` | `train_tar.py` | TARLoco port — contrastive hinge + encoder_critic + num_hist_short=4 (Mousa et al. 2025) |
| **TerAdapt** | `GaitTerRough` | `train_teradapt.py` | VQ-VAE 256-code terrain codebook + dual-horizon proprio (Short MLP + Long 1D CNN) |

## Paper Alignment

Each architecture has its own observation layout, training stages, and hyperparameters drawn directly from the corresponding paper. Nothing is shared across methods.

---

### HIM — Long et al. 2024 · [HIMLoco (arXiv:2312.11460)](https://arxiv.org/abs/2312.11460)

| | |
|---|---|
| **Task** | `GaitHimRough` |
| **Entry** | `train_him.py` |
| **Policy obs** | 5-frame history · 57 dims/frame · [base_ang_vel, projected_gravity, velocity_commands, joint_pos, joint_vel, last_action] |
| **Critic obs** | 5-frame history · above + base_lin_vel(3) + height_scan(187) + contacts + friction + mass |
| **Loss** | PPO + SwAV contrastive on latent + velocity MSE (concurrent) |
| **Stage** | Single-stage end-to-end PPO |
| **PPO** | `num_steps_per_env=48`, `learning_epochs=5`, `mini_batches=4`, `entropy_coef=0.01` |
| **γ / λ / KL** | 0.99 / 0.95 / 0.01 |
| **max_iterations** | 20000 |
| **Terrain** | Thunder native rough |

---

### TAR — Mousa et al. 2025 · [IROS 2025 (arXiv:2503.20839)](https://arxiv.org/abs/2503.20839) · [ammousa/TARLoco](https://github.com/ammousa/TARLoco)

| | |
|---|---|
| **Task** | `GaitTarRough` |
| **Entry** | `train_tar.py` |
| **Policy obs** | 10-frame history · 57 dims/frame (proprio only, vel and height_scan excluded) |
| **Critic obs** | single frame · `[0:3]=base_lin_vel`, `[3:60]=proprio(57)`, `[60:]=height_scan+contacts+friction+mass` |
| **Loss** | PPO + TAR contrastive (pos² + hinge neg) + vel MSE (concurrent) |
| **Triplet anchor** | `next_z_c` = `encoder_critic(next_critic_obs)` |
| **Triplet positive** | `trans(z_a, action)` (dynamics-predicted next latent) |
| **Triplet negative** | batch-shuffled `next_z_c` excluding same-env indices |
| **Stage** | Single-stage end-to-end PPO (no teacher pretraining) |
| **PPO** | `num_steps_per_env=24`, `learning_epochs=5`, `mini_batches=4`, `entropy_coef=0.01` |
| **γ / λ / KL** | 0.99 / 0.95 / 0.01 |
| **Latent / hidden** | `latent_dims=45`, actor/critic `[512, 256, 128]`, dynamics `[64]` |
| **Other** | `empirical_normalization=True`, Adam, `lr_max=1e-3` |
| **max_iterations** | 1500 (TARLoco official Go1 default) |

> **Note**: current TAR implementation is MLP-based; the recurrent `LSTMEnc[256]` variant is not yet implemented.

#### TAR Network (current repo)

```text
encoder_actor:   proprio_history[10x57=570] -> MLP[256,128,64] -> z_a[45]
encoder_critic:  critic_obs                 -> MLP[256,128,64] -> z_c[45]
trans:           [z_a, action[16]]          -> MLP[64]         -> z_a_next[45]
vel_estimator:   cat(z_a, hist_short[4x57=228])[273] -> MLP[64,32] -> v_hat[3]
Actor:  cat(prop[57], z_a.detach()[45], v_hat.detach()[3])[105] -> MLP[512,256,128] -> action[16]
Critic: cat(prop[57], z_c[45], vel_priv[3])[105]                -> MLP[512,256,128] -> value
```

---

### TerAdapt — 2026 · IEEE RA-L (VQ-VAE Codebook Alignment)

| | |
|---|---|
| **Task** | `GaitTerRough` |
| **Entry** | `train_teradapt.py` |
| **Policy obs (short)** | 5-frame history, 57 dims/frame = 285 → `Short Encoder MLP[128,64] → h_short[16]` |
| **Policy obs (long)** | 50-frame history, 57 dims/frame = 2850 → reshape [B, 57, 50] → `Long 1D CNN[32,32,32; k=8,5,5] → h_long[16]` |
| **Critic obs** | 1 frame, full privileged (77 critic + 187 height_scan = 264 dims) |
| **Teacher input** | `height_scan_group` (187 dims) → `Terrain Encoder MLP[64,32] → z_t[16]` → VQ codebook |
| **GT supervision** | `vel_gt` obs group (base_lin_vel, 3 dims, clean no noise) |
| **VQ codebook** | 256 codes × 16 dims, **EMA decay 0.99** (no gradient through codebook) |
| **Loss** | `PPO + L_vel + L_tok + L_vq` all concurrent |
| **Stage** | Single-stage end-to-end PPO with 3 aux losses |
| **PPO** | `num_steps_per_env=48`, `learning_epochs=20`, `mini_batches=16`, `entropy_coef=0.005` |
| **γ / λ / KL** | 0.99 / 0.95 / 0.01 |
| **max_iterations** | 20000 |
| **Terrain** | Thunder native rough (10 rows × 20 cols, slope range up to 0.6 rad / ~34°) |

#### TerAdapt Reward Design (v9 — grouped joint penalty + fall termination)

| Reward | Running weight | Static (cmd≈0) effective | Notes |
|---|---|---|---|
| `track_lin_vel_xy_exp` | +8.0 (std=√2) | — | early gradient signal |
| `track_ang_vel_z_exp` | +3.0 (std=√2) | — | early gradient signal |
| `upward` | +2.0 | — | upright maintenance |
| `joint_pos_penalty_hip` | -1.0 | -5.0 (×5) | suppress body twist |
| `joint_pos_penalty_thigh` | -0.3 | -5.1 (×17) | allow gait swing |
| `joint_pos_penalty_calf` | -0.1 | -5.0 (×50) | allow stair flexion |
| `foot_vel_motion_aware` | -0.05 | -5.0 (×100) | wheels static when not commanded |
| `undesired_contacts` | -3.0 | — | encourage emergent leg lift |
| `feet_stumble` | -5.0 | — | |
| `lin_vel_z_l2` | -2.0 | — | |
| `joint_pos_limits` | -3.0 | — | loose bounds for exploration |

**Termination**: `fall_after_stood_up` — only triggers if base_z < 0.20m or grav_b[z] > 0.5 *after* the robot has been stably upright (base_z > 0.30m AND grav_b[z] < -0.7) for ≥20 consecutive steps. Avoids false termination from initial random-pose drop.

---

## Directory Layout

```
tar/
├── docs/
├── reference/
│   └── TARLoco/                          ← ammousa/TARLoco source (CC-BY-NC-SA 4.0)
├── scripts/
│   ├── him/                              ← HIM algorithm modules
│   │   ├── him_actor_critic.py
│   │   ├── him_estimator.py
│   │   ├── him_ppo.py
│   │   ├── him_on_policy_runner.py
│   │   ├── him_rollout_storage.py
│   │   └── utils/
│   │       ├── observation_reshaper.py
│   │       └── export_him_policy.py
│   ├── tar/                              ← TAR algorithm modules
│   │   ├── tar_actor_critic.py
│   │   ├── tar_ppo.py
│   │   ├── tar_on_policy_runner.py
│   │   └── tar_rollout_storage.py
│   ├── teradapt/                         ← TerAdapt algorithm modules
│   │   ├── teradapt_actor_critic.py
│   │   ├── teradapt_tca.py
│   │   ├── teradapt_ppo.py
│   │   ├── teradapt_on_policy_runner.py
│   │   └── teradapt_rollout_storage.py
│   ├── train_him.py                      ← HIM training entry
│   ├── train_tar.py                      ← TAR training entry
│   ├── train_teradapt.py                 ← TerAdapt training entry
│   ├── play_him.py
│   ├── play_teradapt.py
│   └── teradapt_offline_test.py
├── robot_lab/
│   └── robot_lab/tasks/manager_based/locomotion/velocity/
│       ├── mdp/
│       │   ├── events.py                 ← reset_fall_after_stood_up_state
│       │   └── terminations.py           ← fall_after_stood_up
│       └── config/wheeled/
│           ├── thunder_gait/
│           │   └── rewards.py            ← wheel_vel_motion_aware, joint_pos_penalty_no_fall_filter
│           └── thunder_him_gait/
│               ├── __init__.py           ← gym.register for HIM/TAR/TerAdapt
│               ├── him_rough_env_cfg.py
│               ├── tar_rough_env_cfg.py
│               ├── teradapt_rough_env_cfg.py
│               └── agents/
│                   ├── rsl_rl_ppo_cfg.py
│                   ├── rsl_rl_tar_cfg.py
│                   └── rsl_rl_teradapt_cfg.py
└── README.md
```

## Quick Start

```bash
# HIM
python scripts/train_him.py --task GaitHimRough --num_envs 4096 --headless

# TAR
python scripts/train_tar.py --task GaitTarRough --num_envs 4096 --headless

# TerAdapt
python scripts/train_teradapt.py --task GaitTerRough --num_envs 4096 --headless
```

Resume training:
```bash
python scripts/train_teradapt.py --task GaitTerRough --num_envs 4096 --headless \
  --resume --load_run <run_name> --checkpoint model_<iter>.pt
```

Record video (single env follow camera or multi-env overhead):
```bash
python scripts/play_teradapt.py --task GaitTerRough --num_envs 1 --headless \
  --video --video_length 1500 --enable_cameras \
  --load_run <run_name> --checkpoint model_<iter>.pt \
  --experiment_name thunder_gait_teradapt_rough
```

## License

Project code: see repository LICENSE.
Reference TARLoco source under `reference/TARLoco/` retains its original CC-BY-NC-SA 4.0 license.
