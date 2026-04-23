# TAR · Terrain-Adaptive RL for Thunder Quadruped

[![Paper: TAR (IROS 2025)](https://img.shields.io/badge/Paper-arXiv:2503.20839-9cf.svg)](https://arxiv.org/abs/2503.20839)
[![Paper: TerAdapt (RA-L 2026)](https://img.shields.io/badge/Paper-IEEE%20RA--L%202026-blue.svg)](#)
[![Robot: Thunder 16DOF](https://img.shields.io/badge/Robot-Thunder%2016DOF-orange.svg)](#)
[![Isaac Lab 0.46](https://img.shields.io/badge/IsaacLab-0.46.2-lightgrey.svg)](https://github.com/isaac-sim/IsaacLab)

Training and comparison of proprioception-only terrain adaptation methods for the Thunder wheeled-legged quadruped (16 DOF = 12 leg + 4 wheel).

## Current Status

| Method | Task ID | Entry | Status | Notes |
|---|---|---|---|---|
| **TAR (MLP)** | `GaitTarRough` | `train_tar.py` | Implemented | Fixed-window MLP TAR variant adapted from TARLoco. Default loss weights, PPO settings, latent size, and adaptive LR range are aligned to the current MLP stack in this repo. This is **not** the paper's recurrent TAR student. |
| **TerAdapt** | `GaitTerRough` | `train_teradapt.py` | Implemented | VQ-VAE terrain codebook + dual-horizon proprio encoder. |
| **HIM** | - | - | Not included | Earlier README revisions referenced `train_him.py`, `him_rough_env_cfg.py`, and `rsl_rl_ppo_cfg.py`, but those files are not present in this repository snapshot. |

## TAR Alignment

The current `GaitTarRough` implementation uses these defaults:

- Optimizer: `Adam`
- Adaptive learning rate clamp: `5e-5 ~ 1e-3`
- `gamma=0.99`, `lambda=0.95`, `desired_kl=0.01`
- `num_learning_epochs=5`, `num_mini_batches=4`
- `latent_dims=45`
- actor/critic hidden dims: `[512, 256, 128]`
- dynamics hidden dims: `[64]`
- activation: `ELU`
- history lengths: `num_hist=10`, `num_hist_short=4`

Important scope note:

- The current TAR implementation is MLP-based.
- The recurrent `LSTMEnc [256]` variant from the paper's main model is not implemented in this repo yet.
- `train_tar.py` now defaults to `latent_dims=45` so the CLI matches the implemented config.

### TAR Network (current repo)

```text
encoder_actor:   proprio_history[10x57=570] -> MLP[256,128,64] -> z_a[45]
encoder_critic:  critic_obs                 -> MLP[256,128,64] -> z_c[45]
trans:           [z_a, action[16]]          -> MLP[64]         -> z_a_next[45]
vel_estimator:   cat(z_a, hist_short[4x57=228])[273] -> MLP[64,32] -> v_hat[3]
Actor:  cat(prop[57], z_a.detach()[45], v_hat.detach()[3])[105] -> MLP[512,256,128] -> action[16]
Critic: cat(prop[57], z_c[45], vel_priv[3])[105]                -> MLP[512,256,128] -> value
```

## TerAdapt Alignment

`GaitTerRough` uses its own config in `rsl_rl_teradapt_cfg.py`. The current code defaults are:

- `num_steps_per_env=48`
- `num_learning_epochs=20`
- `num_mini_batches=16`
- `entropy_coef=0.005`
- `gamma=0.99`, `lambda=0.95`, `desired_kl=0.01`
- `max_iterations=20000`

### TerAdapt Network (current repo)

```text
TCA (teacher):
  height_scan[187] -> MLP[64,32] -> z_t[16]
  VQ Codebook (256 x 16, EMA decay=0.99) -> quantize -> indices[0..255], z_q
  MLP[32,64] -> h_hat[187]

Student (dual horizon):
  short_hist[5x57]  -> MLP[128,64]               -> h_short[16]
  long_hist[50x57]  -> 1D CNN[32,32,32, k=8,5,5] -> h_long[16]
  cat(h_short, h_long)[32] -> MLP[64,32] -> l_tilde[16]
  cat(h_short, h_long)[32] -> MLP[64,32] -> v_hat[3]
  l_tilde[16] -> MLP[64,128] -> logits[256]

Actor: cat(obs[57], h_short[16], h_long[16], l_tilde[16], v_hat[3])[109] -> MLP[512,256,128] -> action[16]
```

## Directory Layout

```text
tar/
├── reference/
│   └── TARLoco/                      <- upstream TARLoco reference source
├── scripts/reinforcement_learning/rsl_rl/
│   ├── modules/
│   │   ├── tar_actor_critic.py
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
    ├── __init__.py
    ├── tar_rough_env_cfg.py
    ├── teradapt_rough_env_cfg.py
    └── agents/
        ├── rsl_rl_tar_cfg.py
        └── rsl_rl_teradapt_cfg.py
```

## Training

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate thunder
cd robot_lab

# TAR (current MLP variant)
CUDA_VISIBLE_DEVICES=0 python scripts/reinforcement_learning/rsl_rl/train_tar.py \
    --task GaitTarRough --num_envs 4096 --headless

# TerAdapt
CUDA_VISIBLE_DEVICES=1 python scripts/reinforcement_learning/rsl_rl/train_teradapt.py \
    --task GaitTerRough --num_envs 4096 --headless
```

## Known Gaps

- The HIM baseline is not included in this repository snapshot.
- The recurrent TAR student with `LSTMEnc [256]` is not implemented yet.
- No public evaluation/export/deployment pipeline is included in this repo yet.

## Environment

- Isaac Sim 4.5 + Isaac Lab 0.46.2
- Python 3.10
- PyTorch 2.x
- CUDA 12.x
- RSL-RL PPO base

## References

- [TAR paper (arXiv:2503.20839, IROS 2025)](https://arxiv.org/abs/2503.20839) and upstream code [ammousa/TARLoco](https://github.com/ammousa/TARLoco)
- [TerAdapt (IEEE RA-L, June 2026)](#)
- [HIMLoco (Long et al. 2024)](https://arxiv.org/abs/2312.11460)
