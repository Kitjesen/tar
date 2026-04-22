# NOVA Dog RL

Terrain-adaptive reinforcement learning for **NOVA Dog** (product name) / **Thunder** (internal codename) wheeled-legged quadruped — 16 DOF = 12 leg joints + 4 wheel joints. Three architectures compared, all proprioception-only at deployment (deployed on Horizon S100P board).

## Architectures

### ① HIM — History + SwAV Contrastive
Baseline (existing). 5-10 frame proprioceptive history encoder + SwAV contrastive learning on latent.

- **Paper**: HIMLoco (Long et al., 2024)
- **Task ID**: `RobotLab-Isaac-Velocity-Rough-Thunder-Hist-v0`
- **Entry**: `train_him.py`

### ② TAR — Teacher-Aligned Representations
Contrastive-hinge loss aligning student proprioceptive latent to teacher privileged latent via dynamics-predicted next state.

> **⚠️ 状态**: 当前 `tar_*.py` 是根据论文摘要手写的**简化版**（`latent_dims=45`，classic triplet loss）。官方 `ammousa/TARLoco` 实现在 `reference/TARLoco/` 目录下，关键差异见 `docs/TAR_MIGRATION.md`。需要重写以严格复现论文。

- **Paper**: Teacher-Aligned Representations via Contrastive Learning for Quadrupedal Locomotion (arXiv:2503.20839, IROS 2025)
- **Reported**: -42.2% OOD error, 2× convergence speed vs HIM (on Go2)
- **Task ID**: `GaitTarRough`
- **Entry**: `train_tar.py`

Architecture:
```
Teacher encoder: privileged_obs → z_T[45]
Student encoder: proprio_history[15×57=855] → z_S[45]
Dynamics model:  [z_S, action[16]] → z_S_next[45]
Vel estimator:   z_S → [vel(3), height(1), com_xy(2)] = 6 dims
Actor:           cat(current_obs[57], z_S[45], vel_est[6]) = 108 → MLP → action[16]
Critic:          privileged_obs → MLP → value

Losses:
  L_ppo + c_val·L_value
  + c_triplet · mean(clamp(‖z_T − z_S_next‖² − ‖z_T − z_T_neg‖² + margin, 0))
  + c_est · MSE(vel_est, gt_targets)
```

### ③ TerAdapt — VQ-VAE Terrain Codebook
Learn 256-code discrete terrain vocabulary from height_scan via VQ-VAE, supervise proprio latent via cross-entropy over token indices.

- **Paper**: Proprioceptive Terrain-Adaptive Locomotion via Codebook Aligned Representation Learning (TerAdapt, IEEE RA-L, June 2026)
- **Task ID**: `GaitTerRough`
- **Entry**: `train_teradapt.py`

Architecture:
```
TCA (teacher):
  height_scan[187] → MLP[64,32] → z_t[16]
  VQ Codebook 256 × 16 → quantize → indices, z_q
  MLP[32,64] → ĥ[187]

Student (dual horizon):
  short_hist[5×57] → MLP[128,64] → h_short[16]
  long_hist[50×57] → 1D CNN[32,32,32, k=8,5,5, s=4,1,1] → h_long[16]
  cat(h_short, h_long)[32] → MLP[64,32] → l_tilde[16]
  cat(h_short, h_long)[32] → MLP[64,32] → v_hat[3]
  l_tilde → MLP[64,128] → logits[256]

Actor:  cat(obs[57], h_short, h_long, l_tilde, v_hat)[109] → MLP[512,256,128] → action[16]
Critic: full_privileged → MLP[512,256,128] → value

Losses:
  L_ppo + c_val·L_value
  + c_vel · MSE(v_hat, vel_gt)
  + c_tok · CE(logits, indices.detach())
  + c_vq · (MSE(ĥ, h) + 0.25·MSE(z, sg(z_q)))
  Codebook updated via EMA (decay=0.99)
```

## Directory Layout

```
thunder-adaptive-rl/
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
├── tasks/.../thunder_him_gait/
│   ├── tar_rough_env_cfg.py
│   ├── teradapt_rough_env_cfg.py
│   ├── agents/rsl_rl_tar_cfg.py
│   ├── agents/rsl_rl_teradapt_cfg.py
│   └── __init__.py
└── README.md
```

## Training

### TAR
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate thunder
cd robot_lab
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python scripts/reinforcement_learning/rsl_rl/train_tar.py \
    --task GaitTarRough --num_envs 4096 --headless
```

### TerAdapt
```bash
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python scripts/reinforcement_learning/rsl_rl/train_teradapt.py \
    --task GaitTerRough --num_envs 4096 --headless
```

### Hyperparameters (both)
- `num_envs=4096`, `num_steps_per_env=48`
- `max_iterations=20000`
- PPO: `entropy_coef=0.005`, `clip_param=0.2`, `lr=1e-3` (adaptive KL), `gamma=0.99`, `lam=0.95`
- Optimizer: Adam

## Environment
- Isaac Sim 4.5 + Isaac Lab 0.46.2
- Python 3.10 + PyTorch 2.x + CUDA 12.x
- RSL-RL for PPO

## License
Apache 2.0 — Inovxio (穹沛科技) 2024-2026

## References
- [TAR paper (arXiv:2503.20839)](https://arxiv.org/abs/2503.20839)
- [TerAdapt paper (IEEE RA-L, 2026)](https://ieeexplore.ieee.org/document/TerAdapt) (to be updated)
- [HIM paper (Long et al., 2024)](https://arxiv.org/abs/2312.11460)
