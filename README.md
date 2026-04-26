# Terrain-Adaptive RL


## Architectures

| ID | Task ID | Entry | Key Idea |
|---|---|---|---|
| **HIM** | `RobotLab-Isaac-Velocity-Rough-Thunder-Hist-v0` | `train_him.py` | SwAV contrastive on proprio history (Long et al. 2024) |
| **TAR** | `GaitTarRough` | `train_tar.py` | Paper-accurate port of `ammousa/TARLoco` — contrastive hinge + encoder_critic + num_hist_short=4 |
| **TerAdapt** | `GaitTerRough` | `train_teradapt.py` | VQ-VAE 256-code terrain codebook + dual-horizon proprio (Short MLP + Long 1D CNN) |

TAR uses the official `ammousa/TARLoco` architecture line-by-line; only Thunder's 16-DOF observation slicing is adapted. Full reference source kept in `reference/TARLoco/`.

## Paper Alignment

Every architecture has **its own** obs layout / training stages / hyperparameters taken from that paper's actual config. Nothing copied across.

---

### HIM — Long et al. 2024 · [HIMLoco (arXiv:2312.11460)](https://arxiv.org/abs/2312.11460)

| | |
|---|---|
| **Task** | `RobotLab-Isaac-Velocity-Rough-Thunder-Hist-v0` (Thunder's existing `thunder_hist`) |
| **Entry** | `train_him.py` |
| **Policy obs** | 5-frame history · 57 dims/frame · [base_ang_vel, projected_gravity, velocity_commands, joint_pos, joint_vel, last_action] |
| **Critic obs** | 5-frame history · above + base_lin_vel(3) + height_scan(187) + contacts + friction + mass |
| **Loss** | PPO + SwAV contrastive on latent + velocity MSE (all concurrent) |
| **Stage** | Single-stage end-to-end PPO |
| **PPO** | `num_steps_per_env=48`, `learning_epochs=5`, `mini_batches=4`, `entropy_coef=0.01` |
| **γ / λ / KL** | 0.99 / 0.95 / 0.01 |
| **max_iterations** | 20000 |
| **Terrain** | Thunder native rough (inherited from `thunder_hist`) |

---

### TAR — Mousa et al. 2025 · [IROS 2025 (arXiv:2503.20839)](https://arxiv.org/abs/2503.20839) · [ammousa/TARLoco](https://github.com/ammousa/TARLoco)

| | |
|---|---|
| **Task** | `GaitTarRough` |
| **Entry** | `train_tar.py` |
| **Policy obs** | 10-frame history · 57 dims/frame · [base_ang_vel, projected_gravity, velocity_commands, joint_pos, joint_vel, last_action]. **Vel and height_scan excluded** (proprio only) |
| **Critic obs** | **single frame** · `[0:3]=base_lin_vel`, `[3:60]=proprio(57)`, `[60:]=height_scan+contacts+friction+mass` (layout matters for `extract_critic`) |
| **Loss** | PPO + TAR contrastive (pos² + hinge neg) + vel MSE (concurrent, single stage) |
| **Triplet anchor** | `next_z_c` = `encoder_critic(next_critic_obs)` |
| **Triplet positive** | `trans(z_a, action)` (dynamics-predicted next latent) |
| **Triplet negative** | batch-shuffled `next_z_c` excluding same-env indices |
| **Stage** | Single-stage end-to-end PPO (no teacher pretraining) |
| **PPO** | `num_steps_per_env=24`, `learning_epochs=5`, `mini_batches=4`, `entropy_coef=0.01` (TARLoco `tar_algo_cfg`) |
| **γ / λ / KL** | 0.99 / 0.95 / 0.01 |
| **Other** | `empirical_normalization=True` (obs whitening), Adam, `lr_max=1e-3` |
| **max_iterations** | 200 (smoke); TARLoco official Go1 = 1500 |
| **DR** | friction [0.1, 3.0], payload [-2, 10] kg (paper) |
| **Terrain** | Thunder native rough (paper used `ROUGH_TERRAINS_CFG` with 50% railway tracks — unsuitable for wheeled-legged; keep Thunder's terrain) |

---

### TerAdapt — 2026 · [IEEE RA-L (VQ-VAE Codebook Alignment)](#)

| | |
|---|---|
| **Task** | `GaitTerRough` |
| **Entry** | `train_teradapt.py` |
| **Policy obs (short)** | 5-frame history, 57 dims/frame = 285 → `Short Encoder MLP[128,64] → h_short[16]` |
| **Policy obs (long)** | 50-frame history, 57 dims/frame = 2850 → reshape [B, 57, 50] → `Long 1D CNN[32,32,32; k=8,5,5] → h_long[16]` |
| **Critic obs** | 1 frame, full privileged (77 critic + 187 height_scan = 264 dims) |
| **Teacher input** | `height_scan_group` (187 dims, single frame) → `Terrain Encoder MLP[64,32] → z_t[16]` → VQ codebook |
| **GT supervision** | `vel_gt` obs group (base_lin_vel, 3 dims, clean no noise) |
| **VQ codebook** | 256 codes × 16 dims, **EMA decay 0.99** (no gradient through codebook) |
| **Loss** | `PPO + L_vel + L_tok + L_vq` all concurrent: <br> `L_vel = MSE(v̂, vel_gt)` <br> `L_tok = CE(student logits[256], teacher VQ indices.detach())` <br> `L_vq = MSE(ĥ, h) + 0.25·MSE(z, sg(z_q))` |
| **Stage** | Single-stage end-to-end PPO with 3 aux losses |
| **PPO** | `num_steps_per_env=48`, `learning_epochs=5`, `mini_batches=4`, `entropy_coef=0.005` |
| **γ / λ / KL** | 0.99 / 0.95 / 0.01 |
| **max_iterations** | 20000 |
| **Terrain** | Thunder native rough |

## Directory Layout

```
tar/
├── reference/
│   └── TARLoco/                      ← ammousa/TARLoco source (CC-BY-NC-SA 4.0)
├── scripts/
│   ├── train_tar.py
│   ├── train_teradapt.py
│   ├── tar/
│   │   ├── tar_actor_critic.py              (paper-accurate TARLoco port)
│   │   ├── tar_ppo.py
│   │   ├── tar_on_policy_runner.py
│   │   └── tar_rollout_storage.py
│   └── teradapt/
│       ├── teradapt_actor_critic.py
│       ├── teradapt_tca.py
│       ├── teradapt_ppo.py
│       ├── teradapt_on_policy_runner.py
│       └── teradapt_rollout_storage.py
└── robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/thunder_him_gait/
    ├── __init__.py                    ← registers 3 gym tasks
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

# TAR (paper-accurate)
CUDA_VISIBLE_DEVICES=0 python scripts/train_tar.py \
    --task GaitTarRough --num_envs 4096 --headless

# TerAdapt (VQ-VAE)
CUDA_VISIBLE_DEVICES=1 python scripts/train_teradapt.py \
    --task GaitTerRough --num_envs 4096 --headless
```

Each training takes ~8-12 hours on RTX 3090 (4096 envs; TAR 200-1500 iter for smoke/full, HIM/TerAdapt 20000 iter baseline).

## Network Architecture (detailed)

### TAR — Paper-accurate TARLoco port
```
encoder_actor:   proprio_history[10×57=570] → MLP[256,128,64] → z_a[45]
encoder_critic:  critic_obs              → MLP[256,128,64] → z_c[45]
trans:           [z_a, action[16]]       → MLP[64]         → z_a_next[45]
vel_estimator:   cat(z_a, hist_short[4×57=228])[273] → MLP[64,32] → v̂[3]
Actor:  cat(prop[57], z_a.detach()[45], v̂.detach()[3])[105] → MLP[512,256,128] → action[16]
Critic: cat(prop[57], z_c[45], vel_priv[3])[105]            → MLP[512,256,128] → value

Losses:
  L_tar = ‖next_z_c − trans(z_a, a)‖²_sum.mean()            # positive: MSE dist
        + max(0, 1 − ‖next_z_c − next_neg_z‖²_sum).mean()   # hinge negative
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
