# TAR 迁移到官方实现

## 背景

初版 `scripts/reinforcement_learning/rsl_rl/modules/tar_actor_critic.py` 是根据论文摘要手写的 TAR 简化版，**与官方 TARLoco 实现有多处关键差异**：

| 维度 | 官方 | 我们简化版 | 影响 |
|---|---|---|---|
| latent_dims | **20** | 45 | 网络容量差 2.25× |
| Anchor | **next_z_c**（下一帧 critic encoder）| 当前 Z_T | 时序对齐错位 |
| Loss 公式 | `pos² + max(0, 1-d_neg)` (contrastive hinge) | `clamp(d_pos - d_neg + margin, 0)` (triplet) | 不同的 objective |
| Negative 采样 | 排除同 env 不同时间步（模乘 num_envs） | 简单 batch shuffle | 可能拉近同 env 样本 |
| Critic encoder | **有** `encoder_critic` 把 privileged 压成 z | 直接 privileged → MLP | 缺一级抽象 |
| num_hist_short | 4（给 vel 用） | 无 | vel 信号弱 |
| vel_estimator 输入 | `cat(z, short_history)` | `z` | 少了近期 proprio |
| actor hidden | [256, 128, 128] | [512, 256, 128] | 参数量不同 |
| encoder hidden | [256, 128, 64] | [512, 256, 128] | — |
| dynamics hidden | [32] | [64] | — |
| 变体 | MLP / RNN / TCN 三种 | 仅 MLP | 少两个对比点 |

简化版 paper 复现度大约 50%，**不能声称是 TAR 的对标实验**。

## 方案

### 已做
1. 克隆官方 TARLoco 到 `reference/TARLoco/`（参考源码，MIT + CC-BY-NC-SA 4.0 License）
2. 保留简化版在 `scripts/reinforcement_learning/rsl_rl/modules/tar_actor_critic.py`（标记为 TAR-inspired baseline）

### 待做
1. **重写 tar_actor_critic.py 严格对标官方 TARLoco 架构**：
   - 复制 `reference/TARLoco/exts/tarloco/learning/modules/ac_tar.py` 的 ActorCriticTar
   - 复制 `reference/TARLoco/exts/tarloco/learning/algorithms/tar_ppo.py` 的 PPOTAR
   - 基类 `ActorCriticMlpSlrDblEnc` 依赖，需一并复制（从 `ac_slr.py`）
2. **Thunder 适配**：
   - obs_dim: 45 (Go2) → 57 (Thunder)
   - action_dim: 12 → 16
   - critic_obs extract: 官方 `obs[..., 3:48]` 是 Go2 proprio 切片，Thunder 的 proprio 切片要重新算
3. **重建 env cfg `tar_rough_env_cfg.py`**：
   - policy history_length = num_hist（官方 config 里查）
   - critic_obs 包含 vel(3) + proprio(45) 再加其他 privileged

## 简化版保留用途

保留 `tar_actor_critic.py` 简化版有两个用：
1. **消融实验**：对比 "标准 triplet vs 官方 contrastive hinge" 哪个对 Thunder 更好
2. **快速 baseline**：简化版架构更小，训练更快，可作为 fast iteration baseline

建议重命名为 `tar_simple_actor_critic.py` + 任务名 `GaitTarSimpleRough`，与严格版分开对比。

## 参考文件

- `reference/TARLoco/exts/tarloco/learning/modules/ac_tar.py`（ActorCriticTar 主类）
- `reference/TARLoco/exts/tarloco/learning/modules/ac_slr.py`（父类 ActorCriticMlpSlrDblEnc）
- `reference/TARLoco/exts/tarloco/learning/algorithms/tar_ppo.py`（PPOTAR loss）
- `reference/TARLoco/exts/tarloco/tasks/`（Go1 任务配置参考）
