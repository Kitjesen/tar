# TAR 实现 vs 论文对照

**论文**: Teacher-Aligned Representations via Contrastive Learning for Quadrupedal Locomotion (arXiv:2503.20839, IROS 2025)

## 匹配的部分

| 维度 | 论文 | 我们 | 状态 |
|---|---|---|---|
| z_dim | 45 | 45 | ✅ |
| History frames | 15 | 15 | ✅ |
| Triplet formula | `clamp(d_pos - d_neg + α, 0).mean()` | 一致 | ✅ |
| Positive sample | Dynamics-predicted Z_S_next | Z_S_next | ✅ |
| Actor input | `π(a_t \| o_t, Z_S, v̂_t)` | `cat(current_obs, Z_S, vel_est)` | ✅ |
| Actor gradient stop to encoder | 明确要求 | `Z_S.detach()` + `vel_est.detach()` | ✅ |
| Actor hidden | [512, 256, 128] | [512, 256, 128] | ✅ |
| Critic hidden | [512, 256, 128] | [512, 256, 128] | ✅ |
| Dynamics hidden | [64] 单层 | [64] | ✅ |
| Triplet coefficient | 1.0 | 1.0 | ✅ |
| Activation | ELU | ELU | ✅ |
| Discount γ | 0.99 | 0.99 | ✅ |
| GAE λ | 0.95 | 0.95 | ✅ |

## 主动扩展（偏离论文）

### 1. Velocity Estimator 输出维度
- **论文**: `v̂_t ∈ R³`（仅 base linear velocity）
- **我们**: `R⁶ = [base_lin_vel(3), base_pos_z(1), com_pos_xy(2)]`
- **理由**: 让 latent 通过额外监督学习更多物理信息（height + CoM）
- **风险**: 额外监督可能干扰纯速度估计
- **严格复现**: 设 `num_estimator_targets=3` 并注释掉 env cfg 的 height/com_xy 项

### 2. PPO 超参更激进
| 参数 | 论文 | 我们 |
|---|---|---|
| num_learning_epochs | 5 | 20 |
| num_mini_batches | 4 | 16 |
| entropy_coef | 0.01 | 0.005 |

- **理由**: 沿用 Thunder 既有 HIM 训练的 PPO 配置
- **风险**: 更激进的更新可能提早饱和，但 adaptive KL schedule 会压 LR 补偿

## 实现细节（等价但形式不同）

### Negative sampling
- **论文原文**: "negatives drawn from parallel environments with different domain randomization parameters"
- **我们**: `Z_T[torch.randperm(batch_size)]`
- **说明**: 并行环境每个 env 独立采样 DR 参数；batch shuffle 等于从不同 DR 的环境里取负样本。功能等价。

## 论文未指定项

### Margin α
论文 Equation 中有 `α` 符号但未给数值。我们默认 `triplet_margin=0.5`（triplet loss 标准默认）。
**建议**: sweep `{0.1, 0.3, 0.5, 1.0}` 看效果。

### Teacher Encoder 架构
论文没给出 teacher encoder 的 hidden dims。我们用 `MLP[256, 128] → 45`。
Consideration: teacher 比 student 轻（student 用 [512,256,128]），符合"teacher 只处理已知特权信息，不需要深网络"的直觉。

### Student Encoder 架构（MLP variant）
论文主要展示 TCN variant（`channels=[32,32,32], kernels=[8,5,5], strides=[4,1,1]`），MLP variant 仅提及无具体 dims。我们用 `MLP[512, 256, 128] → 45`。

## Thunder 适配

| 维度 | Go2 (论文) | Thunder (我们) |
|---|---|---|
| DOF | 12 | 16（12 腿 + 4 轮） |
| obs_dim per frame | 45 | 57 |
| action_dim | 12 | 16 |
| Actor input dim | 93 | 109 |
| Critic obs | 199 privileged | 264 (critic group 77 + height_scan 187) |

所有维度按 Thunder 实际 DOF 和 obs 自适应调整。
