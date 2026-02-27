# πRL 论文与代码对应关系详细说明

> 本文档结合论文 *"πRL: Online RL Fine-tuning for Flow-based Vision-Language-Action Models"* 详细说明 RLinf 代码库中各模块与论文原理的对应关系，为后续实现 FlowIPO 算法提供参考。

---

## 1. 整体架构概览

### 1.1 论文 Training Paradigm（图1 & 图2）

论文的训练流程为：**Pretraining → SFT → RL**，RL 阶段核心包括：
- **Policy Rollout**: Flow VLA 与环境交互，收集 trajectory
- **Actor Update**: 基于 PPO/GRPO 更新策略

### 1.2 代码架构映射

| 论文概念 | 代码路径 | 说明 |
|---------|---------|------|
| 整体训练循环 | `rlinf/runners/embodied_runner.py` → `EmbodiedRunner` | 编排 rollout → advantage → actor update 的主循环 |
| Policy Rollout | `rlinf/workers/rollout/hf/huggingface_worker.py` → `MultiStepRolloutWorker` | 推理 worker，负责 rollout 阶段的 action 采样 |
| Actor Update | `rlinf/workers/actor/fsdp_actor_worker.py` → `EmbodiedFSDPActor` | 策略训练 worker，计算 loss 并更新参数 |
| Environment | `rlinf/workers/env/env_worker.py` → `EnvWorker` | 环境交互 worker |
| 数据流结构 | `rlinf/data/embodied_io_struct.py` | 定义 Trajectory、ChunkStepResult 等数据结构 |
| 训练入口 | `examples/embodiment/train_embodied_agent.py` | Hydra 配置 + 启动训练 |
| 启动脚本 | `examples/embodiment/run_embodiment.sh` | Shell 启动脚本 |

---

## 2. Flow-based VLA 模型（论文 Section 3.2）

### 2.1 论文原理

Flow-based VLA（π0/π0.5）通过 flow matching 生成 action：
- VLM 提取 image + language 特征
- Action Expert 通过迭代去噪生成 action chunk `A_t = [a_{t,0}, ..., a_{t,H-1}]`
- 去噪过程：`A^{τ+δ} = A^τ + v_θ(A^τ, o) · δ`（前向 Euler 法）

### 2.2 核心代码

**π0/π0.5 模型定义：**

```
rlinf/models/embodiment/openpi/openpi_action_model.py
```

- **`OpenPi0Config`**（L34-66）：π0 配置，包含：
  - `noise_method`: 噪声注入方法（`flow_sde` / `flow_noise` / `flow_cps`）
  - `num_steps`: 去噪步数（默认10）
  - `action_chunk`: 动作块大小（默认5）
  - `noise_level`: Flow-SDE 噪声水平（默认0.5）
  - `noise_logvar_range`: Flow-Noise 学习噪声范围（默认 [0.08, 0.16]）
  - `joint_logprob`: 是否使用 joint log-probability（Flow-Noise 专用）
  - `double_layer`: 是否使用 two-layer MDP（Flow-SDE 无加速版）
  - `add_value_head`: 是否添加 value head（PPO 使用）
  - `value_after_vlm`: value head 放在 VLM 之后（π0.5 模式）

- **`OpenPi0ForRLActionPrediction`**（L68-790）：主模型类
  - 继承自 `PI0Pytorch`（openpi 官方实现）和 `BasePolicy`（RLinf 接口）
  - `__init__`：初始化 value_head、noise_head（flow-noise 时）
  - `sample_actions`（L378-498）：**核心推理函数**，完整的去噪+采样流程
  - `sample_mean_var_val`（L500-603）：**单步去噪**，计算 mean、std、value
  - `get_log_prob_value`（L672-749）：**训练时重新计算 log prob 和 value**
  - `get_logprob_norm`（L654-667）：**高斯分布 log prob 计算**

**模型配置文件：**
```
examples/embodiment/config/model/pi0.yaml      # π0 默认配置
examples/embodiment/config/model/pi0_5.yaml     # π0.5 默认配置
```

---

## 3. Flow-Noise 方法（论文 Section 4.1）

### 3.1 论文原理

Flow-Noise 将去噪过程建模为离散 MDP，引入可学习噪声网络：
- **随机性注入**（Eq.4）：`p(A^{τ+δ}|A^τ) ~ N(μ_τ, Σ_τ)`
  - `μ_τ = A^τ + v^τ · δ`（Euler 步均值）
  - `Σ_τ = diag(σ²_{θ'})`（噪声网络预测方差）
- **Log-Likelihood**（Eq.5）：整个去噪序列的联合概率
  - `log π(A|o) = log π(A⁰|o) · Π_{k=0}^{K-1} π(A^{τ_{k+1}}|A^{τ_k}, o)`

### 3.2 核心代码

**可学习噪声网络：**
```
rlinf/models/embodiment/modules/explore_noise_net.py → ExploreNoiseNet
```

- `__init__`（L41-57）：
  - `mlp_logvar`：MLP 预测 log variance
  - `noise_logvar_range`：噪声范围约束 [min_std, max_std]
  - `noise_scheduler_type`：`learn`（可学习）或 `const`（固定）
- `forward`（L76-84）：输入 noise_feature，输出 noise_std
- `post_process`（L86-99）：tanh 映射到 [logvar_min, logvar_max] 范围

**在 π0 模型中的调用：**

`openpi_action_model.py` 中 `__init__`（L147-155）：
```python
if self.config.noise_method == "flow_noise":
    self.noise_head = ExploreNoiseNet(
        in_dim=1024, out_dim=self.config.action_dim,
        hidden_dims=[128, 64], activation_type="tanh",
        noise_logvar_range=self.config.noise_logvar_range,
        noise_scheduler_type="learn",
    )
```

**`sample_mean_var_val`** 中 Flow-Noise 分支（L596-599）：
```python
elif self.config.noise_method == "flow_noise":
    x0_weight = 1 - (t_input - delta)
    x1_weight = t_input - delta
    x_t_std = self.noise_head(suffix_out)  # 噪声网络预测 std
```

**Joint Log-Probability 计算**（对应论文 Eq.5）：

`sample_actions` 中（L425-429 & L480-481）：
```python
if self.config.joint_logprob:
    # 初始噪声的 log prob
    initial_log_prob = self.get_logprob_norm(x_t, zeros, ones)
    log_probs.append(initial_log_prob)
    ...
    # 所有步的 log prob 取平均
    log_probs = log_probs.mean(dim=1)
```

**熵计算**（仅 Flow-Noise 使用，用于 entropy bonus）：

`gaussian_entropy`（L779-783）：
```python
def gaussian_entropy(self, sigma):
    entropy = 0.5 * torch.log(2 * π * e * (sigma²))
```

---

## 4. Flow-SDE 方法（论文 Section 4.2）

### 4.1 论文原理

Flow-SDE 将 ODE 转为 SDE，引入固定噪声进行探索：
- **随机性注入**（Eq.8-9）：
  - `σ_τ = a * sqrt(τ/(1-τ))`
  - `μ_τ = A^τ + [v^τ + σ²_τ/(2τ) · (A^τ + (1-τ)v^τ)] · δ`
  - `Σ_τ = σ²_τ · δ · I`
- **Two-Layer MDP**（Eq.10-12）：将 denoising 与环境交互耦合
- **Hybrid ODE-SDE Sampling**：只在一步随机注入噪声，其余确定性

### 4.2 核心代码

**`sample_mean_var_val`** 中 Flow-SDE 分支（L577-588）：
```python
elif self.config.noise_method == "flow_sde":
    # 计算 sigma_i = a * sqrt(τ / (1-τ))（对应论文 Eq.8 的 σ_τ）
    sigmas = noise_level * torch.sqrt(
        timesteps / (1 - torch.where(timesteps == 1, timesteps[1], timesteps))
    )[:-1]
    sigma_i = sigmas[idx][:, None, None].expand_as(x_t)

    # 对应论文 Eq.9 的均值和方差
    x0_weight = torch.ones_like(t_input) - (t_input - delta)
    x1_weight = t_input - delta - sigma_i**2 * delta / (2 * t_input)
    x_t_std = torch.sqrt(delta) * sigma_i
```

**Hybrid ODE-SDE Sampling**（论文 Section 4.2.3）：

`sample_actions` 中 denoise_inds 的选择（L434-447）：
```python
if mode == "train":
    if self.config.joint_logprob:
        # Flow-Noise: 所有步都用 SDE
        denoise_inds = torch.arange(num_steps)
    else:
        # Flow-SDE + Hybrid: 只随机选一步用 SDE，其余 ODE
        denoise_inds = torch.tensor([random.randint(0, num_steps - 1)] * num_steps)
else:
    # 评估时完全 ODE（确定性）
    denoise_inds = torch.tensor([-1] * num_steps)
```

每步去噪时判断是 train 还是 eval 模式（L453-456）：
```python
if idx == denoise_inds[0][idx]:
    sample_mode = "train"   # SDE 步（注入噪声）
else:
    sample_mode = "eval"    # ODE 步（确定性）
```

---

## 5. Policy Optimization — PPO（论文 Section 4.3）

### 5.1 论文原理

- **PPO Objective**（Eq.14）：`J(π_θ) = E_t[min(ρ_t · Â_t, clip(ρ_t, 1-ε, 1+ε) · Â_t)]`
- **GAE Advantage**（Eq.13）：`Â_t = Σ_{k=0}^{T-t} (γλ)^k · T_{t+k}`
- **Probability Ratio**（Eq.15）：one-layer 或 two-layer MDP 形式

### 5.2 核心代码

**PPO Actor Loss：**
```
rlinf/algorithms/losses.py → compute_ppo_actor_loss()（L24-138）
```
- 计算 ratio: `ratio = exp(logprobs - old_logprobs)`
- 计算 clipped ratio: `clipped_ratio = clamp(ratio, 1-ε, 1+ε)`
- PPO loss: `max(-advantages * ratio, -advantages * clipped_ratio)`
- 支持 dual clipping（`clip_ratio_c`）
- 返回 metrics: clip_fraction, approx_kl, ratio 等

**PPO Critic Loss：**
```
rlinf/algorithms/losses.py → compute_ppo_critic_loss()（L141-216）
```
- 值函数裁剪 + Huber Loss
- explained variance 计算

**PPO 联合 Loss：**
```
rlinf/algorithms/losses.py → compute_ppo_actor_critic_loss()（L219-247）
```
- `loss = actor_loss + critic_loss`

**GAE Advantage 计算：**
```
rlinf/algorithms/advantages.py → compute_gae_advantages_and_returns()（L24-86）
```
- 标准 GAE: `δ_t = r_t + γ · V(s_{t+1}) · (1-done) - V(s_t)`
- 递推: `gae = δ + γλ · (1-done) · gae`
- 支持归一化（`normalize_advantages`）

**GRPO Advantage 计算：**
```
rlinf/algorithms/advantages.py → compute_grpo_advantages()（L89-121）
```
- 组内奖励标准化: `Â_i = (R_i - mean) / std`

**算法注册机制：**
```
rlinf/algorithms/registry.py
```
- `@register_policy_loss("actor_critic")` → PPO
- `@register_policy_loss("actor")` → GRPO
- `@register_advantage("gae")` → GAE
- `@register_advantage("grpo")` → GRPO advantages

---

## 6. Critic Design（论文 Section 4.3.2 & 图4）

### 6.1 论文原理

两种 Critic 放置策略：
- **V_expert**（图4a，π0 使用）：Critic 接在 Action Expert 输出后，平均去噪轨迹估值
- **V_vlm**（图4b，π0.5 使用）：Critic 接在 VLM 输出后，直接用观测估值

### 6.2 核心代码

**Value Head 定义：**
```
rlinf/models/embodiment/modules/value_head.py → ValueHead
```
- 可配置 MLP（默认 hidden_sizes=(512, 128)）
- 支持 relu/gelu/tanh 激活

**在 π0 模型中初始化**（`openpi_action_model.py` L130-145）：
```python
if self.config.add_value_head:
    self.value_head = ValueHead(
        input_dim=proj_width,  # 1024 for π0, 2048 for π0.5
        hidden_sizes=value_head_hidden_sizes,
        output_dim=1,
    )
```

**V_expert 路径**（`sample_mean_var_val` L549-564）：
```python
if self.config.add_value_head and not self.config.value_after_vlm:
    # 对 action expert 输出取平均后送入 value_head
    suffix_out_value = torch.mean(suffix_out, dim=1)
    if self.config.detach_critic_input:
        suffix_out_value = suffix_out_value.detach()  # 可选: 断梯度
    value_t = self.value_head(suffix_out_value)[:, 0]
```

**V_vlm 路径**（`get_value_from_vlm` L751-777）：
```python
# 取 VLM prefix output 的 mean token 作为输入
prefix_out_value = prefix_output[:, prefix_mask, :].mean(dim=1)
values_vlm = self.value_head(prefix_out_value)[:, 0]
```

---

## 7. Action Chunking（论文 Section 4.3.1）

### 7.1 论文原理

π系列模型采用 chunk-based action 生成：
- 每次观测输出 H 步动作 `A_t = [a_{t,0}, ..., a_{t,H-1}]`
- 对应奖励 `R_t = Σ_{j=0}^{H-1} r_{t,j}`

### 7.2 核心代码

**配置：**
```yaml
# examples/embodiment/config/libero_spatial_ppo_openpi.yaml
actor:
  model:
    num_action_chunks: 5     # action chunk size H
    action_dim: 7            # 单步 action 维度
env:
  train:
    max_episode_steps: 240   # 环境最大步数
algorithm:
  reward_type: chunk_level   # chunk 级别奖励
  logprob_type: chunk_level  # chunk 级别 log prob
```

**Log prob 裁剪到 chunk 范围**（`openpi_action_model.py` L278-280）：
```python
log_probs = log_probs[:, :, :self.config.action_chunk, :self.config.action_env_dim]
```

---

## 8. 训练循环（论文 Figure 2 右侧）

### 8.1 代码流程

```
rlinf/runners/embodied_runner.py → EmbodiedRunner.run()（L157-268）
```

每个 training step 的流程：

```
Step 1: 权重同步
   actor → rollout（将最新参数同步到推理模型）

Step 2: Rollout 数据收集
   env.interact() ← → rollout.generate()
   环境与 rollout worker 交替执行 interaction_steps 步
   收集: actions, rewards, log_probs, values, dones, forward_inputs

Step 3: 计算 Advantage 和 Returns
   actor.compute_advantages_and_returns()
   → compute_gae_advantages_and_returns()（PPO）
   → compute_grpo_advantages()（GRPO）

Step 4: Actor 训练
   actor.run_training()
   → 多轮 update_epoch 迭代（默认4轮）
   → 每轮: 前向传播 → 计算 loss → 反向传播 → 参数更新

Step 5: 评估（定期）
   env.evaluate() + rollout.evaluate()
   → 统计 success rate 等指标
```

---

## 9. 环境集成（论文 Section 5.1）

### 9.1 论文 Benchmark

| Benchmark | 类型 | 代码路径 |
|-----------|------|---------|
| LIBERO | MuJoCo，4个 task suite | `rlinf/envs/libero/` |
| ManiSkill | GPU 并行仿真 | `rlinf/envs/maniskill/` |
| MetaWorld | MuJoCo，MT50 | `rlinf/envs/metaworld/` |

### 9.2 关键配置文件

```
examples/embodiment/config/env/libero_spatial.yaml   # LIBERO-Spatial
examples/embodiment/config/env/libero_object.yaml    # LIBERO-Object
examples/embodiment/config/env/libero_goal.yaml      # LIBERO-Goal
examples/embodiment/config/env/libero_10.yaml        # LIBERO-Long
examples/embodiment/config/env/maniskill_put_on_plate_in_scene_25_main.yaml
examples/embodiment/config/env/metaworld_50.yaml
```

### 9.3 完整训练配置示例

```
# π0 + PPO + LIBERO-Spatial
examples/embodiment/config/libero_spatial_ppo_openpi.yaml

# π0.5 + PPO + LIBERO-Spatial
examples/embodiment/config/libero_spatial_ppo_openpi_pi05.yaml

# π0 + GRPO + LIBERO-Spatial
examples/embodiment/config/libero_spatial_grpo_openpi.yaml

# π0 + PPO + ManiSkill
examples/embodiment/config/maniskill_ppo_openpi.yaml

# π0 + PPO + MetaWorld
examples/embodiment/config/metaworld_50_ppo_openpi.yaml
```

---

## 10. 关键超参数对照（论文 Table 7 & 8）

### LIBERO 超参数（`libero_spatial_ppo_openpi.yaml`）

| 论文参数 | 配置键 | 默认值 |
|---------|--------|--------|
| Train epochs | `runner.max_epochs` | 1000 |
| Batch size | `actor.global_batch_size` | 2048 |
| Update epochs | `algorithm.update_epoch` | 4 |
| Actor lr | `actor.optim.lr` | 5e-6 |
| Critic lr | `actor.optim.value_lr` | 1e-4 |
| γ (discount) | `algorithm.gamma` | 0.99 |
| λ (GAE) | `algorithm.gae_lambda` | 0.95 |
| ε (clip ratio) | `algorithm.clip_ratio_high` | 0.2 |
| Interaction steps | `env.train.max_episode_steps` | 240 |
| Parallel envs | `env.train.total_num_envs` | 64 |
| Rollout epochs | `algorithm.rollout_epoch` | 8 |
| Action chunk H | `actor.model.num_action_chunks` | 5 |
| Denoise steps | `actor.model.num_steps` | 4 |
| Noise level σ | `actor.model.openpi.noise_level` | 0.5 |
| Noise method | `actor.model.openpi.noise_method` | flow_sde |

---

## 11. 关键文件速查表

### 核心算法
| 文件 | 作用 |
|------|------|
| `rlinf/algorithms/losses.py` | PPO/GRPO actor+critic loss 实现 |
| `rlinf/algorithms/advantages.py` | GAE/GRPO advantage 计算 |
| `rlinf/algorithms/registry.py` | 算法注册机制 |

### 模型
| 文件 | 作用 |
|------|------|
| `rlinf/models/embodiment/openpi/openpi_action_model.py` | **π0/π0.5 模型核心**：去噪采样、log prob、value、noise 注入 |
| `rlinf/models/embodiment/modules/explore_noise_net.py` | **Flow-Noise** 可学习噪声网络 |
| `rlinf/models/embodiment/modules/value_head.py` | **Critic** value head |
| `rlinf/models/embodiment/modules/flow_actor.py` | 独立 flow matching actor（用于 SAC） |

### 训练系统
| 文件 | 作用 |
|------|------|
| `rlinf/runners/embodied_runner.py` | 训练主循环 |
| `rlinf/workers/actor/fsdp_actor_worker.py` | Actor worker：loss 计算 + 参数更新 |
| `rlinf/workers/rollout/hf/huggingface_worker.py` | Rollout worker：推理采样 |
| `rlinf/workers/env/env_worker.py` | Environment worker：环境交互 |

### 数据结构
| 文件 | 作用 |
|------|------|
| `rlinf/data/embodied_io_struct.py` | Trajectory、ChunkStepResult 等数据结构 |

### 配置
| 文件 | 作用 |
|------|------|
| `rlinf/config.py` | 全局配置定义（SupportedModel, SupportedEnv 等） |
| `examples/embodiment/config/model/pi0.yaml` | π0 模型默认配置 |
| `examples/embodiment/config/model/pi0_5.yaml` | π0.5 模型默认配置 |

---

## 12. 为 FlowIPO 改造的关键切入点

基于以上分析，如果要在此代码基础上实现 FlowIPO 算法，建议关注以下核心修改点：

1. **策略优化算法**：`rlinf/algorithms/losses.py`
   - 在此文件中新增 FlowIPO 的 loss 函数
   - 通过 `@register_policy_loss("flow_ipo")` 注册新算法

2. **Advantage 计算**：`rlinf/algorithms/advantages.py`
   - 如果 FlowIPO 使用不同的 advantage 估计方式，在此添加

3. **模型前向传播**：`rlinf/models/embodiment/openpi/openpi_action_model.py`
   - `sample_actions`：修改采样策略
   - `sample_mean_var_val`：修改噪声注入方式
   - `get_log_prob_value`：修改 log prob 计算

4. **训练配置**：`examples/embodiment/config/`
   - 创建新的 YAML 配置文件指定 FlowIPO 相关超参数

5. **训练循环**：`rlinf/runners/embodied_runner.py`
   - 如果 FlowIPO 需要额外数据（如 preference data），可能需要修改数据收集流程

6. **数据结构**：`rlinf/data/embodied_io_struct.py`
   - 如果需要存储额外信息（如 paired trajectories），扩展此处
