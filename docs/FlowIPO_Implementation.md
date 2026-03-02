# FlowIPO 代码实现文档

> 本文档记录 FlowIPO（Reward-Directed Velocity Interpolation）算法在 RLinf 代码库上的全部代码改动，供日后参照。

---

## 1. 改动总览

| 文件 | 类型 | 说明 |
|------|------|------|
| `rlinf/algorithms/credit_assignment.py` | **新建** | FlowIPO 信用分配：计算 $\delta_i$, $A_i$, $w_i$ |
| `rlinf/algorithms/losses.py` | 修改 | 新增 `@register_policy_loss("flow_ipo")` 损失函数 |
| `rlinf/algorithms/registry.py` | 修改 | `policy_loss()` 中为 flow_ipo 跳过 logprob 预处理 |
| `rlinf/algorithms/__init__.py` | 修改 | 注册 `credit_assignment` 模块 |
| `rlinf/models/embodiment/openpi/openpi_action_model.py` | 修改 | 存储 initial_noise 和 action；新增 `forward_velocity()` 方法 |
| `rlinf/workers/actor/fsdp_actor_worker.py` | 修改 | **最大改动**：EMA 参考模型、信用分配、FlowIPO 训练分支 |
| `rlinf/data/embodied_io_struct.py` | 修改 | 修复 `convert_trajectories_to_batch()` 中变量作用域 bug |
| `examples/embodiment/config/libero_object_flowipo_openpi.yaml` | **新建** | FlowIPO 专用 Hydra 配置文件 |

**一键启用方式**：将训练配置从 `libero_object_ppo_openpi.yaml` 换成 `libero_object_flowipo_openpi.yaml`，核心开关为 `loss_type: flow_ipo`。

---

## 2. 各文件详细改动

### 2.1 `rlinf/algorithms/credit_assignment.py`（新建）

FlowIPO 信用分配核心模块，实现从 rollout 数据到插值权重 $w_i$ 的计算。

**核心函数：** `compute_flow_ipo_weights(actions, ref_actions, rewards, alpha, eps)`

**算法流程：**

1. **策略偏离度** $\delta_i = \|a_i - a_i^{ref}\|_2$
   - 对 chunk × action_dim 维度做 L2 范数
   - 含义：当前策略与参考策略在第 $i$ 步的行为差异

2. **奖励方向** $\text{sign} = 2R - 1 \in \{-1, +1\}$
   - 成功 episode：sign = +1（鼓励偏离参考策略的行为）
   - 失败 episode：sign = -1（抑制偏离参考策略的行为）

3. **归一化信用** $A_i = \text{sign} \times \text{normalize}(\delta_i)$
   - 在每个 episode 内对 $\delta_i$ 做 Z-score 归一化（跨步骤）

4. **插值权重** $w_i = \sigma(\alpha \cdot A_i)$
   - $w_i \to 1$：保留当前动作（成功 + 大偏离）
   - $w_i \to 0$：回退参考策略（失败 + 大偏离）

**输入输出：**

```
输入：
  actions:     [n_steps, batch, chunk, action_dim]  当前策略动作
  ref_actions: [n_steps, batch, chunk, action_dim]  参考策略动作
  rewards:     [batch]                               episode 级奖励（0 或 1）
  alpha:       float                                 温度系数（默认 2.0）

输出：
  weights:     [n_steps, batch]                      每步插值权重 w_i ∈ (0, 1)
```

---

### 2.2 `rlinf/algorithms/losses.py`（修改）

在文件末尾新增 FlowIPO 损失函数：

```python
@register_policy_loss("flow_ipo")
def compute_flow_ipo_loss(v_theta, interpolated_target, loss_mask=None, **kwargs):
```

**损失公式：**

$$L = \mathbb{E}\left[\|v_\theta(x_t, t, s_i) - \tilde{v}_i\|^2\right]$$

其中 $\tilde{v}_i = w_i \cdot u_i + (1 - w_i) \cdot v_{ref,i}$ 是插值速度目标。

**计算方式：**
- 逐元素 MSE：`mse = (v_theta - interpolated_target).pow(2)`
- 带 mask 的平均：`loss = (mse * mask).sum() / mask.sum()`

**输出 metrics：**
- `actor/flow_ipo_loss`：最终 MSE 损失
- `actor/velocity_mse`：逐元素 MSE 均值

---

### 2.3 `rlinf/algorithms/registry.py`（修改）

在 `policy_loss()` 函数中添加 FlowIPO 分支：

```python
def policy_loss(**kwargs) -> tuple[torch.Tensor, dict]:
    loss_type = kwargs["loss_type"]
    loss_fn = get_policy_loss(loss_type)

    # FlowIPO 使用自己的前向路径（速度 MSE），跳过 logprob 预处理
    if loss_type == "flow_ipo":
        loss, metrics_data = loss_fn(**kwargs)
        return loss, metrics_data

    # 原有 PPO/GRPO 逻辑不变 ...
```

**原因：** FlowIPO 不依赖 logprob，`preprocess_loss_inputs()` 中的 logprob 相关处理（如 chunk-level 聚合、ratio 计算）与 FlowIPO 无关，跳过以避免 KeyError。

---

### 2.4 `rlinf/algorithms/__init__.py`（修改）

```python
# 修改前
from . import advantages, losses, registry  # noqa: F401

# 修改后
from . import advantages, credit_assignment, losses, registry  # noqa: F401
```

确保 `credit_assignment.py` 在 import 时被加载。

---

### 2.5 `rlinf/models/embodiment/openpi/openpi_action_model.py`（修改）

#### 改动 1：存储 initial_noise 和 action

在 `predict_action_batch()` 中，将 ODE 采样的初始噪声 $\varepsilon_0$ 和去噪后的 raw action 存入 `forward_inputs`：

```python
# 在 forward_inputs 字典中新增
"initial_noise": outputs["chains"][:, 0],  # 保存初始噪声用于 FlowIPO 参考动作生成
"action": outputs["actions"],              # 保存去噪后的 raw action 用于 FlowIPO 信用分配
```

**用途：**
- `initial_noise`：FlowIPO 需要参考策略从**相同的初始噪声**出发生成动作，才能让 $\delta_i$ 纯粹反映策略差异而非噪声差异。
- `action`：FlowIPO 信用分配阶段需要从 `rollout_batch["actions"]` 获取当前策略的动作来计算 $\delta_i = \|a_i - a_i^{ref}\|_2$。原始代码中 `forward_inputs` 未包含 `"action"` key，导致 rollout worker 通过 `result["forward_inputs"].get("action", None)` 得到 `None`，`Trajectory.actions` 不被赋值，最终 `rollout_batch` 缺少 `"actions"` 字段。PPO/GRPO 不直接访问此字段（它们通过 `forward_inputs["chains"]` 重算 logprob），所以该问题仅在 FlowIPO 路径暴露。

#### 改动 2：缓存已处理的观测数据

在 `predict_action_batch()` 中，`input_transform` 输出后、`precision_processor` 调用前，将已处理的观测数据以 `"~"` 前缀扁平化存入 `forward_inputs`：

```python
# 缓存 processed obs（CPU tensors），避免参考动作计算时重跑 input_transform
_cached_proc = {}
for _k, _v in processed_obs.items():
    if isinstance(_v, dict):  # e.g., "image" → {"base_0_rgb": tensor}
        for _sk, _sv in _v.items():
            _cached_proc[f"~{_k}~{_sk}"] = _sv
    elif torch.is_tensor(_v):
        _cached_proc[f"~{_k}"] = _v
forward_inputs.update(_cached_proc)
```

**设计要点：**
- `"~"` 前缀不含 `"/"`，`input_transform` 的 key 过滤（`"/" in key`）不会误选这些 key
- 所有值均为 tensor，兼容 `stack_list_of_dict_tensor`、`convert_trajectories_to_batch`、`process_nested_dict_for_adv`
- `_compute_reference_actions_batch` 通过 `"~"` key 重建 `{"image": {...}, "state": ...}` dict，直接调用 `Observation.from_dict`，跳过 `input_transform` 的逐样本 Python 循环

#### 改动 3：新增 `forward_velocity()` 方法

```python
def forward_velocity(self, forward_inputs, x_t, timestep) -> torch.Tensor:
    """
    给定观测 + x_t + 时间步，返回速度预测 v_θ(x_t, t, s)。
    用于 FlowIPO 训练中计算当前模型和参考模型的速度。
    """
```

**流程：**
1. 输入变换：`forward_inputs` → observation
2. 编码前缀（图像 + 语言）并缓存 KV
3. 通过 suffix out 处理 state + $x_t$ + timestep
4. 投影到速度空间：`v_theta = action_out_proj(suffix_out)`

**与 SFT forward 的区别：**
- SFT forward：内部采样 $t$、构造 $x_t$、计算 MSE loss，一站式完成
- `forward_velocity()`：外部传入 $x_t$ 和 $t$，只返回 $v_\theta$，由训练循环构造插值目标

---

### 2.6 `rlinf/workers/actor/fsdp_actor_worker.py`（修改，最大改动）

#### 2.6.1 初始化

在 `init_worker()` 中：

```python
self._is_flow_ipo = self.cfg.algorithm.loss_type == "flow_ipo"
if self._is_flow_ipo:
    self._init_flow_ipo_ref_model()
```

#### 2.6.2 EMA 参考模型管理

**`_init_flow_ipo_ref_model()`：**
- 将当前模型权重深拷贝到 CPU 作为参考模型 `_ref_weights_cpu`
- 读取 FlowIPO 配置：`alpha`, `beta_ref`, `t_min`, `t_max`

**`_ema_update_ref_model()`：**
- 每个训练 iteration 后调用
- 更新公式：$\theta_{ref} \leftarrow \beta \cdot \theta_{ref} + (1 - \beta) \cdot \theta$
- 在 CPU 上执行，不占 GPU 显存

#### 2.6.3 参考策略推理

**`_compute_reference_actions_batch(forward_inputs, initial_noise, max_chunk_size)`：**
- 将 `[n_steps, batch, ...]` 展平为 `[n_steps * batch, ...]` mega-batch，分 chunk 处理
- **性能关键**：利用 rollout 阶段缓存的已处理观测数据（`"~"` 前缀 key），直接构造 `Observation` 对象，**完全跳过 `input_transform` 的逐样本 Python 循环**（3072+ 次迭代 → 0 次）
- 缓存命中路径：从 `"~"` key 重建 `{"image": {...}, "state": ...}` dict → `Observation.from_dict` → `sample_actions`（ODE 推理）
- 缓存未命中路径（fallback）：`input_transform` → `precision_processor` → `Observation.from_dict` → `sample_actions`
- 调用方通过 `cpu_weight_swap` 管理权重交换（仅一次 swap，而非逐步 swap）

**`_compute_reference_velocity(forward_inputs, x_t, timestep)`：**
- 使用 `cpu_weight_swap` 加载参考权重
- 单次前向传播计算 $v_{ref}(x_t, t, s)$
- 无梯度计算（`torch.no_grad()`）

#### 2.6.4 信用分配

**`_compute_flow_ipo_credit_assignment(rollout_batch)`：**

1. 提取数据：actions, rewards, forward_inputs, initial_noise
2. 计算 episode 级奖励：对步骤和 chunk 维度求和，clamp 到 [0, 1]
3. **单次** `cpu_weight_swap` 加载参考权重，循环内逐步生成参考动作（使用相同 initial_noise）。权重交换必须在循环外执行，否则 n_steps × 2 次 FSDP `load_state_dict` 会导致训练卡死
4. 调用 `compute_flow_ipo_weights()` 得到 $w_i$
5. 存入 `rollout_batch["flow_ipo_weights"]`，并填充 dummy advantages（兼容数据 reshape）

**在 `compute_advantages_and_returns()` 中的调用：**
```python
if self._is_flow_ipo:
    self._compute_flow_ipo_credit_assignment(rollout_batch)
    return  # 跳过 GAE 计算
```

#### 2.6.5 训练循环

在 `run_training()` 中新增完整的 FlowIPO 训练分支：

```python
if self._is_flow_ipo:
    # ---- FlowIPO Training ----
    # 1. 采样时间步和噪声
    t ~ U[t_min, t_max],  shape: [batch, 1, 1]
    ε ~ N(0, I),          shape: [batch, chunk, action_dim]

    # 2. 构造带噪动作
    x_t = (1 - t) * actions + t * ε

    # 3. 当前速度目标（π0 约定）
    u_i = ε - actions    # noise - data

    # 4. 参考速度（无梯度）
    v_ref = _compute_reference_velocity(forward_inputs, x_t, t)

    # 5. 插值速度目标
    w = flow_ipo_weights.unsqueeze(-1).unsqueeze(-1)  # [batch] → [batch, 1, 1]
    v_target = w * u_i + (1 - w) * v_ref

    # 6. 模型前向
    v_theta = model.forward_velocity(forward_inputs, x_t, t)

    # 7. 计算损失
    loss = compute_flow_ipo_loss(v_theta, v_target, loss_mask)
else:
    # ---- 原有 PPO/GRPO 训练（不变）----
```

#### 2.6.6 训练后更新

```python
# 训练循环结束后
if self._is_flow_ipo:
    self._ema_update_ref_model()
```

---

### 2.7 `examples/embodiment/config/libero_object_flowipo_openpi.yaml`（新建）

基于 PPO 配置，关键区别：

```yaml
algorithm:
  loss_type: flow_ipo                # 核心开关
  flow_ipo_alpha: 2.0                # 权重锐度温度
  flow_ipo_beta_ref: 0.995           # EMA 更新率（慢）
  flow_ipo_t_min: 0.2                # flow matching 时间下界
  flow_ipo_t_max: 0.8                # flow matching 时间上界

actor:
  model:
    add_value_head: False             # 无 critic

critic:
  use_critic_model: False
```

**与 PPO 配置的不同：**

| 配置项 | PPO | FlowIPO |
|--------|-----|---------|
| `loss_type` | `actor_critic` | `flow_ipo` |
| `add_value_head` | `True` | `False` |
| `use_critic_model` | `False` | `False` |
| `detach_critic_input` | `True` | `False`（无关） |
| `entropy_bonus` | 可设非零 | `0` |
| FlowIPO 超参 | 无 | `alpha`, `beta_ref`, `t_min`, `t_max` |

---

## 3. 数据流总览

```
┌─────────────────────────────────────────────────────────────────┐
│  Phase 1: Rollout（不变）                                        │
│  EnvWorker ──→ MultiStepRolloutWorker ──→ rollout_batch         │
│  产出: actions, rewards, forward_inputs, initial_noise          │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  Phase 2: Credit Assignment（FlowIPO 新增）                      │
│  _compute_flow_ipo_credit_assignment()                          │
│                                                                 │
│  actions ─┐                                                     │
│           ├──→ δ_i = ||a_i - a_i^ref||₂                        │
│  ref_actions ─┘                                                 │
│                                                                 │
│  rewards ──→ sign = 2R - 1                                      │
│                                                                 │
│  δ_i + sign ──→ A_i = sign × normalize(δ_i)                    │
│             ──→ w_i = sigmoid(α × A_i)                          │
│                                                                 │
│  产出: flow_ipo_weights [n_steps, batch]                        │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  Phase 3: FlowIPO Training                                      │
│  run_training() FlowIPO 分支                                    │
│                                                                 │
│  采样: t ~ U[t_min, t_max],  ε ~ N(0, I)                       │
│  构造: x_t = (1-t) × a_i + t × ε                               │
│  目标: u_i = ε - a_i                                            │
│  参考: v_ref = ref_model.forward_velocity(x_t, t)              │
│  插值: ṽ_i = w_i × u_i + (1 - w_i) × v_ref                    │
│  预测: v_θ = model.forward_velocity(x_t, t)                    │
│  损失: L = ||v_θ - ṽ_i||²                                      │
│                                                                 │
│  反向传播 + 优化器更新                                           │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  Phase 4: EMA Update                                            │
│  θ_ref ← β × θ_ref + (1 - β) × θ                              │
│  在 CPU 上执行                                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. 与 PPO/GRPO 的核心区别

| 方面 | PPO/GRPO | FlowIPO |
|------|----------|---------|
| **损失函数** | Policy gradient（logprob ratio） | Velocity MSE |
| **信用分配** | GAE / GRPO advantages | Flow-matching 权重 $w_i$ |
| **参考模型** | KL penalty（$\beta_{kl}$） | EMA 参考（$\beta_{ref}=0.995$） |
| **Critic** | 需要 value head | 不需要 |
| **训练目标** | 最大化 $r(\theta)A$ | 最小化 $\|v_\theta - \tilde{v}\|^2$ |
| **时间步采样** | 无（单步确定性） | 随机 $t \sim U[t_{min}, t_{max}]$ |
| **速度插值** | 无 | $\tilde{v}_i = w_i u_i + (1-w_i) v_{ref}$ |
| **Metrics** | policy_loss, kl, clip_fraction | flow_ipo_loss, velocity_mse, mean_weight |

---

## 5. 超参数说明

| 超参数 | 默认值 | 含义 |
|--------|--------|------|
| `flow_ipo_alpha` | 2.0 | sigmoid 温度，越大则 $w_i$ 越接近 0/1（更极端） |
| `flow_ipo_beta_ref` | 0.995 | EMA 系数，越大参考模型更新越慢 |
| `flow_ipo_t_min` | 0.2 | flow matching 时间采样下界，避免 $t \approx 0$（纯数据）区间 |
| `flow_ipo_t_max` | 0.8 | flow matching 时间采样上界，避免 $t \approx 1$（纯噪声）区间 |

---

## 6. 使用方式

### 启动 FlowIPO 训练

将训练脚本的配置文件替换为 FlowIPO 版本：

```bash
# PPO 训练（原有）
python train.py --config-name=libero_object_ppo_openpi

# FlowIPO 训练（新增，一键切换）
python train.py --config-name=libero_object_flowipo_openpi
```

### 自定义超参数

可通过命令行 override：

```bash
python train.py --config-name=libero_object_flowipo_openpi \
  algorithm.flow_ipo_alpha=3.0 \
  algorithm.flow_ipo_beta_ref=0.99 \
  algorithm.flow_ipo_t_min=0.1 \
  algorithm.flow_ipo_t_max=0.9
```

---

## 7. 设计决策记录

1. **ODE 而非 SDE**：Rollout 和参考动作生成都使用 ODE。因为 FlowIPO 的信用分配需要 $\delta_i$ 纯粹反映策略差异，SDE 的中间噪声会污染这个信号。探索多样性通过初始噪声 $\varepsilon_0$ 的随机采样实现。

2. **EMA 参考模型在 CPU**：参考权重存在 CPU，通过 `cpu_weight_swap` 按需加载到 GPU。这避免了在 GPU 上同时存两份模型权重的显存开销。

3. **训练分支而非仅替换 loss**：FlowIPO 根本性地改变了训练范式（从 policy gradient 到 modified flow matching），需要完整的新训练分支（采样 $t$、构造 $x_t$、计算 $v_{ref}$ 等），不能仅仅换 loss 函数。

4. **dummy advantages 兼容**：FlowIPO 不需要 advantages，但数据 reshape 管线假设 advantages 字段存在，因此填充全零 tensor 保持兼容。

5. **无 Critic**：FlowIPO 不需要 value head，`add_value_head: False`，减少模型参数和显存占用。

6. **forward_inputs 必须包含 `"action"` key**：`rollout_batch["actions"]` 的来源链路为：`forward_inputs["action"]` → `ChunkStepResult.actions` → `Trajectory.actions` → `rollout_batch["actions"]`。原始 RLinf 代码中 `forward_inputs` 不含 `"action"`（PPO/GRPO 不需要直接访问 actions，而是通过 `chains` 重算 logprob），导致 FlowIPO 信用分配阶段 `KeyError: 'actions'`。修复：在 `predict_action_batch()` 的 `forward_inputs` 中添加 `"action": outputs["actions"]`。

7. **`convert_trajectories_to_batch()` 变量作用域修复**：原始代码在遍历 dataclass fields 时使用了前一个 for 循环泄漏的 `traj` 变量（指向 `trajectories[-1]`），应使用 `trajectories[0]` 进行类型检查。该 bug 在 PPO 中不会触发（因为所有 trajectory 的 actions 要么全为 tensor，要么全为 None），但在边界情况下可能导致字段丢失。
