# FlowSAR 代码实现文档

> 本文档记录 FlowSAR（Self-Annotated Credit Assignment）算法在 RLinf 代码库上的全部代码改动，供日后参照。

---

## 1. 改动总览

| 文件 | 类型 | 说明 |
|------|------|------|
| `rlinf/algorithms/credit_assignment.py` | 修改 | 新增 `compute_flow_sar_weights()` 信用分配函数 |
| `rlinf/algorithms/losses.py` | 修改 | 新增 `@register_policy_loss("flow_sar")` 对比损失函数 |
| `rlinf/algorithms/registry.py` | 修改 | `policy_loss()` 中为 flow_sar 跳过 logprob 预处理 |
| `rlinf/workers/rollout/hf/huggingface_worker.py` | 修改 | 新增 self-annotation + v_old 预计算 |
| `rlinf/workers/actor/fsdp_actor_worker.py` | 修改 | FlowSAR 信用分配 + 对比训练分支 |
| `examples/embodiment/config/libero_object_flowsar_openpi_quickstart.yaml` | **新建** | FlowSAR 专用 Hydra 配置文件 |

**一键启用方式**：将训练配置从 `libero_object_flowipo_openpi_quickstart.yaml` 换成 `libero_object_flowsar_openpi_quickstart.yaml`，核心开关为 `loss_type: flow_sar`。

---

## 2. 各文件详细改动

### 2.1 `rlinf/algorithms/credit_assignment.py`（修改）

新增 FlowSAR 信用分配函数，基于重建误差和 episode reward 计算 step-level 权重。

**核心函数：** `compute_flow_sar_weights(recon_errors, rewards, temperature, w_min, w_max)`

**算法流程：**

1. **重建误差** $e_i$：已由 rollout worker 预计算（self-annotation）
2. **Episode reward 方向** $R \in \{0, 1\}$
3. **信用分配权重**：
   - 成功轨迹 ($R = 1$)：$w_i = \text{softmax}(e_i / T)$ — 不确定但正确的步获高权重
   - 失败轨迹 ($R = 0$)：$w_i = \text{softmax}(-e_i / T)$ — 确信但错误的步获高权重
4. **可选权重裁剪**：$w_i = \text{clip}(w_i, w_{\min}, w_{\max})$ 后重新归一化

**输入输出：**

```
输入：
  recon_errors: [n_steps, batch]        每步的重建误差 e_i
  rewards:      [batch]                  episode 级奖励（0 或 1）
  temperature:  float                    softmax 温度（默认 0.5）
  w_min:        float                    权重下界（默认 0.05）
  w_max:        float                    权重上界（默认 0.5）

输出：
  weights:      [n_steps, batch]         每步信用分配权重 w_i ∈ (0, 1)
  labels:       [batch]                  y_i = 2R - 1 ∈ {-1, +1}
```

---

### 2.2 `rlinf/algorithms/losses.py`（修改）

在文件末尾新增 FlowSAR 损失函数，支持**两种损失变体**和**两种能量类型**：

```python
@register_policy_loss("flow_sar")
def compute_flow_sar_loss(v_theta, v_old, u_target, weights, labels, beta,
                          energy_type="mse", loss_variant="mse_branch",
                          kl_coeff=0.5, flow_t=None, loss_mask=None, **kwargs):
```

**共享步骤：**

1. 构造正负分支速度：
   $$v^+ = (1 - \beta) v^{\text{old}} + \beta \, v_\theta, \quad v^- = (1 + \beta) v^{\text{old}} - \beta \, v_\theta$$

2. 计算正负分支能量：
   $$E^+ = \|v^+ - u_i\|^2, \quad E^- = \|v^- - u_i\|^2$$

3. （可选）SDE 模式 Mahalanobis 缩放：
   $$E^{\pm}_{\text{sde}} = \frac{1}{2t} \cdot E^\pm$$

**损失变体一：`mse_branch`（DiffusionNFT 风格）**

$$\ell_i = w_i \cdot [ R_i \cdot E^+ + (1-R_i) \cdot E^- ]$$

**损失变体二：`softplus_kl`（π-StepNFT 风格）**

$$\ell_i = w_i \cdot \text{softplus}\!\left(\tfrac{1}{2} y_i \cdot (E^+ - E^-)\right) + \lambda_{\text{KL}} \|v_\theta - v^{\text{old}}\|^2$$

**输出 metrics：**
- `actor/flow_sar_loss`：总损失
- `actor/flow_sar_E_pos` / `E_neg`：正/负分支误差均值
- `actor/flow_sar_weight_mean`：SAR 权重均值
- `actor/flow_sar_energy_type`：1.0 (sde) / 0.0 (mse)
- `actor/flow_sar_loss_variant`：1.0 (softplus_kl) / 0.0 (mse_branch)
- `actor/flow_sar_contrastive`：softplus 对比项均值（仅 softplus_kl）
- `actor/flow_sar_kl_penalty`：KL 惩罚项均值（仅 softplus_kl）
- `actor/flow_sar_success_ratio`：batch 成功比例（仅 mse_branch）

---

### 2.3 `rlinf/algorithms/registry.py`（修改）

在 `policy_loss()` 函数中添加 FlowSAR 分支（与 FlowIPO 同理）：

```python
if loss_type in ("flow_ipo", "flow_sar"):
    loss, metrics_data = loss_fn(**kwargs)
    return loss, metrics_data
```

**原因：** FlowSAR 不依赖 logprob，跳过 `preprocess_loss_inputs()` 避免 KeyError。

---

### 2.4 `rlinf/workers/rollout/hf/huggingface_worker.py`（修改）

#### 改动 1：初始化 FlowSAR 配置

在 `__init__` 中，同 FlowIPO 一样维护 EMA 参考权重，并支持动态 EMA 调度：

```python
self._is_flow_sar = cfg.algorithm.get("loss_type", "") == "flow_sar"
if self._is_flow_sar:
    self._ref_weights_cpu = None
    self._ref_swap_buffer = None
    self._flow_sar_ema_beta = cfg.algorithm.get("flow_sar_ema_beta", 0.995)
    # DiffusionNFT-style 动态 EMA 调度
    self._flow_sar_ema_schedule = cfg.algorithm.get("flow_sar_ema_schedule", "fixed")
    self._flow_sar_ema_eta_rate = cfg.algorithm.get("flow_sar_ema_eta_rate", 0.001)
    self._flow_sar_ema_eta_max = cfg.algorithm.get("flow_sar_ema_eta_max", 0.5)
    self._flow_sar_tau_mid = cfg.algorithm.get("flow_sar_tau_mid", 0.5)
```

#### 改动 2：`sync_model_from_actor` 中支持动态 EMA

FlowSAR 复用 FlowIPO 的 EMA 机制，但新增 `linear` 调度模式：

```python
if self._is_flow_sar and self._flow_sar_ema_schedule == "linear":
    # DiffusionNFT-style: β_i = min(eta_rate * i, eta_max)
    # 早期 β ≈ 0（激进更新），后期 β → eta_max（保守稳定）
    beta = min(self._flow_sar_ema_eta_rate * self.count_update,
               self._flow_sar_ema_eta_max)
else:
    beta = self._flow_sar_ema_beta  # 固定模式
```

#### 改动 3：新增 `_compute_self_annotation()` 方法

**核心功能**：对每个 rollout step 计算重建误差 $e_i$，并预计算训练阶段所需的 $(t, \epsilon, v^{\text{old}})$。

```python
@torch.no_grad()
def _compute_self_annotation(self, rollout_result):
    """
    FlowSAR Phase 2: Self-Annotation
    - 计算重建误差 e_i（策略置信度）
    - 预采样 (t, ε) 并计算 v_old 用于对比训练
    """
```

**流程：**
1. 使用 `cpu_weight_swap` 加载 EMA 参考权重
2. 对每个 step 的 `forward_inputs`：
   a. 从 `"~"` 前缀 key 重建 observation（复用 FlowIPO 的缓存机制）
   b. **Self-Annotation**（模型约定：$t=0$ 为干净，$t=1$ 为噪声）：
      - 前向加噪：$x_{t_{\text{mid}}} = (1 - t_{\text{mid}}) \cdot a_i + t_{\text{mid}} \cdot \epsilon$
      - 反向重建（$t_{\text{mid}} \to 0$）：$\hat{a}_i = x_{t_{\text{mid}}} - v_{\theta_{\text{old}}}(x_{t_{\text{mid}}}, t_{\text{mid}}, s_i) \cdot t_{\text{mid}}$
      - 重建误差：$e_i = \|a_i - \hat{a}_i\|^2$
   c. **预计算训练数据**：
      - 采样 $t \sim U[t_{\min}, t_{\max}]$，$\epsilon \sim \mathcal{N}(0, I)$
      - 构造 $x_t = (1-t) \cdot a_i + t \cdot \epsilon$
      - 计算 $v^{\text{old}} = v_{\theta_{\text{old}}}(x_t, t, s_i)$
3. 存入 `forward_inputs`：`recon_error`, `flow_t`, `flow_epsilon`, `v_old`

#### 改动 4：`generate()` 中调用 self-annotation

```python
if (self._is_flow_ipo or self._is_flow_sar) and self._ref_weights_cpu is not None:
    for stage_id in range(self.num_pipeline_stages):
        if self._is_flow_ipo:
            self._compute_reference_actions(self.rollout_results[stage_id])
        elif self._is_flow_sar:
            self._compute_self_annotation(self.rollout_results[stage_id])
```

---

### 2.5 `rlinf/workers/actor/fsdp_actor_worker.py`（修改）

#### 2.5.1 初始化

在 `init_worker()` 中新增 FlowSAR 配置读取：

```python
self._is_flow_sar = self.cfg.algorithm.loss_type == "flow_sar"
if self._is_flow_sar:
    self._flow_sar_cfg = {
        "temperature": self.cfg.algorithm.get("flow_sar_temperature", 0.5),
        "beta": self.cfg.algorithm.get("flow_sar_beta", 1.0),
        "energy_type": self.cfg.algorithm.get("flow_sar_energy_type", "mse"),
        "loss_variant": self.cfg.algorithm.get("flow_sar_loss_variant", "mse_branch"),
        "kl_coeff": self.cfg.algorithm.get("flow_sar_kl_coeff", 0.5),
        "w_min": self.cfg.algorithm.get("flow_sar_w_min", 0.0),
        "w_max": self.cfg.algorithm.get("flow_sar_w_max", 1.0),
    }
```

#### 2.5.2 信用分配

**`_compute_flow_sar_credit_assignment()`：**

1. 提取 `recon_error` 从 `forward_inputs`
2. 计算 episode-level reward
3. 调用 `compute_flow_sar_weights()` 得到 $w_i$ 和 $y_i$
4. 存入 `rollout_batch["flow_sar_weights"]` 和 `rollout_batch["flow_sar_labels"]`
5. 填充 dummy advantages（兼容数据 reshape）

**在 `compute_advantages_and_returns()` 中的调用：**
```python
if self._is_flow_ipo:
    return self._compute_flow_ipo_credit_assignment()
if self._is_flow_sar:
    return self._compute_flow_sar_credit_assignment()
```

#### 2.5.3 训练循环

在 `run_training()` 中新增 FlowSAR 训练分支：

```python
if self._is_flow_ipo:
    # ... FlowIPO branch (unchanged) ...
elif self._is_flow_sar:
    # ============ FlowSAR Training Branch ============
    # 1. 读取预计算数据
    actions = batch["actions"]
    flow_sar_weights = batch["flow_sar_weights"]
    flow_sar_labels = batch["flow_sar_labels"]
    t = forward_inputs["flow_t"]
    epsilon = forward_inputs["flow_epsilon"]
    v_old = forward_inputs["v_old"]

    # 2. 重建 x_t 和 u_i
    x_t = (1 - t) * actions + t * epsilon
    u_i = epsilon - actions

    # 3. 模型前向 v_θ(x_t, t, s_i)（有梯度，通过 FSDP forward）
    v_theta = self.model(
        forward_type=ForwardType.VELOCITY,
        forward_inputs=forward_inputs,
        x_t=x_t, timestep=t.reshape(bsz),
    )

    # 4. Compute FlowSAR loss (支持 mse_branch / softplus_kl × mse / sde)
    loss, metrics_data = compute_flow_sar_loss(
        v_theta=v_theta_chunk,
        v_old=v_old_chunk,
        u_target=u_i_chunk,
        weights=flow_sar_weights,
        labels=flow_sar_labels,
        beta=self._flow_sar_cfg["beta"],
        energy_type=self._flow_sar_cfg["energy_type"],
        loss_variant=self._flow_sar_cfg["loss_variant"],
        kl_coeff=self._flow_sar_cfg["kl_coeff"],
        flow_t=t, loss_mask=loss_mask,
    )
    # ============ End FlowSAR Branch ============
else:
    # ============ Original PPO/GRPO Branch ============
```

---

### 2.6 `examples/embodiment/config/libero_object_flowsar_openpi_quickstart.yaml`（新建）

基于 FlowIPO 配置，关键区别：

```yaml
algorithm:
  loss_type: flow_sar                    # 核心开关
  flow_sar_beta: 1.0                     # 镜像构造信任域 β
  flow_sar_temperature: 0.5              # 信用分配温度
  flow_sar_tau_mid: 0.5                  # 重建时刻
  flow_sar_ema_beta: 0.995               # EMA 系数（仅 fixed 模式）
  flow_sar_ema_schedule: linear          # "fixed" 或 "linear"（DiffusionNFT 动态调度）
  flow_sar_ema_eta_rate: 0.001           # linear: β_i = min(rate * i, max)
  flow_sar_ema_eta_max: 0.5              # linear: 最大保留率
  flow_sar_t_min: 0.2                    # flow matching 时间下界
  flow_sar_t_max: 0.8                    # flow matching 时间上界
  flow_sar_energy_type: mse              # "mse"（ODE）或 "sde"（1/2t 缩放）
  flow_sar_loss_variant: softplus_kl     # "mse_branch"（DiffusionNFT）或 "softplus_kl"（π-StepNFT）
  flow_sar_kl_coeff: 1.0                 # KL 惩罚系数（仅 softplus_kl）
  flow_sar_w_min: 0.0                    # 权重裁剪下界
  flow_sar_w_max: 1.0                    # 权重裁剪上界

actor:
  optim:
    lr: 1.0e-6                            # 学习率
  model:
    add_value_head: False                 # 无 critic
critic:
  use_critic_model: False
```

---

## 3. 数据流总览

```
┌─────────────────────────────────────────────────────────────────┐
│  Phase 1: Rollout（不变）                                        │
│  EnvWorker ──→ MultiStepRolloutWorker ──→ rollout_result        │
│  产出: actions, rewards, forward_inputs                         │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  Phase 2: Self-Annotation（Rollout Worker, no grad）              │
│  _compute_self_annotation()                                      │
│  模型约定: t=0 → 干净, t=1 → 噪声, v = ε - a                    │
│                                                                  │
│  对每步 (s_i, a_i):                                              │
│    ε ~ N(0,I)                                                    │
│    x_t = (1-t_mid) * a_i + t_mid * ε                            │
│    v_pred = v_θ_old(x_t, t_mid, s_i)                            │
│    â_i = x_t - v_pred * t_mid                                   │
│    e_i = ||a_i - â_i||²                                          │
│                                                                  │
│  预计算训练数据:                                                  │
│    t ~ U[t_min, t_max], ε ~ N(0,I)                              │
│    x_t = (1-t) * a_i + t * ε                                    │
│    v_old = v_θ_old(x_t, t, s_i)                                 │
│                                                                  │
│  产出: recon_error, flow_t, flow_epsilon, v_old                  │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  Phase 3: Credit Assignment（Actor Worker）                      │
│  _compute_flow_sar_credit_assignment()                           │
│                                                                  │
│  e_i + R ──→ 成功: w_i = softmax(e_i / T)                      │
│          ──→ 失败: w_i = softmax(-e_i / T)                      │
│  y_i = 2R - 1 ∈ {-1, +1}                                       │
│                                                                  │
│  产出: flow_sar_weights [n_steps, batch], labels [batch]        │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  Phase 4: Weighted Mirror Update（Actor Worker）                  │
│  run_training() FlowSAR 分支                                    │
│                                                                  │
│  v_θ = model.forward_velocity(x_t, t)        (有梯度)           │
│  v⁺ = (1-β)v_old + β·v_θ,  v⁻ = (1+β)v_old - β·v_θ           │
│  E⁺ = ||v⁺ - u_i||², E⁻ = ||v⁻ - u_i||²                      │
│  (可选 SDE: E± *= 1/(2t))                                       │
│                                                                  │
│  mse_branch:   ℓ = w_i · [R·E⁺ + (1-R)·E⁻]                   │
│  softplus_kl:  ℓ = w_i · softplus(½y(E⁺-E⁻)) + λ·||v_θ-v_old||² │
│                                                                  │
│  反向传播 + 优化器更新                                           │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  Phase 5: Dynamic EMA Update（Rollout Worker）                   │
│  linear:  β_i = min(η_rate · i, η_max)                         │
│  fixed:   β_i = flow_sar_ema_beta                               │
│  θ_old ← β_i · θ_old + (1-β_i) · θ （CPU 上执行）              │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. 与 FlowIPO 的核心区别

| 方面 | FlowIPO | FlowSAR |
|------|---------|---------|
| **信用分配** | 策略偏离度 $\delta_i$ + sigmoid | 重建误差 $e_i$ + softmax |
| **信用分配来源** | 需要参考动作（额外 ODE 推理） | 模型自身重建误差（单次 forward） |
| **损失函数** | 速度插值 MSE $\|v_\theta - \tilde{v}\|^2$ | 可切换：MSE 分支选择 / Softplus+KL |
| **能量类型** | 无 | 可切换：ODE (mse) / SDE (1/2t) |
| **EMA 策略** | 固定 beta | 可切换：固定 / DiffusionNFT 动态调度 |
| **失败样本处理** | 回退到参考策略速度 | 镜像构造主动推远 |
| **负样本利用** | 被动（$w_i \to 0$ 使目标趋近 $v_{\text{ref}}$） | 主动（$v^-$ 方向与 $v_\theta$ 相反） |
| **Rollout 额外开销** | 完整 ODE 推理（参考动作） | 2 次 forward pass（重建 + v_old） |

---

## 5. 超参数说明

| 超参数 | 默认值 | 含义 |
|--------|--------|------|
| `flow_sar_beta` | 1.0 | 镜像构造信任域 β，越大正负分支偏离越远 |
| `flow_sar_temperature` | 0.5 | 信用分配温度，越小权重越极端 |
| `flow_sar_tau_mid` | 0.5 | self-annotation 重建时刻 |
| `flow_sar_ema_schedule` | `linear` | EMA 调度："fixed"（固定）或 "linear"（DiffusionNFT 动态） |
| `flow_sar_ema_beta` | 0.995 | 固定 EMA 系数（仅 `fixed` 模式） |
| `flow_sar_ema_eta_rate` | 0.001 | 动态 EMA 增长率（仅 `linear` 模式） |
| `flow_sar_ema_eta_max` | 0.5 | 动态 EMA 上限（仅 `linear` 模式） |
| `flow_sar_energy_type` | `mse` | 能量类型："mse"（ODE）或 "sde"（1/2t Mahalanobis） |
| `flow_sar_loss_variant` | `softplus_kl` | 损失变体："mse_branch"（DiffusionNFT）或 "softplus_kl"（π-StepNFT） |
| `flow_sar_kl_coeff` | 1.0 | KL 惩罚系数（仅 `softplus_kl` 模式） |
| `flow_sar_t_min` | 0.2 | 训练阶段 flow matching 时间下界 |
| `flow_sar_t_max` | 0.8 | 训练阶段 flow matching 时间上界 |
| `flow_sar_w_min` | 0.0 | 权重裁剪下界（0=不裁剪） |
| `flow_sar_w_max` | 1.0 | 权重裁剪上界（1=不裁剪） |

---

## 6. 使用方式

### 启动 FlowSAR 训练

```bash
# FlowSAR 训练（一键切换）
python train.py --config-name=libero_object_flowsar_openpi_quickstart
```

### 自定义超参数

```bash
python train.py --config-name=libero_object_flowsar_openpi_quickstart \
  algorithm.flow_sar_beta=1.5 \
  algorithm.flow_sar_temperature=0.3 \
  algorithm.flow_sar_tau_mid=0.4
```

---

## 7. 设计决策记录

1. **Self-Annotation 在 Rollout Worker 上执行**：与 FlowIPO 参考动作计算一致，利用非 FSDP 模型 + cpu_weight_swap，避免 actor worker 的 FSDP 限制。重建误差计算只需 1 次 forward pass（无 ODE 多步推理），比 FlowIPO 的参考动作生成更快。

2. **v_old 预计算也在 Rollout Worker 上**：训练阶段的对比损失需要 $v^{\text{old}}(x_t, t, s_i)$。由于 actor worker 运行在 FSDP 下，加载 EMA 权重会有 handle state 冲突。因此预采样 $(t, \epsilon)$ 并在 rollout worker 上预计算 $v^{\text{old}}$，存入 forward_inputs 传给 actor。

3. **EMA 机制复用 FlowIPO**：FlowSAR 和 FlowIPO 使用完全相同的 EMA reference model 机制（`_ref_weights_cpu` + `cpu_weight_swap` + `sync_model_from_actor` 中的指数移动平均更新）。

4. **双损失变体可切换**：`loss_variant` 支持 `mse_branch`（DiffusionNFT 风格，有界回归，无需额外正则）和 `softplus_kl`（π-StepNFT 风格，softplus 排序 + 显式 KL 惩罚）。两者共享镜像构造，通过 YAML 一键切换。π-StepNFT Theorem 4.5 指出 wMSE 有隐式分离惩罚，softplus 排序理论上更优。

5. **双能量类型可切换**：`energy_type` 支持 `mse`（ODE，推荐）和 `sde`（1/2t Mahalanobis 缩放）。DiffusionNFT 的消融（Fig. 9）表明类似方向的时间加权 $w(t)=1-t$ 导致训练崩溃，建议默认用 `mse`。

6. **DiffusionNFT 动态 EMA 调度**：`ema_schedule: linear` 实现 $\beta_i = \min(\eta_{\text{rate}} \cdot i, \eta_{\text{max}})$。DiffusionNFT Figure 8 验证了此调度优于固定 EMA：早期快速跟踪防止 off-policy 崩溃，后期保守保持稳定。

7. **`_is_flow_ipo or _is_flow_sar` 统一判断**：EMA 维护、`generate()` 中的额外计算等共享逻辑，使用 `self._needs_ema_ref` 属性统一判断。

8. **dummy advantages 兼容**：与 FlowIPO 一致，FlowSAR 不需要 advantages，填充全零 tensor 保持数据管线兼容。

9. **Self-Annotation 使用模型约定**（$t=0$ 为干净，$t=1$ 为噪声）：前向加噪 $x_t = (1-t) a + t \epsilon$，反向重建 $\hat{a} = x_t - v \cdot t$。注意这与 FlowSAR.md 早期理论描述中的 $\tau$ 约定（$\tau=0$ 为噪声）相反，实现以模型实际约定为准。
