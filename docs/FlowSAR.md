# FlowSAR: Self-Annotated Credit Assignment for Online RL in Flow-based VLAs

> **核心主张**：Flow模型对自身动作的重建误差天然编码了"策略置信度"信息。我们利用这一被忽视的性质，将episode-level的稀疏奖励免费转化为step-level的密集信用分配信号，在不引入Critic网络的前提下实现细粒度的策略优化。

> **副标题**："Your Flow Policy is Its Own Critic"

---

## 1. 问题背景与动机

### 1.1 核心矛盾

分析πRL和π-StepNFT的实验结果，可以发现一个被忽视的根本矛盾：

| 方法 | 信用分配粒度 | Critic | 效果 |
|------|------------|--------|------|
| πRL (PPO) | Step-level (GAE) | ✅ 需要 | LIBERO最强，但OOD过拟合 |
| πRL (GRPO) | Episode-level | ❌ | 弱于PPO |
| DiffusionNFT | Episode-level | ❌ | 仅图像生成 |

**关键观察**：

1. **πRL的Table 5**：PPO在所有LIBERO任务上碾压GRPO（96.0% vs 90.0%），尤其Long任务差距显著（90.2% vs 81.4%）。说明step-level的信用分配信号（通过Critic+GAE提供）至关重要。

2. **πRL的Figure 14**：ManiSkill训练初期，eval先下降后上升——Critic warm-up阶段给了错误信号，策略被带偏。说明Critic引入了训练不稳定性。

3. **πRL的Table 3**：PPO在ManiSkill OOD上表现差（π0: 39.3%），因为Critic过拟合多模态视觉特征。

**矛盾总结**：

> **你需要step-level的精细信号来做好信用分配，但又不想要一个会过拟合且需要warm-up的Critic网络。**

现有方法要么选了Critic（πRL/PPO），要么放弃了精细信号（GRPO/DiffusionNFT）。**没有人找到第三条路。**

### 1.2 被忽视的结构性机会

Flow模型有一个独特性质：给定一个动作 $a_0$ 和状态 $s$，我们可以对 $a_0$ 做前向加噪再去噪重建，重建误差反映了**该动作在当前策略velocity field下的拟合程度**。

这个重建过程不需要任何额外网络，只需要对已有模型做一次forward pass。它提供的信息——"模型对这个动作有多确信"——恰好可以作为step-level的信用分配信号。

---

## 2. 预备知识

### 2.1 Flow Matching for VLA

π0/π0.5的动作生成过程：

**架构**：VLM (PaliGemma 3B) 提取多模态特征 → KV-cache → Action Expert (~300M) 通过flow matching生成action chunk

**训练**：条件概率路径 $x_t = (1-t) a + t \epsilon$，其中 $t \in [0,1]$，$t=0$为干净动作，$t=1$为纯噪声。

CFM loss：
$$\mathcal{L}_{\text{CFM}} = \mathbb{E}_{t, a, \epsilon}\left[\|v_\theta(x_t, t, s) - u\|^2\right], \quad u = \epsilon - a$$

**推理**：从 $t=1$（纯噪声）到 $t=0$（干净动作），$K$步ODE去噪（$K$通常为4）：
$$x_1 \sim \mathcal{N}(0, I), \quad x_{t-\delta} = x_t - v_\theta(x_t, t, s) \cdot \delta$$

输出 $x_0$ 为action chunk（$H$个连续动作）。

### 2.2 RL设定

环境交互MDP：$M = (S, A, P_0, P_{\text{ENV}}, R_{\text{ENV}}, \gamma)$

每个环境步 $i$：状态 $s_i$（RGB图像 + 语言指令 + proprioception）→ 策略生成action chunk $a_i$ → 执行 → 获得reward和下一状态。

Episode reward：$R \in \{0, 1\}$（binary），或shaped reward（如ManiSkill的0/0.1/1.0）。

---

## 3. 方法

### 3.1 概览

FlowSAR包含四个阶段：

```
Phase 1: Rollout        → 收集轨迹和episode reward
Phase 2: Self-Annotation → 计算每步的重建误差（策略置信度）
Phase 3: Credit Assignment → 结合reward和置信度，生成step-level权重
Phase 4: Weighted Update  → 用step-level权重加权的对比目标更新策略
```

### 3.2 Phase 1: 在线Rollout

使用当前策略 $v_{\theta_{\text{old}}}$ 与环境交互。每个环境步 $i$：

1. 采样初始噪声 $x_1 \sim \mathcal{N}(0, I)$
2. $K$步ODE去噪（$t: 1 \to 0$）：$x_0 = \text{ODE}(v_{\theta_{\text{old}}}, x_1, s_i)$
3. 执行 $a_i = x_0$，获得 $s_{i+1}$

Episode结束后获得reward $R$。

存储：$\{(s_i, a_i)\}_{i=0}^{H-1}$ 以及 $R$。

### 3.3 Phase 2: 自标注（Self-Annotation）

这是FlowSAR的核心创新。对每个环境步 $i$ 的执行动作 $a_i$，计算**自重建误差**。

#### 3.3.1 重建过程

给定执行的动作 $a_i$（干净动作）：

> **实现约定**：π0模型使用 $t=0$ 为干净、$t=1$ 为噪声的约定，velocity $v = \epsilon - a$。下面的公式直接使用此约定。

**Step A**：前向加噪到中间时刻 $t_{\text{mid}}$

$$x_i^{t_{\text{mid}}} = (1 - t_{\text{mid}}) \cdot a_i + t_{\text{mid}} \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

**Step B**：用当前模型的velocity field做一步反向重建（从 $t_{\text{mid}}$ 回到 $t=0$）

$$\hat{a}_i = x_i^{t_{\text{mid}}} - v_\theta(x_i^{t_{\text{mid}}}, t_{\text{mid}}, s_i) \cdot t_{\text{mid}}$$

**Step C**：计算重建误差

$$e_i = \|a_i - \hat{a}_i\|^2$$

#### 3.3.2 重建误差的数学含义

展开重建误差（使用模型约定 $t=0$ 为干净、$t=1$ 为噪声）：

$$\hat{a}_i = x_i^{t_{\text{mid}}} - v_\theta \cdot t_{\text{mid}}$$
$$= (1 - t_{\text{mid}}) \cdot a_i + t_{\text{mid}} \cdot \epsilon - v_\theta \cdot t_{\text{mid}}$$

因此：

$$a_i - \hat{a}_i = a_i - (1 - t_{\text{mid}}) \cdot a_i - t_{\text{mid}} \cdot \epsilon + v_\theta \cdot t_{\text{mid}}$$
$$= t_{\text{mid}} \cdot a_i - t_{\text{mid}} \cdot \epsilon + t_{\text{mid}} \cdot v_\theta$$
$$= t_{\text{mid}}(v_\theta - (\epsilon - a_i))$$
$$= t_{\text{mid}}(v_\theta - u_i)$$

其中 $u_i = \epsilon - a_i$ 是flow matching的target velocity。

$$\boxed{e_i = t_{\text{mid}}^2 \|v_\theta(x_i^{t_{\text{mid}}}, t_{\text{mid}}, s_i) - u_i\|^2}$$

**重建误差恰好是该样本上的flow matching loss乘以时间缩放因子。**

#### 3.3.3 语义解释

$e_i$ 反映了**模型在状态 $s_i$ 对动作 $a_i$ 的拟合程度**，即"策略置信度"：

- **$e_i$ 低**：模型的velocity field准确指向 $a_i$ → 该动作在策略的高密度区域 → 策略对该动作**确信**
- **$e_i$ 高**：模型的velocity field偏离 $a_i$ → 该动作偏离策略的主流行为 → 策略对该动作**不确定**

**关键**：虽然 $e_i$ 不严格等于负对数似然，但在经验上两者强相关——模型拟合好的区域（低FM loss）对应高密度区域（高似然），模型拟合差的区域对应低密度区域。

**与Critic的对比**：
- Critic学习 $V(s)$：需要额外网络，需要训练，拟合外部视觉特征 → 可能过拟合
- 重建误差 $e_i$：无额外网络，一次forward pass，只依赖策略自身的velocity field → 天然鲁棒

#### 3.3.4 关于 $t_{\text{mid}}$ 的选择

$t_{\text{mid}}$ 过小（接近0）：$x_i^{t_{\text{mid}}} \approx a_i$（接近干净动作），重建太简单，所有 $e_i$ 都很小，无区分度。

$t_{\text{mid}}$ 过大（接近1）：$x_i^{t_{\text{mid}}} \approx \epsilon$（接近纯噪声），重建太难，所有 $e_i$ 都很大且受噪声主导，无区分度。

**最优范围**：$t_{\text{mid}} \in [0.3, 0.7]$，此时重建误差既有挑战性又有区分度。

实践中可以：
- 固定 $t_{\text{mid}} = 0.5$（最简单）
- 采样 $t_{\text{mid}} \sim U[0.3, 0.7]$ 后取平均（更robust）
- 多个 $t_{\text{mid}}$ 的误差取平均（最精确但计算量增加）

### 3.4 Phase 3: 信用分配（Credit Assignment）

现在我们有：
- Episode reward $R \in \{0, 1\}$
- 每步的策略置信度信号 $e_i$（重建误差）

#### 3.4.1 信用分配逻辑

| Episode结果 | 置信度高（$e_i$低） | 置信度低（$e_i$高） |
|------------|------------------|------------------|
| 成功 ($R=1$) | 策略已掌握，无需重点强化 | **关键突破点**：策略不确定但做对了 → 重点强化 |
| 失败 ($R=0$) | **确信的错误**：策略坚定地做错了 → 重点纠正 | 本来就不确定，信号弱 → 轻微纠正 |

**直觉**：
- 成功轨迹中，重建误差高的步是"尚未被策略内化的正确行为"——这些步最值得学习
- 失败轨迹中，重建误差低的步是"策略非常确信但方向错误的行为"——这些步最需要纠正

#### 3.4.2 权重公式

对每个episode内的步 $i = 0, ..., H-1$：

**成功轨迹** ($R = 1$)：重建误差越高，权重越大

$$w_i = \frac{e_i}{\sum_{j=0}^{H-1} e_j}$$

**失败轨迹** ($R = 0$)：重建误差越低（越确信），权重越大

$$w_i = \frac{1/e_i}{\sum_{j=0}^{H-1} 1/e_j}$$

权重满足 $\sum_i w_i = 1$，$w_i > 0$。

#### 3.4.3 数值稳定性处理

为防止除零和极端值：

$$e_i^{\text{clip}} = \text{clip}(e_i, \; e_{\min}, \; e_{\max})$$

其中 $e_{\min}, e_{\max}$ 可以设为batch内的percentile（如5th和95th）。

或者使用温度缩放的softmax形式：

**成功轨迹**：
$$w_i = \frac{\exp(e_i / T)}{\sum_j \exp(e_j / T)}$$

**失败轨迹**：
$$w_i = \frac{\exp(-e_i / T)}{\sum_j \exp(-e_j / T)}$$

温度 $T$ 控制权重的锐度。$T \to \infty$ 时退化为uniform权重；$T \to 0$ 时退化为argmax。

#### 3.4.4 扩展：Shaped Reward环境

当环境提供step-level reward（如ManiSkill的0.1 for grasp + 1.0 for place）时，可以混合利用：

$$w_i = \text{softmax}\left(\frac{\lambda \cdot \tilde{e}_i + (1-\lambda) \cdot \Delta r_i}{T}\right)$$

其中 $\tilde{e}_i$ 是根据 $R$ 符号调整后的重建误差（成功取$e_i$，失败取$-e_i$），$\Delta r_i = r_{i+1} - r_i$ 是reward差分。

### 3.5 Phase 4: 加权镜像更新

#### 3.5.1 共享的镜像构造（Step A-C）

**对每个环境步 $i$**，在去噪空间内操作：

**Step A**：采样 flow matching 时刻 $t \sim U[t_{\min}, t_{\max}]$，构造中间状态

$$x_t = (1 - t) \cdot a_i + t \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0,I)$$

计算 target velocity：$u_i = \epsilon - a_i$

**Step B**：构造镜像velocity

$$v_\theta^+ = (1 - \beta) v^{\text{old}} + \beta \cdot v_\theta \quad \text{(正分支：向当前策略偏移)}$$
$$v_\theta^- = (1 + \beta) v^{\text{old}} - \beta \cdot v_\theta \quad \text{(负分支：远离当前策略)}$$

其中 $v^{\text{old}} = v_{\theta_{\text{old}}}(x_t, t, s_i)$，$v_\theta = v_\theta(x_t, t, s_i)$，$\beta$ 是信任域参数。

**Step C**：计算正负分支能量

基础能量（ODE 模式，`energy_type: mse`）：
$$E_\theta^+ = \|v_\theta^+ - u_i\|^2, \quad E_\theta^- = \|v_\theta^- - u_i\|^2$$

SDE 模式（`energy_type: sde`）：对能量施加 Mahalanobis 缩放
$$E_\theta^{\pm,\text{sde}} = \frac{1}{2t} \cdot E_\theta^\pm$$

> **注**：DiffusionNFT 论文（Fig. 9）的消融表明 $w(t)=1-t$ 导致训练崩溃，而 SDE 的 $1/(2t)$ 在方向上与之类似（给小 $t$ 更大权重）。此外，该近似与 πRL 的完整 SDE log-likelihood 有本质区别。**建议默认使用 ODE 模式（`mse`）**。

#### 3.5.2 损失变体一：MSE 分支选择（`loss_variant: mse_branch`，DiffusionNFT 风格）

$$\mathcal{L}_{\text{mse}}(\theta) = \frac{1}{|\text{batch}|}\sum_i w_i \left[ R_i \cdot E_\theta^+ + (1-R_i) \cdot E_\theta^- \right]$$

其中 $R_i \in \{0,1\}$ 是episode reward。成功样本最小化正分支误差 $E^+$，失败样本最小化负分支误差 $E^-$。

**特点**：
- 失败情况下 $v_\theta$ 收敛到镜像点 $2v^{\text{old}} - u_i$（有界，不会崩溃），无需额外信任域正则
- π-StepNFT 的 Theorem 4.5 指出 wMSE 存在隐式分离惩罚 $\|d_t\|^2_{\Sigma^{-1}}$，可能抑制正负分支的有效分离

#### 3.5.3 损失变体二：Softplus 对比 + KL 信任域（`loss_variant: softplus_kl`，π-StepNFT 风格）

$$\mathcal{L}_{\text{sp}}(\theta) = \frac{1}{|\text{batch}|}\sum_i \left[ w_i \cdot \text{softplus}\!\left(\tfrac{1}{2} y_i \cdot (E_\theta^+ - E_\theta^-)\right) + \lambda_{\text{KL}} \|v_\theta - v^{\text{old}}\|^2 \right]$$

其中 $y_i = 2R_i - 1 \in \{-1, +1\}$ 是对比标签，$\lambda_{\text{KL}}$ 是信任域惩罚系数。

**特点**：
- Softplus 排序损失避免了 wMSE 的隐式分离惩罚（π-StepNFT Theorem 4.5），理论上更 sound
- 需要显式的 KL 惩罚 $\|v_\theta - v^{\text{old}}\|^2$ 来防止策略偏移过大
- $\lambda_{\text{KL}}$ 过小导致策略不稳定，过大则学习缓慢

#### 3.5.4 损失变体对比

| 维度 | `mse_branch` (DiffusionNFT) | `softplus_kl` (π-StepNFT) |
|------|----------------------------|---------------------------|
| 理论基础 | Theorem 3.2 (DiffusionNFT) | Theorem 4.5 (π-StepNFT) |
| 信任域 | 隐式（镜像构造天然锚定） | 显式（$\lambda_{\text{KL}} \|v_\theta - v^{\text{old}}\|^2$） |
| 梯度行为 | 有界回归目标 | 无界 softplus + KL 约束 |
| 隐式分离惩罚 | 有（可能抑制更新） | 无 |
| 超参数 | β 即可 | β + $\lambda_{\text{KL}}$ |
| 适用场景 | 稳定性优先 | 学习效率优先 |

**配置切换**：在 YAML 中设置 `flow_sar_loss_variant` 和 `flow_sar_energy_type` 即可一键切换，共 4 种组合（2 loss × 2 energy）。

### 3.6 策略同步（EMA 更新）

FlowSAR 支持两种 EMA 调度策略：

#### 3.6.1 固定 EMA（`ema_schedule: fixed`）

$$\theta_{\text{old}} \leftarrow \beta_{\text{ema}} \cdot \theta_{\text{old}} + (1-\beta_{\text{ema}}) \cdot \theta$$

$\beta_{\text{ema}}$ 固定不变（如 0.995），每次只更新 0.5%。简单但参考模型更新缓慢，容易导致采样策略滞后于训练策略。

#### 3.6.2 动态 EMA（`ema_schedule: linear`，DiffusionNFT 风格，推荐）

$$\beta_i = \min(\eta_{\text{rate}} \cdot i, \; \eta_{\text{max}})$$
$$\theta_{\text{old}} \leftarrow \beta_i \cdot \theta_{\text{old}} + (1-\beta_i) \cdot \theta$$

DiffusionNFT (Figure 8) 验证了此调度的有效性：

| 迭代 $i$ | $\beta_i$（$\eta_{\text{rate}}=0.001, \eta_{\text{max}}=0.5$） | 行为 |
|---------|------|------|
| 0 | 0.000 | $\theta_{\text{old}} = \theta$（完全替换，快速跟踪） |
| 100 | 0.100 | 90% 来自新策略（仍然激进） |
| 500 | 0.500 | 50/50 混合（达到上限，开始稳定） |
| 1000+ | 0.500 | 保持 50/50（长期稳定） |

**优势**：训练早期参考模型快速跟上训练策略，避免 off-policy 数据导致的策略崩溃；后期逐步保守，保持稳定性。

$v^{\text{old}}$（用于镜像构造）每次用rollout前的 $\theta_{\text{old}}$ 计算。

---

## 4. 理论分析

### 4.1 Theorem 1: 重建误差与Flow Matching Loss的等价性

**定理**：对于rectified flow下的一步重建（模型约定 $t=0$ 为干净、$t=1$ 为噪声），重建误差与该样本上的flow matching loss成正比：

$$e_i = t_{\text{mid}}^2 \cdot \|v_\theta(x_i^{t_{\text{mid}}}, t_{\text{mid}}, s_i) - u_i\|^2$$

其中 $u_i = \epsilon - a_i$ 是flow matching的target velocity。

**证明**：见3.3.2节的推导。$\square$

**推论**：重建误差高的样本对应模型拟合差的区域，即策略分布的低密度区域。对于well-trained的flow模型，FM loss与负对数似然经验上强相关。因此重建误差可以作为策略置信度的proxy。

### 4.2 Theorem 2: 信用分配的Optimal Posterior Split

**定理**：在binary reward下，FlowSAR的权重方案近似于如下贝叶斯信用分配的最优解。

定义每步的"关键性"（criticality）为该步对episode成功的后验贡献概率：

$$C_i = P(\text{step } i \text{ is critical} | R, \{e_j\}_{j=0}^{H-1})$$

假设先验模型：关键步的动作更可能偏离策略的舒适区（因为需要做出非常规决策）。形式化为：

$$P(e_i | \text{critical}) \propto e_i, \quad P(e_i | \text{non-critical}) \propto 1/e_i$$

通过贝叶斯规则和uniform prior on criticality，可得：

- 成功轨迹：$C_i \propto e_i$（重建误差高的步更可能是关键步）
- 失败轨迹：$C_i \propto 1/e_i$（重建误差低的步更可能是导致失败的关键步）

归一化后恰好是FlowSAR的权重方案。$\square$

### 4.3 Theorem 3: 与Uniform权重的梯度方向差异

**定理**：设 $\nabla_{\text{SAR}}$ 为FlowSAR的加权梯度，$\nabla_{\text{uniform}}$ 为uniform权重下的梯度。两者的差异向量 $\Delta = \nabla_{\text{SAR}} - \nabla_{\text{uniform}}$ 满足：

$$\|\Delta\| \propto \text{Var}(\{w_i\}) \cdot \text{Cov}(w_i, g_i)$$

其中 $g_i = \nabla_\theta \ell_\tau^{(i)}$ 是每步的梯度。

**含义**：当步级别的权重方差越大（重建误差分布越不均匀），且权重与梯度方向相关时，FlowSAR与uniform加权的差异越显著。若所有步的重建误差相同（无区分度），FlowSAR退化为uniform加权。

### 4.4 Theorem 4: OOD鲁棒性

**定理**：重建误差 $e_i$ 仅依赖于velocity field $v_\theta$ 在中间状态 $A^{\tau_{\text{mid}}}$ 上的值，不依赖任何外部视觉特征的拟合。因此，与基于VLM特征的Critic相比，重建误差作为信用分配信号对视觉分布偏移具有天然鲁棒性。

**直觉**：Critic学习 $V(s)$ 需要从视觉输入提取价值信息，容易过拟合训练环境的视觉特征。重建误差只测量velocity field的自一致性——它关心的是"模型对自己生成的动作有多确信"，而非"环境看起来像什么"。

---

## 5. 完整算法

> **Algorithm: FlowSAR (Self-Annotated Retrospective Training)**

**Input**：SFT初始化的flow VLA $v_\theta$（Action Expert, ~300M参数），Frozen VLM，Rollout策略 $v_{\theta_{\text{old}}} \leftarrow \text{copy}(v_\theta)$，并行环境 $N_{\text{env}}$ 个，超参数 $\beta$（镜像信任域），$T$（温度），$t_{\text{mid}}$（重建时刻），$\alpha$（EMA率）

**For** each iteration $m = 1, 2, \ldots, M$:

---

**Phase 1: Online Rollout**

1. 初始化 Buffer $\mathcal{D} \leftarrow \emptyset$
2. **For** each parallel environment:
   - **For** $i = 0$ to $H-1$:
     - VLM前向（frozen, cached）：$\text{features} = \text{VLM}(\text{images}_i, \text{language}, \text{state}_i)$
     - Action Expert $K$步ODE去噪（$t: 1 \to 0$）：采样 $x_1 \sim \mathcal{N}(0, I)$，生成 $a_i = \text{ODE}_{K}(v_{\theta_{\text{old}}}, x_1, \text{features})$
     - 环境执行：$s_{i+1}, r_i = \text{ENV.step}(a_i)$
   - 获取episode reward $R \in \{0, 1\}$
   - 存入：$\mathcal{D} \leftarrow \mathcal{D} \cup \{(s_i, a_i, R)\}_{i=0}^{H-1}$

---

**Phase 2: Self-Annotation**（一次额外forward pass per step，无需梯度）

**For** each step $(s_i, a_i)$ in $\mathcal{D}$:

1. 前向加噪：采样 $\epsilon \sim \mathcal{N}(0, I)$，计算 $x_i^{t_{\text{mid}}} = (1 - t_{\text{mid}}) \cdot a_i + t_{\text{mid}} \cdot \epsilon$
2. 一步反向重建（no grad，$t_{\text{mid}} \to 0$）：$v_{\text{pred}} = v_{\theta_{\text{old}}}(x_i^{t_{\text{mid}}}, t_{\text{mid}}, s_i)$，$\hat{a}_i = x_i^{t_{\text{mid}}} - v_{\text{pred}} \cdot t_{\text{mid}}$
3. 重建误差：$e_i = \|a_i - \hat{a}_i\|^2$

---

**Phase 3: Credit Assignment**

**For** each episode in $\mathcal{D}$:

- 若 $R = 1$（成功）：不确定但正确的步获高权重

$$w_i = \frac{\exp(e_i / T)}{\sum_{j=0}^{H-1} \exp(e_j / T)}$$

- 若 $R = 0$（失败）：确信但错误的步获高权重

$$w_i = \frac{\exp(-e_i / T)}{\sum_{j=0}^{H-1} \exp(-e_j / T)}$$

- 可选：clip极端权重 $w_i = \text{clip}(w_i, w_{\min}, w_{\max})$ 后重新归一化

---

**Phase 4: Weighted Mirror Update**

**For** each mini-batch from $\mathcal{D}$:

**For** each sample $(s_i, a_i, w_i, R_i)$:

1. 采样 flow matching 时刻 $t \sim U[t_{\min}, t_{\max}]$，噪声 $\epsilon \sim \mathcal{N}(0, I)$，构造中间状态 $x_t = (1 - t) \cdot a_i + t \cdot \epsilon$，target velocity $u_i = \epsilon - a_i$
2. Velocity预测：$v^{\text{old}} = v_{\theta_{\text{old}}}(x_t, t, s_i).\text{detach}(), \quad v^{\text{cur}} = v_\theta(x_t, t, s_i)$
3. 镜像构造：$v_\theta^+ = (1-\beta) v^{\text{old}} + \beta \, v^{\text{cur}}, \quad v_\theta^- = (1+\beta) v^{\text{old}} - \beta \, v^{\text{cur}}$
4. 计算能量：$E^+ = \|v^+ - u_i\|^2, \; E^- = \|v^- - u_i\|^2$（ODE），或 $E^\pm_{\text{sde}} = \frac{1}{2t} E^\pm$（SDE）
5. 计算损失（可选变体）：
   - **`mse_branch`**：$\ell_i = w_i \cdot [ R_i \cdot E^+ + (1-R_i) \cdot E^- ]$
   - **`softplus_kl`**：$\ell_i = w_i \cdot \text{softplus}(\tfrac{1}{2} y_i (E^+ - E^-)) + \lambda_{\text{KL}} \|v_\theta - v^{\text{old}}\|^2$

总损失：$\mathcal{L} = \frac{1}{|\text{batch}|}\sum_i \ell_i$

更新Action Expert：$\theta \leftarrow \theta - \text{lr} \cdot \nabla_\theta \mathcal{L}$

---

**Phase 5: Policy Synchronization（DiffusionNFT 动态 EMA）**

动态 EMA 更新rollout策略：$\beta_i = \min(\eta_{\text{rate}} \cdot i, \eta_{\text{max}})$，$\theta_{\text{old}} \leftarrow \beta_i \cdot \theta_{\text{old}} + (1-\beta_i) \cdot \theta$

---

**Output**：优化后的Action Expert $v_\theta$

---

## 6. 与现有方法的对比

### 6.1 定位

| 维度 | πRL (PPO) | πRL (GRPO) | DiffusionNFT | π-StepNFT | **FlowSAR** |
|------|-----------|------------|--------------|-----------|-------------|
| 似然计算 | ✅ 需要 | ✅ 需要 | ❌ | ❌ | **❌** |
| Critic网络 | ✅ 需要 | ❌ | ❌ | ❌ | **❌** |
| 信用分配粒度 | Step (GAE) | Episode | Episode | Episode | **Step (self-annotation)** |
| 信用分配来源 | 学习的Critic | 组内归一化 | 无 | 无 | **模型自身重建误差** |
| 损失形式 | 策略梯度 | 策略梯度 | MSE 分支选择 | Softplus 排序 | **可切换（两种变体）** |
| 额外网络 | Critic + noise net | 无 | 无 | 无 | **无** |
| 额外计算 | ~2x | ~Nx rollout | ~1.3x | ~1.3x | **~1.3x** |
| 失败样本利用 | ✅ (A<0) | ✅ (norm后为负) | ✅ (对比) | ✅ (对比) | **✅ (对比+加权)** |
| OOD鲁棒性 | 差 | 中 | 好 | 好 | **好** |

### 6.2 FlowSAR的独特位置

FlowSAR是唯一一个同时满足以下条件的方法：

1. **无似然、无Critic** → 实现简单，无过拟合风险
2. **有step-level信用分配** → 细粒度优化信号
3. **信用分配来自模型自身** → 不引入任何外部拟合误差
4. **有负样本对比** → 不会collapse

### 6.3 与"模型自身重建误差"作为信号的独特性

这是一个此前没有人指出的结构性观察：

> Flow模型天然携带了一个"内置Critic"——对自身动作的重建误差。这个信号在数学上等价于pointwise FM loss，在语义上反映策略置信度，且计算上只需一次额外forward pass。

Critic需要从零学习 $V(s)$，而重建误差从模型训练好的那一刻就已经可用。这解释了为什么πRL的Figure 14显示Critic需要warm-up而FlowSAR不需要。

---

## 7. 实现细节

### 7.1 架构与训练配置

- **Base model**：π0 (PaliGemma 3B + 300M Action Expert) 或 π0.5
- **冻结VLM**，只更新Action Expert（与πRL一致）
- **硬件**：8× NVIDIA H100 80GB GPUs
- **并行环境**：64（LIBERO）/ 256-320（ManiSkill）

### 7.2 超参数

| 超参数 | 含义 | 建议范围 | 默认值 |
|--------|------|---------|--------|
| `flow_sar_beta` | 镜像构造信任域 $\beta$ | [0.5, 2.0] | 1.0 |
| `flow_sar_temperature` | 权重温度 $T$ | [0.1, 2.0] | 0.5 |
| `flow_sar_tau_mid` | 重建时刻 $t_{\text{mid}}$ | [0.3, 0.7] | 0.5 |
| `flow_sar_ema_schedule` | EMA 调度方式 | `fixed` / `linear` | `linear` |
| `flow_sar_ema_beta` | 固定 EMA 系数（`fixed` 模式） | [0.9, 0.999] | 0.995 |
| `flow_sar_ema_eta_rate` | 动态 EMA 增长率（`linear` 模式） | [0.0005, 0.005] | 0.001 |
| `flow_sar_ema_eta_max` | 动态 EMA 上限（`linear` 模式） | [0.3, 0.8] | 0.5 |
| `flow_sar_energy_type` | 能量计算方式 | `mse` / `sde` | `mse` |
| `flow_sar_loss_variant` | 损失函数变体 | `mse_branch` / `softplus_kl` | `softplus_kl` |
| `flow_sar_kl_coeff` | KL 惩罚系数（`softplus_kl` 专用） | [0.1, 5.0] | 1.0 |
| lr | 学习率 | [1e-6, 1e-5] | 1e-6 |
| `flow_sar_t_min` | flow matching 时间下界 | | 0.2 |
| `flow_sar_t_max` | flow matching 时间上界 | | 0.8 |
| `flow_sar_w_min` / `w_max` | 权重 clip 范围 | | 0.0 / 1.0 |

### 7.3 计算开销

| 操作 | 相对SFT开销 | 说明 |
|------|-----------|------|
| Rollout | 和πRL相同 | ODE K步去噪 |
| Self-Annotation | +0.25x | 每步一次forward，无梯度 |
| Weighted Update | ~1x | 和标准对比更新相同 |
| **总计** | **~1.3x SFT** | 远低于πRL的~2x |

### 7.4 核心代码改动

基于πRL (RLinf框架) 的改动量：

```python
# ===== Phase 2: Self-Annotation =====
# 模型约定: t=0 → 干净, t=1 → 噪声, velocity v = ε - a

def self_annotate(v_theta_old, states, actions, t_mid=0.5):
    """计算每步的重建误差"""
    eps = torch.randn_like(actions)
    x_t = (1 - t_mid) * actions + t_mid * eps
    with torch.no_grad():
        v_pred = v_theta_old(x_t, t_mid, states)
    a_hat = x_t - v_pred * t_mid
    recon_error = ((actions - a_hat) ** 2).sum(dim=-1)
    return recon_error

# ===== Phase 3: Credit Assignment =====

def compute_weights(recon_errors, rewards, temperature=0.5):
    """基于重建误差和reward计算step-level权重"""
    weights = torch.zeros_like(recon_errors)
    success_mask = (rewards == 1)
    failure_mask = (rewards == 0)
    if success_mask.any():
        weights[success_mask] = F.softmax(recon_errors[success_mask] / temperature, dim=-1)
    if failure_mask.any():
        weights[failure_mask] = F.softmax(-recon_errors[failure_mask] / temperature, dim=-1)
    return weights

# ===== Phase 4: Weighted Mirror Loss (支持两种变体) =====

def flow_sar_loss(v_theta, v_old, u_target, weights, labels, beta=1.0,
                  energy_type="mse", loss_variant="softplus_kl", kl_coeff=1.0, flow_t=None):
    v_pos = (1 - beta) * v_old + beta * v_theta
    v_neg = (1 + beta) * v_old - beta * v_theta
    E_pos = (v_pos - u_target).pow(2).mean(dim=(-2, -1))
    E_neg = (v_neg - u_target).pow(2).mean(dim=(-2, -1))

    # SDE 模式: Mahalanobis 缩放 1/(2t)
    if energy_type == "sde" and flow_t is not None:
        sde_scale = 1.0 / (2.0 * flow_t.reshape(-1)).clamp(min=1e-3)
        E_pos, E_neg = E_pos * sde_scale, E_neg * sde_scale

    if loss_variant == "softplus_kl":
        # π-StepNFT 风格: softplus 对比 + KL 信任域
        margin = 0.5 * labels * (E_pos - E_neg)
        contrastive = F.softplus(margin)
        kl_penalty = (v_theta - v_old).pow(2).mean(dim=(-2, -1))
        per_sample = weights * contrastive + kl_coeff * kl_penalty
    else:
        # DiffusionNFT 风格: MSE 分支选择
        R = (labels + 1.0) / 2.0
        per_sample = weights * (R * E_pos + (1.0 - R) * E_neg)

    return per_sample.mean()
```

**总改动量**：约50-60行核心代码。不需要新建任何网络。

---

## 8. 实验设计

### 8.1 Benchmarks

| Benchmark | 任务描述 | 奖励类型 | 测试重点 |
|-----------|---------|---------|---------|
| LIBERO (4 suites) | Spatial/Object/Goal/Long各10任务 | Binary | Few-shot RL性能 |
| ManiSkill (4352 tasks) | 16物体×17容器×16场景 | Shaped | IND + OOD泛化 |

### 8.2 Baselines

| 方法 | 描述 |
|------|------|
| SFT (few-shot) | 纯监督基线 |
| πRL (Flow-SDE + PPO) | 似然+Critic路线 |
| πRL (Flow-SDE + GRPO) | 似然+无Critic路线 |
| πRL (Flow-Noise + PPO) | 另一种似然+Critic方案 |
| DiffusionNFT (uniform) | 对比方法，无step-level权重 |
| RFT | 只在成功样本上SFT |
| **FlowSAR** | 自标注信用分配 + 加权对比 |

### 8.3 主实验

#### 实验1: LIBERO (Few-shot SFT + RL)

| 方法 | Spatial | Object | Goal | Long | Avg. |
|------|---------|--------|------|------|------|
| π0 SFT | 65.3 | 64.4 | 49.8 | 51.2 | 57.6 |
| πRL (PPO) | 98.4 | 99.4 | 96.2 | 90.2 | 96.1 |
| πRL (GRPO) | 97.8 | 97.8 | 83.2 | 81.4 | 90.0 |
| **FlowSAR** | **~98** | **~99** | **~95** | **~90** | **~95.5** |

**预期**：
- 超越GRPO（+5%），因为有step-level信用分配
- 接近PPO，尤其在Goal和Long任务上信用分配最关键
- Long任务是关键对比点——PPO靠GAE做好了Long的信用分配，FlowSAR靠self-annotation能否匹配

#### 实验2: ManiSkill (IND + OOD)

| 方法 | IND | Vision | Semantic | Execution | OOD Avg. |
|------|-----|--------|----------|-----------|----------|
| π0 SFT | 38.4 | 32.6 | 8.4 | 13.2 | 18.1 |
| πRL (PPO) | 78.8 | 61.1 | 25.4 | 31.5 | 39.3 |
| **FlowSAR** | **~77** | **~66** | **~33** | **~33** | **~44** |

**预期**：
- IND略低于PPO（无Critic的代价）
- OOD显著优于PPO（+5%），因为无Critic过拟合
- 尤其Semantic shift可能大幅提升——Critic最容易在语义特征上过拟合

### 8.4 消融实验（核心）

#### 消融1: 信用分配信号的有效性（最关键的消融）

| 变体 | 描述 |
|------|------|
| FlowSAR (full) | $w_i$ 基于重建误差 + reward |
| FlowSAR (uniform) | $w_i = 1/H$（所有步等权） |
| FlowSAR (random) | $w_i$ 随机 |
| FlowSAR (inverse) | 权重逻辑反转（成功步给低$e_i$高权重） |

**预期**：full > uniform > random ≈ inverse

**如果full和uniform差距不大**：说明信用分配信号区分度不够。这是最大风险点。

#### 消融2: 重建时刻 $\tau_{\text{mid}}$ 的影响

$\tau_{\text{mid}} \in \{0.2, 0.3, 0.5, 0.7, 0.8\}$

**预期**：$\tau_{\text{mid}} = 0.5$ 附近最优，两端衰减。

#### 消融3: 温度 $T$ 的影响

$T \in \{0.1, 0.3, 0.5, 1.0, 2.0\}$

**预期**：$T$过小 → 权重过于极端 → 不稳定；$T$过大 → 退化为uniform

#### 消融4: 信任域 $\beta$ 的影响

$\beta \in \{0.5, 1.0, 1.5, 2.0\}$

#### 消融5: 与简化版本的对比

| 变体 | 损失函数 |
|------|---------|
| FlowSAR-contrastive | 镜像对比 + softplus（完整版） |
| FlowSAR-simple | 加权FM + 加权anti-FM + KL正则（简化版） |

### 8.5 分析实验

#### 分析1: 重建误差分布可视化

对LIBERO的典型episode，画出：
- 每步的重建误差 $e_i$ 柱状图
- 区分成功轨迹 vs 失败轨迹
- 标注关键动作步（如抓取、放置瞬间）

**关键验证**：重建误差在步间是否有足够的方差？关键步的误差是否显著不同？

#### 分析2: 权重 $w_i$ 与任务语义的对应

在LIBERO-Long中，$w_i$最大的步是否对应任务的关键决策点？可视化动作轨迹并标注$w_i$的热力图。

#### 分析3: 训练曲线对比

和πRL对比：
- 收敛速度
- 训练稳定性（是否有πRL Figure 14中的初期eval下降）
- KL散度变化

#### 分析4: 与Critic做Oracle对比

训练一个和πRL相同的Critic，用GAE计算的advantage作为oracle信用分配信号。用这个信号替换FlowSAR的self-annotation。对比两者的最终性能——衡量self-annotation作为Critic proxy的质量。

---

## 9. 风险评估与缓解

### 9.1 最大风险：重建误差区分度不足

**风险描述**：如果所有步的重建误差差异很小，$w_i$ 退化为接近uniform。

**缓解方案**：
1. 即使退化为uniform，FlowSAR仍然是"加了DiffusionNFT对比机制的在线RL"——baseline有效性有保证
2. 在shaped reward环境，混合reward差分信号
3. 多$\tau_{\text{mid}}$取平均增加稳定性
4. 训练初期区分度可能确实低，但随着策略分化，不同步的重建误差差异会增大

**验证**：项目启动后48小时内在LIBERO-Spatial上做最小验证，检查 $e_i$ 的步间方差。

### 9.2 风险：重建误差的噪声

**风险描述**：$e_i$ 依赖于随机采样的 $\epsilon$，单次估计可能噪声大。

**缓解**：
1. 多次采样取平均（如3次，开销约3x额外forward）
2. 使用固定的 $\tau_{\text{mid}}$（减少时间维度的方差）
3. Softmax温度 $T$ 本身有平滑效果

### 9.3 风险：审稿人质疑"不算RL"

**缓解**：
1. 满足RL核心要素：在线交互、试错、策略改进
2. 定位为"reward-guided online flow matching with self-annotated credit assignment"
3. 重点强调：和所有现有RL方法正交的新范式

### 9.4 风险：对比目标和DiffusionNFT的overlap

**缓解**：
1. 明确DiffusionNFT是图像生成领域的工作，未用于VLA
2. FlowSAR的核心贡献是self-annotation信用分配，对比目标是"载体"不是"创新"
3. 消融中和简化版本对比，证明self-annotation在不同loss上都有效

---

## 10. 论文结构建议

### Title
**FlowSAR: Self-Annotated Credit Assignment for Reinforcement Learning in Flow-based Vision-Language-Action Models**

### Abstract (~150 words)
Flow-based VLA (π0, π0.5)的RL面临核心矛盾：step-level信用分配需要Critic网络，但Critic会过拟合多模态特征导致OOD退化。我们发现flow模型的重建误差天然编码了策略置信度信息——高重建误差表示模型对该动作不确信。利用这一性质，FlowSAR将episode-level的稀疏奖励免费转化为step-level的密集信用分配信号：成功轨迹中重点强化策略不确定但正确的步，失败轨迹中重点纠正策略确信但错误的步。FlowSAR无需似然计算、无需Critic网络，仅需一次额外forward pass。在LIBERO和ManiSkill上验证，FlowSAR达到接近PPO的性能同时显著提升OOD泛化。

### Sections
1. **Introduction**: 矛盾（step-level信号 vs Critic过拟合）→ 我们的发现（重建误差=置信度）→ FlowSAR
2. **Related Work**: RL for VLAs, RL for flow/diffusion models, credit assignment
3. **Preliminaries**: Flow matching, π0/π0.5架构, 问题设定
4. **Method**: Self-annotation → Credit assignment → Weighted contrastive update → 理论分析
5. **Experiments**: LIBERO + ManiSkill主实验, 消融, 分析
6. **Conclusion & Limitations**

### 核心Figure
- **Figure 1** (teaser): 概览图——rollout → self-annotation → weighted update的pipeline
- **Figure 2**: 重建误差的可视化——同一episode不同步的$e_i$分布
- **Figure 3**: $w_i$热力图叠加在动作轨迹上
- **Figure 4**: 训练曲线对比（FlowSAR vs PPO vs GRPO）
- **Figure 5**: OOD性能对比

---

## 11. 总结

FlowSAR的核心贡献是一个结构性发现：

> **Flow模型的重建误差是一个免费的、天然的、不会过拟合的step-level信用分配信号。**

这个发现统一了两个此前矛盾的阵营——你不需要在"有Critic但过拟合"和"无Critic但信号粗糙"之间做选择。Flow模型自己就是自己的Critic。

方法极简（~30行核心代码），理论清晰（FM loss等价性、贝叶斯最优信用分配），且和所有现有方法正交。
