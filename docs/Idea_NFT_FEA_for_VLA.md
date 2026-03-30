# π-StepNFT + Frozen Embedding Advantage for VLA

## NFT-FEA：Critic-Free对比式Flow RL + 冻结嵌入时序信用分配

---

## 一、Problem Statement

### 1.1 当前Flow RL方法的困境

我们在FlowIPO框架中实现了多种flow-based RL方法，全部收敛失败或极慢：

| 方法 | 核心思想 | 结果 |
|------|---------|------|
| FlowIPO | ODE插值+对比 | 不收敛 |
| FlowSAR | 自标注+能量加权 | 收敛慢 |
| FPI | $\exp(A/\lambda)$ 加权FM | 不收敛（exp≈0） |
| AWM-VLA | 归一化A加权FM | 收敛但极慢 |
| GFN-Flow | SubTB轨迹平衡 | 不收敛 |

唯一有效的基线是 Flow-SDE + PPO（π-RL），但它需要完整的value network，且在长horizon任务上仍有明显gap。

### 1.2 根因分析

通过系统对比分析，我们发现上述方法失败的根因归结为**两个结构性缺陷**：

**根因A：加权回归的梯度消失。** FPI、AWM等方法的loss形式均为 $f(A) \cdot \|v_\theta - u\|^2$。由于VLA模型已经过大量BC预训练，flow matching MSE $\|v_\theta - u\|^2$ 对on-policy action已接近零。RL信号（advantage）乘以一个近零量，导致梯度极其微弱。AWM的实验曲线完美验证了这一点：raw_advantage_mean ≈ ±0.006，pos_frac ≈ neg_frac ≈ 0.5，advantage本质上是噪声。

**根因B：Value Network的鸡生蛋问题。** AWM/FPI依赖 $V(o)$ 提供advantage信号。但在sparse binary reward下，$V(o)$ 需要通过TD bootstrapping学习——初期 $V \approx \text{const}$ → $A \approx 0$ → 无RL信号 → $V$ 学不好。这是一个恶性循环。FlowSAR虽然不用 $V(o)$，但其reconstruction error权重在velocity-space计算，同样受梯度消失影响。

**核心洞察**：需要一种loss，其梯度**不依赖**绝对预测误差 $\|v_\theta - u\|^2$。

---

## 二、Background：π-StepNFT

### 2.1 π-StepNFT的核心设计

π-StepNFT（ICML 2026）是一种critic-free、likelihood-free的online RL方法，专为flow-based VLA设计。其关键创新：

1. **SDE链快照**：在K步SDE去噪过程中，随机采样一步的 $(x_t,\; v_\theta,\; x_{t^-})$ 作为训练数据
2. **对比镜像构造**：$v^+ = v_{\text{old}} + \beta \cdot \Delta v$，$v^- = v_{\text{old}} - \beta \cdot \Delta v$（$\Delta v$ 是策略改进方向）
3. **Sample-space能量**：用flow mean预测在sample space计算能量，并用Mahalanobis归一化 $E = \sum (x_{\text{next}} - \mu)^2 / \sigma^2$
4. **Softplus contrastive loss**：$\ell_t(\theta) = \text{softplus}\!\left(\tfrac{1}{2}\, y \cdot (E^+ - E^-)\right)$
5. **Terminal binary advantage**：成功 $y=+1$，失败 $y=-1$，所有env step统一

### 2.2 为什么π-StepNFT成功

π-StepNFT精确解决了根因A和根因B：

| 问题 | 失败方法 | π-StepNFT的解决 |
|------|---------|----------------|
| 梯度消失 | 梯度 $\propto (v_\theta - u) \to 0$ | 梯度 $\propto (v_\theta - v_{\text{old}})$，从零开始增长 |
| Value瓶颈 | 需要 $V(o)$ 做advantage | Terminal binary，无需 $V(o)$ |
| 噪声水平不均衡 | 原始MSE，高噪声主导 | Mahalanobis归一化，各层贡献均等 |
| 信号方差 | 跨样本加权回归 | 同样本内push-pull，方差低 |
| On-policy数据 | MSE≈0无信号 | 对比信号始终 $O(1)$ |

**梯度分析**（Theorem 4.4）：

$$\nabla_\theta \mathcal{L}_{\text{NFT}} \propto \sigma(z_t) \cdot y \cdot \left(\frac{\partial v_\theta}{\partial \theta}\right)^\top B_t \, \Sigma_t^{-1} \, e_t$$

其中 $e_t$ 是SDE噪声残差，**始终为 $O(\sigma)$**，不随模型收敛而消失。

### 2.3 π-StepNFT的唯一短板：Long-horizon任务

从π-StepNFT论文Table 1：
- Short任务：与PPO基本持平
- Long任务：有明显gap（~4%）
- OOD泛化：反超PPO（50.4% vs 39.3%）

Long任务gap的原因：terminal binary $y=\pm 1$ 对所有env step施加相同梯度权重。在300步长的任务中，关键决策点（抓取、放置）只占~30步，但与routine步骤（接近、等待）获得相同梯度 → 梯度被稀释10倍。

---

## 三、Method：端到端系统架构

### 3.1 系统总览

整个系统分为三个阶段，对应代码中的三个模块：

```
┌─ Stage 1: Rollout（数据收集）─────────────────────────────────────────────┐
│  openpi_action_model.py :: sample_actions(collect_flow_snap=True)         │
│                                                                           │
│  独立 SDE 去噪循环 (ALL K steps use mode="train")                         │
│     ↓ 随机选一步 k                                                        │
│  输出: (nft_xt, nft_v, nft_xnext, nft_step_index, nft_noise_level)       │
│  附带: vlm_embedding = mean_pool(prefix_output).detach()                  │
│  附带: episode_reward (env 返回的 0/1 binary reward)                      │
└───────────────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─ Stage 2: Credit Assignment（标签生成）────────────────────────────────────┐
│  fsdp_actor_worker.py :: _compute_flow_nft_credit_assignment()            │
│                                                                           │
│  输入: episode_reward, (可选) vlm_embedding                               │
│  根据 flow_nft_adv_type 选择:                                             │
│    "terminal_binary" → advantages = ±adv_clip_max (broadcast all steps)  │
│    "embedding_change" → advantages = terminal_sign × emb_weight × clip   │
│                                                                           │
│  输出: advantages [T, B] → 存入训练 buffer                                │
└───────────────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─ Stage 3: Training（loss 计算 + 反向传播）────────────────────────────────┐
│  losses.py :: compute_flow_nft_loss()                                     │
│                                                                           │
│  输入: nft_xt, nft_v(=v_old), nft_xnext, advantages, 当前模型 v_θ        │
│                                                                           │
│  (a) 镜像构造: v⁺ = v_old + β·Δv_clip,  v⁻ = v_old - β·Δv_clip         │
│  (b) Sample-space Mahalanobis 能量: E⁺, E⁻                              │
│  (c) Label: y = advantage / c_adv ∈ [-1, 1]                              │
│  (d) Loss: ℓ = softplus(½ y·(E⁺-E⁻)) + β_kl·||Δv||²                   │
│                                                                           │
│  关键: y 的符号必须正确，Stage 2 保证这一点                                │
└───────────────────────────────────────────────────────────────────────────┘
```

**三个Stage的连接点是 `advantages` 张量** $\in \mathbb{R}^{[T, B]}$：
- Stage 2 生成它（每个 env step 一个标量 $A_t$）
- Stage 3 消费它：$y = A_t / c_{\text{adv}} \in [-1, 1]$，直接代入 $\ell_t(\theta) = \text{softplus}\!\left(\tfrac{1}{2}\, y \cdot \Delta E\right)$
- $y$ 的**符号**决定梯度方向（推向好行为 or 推离坏行为），$|y|$ 决定**信号强度**（$|y|$ 越大 softplus 越陡 → 梯度越大）

### 3.2 Stage 1：SDE链快照采集

在K步SDE去噪过程中，**独立循环**（不与正常ODE推理共享），随机选择一步 $k$ 保存快照：

$$x_t^{(k)},\quad v_{\text{old}}^{(k)} = v_\theta(x_t^{(k)}, t_k).\text{detach}(),\quad x_{\text{next}}^{(k)} = \text{Euler}(x_t^{(k)}, v_\theta) + \sigma \cdot z$$

同时缓存冻结的VLM嵌入（用于Stage 2的temporal credit，零额外计算）：

$$e_t = \text{mean\_pool}(\text{prefix\_output}).\text{detach}() \quad \in \mathbb{R}^{B \times d}$$

> **关键实现细节**：所有K步必须使用 `mode="train"`（SDE噪声），不能混用ODE模式。否则快照数据的分布与Stage 3的loss计算假设不匹配，导致训练崩溃。

### 3.3 Stage 2：Credit Assignment（Temporal Credit）

这是**唯一可替换的模块**——决定每个env step获得多少梯度。Stage 3的softplus loss对此有一个硬约束：

> **Softplus安全约束**：advantage的**符号（方向）必须正确**。softplus的不对称性意味着一个错误符号造成的damage远大于正确符号的benefit。因此任何temporal credit方案都**不能改变label方向**，只能改变**权重（幅度）**。

$$\text{softplus}(x) \approx \begin{cases} e^x \to 0 & x \ll 0 \quad \text{（正确label，快速饱和）} \\ x & x \gg 0 \quad \text{（错误label，线性增长无上界）} \end{cases}$$

#### 方案A：Terminal Binary（当前默认 ✅）

最简单的方案，所有env step获得相同的信号：

$$A_t = \begin{cases} +c_{\text{adv}} & \text{if success (reward = 1)} \\ -c_{\text{adv}} & \text{if failure (reward = 0)} \end{cases} \quad \forall\, t \in [0, T)$$

- 符号永远正确 → softplus安全
- 所有step均匀权重 → Long-horizon任务中关键step被稀释（NFT唯一短板）

#### 方案B：Embedding Change Rate（提出的改进 — 待实验）

利用冻结VLM embedding的**帧间变化率**作为step importance weight。核心洞察：**场景剧变的帧（抓取、放置）对应关键决策点**。

$$w_t = \|e_{t+1} - e_t\|_2$$

$$\tilde{w}_t = \text{clamp}\!\left(\frac{w_t}{\bar{w}},\; 0.2,\; 2.0\right) \quad \text{（归一化到均值1，裁剪极端值）}$$

$$A_t = s_{\text{terminal}} \times \tilde{w}_t \times c_{\text{adv}}$$

其中 $s_{\text{terminal}} = \pm 1$ 始终来自terminal binary。

**为什么安全**：
- 符号来自terminal binary → 永远正确 → softplus安全
- $\tilde{w}_t$ 只调节幅度 → 关键step权重↑，routine step权重↓
- 不需要任何regression → 没有TD噪声 → 没有标签污染
- 对成功和失败trajectory同样有效（场景变化与outcome无关）
- 基础设施已有（vlm_embedding在rollout时已收集）

**直觉**：
| 阶段 | $\|e_{t+1} - e_t\|$ | 权重 | 说明 |
|------|---------------------|------|------|
| 手臂接近物体 | 小 | 低 | 视觉缓慢变化，routine |
| 抓取物体 | **大** | **高** | 物体位置突变，关键决策 |
| 搬运中 | 中 | 中 | 稳定移动 |
| 放置物体 | **大** | **高** | 又一次场景突变 |

#### 方案C：Cross-Trajectory Divergence（更强的方案 — 待实验）

利用成功/失败trajectory在embedding空间的**分叉点**识别关键决策步。

对于一个batch中的成功trajectory $i$ 和失败trajectory $j$，计算逐步embedding距离：

$$d_t^{(i,j)} = 1 - \cos(e_t^{(i)},\; e_t^{(j)})$$

分叉检测：$d_t$ 从低到高的跳变点 = 关键决策点：

$$\Delta d_t = d_{t+1} - d_t, \quad c_t = \max(0,\; \Delta d_t) \quad \text{（只取发散方向）}$$

$$\tilde{w}_t = 1 + \alpha \cdot \frac{c_t}{\bar{c}} \quad \text{（基础权重1 + 发散加权）}$$

$$A_t = s_{\text{terminal}} \times \tilde{w}_t \times c_{\text{adv}}$$

**为什么比方案B更强**：方案B检测"场景变了"，方案C检测"这一步之后成功和失败走向不同" —— 后者是temporal credit的精确定义。

**实际操作**：batch内对成功/失败trajectory做平均嵌入，不需要逐对匹配：

$$\bar{e}_t^{\text{succ}} = \text{mean}(e_t^{(i)} \mid i \in \text{success}), \quad \bar{e}_t^{\text{fail}} = \text{mean}(e_t^{(j)} \mid j \in \text{failure})$$

$$d_t = \|\bar{e}_t^{\text{succ}} - \bar{e}_t^{\text{fail}}\|_2$$

### 3.4 Stage 3：NFT对比Loss

这部分完全移植自π-StepNFT，不做修改。输入来自Stage 1（快照数据）和Stage 2（advantage标签）。

**Step 1: 对比镜像构造**

$$\Delta v = v_\theta - v_{\text{old}}, \quad \Delta v_{\text{clip}} = \Delta v \cdot \min\!\left(\frac{\text{max\_drift}}{\|\Delta v\|},\; 1\right)$$

$$v^+ = v_{\text{old}} + \beta \cdot \Delta v_{\text{clip}}, \quad v^- = v_{\text{old}} - \beta \cdot \Delta v_{\text{clip}}$$

**Step 2: Sample-space Mahalanobis能量**

$$\mu^{\pm} = \text{flow\_mean}(x_t, v^{\pm}), \quad \sigma^2 = \delta \cdot \sigma_i^2 + \varepsilon$$

$$E^{\pm} = \sum_{\text{dim}} \frac{(x_{\text{next}} - \mu^{\pm})^2}{\sigma^2}$$

**Step 3: 对比loss + Trust Region**（Definition 4.1, π-StepNFT）

$$y = A_t / c_{\text{adv}} \in [-1, +1] \quad \text{（来自 Stage 2，符号=方向，幅度=信号强度）}$$

$$\ell_t(\theta) = \text{softplus}\!\left(\tfrac{1}{2}\, y \cdot (E_{\theta,t}^+ - E_{\theta,t}^-)\right)$$

$$\mathcal{L} = \ell_t(\theta) + \beta_{\text{kl}} \|v_\theta - v_{\text{old}}\|^2$$

### 3.5 FEA的演化：公式级别Diff

FEA的改动**只在Stage 2**（$A_t$ 的计算），Stage 1和Stage 3完全不变。以下是三个版本的精确公式对比。

#### π-StepNFT 原始（方案A = Terminal Binary）

$$A_t = \begin{cases} +c_{\text{adv}} & \text{success} \\ -c_{\text{adv}} & \text{failure} \end{cases} \quad \forall\, t \in [0,T)$$

一行公式，所有step统一值。

#### FEA v1（第一版 — 训练崩溃 ❌）

将上面一行替换为以下四步：

$$e_t = \text{mean\_pool}(\text{prefix\_output}).\text{detach}() \quad \text{（Stage 1收集）}$$

$$w = (E^\top E + \lambda I)^{-1} E^\top G, \quad G_t = \gamma^{T-t} R \quad \text{（Ridge回归）}$$

$$V(e_t) = w^\top e_t + b, \quad \text{TD}_t = \gamma \, V(e_{t+1}) - V(e_t) \quad \text{（TD advantage）}$$

$$A_t = \text{normalize}(\text{TD}_t) \times c_{\text{adv}} \quad \text{（归一化后作为label）}$$

**改了什么**：advantage从固定 $\pm c$ 变成逐step不同的连续值。归一化后TD的**符号**成为label $y$。

**为什么崩溃**：

1. $G_t = \gamma^{T-t} R$ 是 $t$ 的确定性函数，完美拟合后 $V(e_t) = G_t$
2. $\text{TD}_t = \gamma \cdot \gamma^{T-t-1} R - \gamma^{T-t} R = 0$（理论恒等于零）
3. 实际回归不完美 → TD残差 = 拟合噪声 → 归一化放大为随机 $\pm 1$
4. softplus不对称性：错误label的loss $\approx |\Delta E|$（线性增长），正确label的loss $\approx 0$（饱和）
5. ~50%错误label造成的damage远超~50%正确label的benefit → 训练方向反转

#### FEA v2（修正版 — 慢于原始 ⚠️，当前代码）

将FEA v1的最后一步替换为：

$$s_{\text{terminal}} = \text{sign}(R - 0.5) \quad \text{（符号锁定为terminal binary）}$$

$$\text{importance}_t = \text{clamp}\!\left(\frac{|\text{TD}_t|}{\text{mean}(|\text{TD}|)},\; 0.2,\; 2.0\right) \quad \text{（幅度来自FEA）}$$

$$A_t = s_{\text{terminal}} \times \text{importance}_t \times c_{\text{adv}}$$

**改了什么（相比v1）**：不再用TD符号做label，符号锁定为terminal binary。FEA仅通过 $|\text{TD}|$ 调节幅度。

**为什么比原始慢**：失败trajectory $R=0 \Rightarrow G_t=0 \Rightarrow V \approx 0 \Rightarrow |\text{TD}| \approx 0 \Rightarrow \text{importance} = 0.2$（下限）。失败信号被削弱到原始的 $\frac{0.2}{1.0} = \frac{1}{5}$，而NFT需要均衡的正负信号。

#### 版本对比总表

| 版本 | $A_t$ 公式 | label $y$ 来源 | 结果 | 原因 |
|------|-----------|---------------|------|------|
| 原始 (方案A) | $\pm c_{\text{adv}}$ | terminal binary | 正常 ✅ | 符号永远正确 |
| FEA v1 | $\text{norm}(\text{TD}_t) \times c$ | TD advantage符号 | 崩溃 ❌ | TD≈0 → 随机符号 → softplus放大错误 |
| FEA v2 | $s_{\text{term}} \times \text{imp}_t \times c$ | terminal binary | 慢于原始 ⚠️ | 失败信号被削弱到1/5 |
| 方案B | $s_{\text{term}} \times \tilde{w}_t \times c$ | terminal binary | 待验证 | 无regression，无信号削弱 |
| 方案C | $s_{\text{term}} \times \tilde{w}_t \times c$ | terminal binary | 待验证 | 直接检测分叉点 |

#### 方案B/C vs FEA的本质区别

| | FEA v1/v2 | 方案B/C |
|---|---|---|
| Embedding用途 | Value regression → TD → label或weight | 距离度量 → weight |
| Label $y$ 是否被改 | v1改了（崩溃），v2没改 | 不改 |
| 对失败trajectory | v2: $V \approx 0$ → importance ≈ 0.2（信号削弱） | 场景变化/分叉与outcome无关（信号均衡） |
| 核心任务难度 | 回归（需精确拟合） | 距离（只需相对大小，鲁棒） |

---

## 四、NFT vs 现有方法对比

| 特性 | FlowIPO | FlowSAR | FPI | AWM | NFT |
|------|---------|---------|-----|-----|-----|
| 梯度来源 | $(v_\theta - u_{\text{interp}})$ | $(v - u)$ velocity-space | $(v_\theta - u) \cdot \tilde{A}$ | $(v_\theta - u) \cdot \tilde{A}$ | $(v_\theta - v_{\text{old}})$ **sample-space** |
| 梯度随预训练衰减? | 是 | 是 | 是 | 是 | **否** |
| 需要Value Network | 否 | 否 | 是 | 是 | **否** |
| Temporal Credit | 无 | recon error | GAE($V$) | $V$-based | **Terminal binary** |
| Push-Pull | 插值 | 镜像(velocity) | 单向 | 双向($A$正负) | **镜像(sample-space)** |
| 方差归一化 | 无 | 无/$1/(2t)$ | 无 | 无 | **Mahalanobis $\sigma^2$** |
| On-policy梯度幅度 | $O(\varepsilon)$ | $O(\varepsilon)$ | $O(\varepsilon^2)$ | $O(\varepsilon^2)$ | $O(1)$ |
| 数据来源 | random $(t, \varepsilon)$ | random $(t, \varepsilon)$ | random $(t, \varepsilon)$ | random $(t, \varepsilon)$ | **实际SDE链** |
| 理论保证 | 启发式 | 启发式 | Policy improvement | ≈PG | **≈PG (Thm 4.4)** |

---

## 五、Risk Assessment & 实验发现

### 5.1 成功因素（已验证 ✅）

1. **π-StepNFT移植成功**：NFT + terminal_binary在LIBERO-Object上训练正常，pref_acc从0.5上升，success_once稳步提升
2. **无需EMA/Annotation Pass**：NFT数据全部来自rollout的SDE链快照，简化了pipeline

### 5.2 已发现的问题

#### 5.2.1 ❌ FEA与softplus contrastive loss不兼容（已确认）

**现象**：启用FEA后，训练曲线与terminal_binary**完全相反**——reward下降、success_once不升反降。

**根因分析（详见§3.3）**：
1. MC return下TD advantage理论为零 → 归一化后为随机 $\pm 1$ 标签
2. softplus contrastive loss对错误标签的惩罚远大于正确标签的奖励 → 净效果是反方向训练
3. 即使修正为重要性加权模式，FEA仍系统性削弱失败信号

**当前状态**：代码中FEA已修正为重要性加权模式（方向来自terminal_binary，幅度来自FEA），但**不推荐使用**。默认配置已切换为`terminal_binary`。

#### 5.2.2 ⚠️ Denoising循环必须是独立SDE分支

**现象**：最初将NFT快照收集钩入正常denoising循环（ODE/SDE混合模式），导致训练比原始版本崩溃更快。

**根因**：π-StepNFT要求所有K步去噪都使用SDE噪声（`mode="train"`），但正常推理循环仅最后几步加噪声。数据分布不匹配导致loss异常。

**修复**：在`sample_actions()`中添加独立的NFT去噪分支，所有步骤强制`mode="train"`，与π-StepNFT原始实现完全对齐。

### 5.3 剩余风险

1. **Long-horizon任务gap**：terminal_binary对所有步施加相同权重的问题仍然存在，但目前没有找到在NFT框架下解决此问题的有效方案
2. **FlowIPO框架差异**：已通过逐行对比验证核心数学一致，但solver API等适配层可能引入微小差异

### 5.4 成功/失败判据

| 指标 | 成功标准 | 失败标准 | 当前状态 |
|------|---------|---------|---------|
| NFT收敛速度 | success_once 100步内开始上升 | 200步后无变化 | ✅ 已达标 |
| NFT最终性能 | 接近π-StepNFT论文结果 | 显著低于FlowSAR | 🔄 测试中 |
| FEA vs terminal_binary | Long任务提升≥2% | Long任务无变化或下降 | ❌ FEA不兼容 |

---

## 六、Experimental Plan

### 6.1 Phase 1: NFT验证 ✅（已完成）

- 在LIBERO-Object上训练flow_nft（terminal_binary advantage）
- 确认NFT收敛正常，pref_acc从0.5上升，success_once稳步提升
- 修复了denoising循环问题（独立SDE分支）

### 6.2 Phase 2: FEA验证 ❌（已完成 — 不可行）

- FEA导致训练方向完全反转
- 根因：softplus contrastive loss与FEA的连续advantage不兼容（详见§3.3和§5.2.1）
- 修正为重要性加权模式后仍不推荐使用（削弱失败信号）
- **结论：使用terminal_binary**

### 6.3 Phase 3: Full Benchmark

- LIBERO 5个suite: Short, Long, Spatial, Object, Goal
- 对比：NFT-terminal_binary, FlowSAR, PPO
- NFT在Short任务上应与PPO持平，Long任务可能有gap（terminal_binary的固有限制）

### 6.4 超参数

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `flow_nft_beta` | 1.0 | 镜像构造缩放 $\beta$（与π-StepNFT论文一致） |
| `flow_nft_kl_beta` | 0.0001 | KL trust region权重（很小，主靠max_drift） |
| `flow_nft_max_drift` | 0.5 | 速度变化裁剪（关键稳定性参数） |
| `flow_nft_dpo_beta` | 1.0 | softplus logit缩放（原论文固定为1） |
| `flow_nft_noise_level` | 0.2 | SDE探索噪声 |
| `flow_nft_adv_clip_max` | 1.0 | Advantage裁剪 $c_{\text{adv}}$ |
| `flow_nft_adv_type` | terminal_binary | **推荐terminal_binary**（FEA不推荐，详见§5.2.1） |
| `flow_nft_fea_gamma` | 0.99 | FEA时间折扣 $\gamma$（仅experimental） |
| `flow_nft_fea_ridge_lambda` | 1.0 | Ridge回归正则化 $\lambda$（仅experimental） |

---

## 七、论文故事

### 方向A：统一框架对比（最可行）

**Title**: "A Unified Benchmark for Flow-Based RL in Vision-Language-Action Models"

**Contributions**:
1. 诊断了加权回归范式在预训练VLA上的结构性梯度消失问题（FlowIPO/FPI/AWM失败根因）
2. 在FlowIPO统一框架中实现并对比6种flow RL方法（IPO, SAR, FPI, AWM, GFN, NFT）
3. 验证π-StepNFT的对比式loss是唯一不受梯度消失影响的方案
4. 提供系统的超参数分析和实践指南

### 方向B：Temporal Credit for Contrastive RL（需要新方案）

FEA（冻结embedding做temporal credit）的概念是合理的，但与softplus contrastive loss不兼容。如果要发表FEA相关工作，需要：
- 新的loss设计：不使用softplus contrastive，而是将FEA重要性作为sample-level权重（如importance-weighted NFT）
- 或限制FEA只在成功trajectory内部做步间区分，不改变成功/失败方向

**FEA失败的教训**：softplus对比loss对标签噪声极度敏感（softplus不对称性），任何产生随机标签的temporal credit方案都会导致灾难性训练崩溃。
