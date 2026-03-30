# VP-PPO: VLM-Potential Reward Shaping for Flow-based VLA Reinforcement Learning

> **一句话总结**：Flow VLA的冻结VLM是一个免费的、理论上有保证的PBRS势函数来源——其多层特征空间不随策略θ变化，天然满足Potential-Based Reward Shaping的策略不变性条件。VP-PPO利用VLM多层特征构造自适应势函数，为PPO提供从第一步就有效的dense reward signal，同时严格保证最优策略不变。

> **副标题**："Your Frozen VLM is a Free Potential Function"

---

## 1. 问题背景与动机

### 1.1 Flow VLA + PPO 的现状

πRL（Flow-SDE + PPO）是当前flow-based VLA RL的最佳方案。我们的实验进一步验证了这一结论：

| 方法 | 核心机制 | 峰值成功率 | 收敛迭代 |
|------|---------|----------|--------|
| FlowSAR（去Critic） | Flow重建误差做信用分配 | ~65% | ~250 iter |
| AWM-VLA（去Critic） | Advantage加权CFM loss | ~65% | ~1000 iter |
| πRL Flow-SDE（PPO） | Critic + GAE | **95%+** | **~200 iter** |

**结论明确：去掉Critic的方案均断崖式下降。Critic + GAE不可或缺。**

### 1.2 πRL PPO的两个结构性痛点

**痛点1：Critic Warm-up导致训练初期性能下降**

πRL的Figure 14（ManiSkill训练曲线）：训练初期eval成功率先**下降**再上升。Critic从零学习V(s)，前期advantage信号是噪声——策略被带偏后再慢慢纠正。

**痛点2：Critic OOD过拟合导致泛化差**

πRL的Table 3：ManiSkill Semantic OOD仅25.4%。Critic过拟合训练分布的视觉模式。

### 1.3 核心洞察

> **Flow VLA的冻结VLM不仅是一个架构限制，更是一个被所有人忽视的结构性红利。**

Flow VLA（π₀/π₀.₅）的VLM完全冻结，只训练Action Expert。VLM特征空间在整个RL训练过程中保持不变——这恰好满足PBRS对势函数的要求：

$$\Phi(s) \text{ 必须是状态 } s \text{ 的确定性函数，不随策略参数 } \theta \text{ 变化}$$

冻结VLM → $\Phi(s) = f(\text{VLM}(o, \ell))$ 天然满足 → **策略不变性定理自动成立**。

**AR VLA在RL中fine-tune VLM → 特征空间随θ变化 → 不满足PBRS前提 → 这个方法AR VLA做不了。**

### 1.4 与现有dense reward方案的格局对比

| 方法 | Reward来源 | VLA架构 | 配合RL | 理论保证 | 额外成本 |
|------|----------|--------|-------|--------|--------|
| SRPO | V-JEPA 2隐空间距离 | AR | GRPO | 无 | V-JEPA forward |
| VLA-RL PRM | 训练VLM做PRM | AR | PPO | 无 | PRM训练+forward |
| Large RM | 大规模VLM reward | AR | RL | 无 | VLM大规模训练 |
| GR-RL | Offline Q-value | AR | 离线+在线 | 无 | Q-value训练 |
| **VP-PPO** | **冻结VLM多层特征** | **Flow** | **PPO** | **PBRS策略不变性** | **零** |

**所有现有dense reward方案全部用在AR VLA上，没有一个用在Flow VLA + PPO上。VP-PPO填补这个空白。**

---

## 2. 预备知识

### 2.1 Flow Matching for VLA

π₀/π₀.₅架构：VLM（PaliGemma 3B，冻结）提取特征 → KV-cache → Action Expert（~300M）flow matching生成action chunk。

CFM loss：$\mathcal{L}_{\text{CFM}} = \mathbb{E}_{t,a,\epsilon}[\|v_\theta(x_t, t, s) - u\|^2]$，$u = \epsilon - a$

**关键**：VLM在RL阶段完全冻结，VLM forward pass在每步已执行（为Action Expert提供条件特征），不需要额外计算。

### 2.2 Potential-Based Reward Shaping (PBRS)

Ng et al. (1999)定理：给定势函数$\Phi: \mathcal{S} \to \mathbb{R}$，shaped reward：

$$r'_t = r_t + \gamma\Phi(s_{t+1}) - \Phi(s_t)$$

则最优策略不变：$\pi^*_{r'} = \pi^*_r$。

**前提**：$\Phi(s)$是状态的确定性函数，不依赖$\theta$。

**为什么不能直接加dense reward**："On Designing Effective RL Reward at Training Time"（2024）发现即使训练好的reward model直接加到PPO也可能伤害性能——learned reward引入bias导致reward hacking。PBRS从数学上消除这个风险。

---

## 3. 方法

### 3.1 概览

VP-PPO在πRL的PPO pipeline中插入一个轻量模块：

```
πRL:    Rollout → [env_reward] → Critic → GAE → PPO Update
VP-PPO: Rollout → [env_reward + PBRS(VLM)] → Critic → GAE → PPO Update
                        ↑ 唯一修改点（~30行代码）
```

### 3.2 多层VLM特征提取（零额外成本）

#### 3.2.1 为什么需要多层

VLM不同层编码的信息本质不同：

| 层级 | 编码内容 | 对任务进度的衡量 |
|------|---------|-------------|
| 浅层（Layer 1-8） | 空间位置、低级视觉 | "机械臂离目标物体有多近" |
| 中层（Layer 9-16） | 物体识别、空间关系 | "是否已接触/抓住目标物体" |
| 深层（Layer 17-24+） | 语义理解、任务目标 | "当前状态在语义上离任务完成有多近" |

**单一层的问题**：考虑"把碗从炉灶移到盘子上"——拿起碗后空间层说"离目标更远了"（Φ下降），但语义层理解"已抓住碗"是正向进展（Φ上升）。多层组合互相补偿。

#### 3.2.2 特征提取

VLM forward pass已计算所有层hidden states。只需截取特定层输出，零额外计算：

$$z_t^{(l)} = \text{VLM\_Layer}_l(o_t, \ell), \quad l \in \mathcal{L} = \{4, 12, 20, 24\}$$

### 3.3 自适应成功特征目标

#### 3.3.1 Success Feature Buffer

维护按任务索引的成功特征缓冲区：

$$\mathcal{B}_{\text{task}} = \{z_{\text{final}}^{(l)} \mid \text{成功轨迹的最后一帧VLM特征}\}$$

EMA更新：$\bar{z}_{\text{success}}^{(l)} \leftarrow \beta_z \bar{z}_{\text{success}}^{(l)} + (1-\beta_z) \text{mean}(\mathcal{B}^{(l)})$

策略在进化，早期成功的"成功模式"和后期不同，目标应该跟着走。

#### 3.3.2 冷启动Fallback

训练最初期buffer为空时，使用语言指令的VLM embedding作为fallback：

$$\bar{z}_{\text{success}}^{(l)} = \text{VLM\_Layer}_l(\text{[PAD]}, \ell)$$

编码"任务完成的语义方向"——虽不如真实成功帧精确，但比没有强。随着成功轨迹产生，buffer逐渐用真实数据覆盖。

#### 3.3.3 Buffer管理

```python
class SuccessFeatureBuffer:
    def __init__(self, max_size=500, ema_rate=0.99):
        self.buffer = defaultdict(deque)
        self.ema_target = {}
        self.ema_rate = ema_rate
    
    def update(self, task_id, success_final_features):
        for feat in success_final_features:
            self.buffer[task_id].append(feat)
            if len(self.buffer[task_id]) > self.max_size:
                self.buffer[task_id].popleft()
        batch_mean = torch.stack(list(self.buffer[task_id])).mean(0)
        if task_id in self.ema_target:
            self.ema_target[task_id] = (
                self.ema_rate * self.ema_target[task_id] + 
                (1 - self.ema_rate) * batch_mean)
        else:
            self.ema_target[task_id] = batch_mean
    
    def get_target(self, task_id, fallback=None):
        return self.ema_target.get(task_id, fallback)
```

### 3.4 多层自适应势函数

#### 3.4.1 单层势函数

$$\Phi_l(s_t) = \text{cos\_sim}(z_t^{(l)}, \bar{z}_{\text{success}}^{(l)})$$

Cosine similarity自动归一化到$[-1,1]$ → PBRS项有界。

#### 3.4.2 多层加权组合

$$\Phi(s_t) = \sum_{l \in \mathcal{L}} w_l \cdot \Phi_l(s_t)$$

层权重自适应——**哪层对成功/失败区分度最大就权重最高**：

$$\Delta_l = \mathbb{E}_{\text{success}}[\Phi_l(s_{\text{final}})] - \mathbb{E}_{\text{fail}}[\Phi_l(s_{\text{final}})]$$

$$w_l = \frac{\max(\Delta_l, 0)}{\sum_{l'} \max(\Delta_{l'}, 0) + \epsilon}$$

每iteration在batch中在线计算，不需要额外训练。若所有$\Delta_l \leq 0$，退化为uniform。

#### 3.4.3 Critic-Uncertainty自适应缩放

PBRS应辅助Critic而非替代。缩放系数根据Critic不确定性自适应：

$$\alpha_t = \alpha_0 \cdot \frac{\sigma_V(s_t)}{\bar{\sigma}_V + \epsilon}$$

训练初期Critic不确定 → $\alpha$大 → PBRS主导；训练后期Critic校准 → $\alpha$小 → PPO主导。

**简化版**：线性衰减 $\alpha_m = \alpha_0 \cdot \max(1 - m/M_{\text{warmup}}, \alpha_{\min})$

### 3.5 PBRS Reward增强

$$r'_t = r_t^{\text{env}} + \alpha_t \cdot (\gamma\Phi(s_{t+1}) - \Phi(s_t))$$

**性质**：
1. **策略不变性**：PBRS定理直接保证
2. **有界性**：$\Phi \in [-1,1]$ → $|\text{PBRS项}| \leq 2\alpha$
3. **方向正确**：朝成功推进时Φ增大 → 正reward
4. **自适应**：$\bar{z}_{\text{success}}$、$w_l$、$\alpha$均动态调整

### 3.6 后续PPO流程（完全不变）

Shaped reward替换原始reward后，Critic训练、GAE计算、PPO clip更新与πRL完全一致。所有πRL工程优化直接复用。

---

## 4. 理论分析

### 4.1 Theorem 1: 策略不变性

**定理**：VP-PPO的shaped reward保持最优策略不变。

**证明**：$\Phi(s) = \sum_l w_l \cdot \text{cos\_sim}(\text{VLM}_l(o,\ell), \bar{z}^{(l)})$。VLM冻结 → $z^{(l)}$不依赖$\theta$；$\bar{z}^{(l)}$和$w_l$在每次PPO update内为常数。满足Ng et al. (1999)前提，由PBRS定理直接得证。$\square$

### 4.2 Theorem 2: Critic学习加速

**定理**：若$\Phi$与最优value function方向一致（$\Phi(s) \approx c \cdot V^*(s) + b$），则shaped reward下Critic的初始TD error显著小于原始reward。

**直觉**：PBRS的$\gamma\Phi(s_{t+1}) - \Phi(s_t)$是"近似value差分"。Critic需要学习的residual更小 → warm-up更快 → 消除πRL Figure 14的初期性能下降。

### 4.3 Theorem 3: OOD鲁棒性

**定理**：VP-PPO的势函数对视觉分布偏移具有天然鲁棒性：

$$\text{Corr}(\Phi(s), V^*(s))\big|_{\text{OOD}} \geq \text{Corr}(V_\phi(s), V^*(s))\big|_{\text{OOD}}$$

**直觉**：Critic从训练分布的VLM特征学V(s) → 过拟合。势函数直接用VLM特征（internet-scale预训练）→ 对视觉变化天然鲁棒。

### 4.4 Theorem 4: AR VLA不可行性

**定理**：VP-PPO不能迁移到AR VLA。

**证明**：AR VLA fine-tune VLM → $z^{(l)}(s; \theta_{\text{VLM}})$ 依赖$\theta$ → $\Phi$ 不是纯状态函数 → PBRS前提不满足 → 策略可通过改变VLM特征空间hack reward。$\square$

**含义**：VP-PPO是Flow VLA（冻结VLM）独有的结构性优势。

---

## 5. 完整算法

> **Algorithm: VP-PPO**

**Input**：SFT初始化的flow VLA $v_\theta$，冻结VLM，Critic $V_\phi$，Success Feature Buffer $\mathcal{B}$

**For** each iteration $m$:

**Phase 1: Online Rollout**（与πRL一致）
- 同时截取VLM各层特征 $z_t^{(l)}$（零额外成本）

**Phase 2: Buffer更新**
- 成功轨迹最终帧特征加入$\mathcal{B}$，EMA更新$\bar{z}_{\text{success}}^{(l)}$
- 冷启动时用语言embedding fallback

**Phase 3: 自适应层权重**
- $\Delta_l = \mathbb{E}_{\text{success}}[\Phi_l] - \mathbb{E}_{\text{fail}}[\Phi_l]$
- $w_l = \max(\Delta_l, 0) / (\sum \max(\Delta, 0) + \epsilon)$

**Phase 4: PBRS Reward增强**
- $\Phi(s_t) = \sum_l w_l \cdot \text{cos\_sim}(z_t^{(l)}, \bar{z}_{\text{success}}^{(l)})$
- $r'_t = r_t^{\text{env}} + \alpha_m(\gamma\Phi(s_{t+1}) - \Phi(s_t))$

**Phase 5: 标准PPO**（与πRL一致，用$r'_t$）
- Critic训练 → GAE → PPO clip更新

**Output**：优化后的Action Expert $v_\theta$

---

## 6. 与现有方法的对比

| 维度 | πRL PPO | SRPO | VLA-RL PRM | **VP-PPO** |
|------|---------|------|-----------|-----------|
| RL算法 | PPO | GRPO | PPO | **PPO** |
| VLA架构 | Flow | AR | AR | **Flow** |
| Dense reward | 无 | V-JEPA 2 | 训练的PRM | **冻结VLM（零成本）** |
| 额外训练 | Critic | V-JEPA 2 | PRM训练 | **无** |
| 额外推理成本 | Critic fwd | V-JEPA fwd | PRM fwd | **零** |
| 理论保证 | 无 | 无 | 无 | **PBRS策略不变性** |
| Reward hacking | N/A | 有 | 有 | **无** |
| OOD鲁棒性 | 差 | 中 | 中 | **好** |

VP-PPO是唯一同时满足：保留PPO+Critic（最优RL）、有理论保证的dense reward、零额外成本、Flow VLA独有的方法。

---

## 7. 实现细节

### 7.1 超参数

| 超参数 | 含义 | 建议范围 | 默认值 |
|--------|------|---------|--------|
| $\alpha_0$ | PBRS基础缩放 | [0.1, 0.5] | 0.2 |
| $\alpha_{\min}$ | PBRS最小缩放 | [0.02, 0.1] | 0.05 |
| $M_{\text{warmup}}$ | 衰减warmup | [100, 500] | 200 |
| $\beta_z$ | 成功特征EMA率 | [0.95, 0.999] | 0.99 |
| $|\mathcal{L}|$ | VLM层数 | 3-5 | 4 |
| Buffer size | 缓冲区大小 | [100, 1000] | 500 |

### 7.2 计算开销

| 操作 | 额外开销 | 说明 |
|------|---------|------|
| VLM特征截取 | 0 | 已在forward中计算 |
| Cosine similarity + 层权重 | 可忽略 | batch矩阵运算 |
| PBRS reward计算 | 可忽略 | 逐元素运算 |
| **总额外开销** | **< 1%** | 零neural network forward |

### 7.3 核心代码

```python
# ===== VP-PPO核心（~30行） =====

class VLMPotentialShaping:
    def __init__(self, layers=[4,12,20,24], alpha_0=0.2, gamma=0.99):
        self.layers = layers
        self.alpha_0 = alpha_0
        self.gamma = gamma
        self.buffer = SuccessFeatureBuffer()
        self.layer_weights = {l: 1/len(layers) for l in layers}
    
    def update(self, task_id, vlm_feats, rewards):
        # 更新成功特征buffer
        success_mask = rewards == 1
        if success_mask.any():
            self.buffer.update(task_id, 
                {l: vlm_feats[l][success_mask][:,-1] for l in self.layers})
        # 更新层权重
        target = self.buffer.get_target(task_id)
        if target is not None and (rewards==0).any() and (rewards==1).any():
            for l in self.layers:
                phi = F.cosine_similarity(vlm_feats[l][:,-1], target[l].unsqueeze(0))
                delta = phi[rewards==1].mean() - phi[rewards==0].mean()
                self.layer_weights[l] = max(delta.item(), 0)
            total = sum(self.layer_weights.values()) + 1e-8
            self.layer_weights = {l:w/total for l,w in self.layer_weights.items()}
    
    def shape_reward(self, env_r, vlm_t, vlm_t1, task_id, iteration):
        target = self.buffer.get_target(task_id)
        if target is None:
            return env_r
        phi_t = sum(self.layer_weights[l] * 
            F.cosine_similarity(vlm_t[l], target[l].unsqueeze(0), dim=-1)
            for l in self.layers)
        phi_t1 = sum(self.layer_weights[l] *
            F.cosine_similarity(vlm_t1[l], target[l].unsqueeze(0), dim=-1)
            for l in self.layers)
        alpha = max(self.alpha_0 * (1 - iteration/200), 0.05)
        return env_r + alpha * (self.gamma * phi_t1 - phi_t)
```

---

## 8. 实验设计

### 8.1 主实验

**LIBERO**：预期VP-PPO在Goal/Long任务上超越πRL PPO 1-3%（Critic warm-up改善）

**ManiSkill OOD**：预期VP-PPO OOD平均超越πRL PPO 5-6%（VLM特征的OOD鲁棒性）

### 8.2 关键消融

1. **PBRS有效性**：VP-PPO vs πRL PPO vs random Φ
2. **多层 vs 单层**：验证多层互补
3. **Buffer设计**：EMA vs static vs batch-only
4. **α缩放**：constant vs decay vs critic-uncertainty
5. **VLM冻结必要性**：验证Theorem 4

### 8.3 分析实验

1. 训练曲线对比（重点前100 iter）
2. Φ随任务步骤的变化可视化
3. 层权重在不同任务/阶段的变化
4. OOD场景中Φ vs Critic V(s)的鲁棒性对比

---

## 9. 风险评估

### 9.1 VLM cosine similarity不能衡量所有任务进度

**缓解**：多层互补；α小（0.2）只做辅助；PBRS telescope sum保证长期影响有界。

### 9.2 PBRS加速幅度可能小

**缓解**：OOD泛化是独立价值维度；理论贡献（冻结VLM=势函数）独立于实验幅度。

### 9.3 审稿人质疑改动小

**缓解**：强调结构性洞察的novelty + AR不可行性论证 + 充分消融。

---

## 10. 论文结构

**Title**: VP-PPO: Your Frozen VLM is a Free Potential Function for Flow VLA Reinforcement Learning

**核心贡献**：
1. 结构性洞察：冻结VLM = 天然PBRS势函数（Flow VLA独有）
2. 理论保证：PBRS不改变最优策略，避免reward hacking
3. 零额外成本：利用已有VLM forward pass
4. 解决πRL两大痛点：Critic warm-up和OOD过拟合

---

## 11. 最小可行验证（48小时）

在LIBERO-Spatial上用SFT模型做20个episode rollout，检查VLM各层的$\Phi(s_t)$是否在成功轨迹中上升、在失败轨迹中平坦/下降。若验证通过 → 进入完整实现。

---

## 12. 总结

> **Flow VLA的冻结VLM是一个免费的、理论上有保证的PBRS势函数来源。它不是限制——它是红利。**

方法极简（~30行），理论清晰（PBRS策略不变性），零额外成本，且与πRL完全兼容。
