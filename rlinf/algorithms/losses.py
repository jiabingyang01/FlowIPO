# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Callable, Optional, Sequence

import torch
import torch.nn.functional as F

from rlinf.algorithms.registry import register_policy_loss
from rlinf.algorithms.utils import huber_loss
from rlinf.utils.utils import masked_mean, masked_mean_ratio


def compute_ppo_actor_loss(
    logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    clip_ratio_low: float,
    clip_ratio_high: float,
    advantages: torch.Tensor,
    loss_mask: Optional[torch.Tensor] = None,
    clip_ratio_c: Optional[float] = None,
    loss_agg_func: Optional[Callable[..., torch.Tensor]] = masked_mean,
    max_episode_steps: Optional[int] = None,
    loss_mask_sum: Optional[torch.Tensor] = None,
    critic_warmup: Optional[bool] = False,
    **kwargs,
) -> tuple[torch.Tensor, dict]:
    """
    Compute PPO actor loss function.

    Args:
        logprobs (torch.FloatTensor): Log probabilities of actions.
        old_logprobs (torch.FloatTensor): Old log probabilities of actions.
        clip_ratio_low (float): Lower bound of clipping ratio.
        clip_ratio_high (float): Upper bound of clipping ratio.
        advantages (torch.FloatTensor): GAE (normalized) advantages.
        loss_mask (Optional[torch.BoolTensor], optional): Mask for valid entries. Defaults to None.
        clip_ratio_c (Optional[float], optional): Optional clipping coefficient. Defaults to None.
        loss_agg_func (callable, optional): Aggregation function (e.g., masked_mean). Defaults to None.
        max_episode_steps (Optional[int], optional): Max episode length for normalization. Defaults to None.

    Returns:
        Tuple[torch.Tensor, Dict]: (actor_loss, metrics_dict)
    """

    loss_mask_ratio = None

    if (
        max_episode_steps is not None
        and loss_mask_sum is not None
        and loss_mask is not None
    ):
        loss_mask_ratio = (loss_mask_sum * 1.0) / max_episode_steps
        loss_agg_func = masked_mean_ratio

    if loss_mask is None:
        loss_mask = torch.ones_like(logprobs).bool()

    assert logprobs.dtype == torch.float32
    assert old_logprobs.dtype == torch.float32
    assert advantages.dtype == torch.float32

    loss_mask_count = loss_mask.count_nonzero() or 1
    # For numerical stability.
    ratio = torch.where(loss_mask, torch.exp(logprobs - old_logprobs), 0)
    approx_kl = torch.where(loss_mask, (logprobs - old_logprobs).detach(), 0.0)

    clipped_ratio = torch.clamp(ratio, 1.0 - clip_ratio_low, 1.0 + clip_ratio_high)
    policy_loss1 = -advantages * ratio
    policy_loss2 = -advantages * clipped_ratio

    clip_mask = policy_loss1.detach() < policy_loss2.detach()

    policy_loss = torch.max(policy_loss1, policy_loss2)
    if clip_ratio_c is not None:
        assert clip_ratio_c > 1.0, clip_ratio_c
        policy_loss3 = torch.sign(advantages) * clip_ratio_c * advantages
        dual_clip_mask = policy_loss3.detach() < policy_loss.detach()
        policy_loss = torch.min(policy_loss, policy_loss3)
    else:
        dual_clip_mask = torch.zeros_like(clip_mask)

    metric_policy_loss_abs = loss_agg_func(
        policy_loss.abs(), loss_mask, loss_mask_ratio
    )
    policy_loss = loss_agg_func(
        policy_loss, loss_mask, loss_mask_ratio
    )  # default max_episode_steps is None

    clip_mask = policy_loss1.detach() < policy_loss2.detach()
    dual_clip_mask = (dual_clip_mask * loss_mask).bool()

    clip_fraction = (clip_mask * loss_mask).sum() / float(loss_mask_count)
    approx_kl = -torch.sum(approx_kl) / float(loss_mask_count)

    dual_cliped_ratio = torch.where(dual_clip_mask, ratio, 0)

    if critic_warmup:
        policy_loss = torch.tensor(0.0, device=policy_loss.device)

    # Compile metrics for logging
    loss_mask_for_metrics = loss_mask
    ratio_for_metrics = ratio.detach()
    ratio_abs_for_metrics = (ratio - 1).abs().detach()
    clipped_ratio_for_metrics = clipped_ratio.detach()
    dual_cliped_ratio_for_metrics = dual_cliped_ratio.detach()

    # Only broadcast when ratio has action_dim dimension and loss_mask's last dim is 1
    # This handles token_level mode: ratio [bsz, num_chunks, action_dim], loss_mask [bsz, num_chunks, 1]
    if len(ratio.shape) > 2 and loss_mask.shape[-1] == 1 and ratio.shape[-1] > 1:
        # Broadcast loss_mask to match ratio's shape for metrics computation
        loss_mask_for_metrics = loss_mask.expand_as(ratio)

    metrics_data = {
        "actor/policy_loss": policy_loss.detach(),
        "actor/policy_loss_abs": metric_policy_loss_abs.detach(),
        "actor/ratio": masked_mean(ratio_for_metrics, loss_mask_for_metrics),
        "actor/ratio_abs": masked_mean(ratio_abs_for_metrics, loss_mask_for_metrics),
        "actor/clipped_ratio": masked_mean(
            clipped_ratio_for_metrics, loss_mask_for_metrics
        ),
        "actor/dual_cliped_ratio": masked_mean(
            dual_cliped_ratio_for_metrics, loss_mask_for_metrics
        ),
        "actor/approx_kl": approx_kl.detach(),
        "actor/clip_fraction": clip_fraction.detach(),
    }
    return policy_loss, metrics_data


def compute_ppo_critic_loss(
    values: torch.Tensor,
    returns: torch.Tensor,
    prev_values: torch.Tensor,
    value_clip: float,
    huber_delta: float,
    loss_mask: Optional[torch.Tensor] = None,
    max_episode_steps: Optional[int] = None,
    loss_mask_sum: Optional[torch.Tensor] = None,
    **kwargs,
) -> tuple[torch.Tensor, dict]:
    """
    Compute PPO critic loss function.

    Args:
        values (torch.Tensor): Current value predictions.
        returns (torch.Tensor): Return values.
        prev_values (torch.Tensor): Previous value predictions.
        value_clip (float): Value clipping threshold.
        huber_delta (float): Huber loss delta parameter.

    Returns:
        Tuple[torch.Tensor, Dict]: (critic_loss, metrics_dict)
    """
    loss_mask_ratio = None
    loss_agg_func = masked_mean

    if (
        max_episode_steps is not None
        and loss_mask_sum is not None
        and loss_mask is not None
    ):
        loss_mask_ratio = (loss_mask_sum * 1.0) / max_episode_steps
        loss_agg_func = masked_mean_ratio

    value_pred_clipped = prev_values + (values - prev_values).clamp(
        -value_clip, value_clip
    )  # [bsz, ] | [bsz, chunk-step]

    value_loss_original = huber_loss(
        returns - values, huber_delta
    )  # [bsz, ] | [bsz, chunk-step]
    value_loss_clipped = huber_loss(
        returns - value_pred_clipped, huber_delta
    )  # [bsz, ] | [bsz, chunk-step]
    value_loss = torch.max(value_loss_original, value_loss_clipped)
    value_loss = loss_agg_func(value_loss, loss_mask, loss_mask_ratio)

    value_clip_indicator = (value_pred_clipped - prev_values).abs() > value_clip
    value_clip_ratio = value_clip_indicator.float().mean()

    # explained variance
    if loss_mask is not None:
        masked_returns = returns[loss_mask]
        masked_values = values[loss_mask]
    else:
        masked_returns = returns
        masked_values = values

    var_returns = torch.var(masked_returns)
    if torch.isnan(var_returns) or var_returns == 0:
        explained_variance = torch.tensor(float("nan"), device=returns.device)
    else:
        var_diff = torch.var(masked_returns - masked_values)
        if torch.isnan(var_diff):
            explained_variance = torch.tensor(float("nan"), device=returns.device)
        else:
            explained_variance = 1 - var_diff / var_returns

    # Compile metrics for logging
    metrics_data = {
        "critic/value_loss": value_loss.detach().item(),
        "critic/value_clip_ratio": value_clip_ratio.detach().item(),
        "critic/explained_variance": explained_variance.detach().item(),
    }
    return value_loss, metrics_data


@register_policy_loss("actor_critic")
def compute_ppo_actor_critic_loss(**kwargs) -> tuple[torch.Tensor, dict]:
    """
    Compute PPO actor loss function.

    Args:
        logprobs (torch.Tensor): Log probabilities of actions
        values (torch.Tensor): Current value predictions
        old_log_prob (torch.Tensor): Previous log probabilities
        advantages (torch.Tensor): Advantage values
        returns (torch.Tensor): Return values
        prev_values (torch.Tensor): Previous value predictions
        clip_ratio_low (float): Lower clipping ratio for PPO
        clip_ratio_high (float): Upper clipping ratio for PPO
        value_clip (float): Value clipping threshold
        huber_delta (float): Huber loss delta parameter

    Returns:
        Tuple[torch.Tensor, Dict]: Loss and metrics dictionary
    """
    metrics_data = {}
    actor_loss, actor_metrics_data = compute_ppo_actor_loss(**kwargs)
    critic_loss, critic_metrics_data = compute_ppo_critic_loss(**kwargs)

    loss = actor_loss + critic_loss
    metrics_data.update(actor_metrics_data)
    metrics_data.update(critic_metrics_data)

    return loss, metrics_data


@register_policy_loss("actor")
def compute_grpo_actor_loss_fn(**kwargs) -> tuple[torch.Tensor, dict]:
    """
    Compute actor loss for Group Relative Policy Optimization (GRPO).

    This function implements the PPO-style actor loss with clipping for GRPO.
    Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppotrainer.py#L1122

    Args:
        log_prob (torch.Tensor): Current log probabilities
        old_log_prob (torch.Tensor): Previous log probabilities
        advantages (torch.Tensor): Advantage values of shape
        clip_ratio_high (float): Upper clipping ratio for PPO
        clip_ratio_low (float): Lower clipping ratio for PPO
        loss_mask (Optional[torch.Tensor]): Mask tensor of shape to apply to the loss

    Returns:
        Tuple[torch.Tensor, Dict]: Policy gradient loss and metrics dictionary containing:
            - actor/loss: Total actor loss
            - actor/policy_loss: Policy gradient loss
            - actor/clip_fraction: Fraction of clipped policy gradient loss
            - actor/ppo_kl: Approximate KL divergence
    """
    metrics_data = {}
    actor_loss, actor_metrics_data = compute_ppo_actor_loss(**kwargs)
    metrics_data.update(actor_metrics_data)

    return actor_loss, metrics_data


@register_policy_loss("flow_ipo")
def compute_flow_ipo_loss(
    v_theta: torch.Tensor,
    interpolated_target: torch.Tensor,
    loss_mask: Optional[torch.Tensor] = None,
    **kwargs,
) -> tuple[torch.Tensor, dict]:
    """
    Compute FlowIPO loss: MSE between model velocity prediction and
    interpolated velocity target.

    L = E[ ||v_θ(x_t, t, s_i) - ṽ_i||² ]

    where ṽ_i = w_i * u_i + (1 - w_i) * v_ref_i is the interpolated target
    constructed in the training loop (fsdp_actor_worker).

    Args:
        v_theta: Model's velocity prediction, shape [batch, chunk, action_dim].
        interpolated_target: Interpolated velocity target ṽ_i, same shape.
        loss_mask: Optional mask for valid entries, shape [batch, ...].

    Returns:
        Tuple[torch.Tensor, Dict]: (flow_ipo_loss, metrics_dict)
    """
    # Per-element MSE
    mse = (v_theta - interpolated_target).pow(2)

    # Aggregate: masked mean over all dims
    if loss_mask is not None:
        # Expand loss_mask to match mse shape if needed
        while loss_mask.dim() < mse.dim():
            loss_mask = loss_mask.unsqueeze(-1)
        loss_mask = loss_mask.expand_as(mse)
        loss = (mse * loss_mask).sum() / (loss_mask.sum().clamp(min=1.0))
    else:
        loss = mse.mean()

    metrics_data = {
        "actor/flow_ipo_loss": loss.detach().item(),
        "actor/velocity_mse": mse.detach().mean().item(),
    }
    return loss, metrics_data


@register_policy_loss("flow_sar")
def compute_flow_sar_loss(
    v_theta: torch.Tensor,
    v_old: torch.Tensor,
    u_target: torch.Tensor,
    weights: torch.Tensor,
    labels: torch.Tensor,
    beta: float = 1.0,
    energy_type: str = "mse",
    loss_variant: str = "mse_branch",
    kl_coeff: float = 0.5,
    flow_t: Optional[torch.Tensor] = None,
    loss_mask: Optional[torch.Tensor] = None,
    **kwargs,
) -> tuple[torch.Tensor, dict]:
    """
    Compute FlowSAR loss with mirror construction.

    v⁺ = (1 - β) * v_old + β * v_θ      (positive branch: toward current policy)
    v⁻ = (1 + β) * v_old - β * v_θ      (negative branch: away from current policy)

    Energy modes (energy_type):
      - "mse" (ODE): E = ||v - u||²  (velocity-space MSE, all timesteps equal)
      - "sde" (Flow-SDE Mahalanobis): E = ||v - u||² / (2t)
        Derived from Flow-SDE transition kernel variance σ_t² ≈ 2tδ.

    Loss variants (loss_variant):
      - "mse_branch" (DiffusionNFT-style): L = w_i * [R_i * E⁺ + (1 - R_i) * E⁻]
        Reward-conditioned branch selection. Bounded by mirror construction.
      - "softplus_kl" (π-StepNFT-style): L = w_i * softplus(½ y (E⁺ - E⁻)) + kl_coeff * ||v_θ - v_old||²
        Contrastive softplus ranking + trust region KL penalty.

    Args:
        v_theta: Current model velocity prediction, shape [batch, chunk, action_dim].
        v_old: Reference (EMA) velocity prediction (detached), same shape.
        u_target: Flow matching target velocity (ε - a), same shape.
        weights: Per-step credit assignment weights, shape [batch].
        labels: Contrastive labels y_i = 2R-1 ∈ {-1, +1}, shape [batch].
        beta: Trust region parameter for mirror construction.
        energy_type: "mse" for ODE velocity MSE, "sde" for Flow-SDE Mahalanobis.
        loss_variant: "mse_branch" for DiffusionNFT-style, "softplus_kl" for π-StepNFT-style.
        kl_coeff: KL penalty coefficient (only used when loss_variant="softplus_kl").
        flow_t: Flow matching timestep per sample, shape [batch]. Required for "sde".
        loss_mask: Optional mask for valid entries.

    Returns:
        Tuple[torch.Tensor, Dict]: (flow_sar_loss, metrics_dict)
    """
    # Mirror velocity construction
    v_pos = (1 - beta) * v_old + beta * v_theta           # positive branch
    v_neg = (1 + beta) * v_old - beta * v_theta           # negative branch

    # Branch errors: MSE against flow matching target u_i
    # Mean over chunk and action_dim dimensions -> [batch]
    E_pos = (v_pos - u_target).pow(2).mean(dim=(-2, -1))  # [batch]
    E_neg = (v_neg - u_target).pow(2).mean(dim=(-2, -1))  # [batch]

    # SDE mode: apply Mahalanobis scaling from Flow-SDE transition kernel
    # σ_t² ≈ 2tδ → E_SDE = δ/(2t) * ||v-u||² → scale by 1/(2t)
    if energy_type == "sde" and flow_t is not None:
        t_flat = flow_t.reshape(-1)  # [batch]
        sde_scale = 1.0 / (2.0 * t_flat).clamp(min=1e-3)  # [batch]
        E_pos = E_pos * sde_scale
        E_neg = E_neg * sde_scale

    if loss_variant == "softplus_kl":
        # ---- π-StepNFT-style: softplus contrastive + trust region ----
        # Contrastive ranking loss with softplus
        # y_i = +1 (success): minimize E⁺ - E⁻ (positive branch should be closer)
        # y_i = -1 (failure): minimize -(E⁺ - E⁻) (push away from current policy)
        margin = 0.5 * labels * (E_pos - E_neg)
        contrastive = F.softplus(margin)  # [batch]

        # Trust region KL penalty: ||v_θ - v_old||² (velocity-space divergence)
        kl_penalty = (v_theta - v_old).pow(2).mean(dim=(-2, -1))  # [batch]

        per_sample_loss = weights * contrastive + kl_coeff * kl_penalty  # [batch]
    else:
        # ---- DiffusionNFT-style: reward-conditioned branch selection (default) ----
        # labels: y_i = 2R-1, so R_i = (labels + 1) / 2
        R = (labels + 1.0) / 2.0  # [batch], 0.0 or 1.0
        per_sample_loss = weights * (R * E_pos + (1.0 - R) * E_neg)  # [batch]

    # Apply loss mask if provided
    if loss_mask is not None:
        if loss_mask.dim() > 1:
            loss_mask = loss_mask.reshape(loss_mask.shape[0], -1).any(dim=-1)
        loss_mask = loss_mask.float()
        loss = (per_sample_loss * loss_mask).sum() / loss_mask.sum().clamp(min=1.0)
    else:
        loss = per_sample_loss.mean()

    metrics_data = {
        "actor/flow_sar_loss": loss.detach().item(),
        "actor/flow_sar_E_pos": E_pos.detach().mean().item(),
        "actor/flow_sar_E_neg": E_neg.detach().mean().item(),
        "actor/flow_sar_weight_mean": weights.detach().mean().item(),
        "actor/flow_sar_energy_type": 1.0 if energy_type == "sde" else 0.0,
        "actor/flow_sar_loss_variant": 1.0 if loss_variant == "softplus_kl" else 0.0,
    }
    if loss_variant == "softplus_kl":
        metrics_data["actor/flow_sar_contrastive"] = contrastive.detach().mean().item()
        metrics_data["actor/flow_sar_kl_penalty"] = kl_penalty.detach().mean().item()
    else:
        R = (labels + 1.0) / 2.0
        metrics_data["actor/flow_sar_success_ratio"] = R.detach().mean().item()
    return loss, metrics_data


@register_policy_loss("flow_awm")
def compute_flow_awm_loss(
    v_theta: torch.Tensor,
    u_target: torch.Tensor,
    advantages: torch.Tensor,
    loss_mask: Optional[torch.Tensor] = None,
    **kwargs,
) -> tuple[torch.Tensor, dict]:
    """
    Compute AWM-VLA loss: advantage-weighted flow matching MSE.

    L_AWM = Ã_i · ||v_θ(x_t, t | o, l) - u||²

    where Ã_i = clip(normalize(A_i), -c, c) can be NEGATIVE.
    - Ã > 0: gradient pulls v_θ TOWARD good action's target (learn good actions)
    - Ã < 0: gradient pushes v_θ AWAY from bad action's target (unlearn bad actions)

    This bidirectional signal is the key difference from FPI (exp(A/λ) > 0 always).
    Mathematically equivalent to policy gradient with lower variance (AWM, Xue et al.).

    Args:
        v_theta: Current model velocity prediction, shape [batch, chunk, action_dim].
        u_target: Flow matching target velocity (ε - a), same shape.
        advantages: Per-sample normalized+clipped advantages, shape [batch]. Can be negative.
        loss_mask: Optional mask for valid entries.

    Returns:
        Tuple[torch.Tensor, Dict]: (flow_awm_loss, metrics_dict)
    """
    # Per-sample MSE: ||v_θ - u||², mean over chunk and action_dim
    mse = (v_theta - u_target).pow(2).mean(dim=(-2, -1))  # [batch]

    # Advantage-weighted loss: Ã * MSE (Ã can be negative!)
    per_sample_loss = advantages * mse  # [batch]

    # Apply loss mask if provided
    if loss_mask is not None:
        if loss_mask.dim() > 1:
            loss_mask = loss_mask.reshape(loss_mask.shape[0], -1).any(dim=-1)
        loss_mask = loss_mask.float()
        loss = (per_sample_loss * loss_mask).sum() / loss_mask.sum().clamp(min=1.0)
    else:
        loss = per_sample_loss.mean()

    pos_mask = (advantages > 0).float()
    neg_mask = (advantages < 0).float()
    metrics_data = {
        "actor/awm_loss": loss.detach().item(),
        "actor/awm_mse": mse.detach().mean().item(),
        "actor/awm_adv_mean": advantages.detach().mean().item(),
        "actor/awm_adv_std": advantages.detach().std().item(),
        "actor/awm_adv_pos_frac": pos_mask.mean().item(),
        "actor/awm_adv_neg_frac": neg_mask.mean().item(),
    }
    return loss, metrics_data


@register_policy_loss("flow_gfn")
def compute_flow_gfn_loss(
    log_flows: torch.Tensor,
    log_pf: torch.Tensor,
    log_pb: torch.Tensor,
    log_rewards: torch.Tensor,
    subtb_lambda: float = 1.0,
    boundary_coeff: float = 1.0,
    loss_mask: Optional[torch.Tensor] = None,
    **kwargs,
) -> tuple[torch.Tensor, dict]:
    """
    Compute GFN-Flow SubTB + boundary loss.

    SubTB (Sub-Trajectory Balance):
      For each sub-trajectory (j, k) with 0 <= j < k <= K:
        residual = log F(s_j) + Σ_{i=j}^{k-1} log P_F(s_{i+1}|s_i)
                 - log F(s_k) - Σ_{i=j}^{k-1} log P_B(s_i|s_{i+1})
        L_SubTB = Σ_{j<k} λ^{k-j} * residual²  / count

    Boundary condition:
      L_boundary = (log F(s_K) - log R)²

    Args:
        log_flows: Log state flow values, shape [batch, K+1].
        log_pf: Log forward transition probs, shape [batch, K].
        log_pb: Log backward transition probs, shape [batch, K].
        log_rewards: Log rewards, shape [batch].
        subtb_lambda: Geometric weighting for longer sub-trajectories.
        boundary_coeff: Coefficient for boundary loss.
        loss_mask: Optional mask for valid entries, shape [batch].

    Returns:
        Tuple[torch.Tensor, Dict]: (gfn_loss, metrics_dict)
    """
    bsz, K_plus_1 = log_flows.shape
    K = K_plus_1 - 1

    # Cumulative sums for efficient sub-trajectory computation
    # cum_log_pf[k] = Σ_{i=0}^{k-1} log P_F(i)  (cum_log_pf[0] = 0)
    # cum_log_pb[k] = Σ_{i=0}^{k-1} log P_B(i)  (cum_log_pb[0] = 0)
    zeros = torch.zeros(bsz, 1, device=log_pf.device, dtype=log_pf.dtype)
    cum_log_pf = torch.cat([zeros, torch.cumsum(log_pf, dim=1)], dim=1)  # [bsz, K+1]
    cum_log_pb = torch.cat([zeros, torch.cumsum(log_pb, dim=1)], dim=1)  # [bsz, K+1]

    # Enumerate all C(K+1, 2) sub-trajectory pairs (j, k) with j < k
    total_loss = torch.zeros(bsz, device=log_flows.device, dtype=log_flows.dtype)
    count = 0
    weight_sum = 0.0

    for j in range(K_plus_1):
        for k in range(j + 1, K_plus_1):
            length = k - j
            weight = subtb_lambda ** length

            # residual = log F(s_j) + (cum_pf[k] - cum_pf[j])
            #          - log F(s_k) - (cum_pb[k] - cum_pb[j])
            residual = (
                log_flows[:, j]
                + (cum_log_pf[:, k] - cum_log_pf[:, j])
                - log_flows[:, k]
                - (cum_log_pb[:, k] - cum_log_pb[:, j])
            )  # [bsz]

            total_loss = total_loss + weight * residual.pow(2)
            count += 1
            weight_sum += weight

    subtb_loss = total_loss / max(weight_sum, 1e-8)  # normalize by weight sum

    # Boundary loss: (log F(s_K) - log R)²
    boundary_loss = (log_flows[:, -1] - log_rewards).pow(2)  # [bsz]

    # Combined per-sample loss
    per_sample_loss = subtb_loss + boundary_coeff * boundary_loss  # [bsz]

    # Apply loss mask
    if loss_mask is not None:
        if loss_mask.dim() > 1:
            loss_mask = loss_mask.reshape(loss_mask.shape[0], -1).any(dim=-1)
        loss_mask = loss_mask.float()
        loss = (per_sample_loss * loss_mask).sum() / loss_mask.sum().clamp(min=1.0)
    else:
        loss = per_sample_loss.mean()

    metrics_data = {
        "actor/gfn_loss": loss.detach().item(),
        "actor/gfn_subtb_loss": subtb_loss.detach().mean().item(),
        "actor/gfn_boundary_loss": boundary_loss.detach().mean().item(),
        "actor/gfn_log_flow_mean": log_flows.detach().mean().item(),
        "actor/gfn_log_flow_terminal": log_flows[:, -1].detach().mean().item(),
        "actor/gfn_log_reward_mean": log_rewards.detach().mean().item(),
        "actor/gfn_log_pf_mean": log_pf.detach().mean().item(),
        "actor/gfn_log_pb_mean": log_pb.detach().mean().item(),
        "actor/gfn_subtb_pairs": float(count),
    }
    return loss, metrics_data


@register_policy_loss("flow_nft")
def compute_flow_nft_loss(
    v_theta: torch.Tensor,
    v_old: torch.Tensor,
    x_t: torch.Tensor,
    x_next: torch.Tensor,
    schedule: torch.Tensor,
    advantages: torch.Tensor,
    loss_mask: Optional[torch.Tensor] = None,
    step_indices: Optional[torch.Tensor] = None,
    total_denoise_steps: Optional[int] = None,
    noise_level: Optional[torch.Tensor | float] = None,
    std_epsilon: float = 1e-4,
    beta: float = 1.0,
    kl_beta: float = 0.0001,
    adv_clip_max: float = 1.0,
    max_drift: float = 0.5,
    dpo_beta: float = 1.0,
    critic_warmup: bool = False,
    x0_target: Optional[torch.Tensor] = None,
    use_x0_target: bool = False,
    loss_form: str = "dpo",
    **kwargs,
) -> tuple[torch.Tensor, dict]:
    """
    Compute pi-StepNFT contrastive mirror loss for flow-based VLA.

    Faithfully ported from pi-StepNFT reference implementation.
    Core idea: construct mirror velocities v+/v- from (v_theta - v_old),
    compute sample-space Mahalanobis energy, and apply DPO-style softplus loss.
    Uses per-trajectory averaging (_masked_mean_per_traj) for stable gradients.

    All inputs (v_theta, v_old, x_t, x_next) should be pre-cropped to the same
    shape [batch, n_steps, action_dim] before calling this function.

    Args:
        v_theta: Current model velocity [batch, n_steps, action_dim].
        v_old: Reference velocity from rollout (detached) [batch, n_steps, action_dim].
        x_t: SDE chain state at snapshot step [batch, n_steps, action_dim].
        x_next: SDE chain state after snapshot step [batch, n_steps, action_dim].
        schedule: Denoising schedule linspace(1, 0, K+1) [K+1].
        advantages: Per-sample advantages [batch] or [batch, n_steps].
        loss_mask: Optional mask [batch] or [batch, n_steps].
        step_indices: Which denoising step was sampled per batch [batch].
        total_denoise_steps: Total K denoising steps.
        noise_level: SDE noise level per batch [batch] or scalar.
        std_epsilon: Numerical stability for variance.
        beta: Mirror construction scale.
        kl_beta: KL trust region weight.
        adv_clip_max: Advantage clipping range.
        max_drift: Max velocity drift norm (critical stability param).
        dpo_beta: DPO loss temperature.
        critic_warmup: If True, zero out total_loss during critic warmup phase.
        x0_target: Optional target for x0-space energy (alternative to flow mean).
        use_x0_target: Whether to use x0_target instead of flow mean.
        loss_form: "dpo" (softplus) or "weighted" (weighted energy).

    Returns:
        Tuple[torch.Tensor, Dict]: (total_loss, metrics_dict)
    """
    # ---- helpers (from pi-StepNFT reference) ----

    def _align_to_steps(
        x: torch.Tensor | None, target_shape: Sequence[int]
    ) -> torch.Tensor | None:
        if x is None:
            return None
        target_b, target_steps = target_shape
        if x.shape == (target_b, target_steps):
            return x
        if x.ndim == 1 and x.shape[0] == target_b:
            return x.unsqueeze(1).expand(target_b, target_steps)
        if x.ndim == 2 and x.shape[0] == target_b and x.shape[1] == 1:
            return x.expand(target_b, target_steps)
        if x.numel() == target_b * target_steps:
            return x.reshape(target_b, target_steps)
        raise ValueError(f"Cannot align tensor of shape {x.shape} to steps {target_shape}")

    def _masked_mean_per_traj(
        x: torch.Tensor, mask: Optional[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if mask is None:
            traj_mean = x.mean(dim=1)
            valid_mask = torch.ones_like(traj_mean, dtype=torch.bool)
            return traj_mean, valid_mask
        mask = mask.float()
        x = x * mask
        valid = mask.sum(dim=1)
        valid_clamped = valid.clamp_min(1.0)
        traj_sum = x.sum(dim=1)
        valid_mask = valid > 0
        traj_mean = torch.where(
            valid_mask, traj_sum / valid_clamped, torch.zeros_like(traj_sum)
        )
        return traj_mean, valid_mask

    def _pad_right_ndim(x: torch.Tensor, target_ndim: int) -> torch.Tensor:
        while x.ndim < target_ndim:
            x = x.unsqueeze(-1)
        return x

    # ---- shape setup ----
    batch_size, n_steps = x_t.shape[:2]
    step_shape = (batch_size, n_steps)

    advantages = _align_to_steps(advantages, step_shape)
    loss_mask = _align_to_steps(loss_mask, step_shape)

    if advantages is None:
        raise ValueError("NFT loss requires `advantages`.")
    if loss_mask is None:
        loss_mask_float = None
    else:
        loss_mask_float = loss_mask.float()

    if step_indices is None or total_denoise_steps is None or noise_level is None:
        raise ValueError(
            "step_indices, total_denoise_steps, and noise_level must be provided for NFT loss."
        )

    # ---- advantage -> label ----
    advantages_clip = torch.clamp(advantages, -adv_clip_max, adv_clip_max)
    normalized_advantages_clip = (advantages_clip / adv_clip_max) / 2.0 + 0.5
    r = torch.clamp(normalized_advantages_clip, 0, 1)
    y = r * 2.0 - 1.0

    # ---- mirror construction with max_drift clipping ----
    v_old = v_old.detach()
    delta_v = v_theta - v_old

    dims_v = tuple(range(2, delta_v.ndim))
    delta_norm = delta_v.norm(dim=dims_v, keepdim=True) + 1e-8
    clip_coef = (max_drift / delta_norm).clamp(max=1.0)

    delta_v_clipped = delta_v * clip_coef
    v_pos = v_old + beta * delta_v_clipped
    v_neg = v_old - beta * delta_v_clipped

    # ---- schedule-based variance for Mahalanobis normalization ----
    dims = tuple(range(2, x_t.ndim))
    idx = step_indices.long()
    t_cur = schedule[idx]
    t_next = schedule[idx + 1]
    delta = t_cur - t_next

    t_bc = _pad_right_ndim(t_cur, x_t.ndim)
    delta_bc = _pad_right_ndim(delta, x_t.ndim)

    denom = schedule.clone()
    denom[0] = denom[1]
    sigma_base = torch.sqrt(schedule / (1 - denom))[:-1]
    sigma_i = _pad_right_ndim(sigma_base[idx], x_t.ndim)
    nl_tensor = torch.as_tensor(noise_level, device=x_t.device, dtype=x_t.dtype)
    sigma_i = sigma_i * _pad_right_ndim(nl_tensor, sigma_i.ndim)

    std_t = torch.sqrt(delta_bc.clamp_min(0)) * sigma_i
    std_t_detached = std_t.detach()

    # ---- energy computation ----
    if use_x0_target:
        if x0_target is None:
            raise ValueError("use_x0_target=True requires `x0_target` to be provided.")
        if x0_target.shape != x_t.shape:
            raise ValueError(
                f"x0_target shape {x0_target.shape} must match x_t shape {x_t.shape}."
            )
        x0_pos = x_t - t_bc * v_pos
        x0_neg = x_t - t_bc * v_neg
        var = std_t_detached**2 + std_epsilon
        E_pos = ((x0_pos - x0_target) ** 2 / var).sum(dim=dims)
        E_neg = ((x0_neg - x0_target) ** 2 / var).sum(dim=dims)
        delta_E = E_pos - E_neg
    else:
        def _flow_mean(x_cur: torch.Tensor, velocity: torch.Tensor) -> torch.Tensor:
            x0_pred = x_cur - velocity * t_bc
            x1_pred = x_cur + velocity * (1 - t_bc)
            x0_weight = torch.ones_like(t_bc) - (t_bc - delta_bc)
            x1_weight = t_bc - delta_bc - sigma_i**2 * delta_bc / (2 * t_bc)
            return x0_pred * x0_weight + x1_pred * x1_weight

        mean_pos = _flow_mean(x_t, v_pos)
        mean_neg = _flow_mean(x_t, v_neg)
        var = std_t_detached**2 + std_epsilon
        E_pos = ((x_next - mean_pos) ** 2 / var).sum(dim=dims)
        E_neg = ((x_next - mean_neg) ** 2 / var).sum(dim=dims)
        delta_E = E_pos - E_neg

    # ---- contrastive loss with per-trajectory averaging ----
    logit = (dpo_beta / 2.0) * y * delta_E

    if loss_form == "weighted":
        L_step = r * E_pos + (1 - r) * E_neg
        traj_loss, traj_valid = _masked_mean_per_traj(L_step, loss_mask_float)
    else:
        L_step = F.softplus(logit)
        traj_loss, traj_valid = _masked_mean_per_traj(L_step, loss_mask_float)
    nft_loss = traj_loss.sum() / traj_valid.sum().clamp_min(1.0)

    # ---- KL trust region (also per-trajectory averaged) ----
    kl_loss_per_sample = torch.mean((v_theta - v_old) ** 2, dim=dims)
    kl_per_traj, kl_valid = _masked_mean_per_traj(kl_loss_per_sample, loss_mask_float)
    kl_loss = kl_per_traj.sum() / kl_valid.sum().clamp_min(1.0)

    total_loss = nft_loss + kl_beta * kl_loss
    if critic_warmup:
        total_loss = torch.tensor(0.0, device=total_loss.device, dtype=total_loss.dtype)

    # ---- metrics ----
    with torch.no_grad():
        adv_mean = advantages.mean()
        adv_std = advantages.std()
        adv_clip_frac = (advantages.abs() >= adv_clip_max).float().mean()
        r_mean = r.mean()
        r_std = r.std()
        y_abs_mean = y.abs().mean()
        y_sat_frac = ((r < 0.05) | (r > 0.95)).float().mean()

        delta_v_norm = delta_v.norm(dim=dims_v)
        delta_v_clipped_norm = delta_v_clipped.norm(dim=dims_v)
        clip_frac = (clip_coef < 1).float().mean()
        clip_coef_mean = clip_coef.mean()

        std_mean = std_t_detached.mean()
        std_min = std_t_detached.min()
        std_max = std_t_detached.max()
        z2_mean = (
            ((x0_pos - x0_target) / (std_t_detached + std_epsilon)).pow(2).mean()
            if use_x0_target
            else ((x_next - mean_pos) / (std_t_detached + std_epsilon)).pow(2).mean()
        )
        finite_frac = torch.isfinite(delta_E).float().mean()

        logit_mean = logit.mean()
        logit_std = logit.std()
        margin_mean = (-logit).mean()
        pref_acc = (logit < 0).float().mean()
        y_abs = y.abs()
        mask_strong = y_abs > 0.3
        pref_acc_strong = (
            (logit[mask_strong] < 0).float().mean()
            if mask_strong.any()
            else torch.tensor(0.0, device=x_t.device)
        )
        pref_acc_weighted = (
            ((logit < 0).float() * y_abs).sum() / (y_abs.sum() + 1e-8)
        )
        deltaE_pos_mean = (
            delta_E[y > 0].mean()
            if (y > 0).any()
            else torch.tensor(0.0, device=x_t.device)
        )
        deltaE_neg_mean = (
            delta_E[y < 0].mean()
            if (y < 0).any()
            else torch.tensor(0.0, device=x_t.device)
        )
        E_pos_mean = E_pos.mean()
        E_neg_mean = E_neg.mean()
        delta_E_mean = delta_E.mean()

        kl_raw = kl_per_traj.sum() / kl_valid.sum().clamp_min(1.0)
        kl_weighted = kl_beta * kl_raw
        kl_ratio = kl_weighted / (nft_loss + 1e-8)

    metrics_data = {
        "actor/nft_loss": nft_loss.detach().item(),
        "actor/kl_loss": kl_loss.detach().item(),
        "actor/total_loss": total_loss.detach().item(),
        "actor/adv_mean": adv_mean.item(),
        "actor/adv_std": adv_std.item(),
        "actor/adv_clip_frac": adv_clip_frac.item(),
        "actor/r_mean": r_mean.item(),
        "actor/r_std": r_std.item(),
        "actor/y_abs_mean": y_abs_mean.item(),
        "actor/y_sat_frac": y_sat_frac.item(),
        "actor/delta_v_norm_mean": delta_v_norm.mean().item(),
        "actor/delta_v_clipped_norm_mean": delta_v_clipped_norm.mean().item(),
        "actor/clip_coef_mean": clip_coef_mean.item(),
        "actor/clip_frac": clip_frac.item(),
        "actor/std_mean": std_mean.item(),
        "actor/std_min": std_min.item(),
        "actor/std_max": std_max.item(),
        "actor/z2_mean": z2_mean.item(),
        "actor/finite_frac": finite_frac.item(),
        "actor/logit_mean": logit_mean.item(),
        "actor/logit_std": logit_std.item(),
        "actor/margin_mean": margin_mean.item(),
        "actor/pref_acc": pref_acc.item(),
        "actor/pref_acc_strong": pref_acc_strong.item(),
        "actor/pref_acc_weighted": pref_acc_weighted.item(),
        "actor/deltaE_pos_mean": deltaE_pos_mean.item(),
        "actor/deltaE_neg_mean": deltaE_neg_mean.item(),
        "actor/E_pos_mean": E_pos_mean.item(),
        "actor/E_neg_mean": E_neg_mean.item(),
        "actor/delta_E_mean": delta_E_mean.item(),
        "actor/kl_raw": kl_raw.item(),
        "actor/kl_weighted": kl_weighted.item(),
        "actor/kl_ratio": kl_ratio.item(),
    }
    return total_loss, metrics_data


@register_policy_loss("flow_hinge_nft")
def compute_flow_hinge_nft_loss(
    v_theta: torch.Tensor,
    v_old: torch.Tensor,
    x_t: torch.Tensor,
    x_next: torch.Tensor,
    schedule: torch.Tensor,
    advantages: torch.Tensor,
    loss_mask: Optional[torch.Tensor] = None,
    step_indices: Optional[torch.Tensor] = None,
    total_denoise_steps: Optional[int] = None,
    noise_level: Optional[torch.Tensor | float] = None,
    std_epsilon: float = 1e-4,
    beta: float = 1.0,
    kl_beta: float = 0.0001,
    adv_clip_max: float = 1.0,
    max_drift: float = 0.5,
    margin: float = 1.0,
    critic_warmup: bool = False,
    **kwargs,
) -> tuple[torch.Tensor, dict]:
    """
    Compute Hinge-NFT contrastive mirror loss for flow-based VLA.

    Identical to pi-StepNFT (compute_flow_nft_loss) except the outer loss
    function: softplus is replaced by a symmetric hinge (margin) loss.

    Hinge loss: max(0, margin + 0.5 * y * delta_E)

    This makes the loss symmetric w.r.t. label noise, enabling safe use
    of continuous advantages (e.g., from self-annotation temporal credit).

    Args:
        v_theta: Current model velocity [batch, n_steps, action_dim].
        v_old: Reference velocity from rollout (detached) [batch, n_steps, action_dim].
        x_t: SDE chain state at snapshot step [batch, n_steps, action_dim].
        x_next: SDE chain state after snapshot step [batch, n_steps, action_dim].
        schedule: Denoising schedule linspace(1, 0, K+1) [K+1].
        advantages: Per-sample advantages [batch] or [batch, n_steps].
        loss_mask: Optional mask [batch] or [batch, n_steps].
        step_indices: Which denoising step was sampled per batch [batch].
        total_denoise_steps: Total K denoising steps.
        noise_level: SDE noise level per batch [batch] or scalar.
        std_epsilon: Numerical stability for variance.
        beta: Mirror construction scale.
        kl_beta: KL trust region weight.
        adv_clip_max: Advantage clipping range.
        max_drift: Max velocity drift norm.
        margin: Hinge margin m (core hyperparameter).
        critic_warmup: If True, zero out total_loss during warmup.

    Returns:
        Tuple[torch.Tensor, Dict]: (total_loss, metrics_dict)
    """
    # ---- helpers (shared with NFT) ----

    def _align_to_steps(
        x: torch.Tensor | None, target_shape: Sequence[int]
    ) -> torch.Tensor | None:
        if x is None:
            return None
        target_b, target_steps = target_shape
        if x.shape == (target_b, target_steps):
            return x
        if x.ndim == 1 and x.shape[0] == target_b:
            return x.unsqueeze(1).expand(target_b, target_steps)
        if x.ndim == 2 and x.shape[0] == target_b and x.shape[1] == 1:
            return x.expand(target_b, target_steps)
        if x.numel() == target_b * target_steps:
            return x.reshape(target_b, target_steps)
        raise ValueError(f"Cannot align tensor of shape {x.shape} to steps {target_shape}")

    def _masked_mean_per_traj(
        x: torch.Tensor, mask: Optional[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if mask is None:
            traj_mean = x.mean(dim=1)
            valid_mask = torch.ones_like(traj_mean, dtype=torch.bool)
            return traj_mean, valid_mask
        mask = mask.float()
        x = x * mask
        valid = mask.sum(dim=1)
        valid_clamped = valid.clamp_min(1.0)
        traj_sum = x.sum(dim=1)
        valid_mask = valid > 0
        traj_mean = torch.where(
            valid_mask, traj_sum / valid_clamped, torch.zeros_like(traj_sum)
        )
        return traj_mean, valid_mask

    def _pad_right_ndim(x: torch.Tensor, target_ndim: int) -> torch.Tensor:
        while x.ndim < target_ndim:
            x = x.unsqueeze(-1)
        return x

    # ---- shape setup ----
    batch_size, n_steps = x_t.shape[:2]
    step_shape = (batch_size, n_steps)

    advantages = _align_to_steps(advantages, step_shape)
    loss_mask = _align_to_steps(loss_mask, step_shape)

    if advantages is None:
        raise ValueError("Hinge-NFT loss requires `advantages`.")
    if loss_mask is None:
        loss_mask_float = None
    else:
        loss_mask_float = loss_mask.float()

    if step_indices is None or total_denoise_steps is None or noise_level is None:
        raise ValueError(
            "step_indices, total_denoise_steps, and noise_level must be provided."
        )

    # ---- advantage -> label ----
    advantages_clip = torch.clamp(advantages, -adv_clip_max, adv_clip_max)
    normalized_advantages_clip = (advantages_clip / adv_clip_max) / 2.0 + 0.5
    r = torch.clamp(normalized_advantages_clip, 0, 1)
    y = r * 2.0 - 1.0  # y = A_t / c_adv ∈ [-1, 1]

    # ---- mirror construction with max_drift clipping (same as NFT) ----
    v_old = v_old.detach()
    delta_v = v_theta - v_old

    dims_v = tuple(range(2, delta_v.ndim))
    delta_norm = delta_v.norm(dim=dims_v, keepdim=True) + 1e-8
    clip_coef = (max_drift / delta_norm).clamp(max=1.0)

    delta_v_clipped = delta_v * clip_coef
    v_pos = v_old + beta * delta_v_clipped
    v_neg = v_old - beta * delta_v_clipped

    # ---- schedule-based variance for Mahalanobis normalization (same as NFT) ----
    dims = tuple(range(2, x_t.ndim))
    idx = step_indices.long()
    t_cur = schedule[idx]
    t_next = schedule[idx + 1]
    delta = t_cur - t_next

    t_bc = _pad_right_ndim(t_cur, x_t.ndim)
    delta_bc = _pad_right_ndim(delta, x_t.ndim)

    denom = schedule.clone()
    denom[0] = denom[1]
    sigma_base = torch.sqrt(schedule / (1 - denom))[:-1]
    sigma_i = _pad_right_ndim(sigma_base[idx], x_t.ndim)
    nl_tensor = torch.as_tensor(noise_level, device=x_t.device, dtype=x_t.dtype)
    sigma_i = sigma_i * _pad_right_ndim(nl_tensor, sigma_i.ndim)

    std_t = torch.sqrt(delta_bc.clamp_min(0)) * sigma_i
    std_t_detached = std_t.detach()

    # ---- energy computation (same as NFT) ----
    def _flow_mean(x_cur: torch.Tensor, velocity: torch.Tensor) -> torch.Tensor:
        x0_pred = x_cur - velocity * t_bc
        x1_pred = x_cur + velocity * (1 - t_bc)
        x0_weight = torch.ones_like(t_bc) - (t_bc - delta_bc)
        x1_weight = t_bc - delta_bc - sigma_i**2 * delta_bc / (2 * t_bc)
        return x0_pred * x0_weight + x1_pred * x1_weight

    mean_pos = _flow_mean(x_t, v_pos)
    mean_neg = _flow_mean(x_t, v_neg)
    var = std_t_detached**2 + std_epsilon
    E_pos = ((x_next - mean_pos) ** 2 / var).sum(dim=dims)
    E_neg = ((x_next - mean_neg) ** 2 / var).sum(dim=dims)
    delta_E = E_pos - E_neg

    # ---- ★ HINGE loss (the ONLY change from NFT) ----
    logit = 0.5 * y * delta_E
    hinge_raw = margin + logit  # margin + ½ y ΔE
    L_step = torch.clamp(hinge_raw, min=0.0)  # max(0, ...)
    traj_loss, traj_valid = _masked_mean_per_traj(L_step, loss_mask_float)
    hinge_loss = traj_loss.sum() / traj_valid.sum().clamp_min(1.0)

    # ---- KL trust region (same as NFT) ----
    kl_loss_per_sample = torch.mean((v_theta - v_old) ** 2, dim=dims)
    kl_per_traj, kl_valid = _masked_mean_per_traj(kl_loss_per_sample, loss_mask_float)
    kl_loss = kl_per_traj.sum() / kl_valid.sum().clamp_min(1.0)

    total_loss = hinge_loss + kl_beta * kl_loss
    if critic_warmup:
        total_loss = torch.tensor(0.0, device=total_loss.device, dtype=total_loss.dtype)

    # ---- metrics ----
    with torch.no_grad():
        adv_mean = advantages.mean()
        adv_std = advantages.std()
        r_mean = r.mean()
        y_abs_mean = y.abs().mean()

        delta_v_norm = delta_v.norm(dim=dims_v)
        clip_frac = (clip_coef < 1).float().mean()
        clip_coef_mean = clip_coef.mean()

        std_mean = std_t_detached.mean()
        z2_mean = ((x_next - mean_pos) / (std_t_detached + std_epsilon)).pow(2).mean()
        finite_frac = torch.isfinite(delta_E).float().mean()

        # Hinge-specific metrics
        margin_violated = (hinge_raw > 0).float()
        margin_violation_frac = margin_violated.mean()
        margin_satisfied_frac = 1.0 - margin_violation_frac
        active_loss_mean = (
            L_step[hinge_raw > 0].mean()
            if (hinge_raw > 0).any()
            else torch.tensor(0.0, device=x_t.device)
        )

        logit_mean = logit.mean()
        logit_std = logit.std()
        pref_acc = (logit < 0).float().mean()
        y_abs = y.abs()
        mask_strong = y_abs > 0.3
        pref_acc_strong = (
            (logit[mask_strong] < 0).float().mean()
            if mask_strong.any()
            else torch.tensor(0.0, device=x_t.device)
        )

        E_pos_mean = E_pos.mean()
        E_neg_mean = E_neg.mean()
        delta_E_mean = delta_E.mean()

        kl_raw = kl_per_traj.sum() / kl_valid.sum().clamp_min(1.0)
        kl_weighted = kl_beta * kl_raw
        kl_ratio = kl_weighted / (hinge_loss + 1e-8)

    metrics_data = {
        "actor/hinge_loss": hinge_loss.detach().item(),
        "actor/kl_loss": kl_loss.detach().item(),
        "actor/total_loss": total_loss.detach().item(),
        "actor/margin_violation_frac": margin_violation_frac.item(),
        "actor/margin_satisfied_frac": margin_satisfied_frac.item(),
        "actor/active_loss_mean": active_loss_mean.item(),
        "actor/adv_mean": adv_mean.item(),
        "actor/adv_std": adv_std.item(),
        "actor/r_mean": r_mean.item(),
        "actor/y_abs_mean": y_abs_mean.item(),
        "actor/delta_v_norm_mean": delta_v_norm.mean().item(),
        "actor/clip_coef_mean": clip_coef_mean.item(),
        "actor/clip_frac": clip_frac.item(),
        "actor/std_mean": std_mean.item(),
        "actor/z2_mean": z2_mean.item(),
        "actor/finite_frac": finite_frac.item(),
        "actor/logit_mean": logit_mean.item(),
        "actor/logit_std": logit_std.item(),
        "actor/pref_acc": pref_acc.item(),
        "actor/pref_acc_strong": pref_acc_strong.item(),
        "actor/E_pos_mean": E_pos_mean.item(),
        "actor/E_neg_mean": E_neg_mean.item(),
        "actor/delta_E_mean": delta_E_mean.item(),
        "actor/kl_raw": kl_raw.item(),
        "actor/kl_weighted": kl_weighted.item(),
        "actor/kl_ratio": kl_ratio.item(),
    }
    return total_loss, metrics_data


@register_policy_loss("flow_fpi")
def compute_flow_fpi_loss(
    v_theta: torch.Tensor,
    u_target: torch.Tensor,
    weights: torch.Tensor,
    loss_mask: Optional[torch.Tensor] = None,
    **kwargs,
) -> tuple[torch.Tensor, dict]:
    """
    Compute Flow Policy Iteration (FPI) loss: advantage-weighted flow matching MSE.

    L_FPI = w_i · ||v_θ(x_t, t | o, l) - u||²

    where w_i = exp(A_i / λ), self-normalized and clipped.

    Good actions (A > 0) get w > 1: velocity field learns them more.
    Bad actions (A < 0) get w < 1: velocity field ignores them.
    This is the simplest possible policy improvement for flow matching.

    Args:
        v_theta: Current model velocity prediction, shape [batch, chunk, action_dim].
        u_target: Flow matching target velocity (ε - a), same shape.
        weights: Per-sample advantage weights from FPI credit assignment, shape [batch].
        loss_mask: Optional mask for valid entries.

    Returns:
        Tuple[torch.Tensor, Dict]: (flow_fpi_loss, metrics_dict)
    """
    # Per-sample MSE: ||v_θ - u||², mean over chunk and action_dim
    mse = (v_theta - u_target).pow(2).mean(dim=(-2, -1))  # [batch]

    # Weighted loss
    per_sample_loss = weights * mse  # [batch]

    # Apply loss mask if provided
    if loss_mask is not None:
        if loss_mask.dim() > 1:
            loss_mask = loss_mask.reshape(loss_mask.shape[0], -1).any(dim=-1)
        loss_mask = loss_mask.float()
        loss = (per_sample_loss * loss_mask).sum() / loss_mask.sum().clamp(min=1.0)
    else:
        loss = per_sample_loss.mean()

    metrics_data = {
        "actor/fpi_loss": loss.detach().item(),
        "actor/fpi_mse": mse.detach().mean().item(),
        "actor/fpi_weight_mean": weights.detach().mean().item(),
        "actor/fpi_weight_std": weights.detach().std().item(),
        "actor/fpi_weight_max": weights.detach().max().item(),
    }
    return loss, metrics_data


def compute_residual_kinetic_energy(
    u_theta_list: list[torch.Tensor],
    u_pre_list: list[torch.Tensor],
    dt_list: list[float],
) -> torch.Tensor:
    """
    Compute residual kinetic energy E_res = Σ_k ½ ||u_θ_k - u_pre_k||² · dt_k.

    This is the path-space deviation measure from the pretrained policy.
    By Girsanov theorem, D_KL(P_θ || P_pre) = E_res / σ² (exact, not a bound).

    Args:
        u_theta_list: List of [batch, horizon, dim] current policy velocities at each ODE step.
        u_pre_list: List of [batch, horizon, dim] pretrained policy velocities (detached).
        dt_list: List of scalar time step sizes for each ODE step.

    Returns:
        e_res: [batch] residual kinetic energy per sample.
    """
    e_res = torch.zeros(u_theta_list[0].shape[0], device=u_theta_list[0].device)
    for u_theta, u_pre, dt in zip(u_theta_list, u_pre_list, dt_list):
        diff = u_theta - u_pre.detach()
        # sum over action dims, mean over horizon steps
        e_res = e_res + 0.5 * (diff ** 2).sum(dim=-1).mean(dim=-1) * dt
    return e_res


@register_policy_loss("flow_rkfac")
def compute_flow_rkfac_loss(
    actor_loss: torch.Tensor = None,
    **kwargs,
) -> tuple[torch.Tensor, dict]:
    """
    RK-FAC loss placeholder — actual loss is computed directly in actor worker
    (Q update + actor update with E_res and Q gradient through ODE).

    This registered entry just passes through the pre-computed actor loss
    so the registry dispatch system works correctly.
    """
    if actor_loss is None:
        actor_loss = torch.tensor(0.0)
    return actor_loss, {}


@register_policy_loss("flow_qgfm")
def compute_flow_qgfm_loss(
    actor_loss: torch.Tensor = None,
    **kwargs,
) -> tuple[torch.Tensor, dict]:
    """
    QGFM loss placeholder — actual loss is computed directly in actor worker.

    QGFM = Q-Guided Flow Matching: standard flow matching MSE with Q-gradient
    perturbed target action.  L = ||v_θ(x_t, t | s) - (ε - a')||²
    where a' = a + η·∇_a Q(s, a) / ||∇_a Q||.

    The actor worker handles Q-network TD update + target perturbation + flow
    matching forward.  This registered entry just passes through the pre-computed
    loss so the registry dispatch system works correctly.
    """
    if actor_loss is None:
        actor_loss = torch.tensor(0.0)
    return actor_loss, {}
