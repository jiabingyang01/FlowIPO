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

from typing import Callable, Optional

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
