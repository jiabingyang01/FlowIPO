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

import copy
import gc
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from tqdm import tqdm

from rlinf.config import SupportedModel
from rlinf.data.embodied_io_struct import (
    ChunkStepResult,
    EmbodiedRolloutResult,
    Trajectory,
)
from rlinf.models import get_model
from rlinf.models.embodiment.base_policy import BasePolicy
from rlinf.scheduler import Channel, Cluster, CollectiveGroupOptions, Worker
from rlinf.utils.metric_utils import compute_split_num
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.utils.utils import cpu_weight_swap, get_model_weights_id


class MultiStepRolloutWorker(Worker):
    def __init__(self, cfg: DictConfig):
        Worker.__init__(self)

        self.cfg = cfg
        self.should_stop = False

        self.actor_group_name = cfg.actor.group_name
        self.device = torch.cuda.current_device()

        self.num_pipeline_stages = cfg.rollout.pipeline_stage_num
        self.enable_offload = self.cfg.rollout.get("enable_offload", False)

        self.placement = HybridComponentPlacement(cfg, Cluster())

        actor_world_size = self.placement.get_world_size("actor")
        self.actor_weight_src_rank = self._rank % actor_world_size

        self.collect_transitions = self.cfg.rollout.get("collect_transitions", False)
        self.model_weights_id = ""
        self.count_update = 0

        # Sync weight comm options
        max_ctas = cfg.rollout.get("sync_weight_nccl_max_ctas", None)
        min_ctas = cfg.rollout.get("sync_weight_nccl_min_ctas", None)
        self._sync_weight_comm_options = CollectiveGroupOptions(
            accel_max_ctas=max_ctas, accel_min_ctas=min_ctas
        )
        self.total_num_train_envs = cfg.env.train.total_num_envs
        self.total_num_eval_envs = cfg.env.eval.total_num_envs
        self.num_pipeline_stages = cfg.rollout.pipeline_stage_num
        self.train_batch_size = (
            self.total_num_train_envs // self._world_size // self.num_pipeline_stages
        )
        self.eval_batch_size = (
            self.total_num_eval_envs // self._world_size // self.num_pipeline_stages
        )
        self.enable_cuda_graph = cfg.rollout.get("enable_cuda_graph", False)

        # FlowIPO / FlowSAR: EMA reference model for computation on rollout worker
        _loss_type = cfg.algorithm.get("loss_type", "")
        self._is_flow_ipo = _loss_type == "flow_ipo"
        self._is_flow_sar = _loss_type == "flow_sar"
        self._needs_ema_ref = self._is_flow_ipo or self._is_flow_sar
        if self._needs_ema_ref:
            self._ref_weights_cpu = None
            self._ref_swap_buffer = None
        if self._is_flow_ipo:
            self._flow_ipo_beta = cfg.algorithm.get("flow_ipo_beta_ref", 0.995)
        if self._is_flow_sar:
            self._flow_sar_ema_beta = cfg.algorithm.get("flow_sar_ema_beta", 0.995)
            # DiffusionNFT-style dynamic EMA schedule: η_i = min(eta_rate * i, eta_max)
            # "fixed" = use flow_sar_ema_beta constantly
            # "linear" = DiffusionNFT-style: β_i = min(eta_rate * i, eta_max)
            self._flow_sar_ema_schedule = cfg.algorithm.get("flow_sar_ema_schedule", "fixed")
            self._flow_sar_ema_eta_rate = cfg.algorithm.get("flow_sar_ema_eta_rate", 0.001)
            self._flow_sar_ema_eta_max = cfg.algorithm.get("flow_sar_ema_eta_max", 0.5)
            self._flow_sar_tau_mid = cfg.algorithm.get("flow_sar_tau_mid", 0.5)
            self._flow_sar_t_min = cfg.algorithm.get("flow_sar_t_min", 0.2)
            self._flow_sar_t_max = cfg.algorithm.get("flow_sar_t_max", 0.8)

    def init_worker(self):
        rollout_model_config = copy.deepcopy(self.cfg.actor.model)
        with open_dict(rollout_model_config):
            rollout_model_config.precision = self.cfg.rollout.model.precision
            rollout_model_config.model_path = self.cfg.rollout.model.model_path

        self.hf_model: BasePolicy = get_model(rollout_model_config)

        if self.cfg.runner.get("ckpt_path", None):
            model_dict = torch.load(self.cfg.runner.ckpt_path)
            self.hf_model.load_state_dict(model_dict)

        self.hf_model.eval()

        if self.cfg.rollout.get("enable_torch_compile", False):
            mode = self.cfg.rollout.get(
                "torch_compile_mode", "max-autotune-no-cudagraphs"
            )
            self.hf_model.enable_torch_compile(mode=mode)
        if self.enable_cuda_graph and not self.enable_offload:
            self.hf_model.capture_cuda_graph(
                train_batch_size=self.train_batch_size,
                eval_batch_size=self.eval_batch_size,
            )

        self.setup_sample_params()
        if self.enable_offload:
            self.offload_model()

    def setup_sample_params(self):
        # length parameters for rollout
        self._length_params = OmegaConf.to_container(
            self.cfg.algorithm.length_params, resolve=True
        )
        # sampling parameters for rollout
        self._sampling_params = OmegaConf.to_container(
            self.cfg.algorithm.sampling_params, resolve=True
        )
        self._train_sampling_params = {
            "do_sample": self._sampling_params["do_sample"],
            "temperature": self._sampling_params["temperature_train"]
            if self._sampling_params["do_sample"]
            else 1.0,
            "top_k": self._sampling_params["top_k"],
            "top_p": self._sampling_params["top_p"],
            "max_new_tokens": self._length_params["max_new_token"],
        }

        self._eval_sampling_params = {
            "do_sample": True
            if self._sampling_params.get("temperature_eval", -1) > 0
            else False,
            "temperature": self._sampling_params["temperature_eval"],
            "top_k": self._sampling_params["top_k"],
            "top_p": self._sampling_params["top_p"],
            "max_new_tokens": self._length_params["max_new_token"],
        }

    @Worker.timer("predict")
    def predict(self, env_obs, mode="train"):
        kwargs = (
            self._train_sampling_params
            if mode == "train"
            else self._eval_sampling_params
        )

        if SupportedModel(self.cfg.actor.model.model_type) in [
            SupportedModel.OPENPI,
            SupportedModel.MLP_POLICY,
            SupportedModel.GR00T,
            SupportedModel.CNN_POLICY,
        ]:
            kwargs = {"mode": mode}

        if SupportedModel(self.cfg.actor.model.model_type) in [
            SupportedModel.CNN_POLICY,
            SupportedModel.FLOW_POLICY,
            SupportedModel.MLP_POLICY,
        ]:
            kwargs["return_obs"] = not hasattr(self.hf_model, "q_head")

        with torch.no_grad():
            actions, result = self.hf_model.predict_action_batch(
                env_obs=env_obs,
                **kwargs,
            )

        return actions, result

    def get_dones_and_rewards(
        self,
        env_output: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, dict[str, Any] | None]:
        """
        Get dones and rewards from environment batch, handling auto_reset if needed.

        Args:
            env_output: Environment batch containing dones, rewards, and optionally final_obs

        Returns:
            Tuple of (dones, rewards). dones and rewards are tensors.
        """
        # First step: no rewards yet, only dones
        if env_output["rewards"] is None:
            return (
                env_output["dones"].bool().cpu().contiguous(),
                None,
            )

        dones = env_output["dones"].bool().cpu().contiguous()
        rewards = env_output["rewards"].cpu().contiguous()
        bootstrap_type = self.cfg.algorithm.get("bootstrap_type", "standard")

        if bootstrap_type == "standard":
            last_step_truncations = env_output["truncations"].cpu().contiguous()[:, -1]
        else:
            last_step_truncations = dones[:, -1]

        # Handle auto_reset: add bootstrap value ONLY for truncated episodes (not terminated)
        if last_step_truncations.any() and self.cfg.env.train.auto_reset:
            if hasattr(self.hf_model, "value_head") or hasattr(self.hf_model, "q_head"):
                final_obs = env_output["final_obs"]
                with torch.no_grad():
                    actions, result = self.predict(final_obs)
                    if "prev_values" in result:
                        _final_values = result["prev_values"]
                    else:
                        _final_values = torch.zeros_like(actions[:, 0])
                final_values = torch.zeros_like(_final_values[:, 0])  # [bsz, ]
                # bootstrap only on the truncated episode
                final_values[last_step_truncations] = _final_values[:, 0][
                    last_step_truncations
                ]
                # Add bootstrap value to the last step of truncated episodes
                rewards[:, -1] += self.cfg.algorithm.gamma * final_values.cpu()

        return dones, rewards

    async def sync_model_from_actor(self):
        """Sync model parameters from the actor worker."""
        param_state_dict = await self.recv(
            self.actor_group_name,
            src_rank=self.actor_weight_src_rank,
            async_op=True,
            options=self._sync_weight_comm_options,
        ).async_wait()

        # FlowIPO / FlowSAR: maintain EMA reference weights on rollout worker
        if self._needs_ema_ref:
            if self._is_flow_ipo:
                beta = self._flow_ipo_beta
            elif self._is_flow_sar and self._flow_sar_ema_schedule == "linear":
                # DiffusionNFT-style dynamic EMA: β_i = min(eta_rate * i, eta_max)
                # Early iterations: β ≈ 0 → aggressive update (ref ≈ θ_new)
                # Later iterations: β → eta_max → conservative update
                beta = min(
                    self._flow_sar_ema_eta_rate * self.count_update,
                    self._flow_sar_ema_eta_max,
                )
            else:
                beta = self._flow_sar_ema_beta
            if self._ref_weights_cpu is None:
                self._ref_weights_cpu = {
                    k: v.clone().cpu() for k, v in param_state_dict.items()
                }
            else:
                for k in self._ref_weights_cpu:
                    self._ref_weights_cpu[k].mul_(beta).add_(
                        param_state_dict[k].cpu(), alpha=1 - beta
                    )

        self.hf_model.load_state_dict(param_state_dict)
        self.model_weights_id = (
            str(get_model_weights_id(self.hf_model)) + f"_{self.count_update}"
        )
        self.count_update += 1

        del param_state_dict
        gc.collect()
        torch.cuda.empty_cache()

    async def send_rollout_trajectories(
        self, rollout_result: EmbodiedRolloutResult, channel: Channel
    ):
        split_num = self.get_actor_split_num()
        trajectories: Trajectory = rollout_result.to_splited_trajectories(split_num)
        for trajectory in trajectories:
            channel.put(trajectory, async_op=True)

    @Worker.timer("generate_one_epoch")
    async def generate_one_epoch(self, input_channel: Channel, output_channel: Channel):
        n_chunk_steps = (
            self.cfg.env.train.max_steps_per_rollout_epoch
            // self.cfg.actor.model.num_action_chunks
        )

        last_obs = [None for i in range(self.num_pipeline_stages)]
        for _ in range(n_chunk_steps):
            for stage_id in range(self.num_pipeline_stages):
                env_output = await self.recv_env_output(input_channel)

                if env_output["intervene_actions"] is not None:
                    self.rollout_results[stage_id].update_last_actions(
                        env_output["intervene_actions"],
                        env_output["intervene_flags"],
                    )

                dones, rewards = self.get_dones_and_rewards(env_output)

                actions, result = self.predict(env_output["obs"])

                env_output["obs"].pop("task_descriptions", None)
                if env_output["final_obs"] is not None:
                    env_output["final_obs"].pop("task_descriptions", None)
                chunk_step_result = ChunkStepResult(
                    actions=result["forward_inputs"].get("action", None),
                    dones=dones,
                    rewards=rewards,
                    truncations=env_output["truncations"],
                    terminations=env_output["terminations"],
                    prev_logprobs=result["prev_logprobs"]
                    if self.cfg.rollout.get("collect_prev_infos", True)
                    else None,
                    prev_values=result["prev_values"]
                    if self.cfg.rollout.get("collect_prev_infos", True)
                    else None,
                    forward_inputs=result["forward_inputs"],
                )

                self.rollout_results[stage_id].append_step_result(chunk_step_result)
                if self.collect_transitions and last_obs[stage_id] is not None:
                    curr_obs = last_obs[stage_id]
                    next_obs = (
                        env_output["final_obs"]
                        if dones.any() and self.cfg.env.train.auto_reset
                        else env_output["obs"]
                    )
                    self.rollout_results[stage_id].append_transitions(
                        curr_obs, next_obs
                    )

                last_obs[stage_id] = env_output["obs"]

                self.send_chunk_actions(output_channel, actions)

        for stage_id in range(self.num_pipeline_stages):
            env_output = await self.recv_env_output(input_channel)

            if env_output["intervene_actions"] is not None:
                self.rollout_results[stage_id].update_last_actions(
                    env_output["intervene_actions"], env_output["intervene_flags"]
                )

            dones, rewards = self.get_dones_and_rewards(env_output)

            _, result = self.predict(env_output["obs"])

            env_output["obs"].pop("task_descriptions", None)
            if env_output["final_obs"] is not None:
                env_output["final_obs"].pop("task_descriptions", None)

            chunk_step_result = ChunkStepResult(
                dones=dones,
                rewards=rewards,
                truncations=env_output["truncations"],
                terminations=env_output["terminations"],
                prev_logprobs=None,
                prev_values=result["prev_values"]
                if self.cfg.rollout.get("collect_prev_infos", True)
                else None,
                forward_inputs=None,
            )

            self.rollout_results[stage_id].append_step_result(chunk_step_result)
            if self.collect_transitions and last_obs[stage_id] is not None:
                curr_obs = last_obs[stage_id]
                next_obs = (
                    env_output["final_obs"]
                    if dones.any() and self.cfg.env.train.auto_reset
                    else env_output["obs"]
                )
                self.rollout_results[stage_id].append_transitions(curr_obs, next_obs)

    @torch.no_grad()
    def _compute_reference_actions(self, rollout_result: EmbodiedRolloutResult):
        """
        FlowIPO: compute ref_actions and ref_velocity using EMA model on rollout worker.
        Uses cpu_weight_swap on the non-FSDP rollout model.
        Stores ref_action, ref_v, flow_t, flow_epsilon in forward_inputs.
        """
        from openpi.models import model as _model

        t_min = self.cfg.algorithm.get("flow_ipo_t_min", 0.2)
        t_max = self.cfg.algorithm.get("flow_ipo_t_max", 0.8)

        with cpu_weight_swap(self.hf_model, self._ref_weights_cpu, self._ref_swap_buffer):
            for fi in rollout_result.forward_inputs:
                noise = fi.get("initial_noise", None)
                if noise is None:
                    continue
                # Reconstruct observation from cached "~" keys
                obs_dict = {}
                for k, v in fi.items():
                    if not k.startswith("~"):
                        continue
                    parts = k[1:].split("~", 1)
                    val = v.to(self.device) if torch.is_tensor(v) else v
                    if len(parts) == 2:
                        obs_dict.setdefault(parts[0], {})[parts[1]] = val
                    else:
                        obs_dict[parts[0]] = val
                observation = _model.Observation.from_dict(obs_dict)

                # 1) Reference actions via ODE sampling
                outputs = self.hf_model.sample_actions(
                    observation, noise=noise.to(self.device),
                    mode="eval", compute_values=False,
                )
                fi["ref_action"] = outputs["actions"].cpu()

                # 2) Pre-sample t, ε and compute reference velocity
                actions = fi["action"].to(self.device)
                bsz = actions.shape[0]
                t = torch.rand(bsz, 1, 1, device=self.device) * (t_max - t_min) + t_min
                epsilon = torch.randn_like(actions)
                x_t = (1 - t) * actions + t * epsilon
                v_ref = self.hf_model.forward_velocity(
                    None, x_t, t.reshape(bsz), observation=observation,
                )
                fi["ref_v"] = v_ref.cpu()
                fi["flow_t"] = t.cpu()
                fi["flow_epsilon"] = epsilon.cpu()

    @torch.no_grad()
    def _compute_self_annotation(self, rollout_result: EmbodiedRolloutResult):
        """
        FlowSAR Phase 2: Self-Annotation + pre-compute training data.

        For each rollout step:
        1. Self-Annotation: compute reconstruction error e_i (policy confidence proxy)
           - Forward noise: x_τ = τ_mid * a_i + (1 - τ_mid) * ε
           - One-step reconstruction: â_i = x_τ + v_θ_old(x_τ, τ_mid, s_i) * (1 - τ_mid)
           - Error: e_i = ||a_i - â_i||²

        2. Pre-compute training data: sample (t, ε), compute v_old
           - t ~ U[t_min, t_max], ε ~ N(0, I)
           - x_t = (1-t) * a_i + t * ε
           - v_old = v_θ_old(x_t, t, s_i)

        Stores: recon_error, flow_t, flow_epsilon, v_old in forward_inputs.
        """
        from openpi.models import model as _model

        tau_mid = self._flow_sar_tau_mid
        t_min = self._flow_sar_t_min
        t_max = self._flow_sar_t_max

        with cpu_weight_swap(self.hf_model, self._ref_weights_cpu, self._ref_swap_buffer):
            for fi in rollout_result.forward_inputs:
                actions = fi.get("action", None)
                if actions is None:
                    continue

                # Reconstruct observation from cached "~" keys (reuse FlowIPO pattern)
                obs_dict = {}
                for k, v in fi.items():
                    if not k.startswith("~"):
                        continue
                    parts = k[1:].split("~", 1)
                    val = v.to(self.device) if torch.is_tensor(v) else v
                    if len(parts) == 2:
                        obs_dict.setdefault(parts[0], {})[parts[1]] = val
                    else:
                        obs_dict[parts[0]] = val
                observation = _model.Observation.from_dict(obs_dict)

                actions_gpu = actions.to(self.device)
                bsz = actions_gpu.shape[0]

                # ===== Phase 2a: Self-Annotation (reconstruction error) =====
                # Model convention: t=0 → clean, t=1 → noise
                # x_t = (1-t)*a + t*ε,  velocity v = ε - a
                eps_recon = torch.randn_like(actions_gpu)
                x_tau = (1 - tau_mid) * actions_gpu + tau_mid * eps_recon
                t_recon = torch.full((bsz,), tau_mid, device=self.device)

                v_pred = self.hf_model.forward_velocity(
                    None, x_tau, t_recon, observation=observation,
                )
                # Backward integration to t=0 (clean): a_hat = x_tau - v * tau_mid
                a_hat = x_tau - v_pred * tau_mid
                # Per-sample reconstruction error (sum over chunk & action_dim)
                recon_error = (actions_gpu - a_hat).pow(2).sum(dim=(-2, -1))  # [batch]
                fi["recon_error"] = recon_error.cpu()

                # ===== Phase 2b: Pre-compute training data (t, ε, v_old) =====
                t = torch.rand(bsz, 1, 1, device=self.device) * (t_max - t_min) + t_min
                epsilon = torch.randn_like(actions_gpu)
                x_t = (1 - t) * actions_gpu + t * epsilon

                v_old = self.hf_model.forward_velocity(
                    None, x_t, t.reshape(bsz), observation=observation,
                )
                fi["v_old"] = v_old.cpu()
                fi["flow_t"] = t.cpu()
                fi["flow_epsilon"] = epsilon.cpu()

    async def generate(
        self, input_channel: Channel, output_channel: Channel, actor_channel: Channel
    ):
        if self.enable_offload:
            self.reload_model()

        # rollout_results[stage_id]
        self.rollout_results: list[EmbodiedRolloutResult] = [
            EmbodiedRolloutResult(
                max_episode_length=self.cfg.env.train.max_episode_steps,
                model_weights_id=self.model_weights_id,
            )
            for _ in range(self.num_pipeline_stages)
        ]

        for _ in tqdm(
            range(self.cfg.algorithm.rollout_epoch),
            desc="Generating Rollout Epochs",
            disable=(self._rank != 0),
        ):
            await self.generate_one_epoch(input_channel, output_channel)

        # FlowIPO / FlowSAR: compute additional data on rollout worker (non-FSDP)
        if self._needs_ema_ref and self._ref_weights_cpu is not None:
            for stage_id in range(self.num_pipeline_stages):
                if self._is_flow_ipo:
                    self._compute_reference_actions(self.rollout_results[stage_id])
                elif self._is_flow_sar:
                    self._compute_self_annotation(self.rollout_results[stage_id])

        for stage_id in range(self.num_pipeline_stages):
            await self.send_rollout_trajectories(
                self.rollout_results[stage_id], actor_channel
            )

        if self.enable_offload:
            self.offload_model()

    async def evaluate(self, input_channel: Channel, output_channel: Channel):
        if self.enable_offload:
            self.reload_model()

        n_chunk_steps = (
            self.cfg.env.eval.max_steps_per_rollout_epoch
            // self.cfg.actor.model.num_action_chunks
        )
        for _ in tqdm(
            range(self.cfg.algorithm.eval_rollout_epoch),
            desc="Evaluating Rollout Epochs",
            disable=(self._rank != 0),
        ):
            for _ in range(n_chunk_steps):
                for _ in range(self.num_pipeline_stages):
                    env_output = await self.recv_env_output(input_channel, mode="eval")
                    actions, _ = self.predict(env_output["obs"], mode="eval")
                    self.send_chunk_actions(output_channel, actions, mode="eval")

        if self.enable_offload:
            self.offload_model()

    def offload_model(self):
        if self.enable_cuda_graph:
            self.hf_model.release_cuda_graph()
        self.hf_model.to("cpu")
        torch.cuda.empty_cache()

    def reload_model(self):
        self.hf_model.to(self.device)
        if self.enable_cuda_graph:
            self.hf_model.capture_cuda_graph(
                train_batch_size=self.train_batch_size,
                eval_batch_size=self.eval_batch_size,
            )

    async def recv_env_output(
        self, input_channel: Channel, mode="train"
    ) -> dict[str, torch.Tensor]:
        assert mode in ["train", "eval"], f"{mode=} is not supported"
        # Use asyncio so that it can run alongside async weight syncing
        env_output = await input_channel.get(
            key=f"{self._rank}_{mode}", async_op=True
        ).async_wait()
        return env_output

    def send_chunk_actions(self, output_channel: Channel, chunk_actions, mode="train"):
        assert mode in ["train", "eval"], f"{mode=} is not supported"
        output_channel.put(
            item=chunk_actions, key=f"{self._rank}_{mode}", async_op=True
        )

    def get_actor_split_num(self):
        send_num = self.placement.get_world_size("rollout") * self.num_pipeline_stages
        recv_num = self.placement.get_world_size("actor")
        split_num = compute_split_num(recv_num, send_num)
        return split_num

    def set_global_step(self, global_step):
        if hasattr(self.hf_model, "set_global_step"):
            self.hf_model.set_global_step(global_step)
