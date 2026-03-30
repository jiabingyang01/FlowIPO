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

import os
import time
from functools import partial
from typing import Optional

import numpy as np
import torch
from omegaconf import DictConfig
from torch import nn
from torch.distributed.tensor import DTensor
from torch.multiprocessing.reductions import reduce_tensor

import rlinf.algorithms  # noqa: F401
from rlinf.algorithms.registry import calculate_adv_and_returns, policy_loss
from rlinf.algorithms.reward_shaping import (
    SuccessFeatureBuffer,
    compute_pbrs_reward,
    compute_vlm_potential,
)
from rlinf.algorithms.utils import (
    kl_penalty,
)
from rlinf.config import SupportedModel, torch_dtype_from_precision
from rlinf.data.embodied_io_struct import Trajectory, convert_trajectories_to_batch
from rlinf.data.io_struct import BatchResizingIterator, RolloutResult
from rlinf.hybrid_engines.fsdp.fsdp_model_manager import (
    FSDPModelManager,
)
from rlinf.hybrid_engines.fsdp.utils import (
    pack_fsdp_input,
    prepare_pack_fsdp,
    unpack_fsdp_logprobs,
    unpack_sequences,
)
from rlinf.models import get_model
from rlinf.scheduler import Channel, Cluster, CollectiveGroupOptions, Worker
from rlinf.utils.data_iter_utils import (
    get_iterator_k_split,
    get_reverse_idx,
    get_seqlen_balanced_partitions,
    split_dynamic_batch_size,
)
from rlinf.utils.distributed import (
    RolloutDataBalance,
    all_reduce_dict,
    all_reduce_int,
    masked_normalization,
)
from rlinf.utils.distributed import (
    compute_rollout_metrics as compute_math_rollout_metrics,
)
from rlinf.utils.metric_utils import (
    append_to_dict,
    compute_loss_mask,
    compute_rollout_metrics,
    compute_split_num,
)
from rlinf.utils.nested_dict_process import (
    put_tensor_device,
    split_dict_to_chunk,
)
from rlinf.utils.placement import (
    HybridComponentPlacement,
    ModelParallelComponentPlacement,
)
from rlinf.utils.utils import (
    clear_memory,
    compute_entropy_from_logits,
    compute_logprobs_from_logits,
    cpu_weight_swap,
    get_loss_agg_func,
    masked_mean,
    reshape_entropy,
    retrieve_model_state_dict_in_cpu,
)
from rlinf.algorithms.advantages import compute_gae_advantages_and_returns
from rlinf.algorithms.credit_assignment import compute_flow_ipo_weights, compute_flow_sar_weights, compute_flow_fpi_weights, compute_flow_awm_advantages, compute_flow_awm_exp_weights, compute_gfn_log_pf, compute_gfn_log_pb, compute_terminal_binary_advantages, compute_frozen_embedding_advantages
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.workers.rollout.utils import RankMapper


def process_nested_dict_for_adv(nested_dict, rollout_epoch):
    """
    original shape: [rollout_epoch x n_chunk_steps, bsz, num_action_chunks, ...]
    target shape: [n_chunk_steps, rollout_epoch x bsz, num_action_chunks, ...]
    """
    ret_dict = {}
    for key, value in nested_dict.items():
        if isinstance(value, torch.Tensor):
            new_value = value.reshape(
                rollout_epoch, -1, *value.shape[1:]
            )  # [rollout_epoch, n_chunk_step, bsz, ...]
            new_value = new_value.transpose(
                0, 1
            )  # [n_chunk_step, rollout_epoch, bsz, ...]
            new_value = new_value.reshape(new_value.shape[0], -1, *new_value.shape[3:])
            ret_dict[key] = new_value
        elif isinstance(value, dict):
            ret_dict[key] = process_nested_dict_for_adv(value, rollout_epoch)
    return ret_dict


def process_nested_dict_for_train(nested_dict, shuffle_id):
    ret_dict = {}
    for key, value in nested_dict.items():
        if key in ["dones", "terminations", "truncations", "prev_values"]:
            value = value[:-1]
        if "env_info" in key:
            raise NotImplementedError
        if value is None:
            ret_dict[key] = None
        if isinstance(value, torch.Tensor):
            ret_dict[key] = value.reshape(-1, *value.shape[2:])[shuffle_id]
        elif isinstance(value, dict):
            ret_dict[key] = process_nested_dict_for_train(value, shuffle_id)
    return ret_dict


class FSDPActor(FSDPModelManager, Worker):
    def __init__(
        self,
        cfg: DictConfig,
        placement: ModelParallelComponentPlacement,
        cfg_fsdp: Optional[DictConfig] = None,
    ) -> None:
        """
        FSDPActor worker used to train the model with data from rollout workers.

        Args:
            cfg (DictConfig): The global yaml configuration.
            placement (ModelParallelComponentPlacement): The accelerator placement for actor worker.
        """
        if cfg_fsdp is None:
            cfg_fsdp = cfg.actor
        Worker.__init__(self)
        super().__init__(cfg_fsdp, self._world_size, self._rank)

        self.cfg = cfg

        self.response_len = (
            self.cfg.actor.model.encoder_seq_length - self.cfg.data.max_prompt_length
        )
        self.calculate_entropy = self.cfg.algorithm.calculate_entropy
        self.calculate_entropy_loss = (
            self.cfg.algorithm.entropy_bonus > 0 and self.calculate_entropy
        )
        self.kl_beta = self.cfg.algorithm.kl_beta
        self.kl_penalty_type = self.cfg.algorithm.kl_penalty_type
        self.reinpp_kl_beta = cfg.algorithm.get("reinpp_kl_beta", 0.0)
        self.combine_reference_model = cfg.actor.get("combine_reference_model", True)

        self.total_batch_size_per_dp = (
            self.cfg.data.rollout_batch_size
            * self.cfg.algorithm.group_size
            // self._world_size
        )

        self._rollout_group_name = cfg.rollout.group_name
        self._component_placement = placement
        self.is_pipeline = self._component_placement.is_disaggregated
        self.ref_policy_state_dict = None
        if self.is_pipeline:
            self._inference_group_name = cfg.inference.group_name
            self._inference_world_size = self._component_placement.get_world_size(
                "inference"
            )
            self._inference_dst_map: dict[int, list[str]] = {}
        else:
            self._inference_group_name = None
            self._inference_world_size = 0
            self._inference_dst_map = None
        self.loss_agg_func = get_loss_agg_func(self.cfg.algorithm.loss_agg_func)
        self.enable_offload = (
            self.cfg.actor.get("enable_offload", False) and not self.is_pipeline
        )
        self.micro_batch_size = self.cfg.actor.micro_batch_size
        self.n_mini_batches = self.cfg.algorithm.n_minibatches
        self.task_type = self.cfg.runner.task_type
        self.entropy_op_type = self.cfg.algorithm.get("entropy_op_type", "flash_attn")
        self.enable_dp_load_balance = self.cfg.actor.get(
            "enable_dp_load_balance", False
        )
        self.lr_sched_sync_with_optim = self.cfg.actor.get(
            "lr_sched_sync_with_optim", True
        )
        self.enable_dynamic_batch_size = cfg.runner.get(
            "enable_dynamic_batch_size", False
        )
        if self.is_pipeline:
            assert not self.enable_dp_load_balance, (
                "DP load balance is not supported in pipeline mode."
            )
            assert not self.enable_dynamic_batch_size, (
                "Dynamic batch size is not supported in pipeline mode."
            )
        self.max_tokens_per_mbs = cfg.runner.get("max_tokens_per_mbs", 2048)

        self.bucket_capacity = 128 * 1024 * 1024

    def init_worker(self) -> None:
        """
        Initialize the actor worker. build the model and use corresponding training backend
        (FSDP/FSDP2) to wrap it. If needed, offload model parameters and optimizer states to CPU.
        If kl_beta > 0, retrieve the reference policy model state dict to CPU.
        If mode is disaggregated, setup which inference ranks it needs to sync weights to by
        doing a handshake with inference workers.
        """
        self.setup_model_and_optimizer()
        if (
            self.kl_beta > 0 or self.reinpp_kl_beta > 0
        ) and self.combine_reference_model:
            self.ref_policy_state_dict = retrieve_model_state_dict_in_cpu(self.model)
            self.offload_model_buffer = {}

        if self.enable_offload and not self.is_pipeline:
            self.offload_param_and_grad()
            self.offload_optimizer()
        self._setup_rollout_weight_dst_ranks()

    def _setup_rollout_weight_dst_ranks(self) -> None:
        """Setup destination ranks for token and weight communication."""
        rank_map = RankMapper.get_actor_rank_to_rollout_rank_map(
            self._component_placement
        )
        self._weight_dst_rank_in_rollout = rank_map[self._rank]
        self.log_info(
            f"Actor rank {self._rank} will send weights to {self._weight_dst_rank_in_rollout}"
        )

    def del_reshard_state_dict(self) -> None:
        """Just for interface compatibility with MegatronActor."""
        if hasattr(self, "rollout_state_dict"):
            del self.rollout_state_dict
        clear_memory(sync=False)

    def sync_model_to_inference(self) -> None:
        """
        Sync the model's full state dict to the inference worker.
        The model state_dict is the reference of actor's model
        parameters(by setting cpu_offload=False).
        """
        if not self._inference_dst_map:
            self._strategy.setup_actor_sync_inference_ranks(self)

        if self.is_optimizer_offloaded:
            self.offload_optimizer()

        if self.is_weight_offloaded:
            self.load_param_and_grad(self.device, False)

        inference_state_dict = self.get_model_state_dict(
            cpu_offload=False, full_state_dict=False
        )
        # NOTE: we have already know which inference rank needs which params
        # by calling _strategy.setup_actor_sync_inference_ranks() to do handshake
        # with each inference rank. just send them accordingly.
        for rank, needed_params in self._inference_dst_map.items():
            sended_params = {}
            for name in needed_params:
                if name in inference_state_dict:
                    # mentioned again, no ShardedTensor here.
                    sended_params[name] = (
                        inference_state_dict[name].to_local()
                        if isinstance(inference_state_dict[name], DTensor)
                        else inference_state_dict[name]
                    )
            self.send(
                object=sended_params,
                dst_group_name=self._inference_group_name,
                dst_rank=rank,
                async_op=True,
            )

        if self.enable_offload and not self.is_weight_offloaded:
            self.offload_param_and_grad()

        torch.distributed.barrier()

    def divide_model_to_bucket(self, state_dict, has_visual):
        bucket_capacity = self.bucket_capacity
        model_bucket_list = []
        current_capacity = 0
        model_bucket = {}
        for key, val in state_dict.items():
            name = key
            if "_extra_state" in name:
                continue
            if has_visual:
                if name.startswith("model.language_model."):
                    name = "model." + name[21:]
                # NOTE:
                # if transformers version is 4.56.1 or older(not tested),
                # the following line should be uncommented

                # elif name.startswith("model."):
                #     name = name[6:]

            model_bucket[name] = val
            current_capacity += (
                val.numel() * val.element_size() * torch.distributed.get_world_size()
            )

            if current_capacity >= bucket_capacity:
                model_bucket_list.append(model_bucket)
                current_capacity = 0
                model_bucket = {}

        if len(model_bucket) > 0:
            model_bucket_list.append(model_bucket)
        return model_bucket_list

    def sync_model_to_rollout(self) -> None:
        """
        Sync the model's full state dict to the rollout worker.
        """
        if self.enable_offload and not self.is_optimizer_offloaded:
            self.offload_optimizer()

        if self.enable_offload and self.is_weight_offloaded:
            self.load_param_and_grad(self.device, False)

        self.rollout_state_dict = self.get_model_state_dict(
            cpu_offload=False, full_state_dict=False
        )

        has_visual = any("visual." in k for k in self.rollout_state_dict.keys())
        if self._weight_dst_rank_in_rollout is not None:
            rollout_dtype = None
            if self._cfg.get("sync_precision", None) is not None:
                rollout_dtype = torch_dtype_from_precision(self._cfg.sync_precision)
            model_bucket_list = self.divide_model_to_bucket(
                self.rollout_state_dict, has_visual
            )
            self.log_debug(
                f"[sync_model_to_rollout rank-{self._rank}] length of model_bucket_list: {len(model_bucket_list)}"
            )
            for bucket_idx, model_bucket in enumerate(model_bucket_list):
                buffer = {}
                for k, v in model_bucket.items():
                    if isinstance(v, DTensor):
                        v = v.full_tensor()
                    if rollout_dtype is not None:
                        v = v.to(rollout_dtype)
                    if not self.is_pipeline:
                        v = reduce_tensor(v)
                    buffer[k] = v
                if bucket_idx == 0:
                    buffer["bucket_length"] = len(model_bucket_list)
                if not self.is_pipeline:
                    self.send(
                        buffer,
                        self._rollout_group_name,
                        self._weight_dst_rank_in_rollout,
                    )
                else:
                    for weight_dst_rank in self._weight_dst_rank_in_rollout:
                        self.send(
                            buffer,
                            self._rollout_group_name,
                            weight_dst_rank,
                        )
        if self.enable_offload and not self.is_weight_offloaded:
            self.offload_param_and_grad()

    def get_batch(
        self, channel: Channel
    ) -> tuple[dict[str, torch.Tensor], RolloutResult]:
        result: RolloutResult = channel.get()

        batch = result.to_actor_batch(
            self.cfg.data.max_prompt_length,
            self.cfg.actor.model.encoder_seq_length,
            self.tokenizer.eos_token_id,
        )
        return batch, result

    def get_dynamic_batch_as_much(
        self,
        input_channel: Channel,
        min_result_len: int,
        max_result_len: int,
        cliped_results=[],
        unfinished_result=None,
    ):
        assert not input_channel.is_local
        rollout_results = cliped_results
        # get min_result_len
        while len(rollout_results) < min_result_len:
            if unfinished_result is not None:
                rollout_result: RolloutResult = unfinished_result.wait()
                unfinished_result = None
            else:
                rollout_result: RolloutResult = input_channel.get()
            rollout_results.append(rollout_result)

        # try to get result as much
        # get result in every 0.1s and do all reduce to get the min result between dp (result_len)
        # stop at: the min result between dp (result_len) is same as the last min result
        last_result_len = 0
        result_len = len(rollout_results)
        time_until = time.time() + 0.1
        while last_result_len < result_len:
            if len(rollout_results) < max_result_len:
                if unfinished_result is None:
                    unfinished_result = input_channel.get(async_op=True)
                else:
                    time.sleep(0.001)
                if unfinished_result.done():
                    rollout_results.append(unfinished_result.wait())
                    unfinished_result = None
                if time.time() >= time_until:
                    last_result_len = result_len
                    result_len = all_reduce_int(len(rollout_results))
                    if last_result_len < result_len:
                        time_until = time.time() + 0.1
            else:
                last_result_len = result_len
                result_len = all_reduce_int(len(rollout_results))

        batches = []
        for rollout_result in rollout_results:
            batch = rollout_result.to_actor_batch(
                self.cfg.data.max_prompt_length,
                self.cfg.actor.model.encoder_seq_length,
                self.tokenizer.eos_token_id,
            )
            batches.append(batch)

        batch = RolloutResult.merge_batches(batches)
        rollout_result = RolloutResult.merge_result_list(rollout_results)
        return batch, rollout_result, result_len, cliped_results, unfinished_result

    @staticmethod
    def _split_to_micro_batch(
        batch,
        enable_dynamic_batch_size: bool,
        *,
        max_tokens_per_mbs: Optional[int] = None,
        split_num,
    ):
        if enable_dynamic_batch_size:
            (
                micro_batches_iter,
                _,
                micro_batch_cnt,
                dbs_indices,
            ) = split_dynamic_batch_size(
                batch=batch,
                cp_world_size=1,
                vpp_world_size=1,
                max_tokens_per_mbs=max_tokens_per_mbs,
                microbatch_group_size_per_vp_stage=1,
            )
        else:
            micro_batch_cnt = split_num
            micro_batches_iter = get_iterator_k_split(batch, micro_batch_cnt)
            dbs_indices = None
        return micro_batches_iter, micro_batch_cnt, dbs_indices

    def _load_weight_and_optimizer(self) -> None:
        # Acquire the GPUs to ensure that no one is using them before loading models
        # Otherwise, it may lead to OOM
        with self.device_lock:
            if not self.enable_offload:
                return
            if self.is_weight_offloaded:
                self.load_param_and_grad(self.device)
            if self.is_optimizer_offloaded:
                self.load_optimizer(self.device)

    def compute_logprobs(self, logits, target):
        return compute_logprobs_from_logits(
            logits,
            target,
            op_type=self.entropy_op_type,
        )

    def forward_batch(
        self, m_batch: dict[str, torch.Tensor], calculate_entropy: bool = False
    ) -> torch.Tensor:
        input_ids = m_batch["input_ids"]
        attention_mask = m_batch["attention_mask"]
        position_ids = m_batch["position_ids"]

        multi_modal_inputs = {}
        if "multi_modal_inputs" in m_batch.keys():
            for key in m_batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat(
                    [inputs[key] for inputs in m_batch["multi_modal_inputs"]],
                    dim=0,
                ).cuda()

        if self.enable_dynamic_batch_size:
            max_seq_len_pack = self.max_tokens_per_mbs
            max_seq_len_unpack = self.cfg.actor.model.encoder_seq_length
            max_prompt_len = self.cfg.data.max_prompt_length
            max_response_len = max_seq_len_unpack - max_prompt_len
            idx_starts, idx_ends = prepare_pack_fsdp(m_batch, max_prompt_len)

            input_ids, position_ids, attention_mask = pack_fsdp_input(
                input_ids,
                position_ids,
                idx_starts=idx_starts,
                idx_ends=idx_ends,
                max_seq_len_pack=max_seq_len_pack,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        with self.amp_context:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
                **multi_modal_inputs,
            )

        logits: torch.Tensor = outputs.logits
        logits.div_(self.cfg.algorithm.sampling_params.temperature)
        if self.enable_dynamic_batch_size:
            logprobs = unpack_fsdp_logprobs(
                logits,
                input_ids,
                idx_starts=idx_starts,
                idx_ends=idx_ends,
                max_seq_len_unpack=max_seq_len_unpack,
                eos_token_id=self.tokenizer.eos_token_id,
                compute_logprobs_fn=self.compute_logprobs,
            )
            logprobs = logprobs[:, -max_response_len:]
        else:
            # (bsz, response_length, vocab_size)
            logits = logits[:, -self.response_len - 1 : -1, :]
            responses = input_ids[:, -self.response_len :]
            logprobs = self.compute_logprobs(logits, responses)
        if calculate_entropy:
            entropy = compute_entropy_from_logits(logits)
            if self.enable_dynamic_batch_size:
                entropy = unpack_sequences(
                    entropy, idx_starts, idx_ends, max_seq_len_unpack, pad_val=0
                )[:, -self.response_len :]
            return logprobs, entropy
        return logprobs

    def inference_step(
        self,
        batch: dict[str, torch.Tensor],
        rollout_result: RolloutResult,
        compute_ref_logprobs: bool,
    ):
        micro_batches_iter, _, dbs_indices = self._split_to_micro_batch(
            batch,
            self.enable_dynamic_batch_size,
            max_tokens_per_mbs=self.max_tokens_per_mbs,
            split_num=rollout_result.num_sequence
            // self.cfg.algorithm.logprob_forward_micro_batch_size,
        )
        if self.enable_dynamic_batch_size:
            indices = sum(dbs_indices, [])
            revert_indices = torch.tensor(
                get_reverse_idx(indices),
                dtype=torch.long,
            )
        micro_batches = list(micro_batches_iter)

        prev_logprobs, ref_logprobs = None, None

        # Prev logprobs
        prev_logprobs = torch.cat(
            [self.forward_batch(batch) for batch in micro_batches]
        ).cpu()

        if self.enable_dynamic_batch_size:
            assert len(indices) == prev_logprobs.size(0), (
                f"Dynamic batch size indices length {len(indices)} does not equal "
                f"output length {prev_logprobs.size(0)}"
            )
            prev_logprobs = prev_logprobs[revert_indices]

        # Ref logprobs
        if compute_ref_logprobs:
            assert self.ref_policy_state_dict is not None, (
                "Reference policy state dict is None but compute_ref_logprobs is True"
            )
            with cpu_weight_swap(
                self.model,
                self.ref_policy_state_dict,
                self.offload_model_buffer,
            ):
                ref_logprobs = torch.cat(
                    [self.forward_batch(batch) for batch in micro_batches]
                ).cpu()

                if self.enable_dynamic_batch_size:
                    assert len(indices) == ref_logprobs.size(0), (
                        f"Dynamic batch size indices length {len(indices)} does not equal "
                        f"output length {ref_logprobs.size(0)}"
                    )
                    ref_logprobs = ref_logprobs[revert_indices]

        return prev_logprobs, ref_logprobs

    def run_inference(
        self,
        input_channel: Channel,
        output_channel: Channel,
        compute_ref_logprobs: bool,
    ):
        """
        Compute prev/ref logprobs using the actor Model's forward.

        Args:
            input_channel: The input channel to read from.
            output_channel: The output channel to send results to.
            compute_ref_logprobs: Whether to compute reference logprobs.
        """
        inference_split = self.cfg.actor.get("inference_split", None)
        if inference_split is None:
            if not self.is_pipeline:
                inference_split = 1
            else:
                inference_split = self.cfg.algorithm.n_minibatches
        assert self.total_batch_size_per_dp % inference_split == 0, (
            f"FSDPActor: total_batch_size_per_dp[{self.total_batch_size_per_dp}] should be divisible by inference_split[{inference_split}]"
        )

        min_result_len = 1
        max_result_len = (
            self.cfg.data.rollout_batch_size // self._world_size // inference_split
        )
        if not self.is_pipeline:
            min_result_len = max_result_len
            coll_rollout_results = []
        total_result_len = 0
        total_result_len_per_dp = self.cfg.data.rollout_batch_size // self._world_size
        cliped_results, unfinished_result = [], None
        while total_result_len < total_result_len_per_dp:
            batch, rollout_result, result_len, cliped_results, unfinished_result = (
                self.get_dynamic_batch_as_much(
                    input_channel,
                    min(min_result_len, total_result_len_per_dp - total_result_len),
                    min(max_result_len, total_result_len_per_dp - total_result_len),
                    cliped_results,
                    unfinished_result,
                )
            )
            total_result_len += result_len
            self.log_info(
                f"[dynamic inference rank-{self._rank}] inference result_len={result_len}, total_result_len={total_result_len}/{total_result_len_per_dp}"
            )
            self._load_weight_and_optimizer()
            self.model.eval()

            with self.worker_timer():
                with torch.no_grad():
                    prev_logprobs, ref_logprobs = self.inference_step(
                        batch, rollout_result, compute_ref_logprobs
                    )

                if rollout_result.rollout_logprobs is not None:
                    # Rollout has returned logprobs, store the recomputed logprobs in recompute_prev_logprobs
                    rollout_result.recompute_prev_logprobs = prev_logprobs
                else:
                    # Otherwise, directly store the logprobs in prev_logprobs (the final logprobs used for training)
                    rollout_result.prev_logprobs = prev_logprobs

                # Ref logprobs
                if compute_ref_logprobs:
                    rollout_result.ref_logprobs = ref_logprobs

            if self.is_pipeline:
                # for pipeline mode, send after inference to reduce latency.
                # should do split to ensure actor won't get too much batches.
                split_results = RolloutResult.split_results(rollout_result, result_len)
                for split_result in split_results:
                    output_channel.put(split_result, async_op=True)
            else:
                coll_rollout_results.append(rollout_result)

        if not self.is_pipeline:
            # for coll mode, merge results to reduce send time.
            rollout_result = RolloutResult.merge_result_list(coll_rollout_results)
            split_results = RolloutResult.split_results(
                rollout_result,
                min(total_result_len, self.cfg.algorithm.n_minibatches),
            )
            for split_result in split_results:
                output_channel.put(split_result, async_op=True)
        assert total_result_len == total_result_len_per_dp, (
            f"Expected {total_result_len_per_dp} sequences from channel, but got {total_result_len}"
        )

    def training_step(
        self, batch: dict[str, torch.Tensor] | BatchResizingIterator
    ) -> tuple[dict[str, torch.Tensor], float, list[float]]:
        if isinstance(batch, dict):
            global_batch_size = batch["input_ids"].shape[0]
            assert global_batch_size % self.micro_batch_size == 0, (
                f"global batch size {global_batch_size} can not divide micro_batch_size {self.micro_batch_size}"
            )
            micro_batches_iter, micro_batch_cnt, _ = self._split_to_micro_batch(
                batch,
                self.enable_dynamic_batch_size,
                max_tokens_per_mbs=self.max_tokens_per_mbs,
                split_num=global_batch_size // self.micro_batch_size,
            )
            self.gradient_accumulation = micro_batch_cnt
        else:
            global_batch_size = self.total_batch_size_per_dp // self.n_mini_batches
            micro_batch_cnt = global_batch_size // self.micro_batch_size
            self.gradient_accumulation = micro_batch_cnt

            def iterator_wrapper():
                for _ in range(micro_batch_cnt):
                    yield next(batch)

            micro_batches_iter = iterator_wrapper()
        self.optimizer.zero_grad()
        mbs_metrics_list = {}
        for idx, m_batch in enumerate(micro_batches_iter):
            backward_ctx = self.before_micro_batch(
                self.model,
                is_last_micro_batch=(idx + 1) == micro_batch_cnt,
            )
            for k, v in m_batch.items():
                m_batch[k] = v.cuda() if isinstance(v, torch.Tensor) else v

            # batch for forward
            logprobs, entropy = self.forward_batch(m_batch, True)

            # batch for backward
            prev_logprobs = m_batch["prev_logprobs"]
            advantages = m_batch["advantages"]
            ref_logprobs = None
            if "ref_logprobs" in m_batch:
                ref_logprobs = m_batch["ref_logprobs"]

            loss_mask = m_batch["response_mask"][:, -self.response_len :]

            clip_ratio = self.cfg.algorithm.ratio_clip_eps
            clip_ratio_low = self.cfg.algorithm.get("clip_ratio_low", None)
            clip_ratio_high = self.cfg.algorithm.get("clip_ratio_high", None)
            clip_ratio_low = (
                clip_ratio_low if clip_ratio_low is not None else clip_ratio
            )
            clip_ratio_high = (
                clip_ratio_high if clip_ratio_high is not None else clip_ratio
            )
            clip_ratio_c = self.cfg.algorithm.get("clip_ratio_c", 3.0)

            if self.cfg.algorithm.get("importance_sampling_fix", False):
                rollout_prev_logprobs = prev_logprobs
                recompute_prev_logprobs = m_batch["recompute_prev_logprobs"]
                advantages = advantages * torch.clamp(
                    (recompute_prev_logprobs - rollout_prev_logprobs).exp(),
                    min=self.cfg.algorithm.importance_sampling_clip,
                )

            loss, mbs_metrics_data = policy_loss(
                loss_type=self.cfg.algorithm.loss_type,
                loss_agg_func=self.loss_agg_func,
                logprobs=logprobs,
                old_logprobs=prev_logprobs,
                advantages=advantages,
                clip_ratio_low=clip_ratio_low,
                clip_ratio_high=clip_ratio_high,
                clip_ratio_c=clip_ratio_c,
                loss_mask=loss_mask,
                task_type=self.task_type,
            )

            entropy_loss = torch.tensor(0.0, device=torch.cuda.current_device())
            if self.calculate_entropy:
                entropy_loss = self.loss_agg_func(entropy, mask=loss_mask)
                if self.calculate_entropy_loss:
                    loss = loss - self.cfg.algorithm.entropy_bonus * entropy_loss

            kl_loss = torch.tensor(0.0, device=torch.cuda.current_device())
            if self.kl_beta > 0 and ref_logprobs is not None:
                kld = kl_penalty(ref_logprobs, logprobs, self.kl_penalty_type)
                kl_loss = self.loss_agg_func(kld, loss_mask)
                loss = loss + kl_loss * self.kl_beta

            # add to log
            # scale loss for gradient accumulation and backprop
            final_loss_metric = loss.detach()
            loss = loss / self.gradient_accumulation
            with backward_ctx:
                self.grad_scaler.scale(loss).backward()

            mbs_metrics_data.update(
                {
                    "actor/final_loss": final_loss_metric,
                    "actor/entropy_loss": entropy_loss.detach(),
                    "actor/kl_loss": kl_loss.detach(),
                }
            )

            append_to_dict(mbs_metrics_list, mbs_metrics_data)

        grad_norm, lr_list = self.optimizer_step()

        if self.lr_sched_sync_with_optim:
            self.lr_scheduler.step()

        # aggregate metrics across micro-batches
        mean_metric_dict = {
            key: torch.mean(torch.stack(value))
            for key, value in mbs_metrics_list.items()
        }
        mean_metric_dict = all_reduce_dict(
            mean_metric_dict, op=torch.distributed.ReduceOp.AVG
        )

        mean_metric_dict["actor/grad_norm"] = float(grad_norm)
        mean_metric_dict["actor/lr"] = lr_list[0]
        return mean_metric_dict

    def run_training_pipeline(self, input_channel: Channel) -> tuple[dict, list]:
        self.model.train()
        train_batch_iterator = BatchResizingIterator(
            cfg=self.cfg,
            get_batch_fn=partial(self.get_batch, input_channel),
            micro_batch_size=self.micro_batch_size,
            total_batch_size=self.total_batch_size_per_dp,
            num_global_batches=self.n_mini_batches,
            forward_only=False,
        )
        train_batch_iterator.register_get_batch_handler(
            self.compute_advantages_and_returns
        )

        if self.cfg.algorithm.normalize_advantages:

            def normalize_advantages(batch: dict[str, torch.Tensor]):
                mask = batch["response_mask"][:, -self.response_len :]
                batch["advantages"] = masked_normalization(batch["advantages"], mask)
                return batch

            train_batch_iterator.register_global_batch_handler(normalize_advantages)

        self._load_weight_and_optimizer()
        training_metrics_list = []
        with self.worker_timer("run_training"):
            for _ in range(self.n_mini_batches):
                mean_metric_dict = self.training_step(batch=train_batch_iterator)
                training_metrics_list.append(mean_metric_dict)
            if not self.lr_sched_sync_with_optim:
                self.lr_scheduler.step()

        # Rollout metrics
        batch = train_batch_iterator.get_all_batches()
        rollout_metrics, _, _ = compute_math_rollout_metrics(
            batch, self.cfg.data.max_prompt_length, self.response_len
        )

        return rollout_metrics, training_metrics_list

    def _dp_load_balance(self, batch: dict[str, torch.Tensor]):
        batch_size = batch["input_ids"].shape[0]
        assert batch_size == self.total_batch_size_per_dp, (
            f"DP Load balance is only available when a single batch contains all data, e.g., in collocated mode. But got {batch_size=} and {self.total_batch_size_per_dp=}."
        )
        batch = RolloutDataBalance.from_rollout_batches(
            rollout_batches=batch,
            dp_world_size=torch.distributed.get_world_size(),
            dp_rank=torch.distributed.get_rank(),
            dp_group=torch.distributed.group.WORLD,
            partitioning_tool=get_seqlen_balanced_partitions,
        )
        return batch

    def run_training(self, input_channel: Channel) -> tuple[dict, list]:
        # Get all batches for this DP
        if self.is_pipeline:
            return self.run_training_pipeline(input_channel)

        batches = []
        recv_batch_size = 0
        while recv_batch_size < self.total_batch_size_per_dp:
            batch, rollout_result = self.get_batch(input_channel)
            batches.append(batch)
            recv_batch_size += rollout_result.num_sequence
        assert recv_batch_size == self.total_batch_size_per_dp, (
            f"Expected {self.total_batch_size_per_dp} sequences from channel, but got {recv_batch_size}"
        )
        global_batch = RolloutResult.merge_batches(batches)

        # Compute advantages and returns
        global_batch = self.compute_advantages_and_returns(global_batch)

        if self.enable_dp_load_balance:
            global_batch = self._dp_load_balance(global_batch)

        if self.cfg.algorithm.normalize_advantages:
            mask = global_batch["response_mask"][:, -self.response_len :]
            global_batch["advantages"] = masked_normalization(
                global_batch["advantages"], mask
            )

        # Must be called after batch is retrieved, which is when rollout has stopped
        # Otherwise, loading model might cause OOM
        self._load_weight_and_optimizer()

        mini_batches = get_iterator_k_split(
            global_batch,
            num_splits=self.cfg.algorithm.n_minibatches,
            shuffle=self.cfg.algorithm.get("shuffle_rollout", True),
            shuffle_seed=self.cfg.actor.seed,
        )

        self.model.train()
        assert (
            self.cfg.actor.global_batch_size
            % (self.cfg.actor.micro_batch_size * self._world_size)
            == 0
        )

        training_metrics_list = []
        # Global batch iterations
        with self.worker_timer():
            for mini_batch in mini_batches:
                mean_metric_dict = self.training_step(batch=mini_batch)
                training_metrics_list.append(mean_metric_dict)
            if not self.lr_sched_sync_with_optim:
                self.lr_scheduler.step()

        # Rollout metrics
        rollout_metrics, _, _ = compute_math_rollout_metrics(
            global_batch, self.cfg.data.max_prompt_length, self.response_len
        )

        return rollout_metrics, training_metrics_list

    # Advantages and returns
    def compute_advantages_and_returns(self, batch: dict[str, torch.Tensor]):
        """Compute the advantages and returns.

        Args:
            batch (Dict[str, torch.Tensor]): The rollout batch.
        """
        with self.worker_timer():
            if batch.get("advantages", None) is None:
                mask = batch["response_mask"][:, -self.response_len :]
                advantages, _ = calculate_adv_and_returns(
                    task_type=self.task_type,
                    adv_type=self.cfg.algorithm.adv_type,
                    rewards=batch["rewards"].cuda(),
                    loss_mask=mask.cuda(),
                    group_size=self.cfg.algorithm.group_size,
                    kl_beta=self.reinpp_kl_beta,
                    kl_penalty_type=self.kl_penalty_type,
                    logprob=batch["prev_logprobs"].cuda()
                    if "prev_logprobs" in batch
                    else None,
                    ref_logprob=batch["ref_logprobs"].cuda()
                    if "ref_logprobs" in batch
                    else None,
                    use_reinpp_baseline=self.cfg.algorithm.get(
                        "use_reinpp_baseline", False
                    ),
                )
                batch["advantages"] = advantages

        return batch


class EmbodiedFSDPActor(FSDPModelManager, Worker):
    def __init__(self, cfg: DictConfig):
        Worker.__init__(self)
        super().__init__(cfg.actor, self._world_size, self._rank)
        self.cfg = cfg
        self._env_group_name = cfg.env.group_name
        self._rollout_group_name = cfg.rollout.group_name
        self._component_placement = HybridComponentPlacement(cfg, Cluster())

        # stage_num: default to 2, use for pipeline rollout process
        self.stage_num = cfg.rollout.pipeline_stage_num

        self.enable_offload = self.cfg.actor.get("enable_offload", False)
        self.entropy_op_type = self.cfg.algorithm.get("entropy_op_type", "torch")

        # Sync weight comm options
        max_ctas = cfg.rollout.get("sync_weight_nccl_max_ctas", None)
        min_ctas = cfg.rollout.get("sync_weight_nccl_min_ctas", None)
        self._sync_weight_comm_options = CollectiveGroupOptions(
            accel_max_ctas=max_ctas, accel_min_ctas=min_ctas
        )

    def _setup_rollout_weight_dst_ranks(self) -> None:
        """
        Setup destination ranks for weight communication.
        It can support any topology between actor and rollout workers.
        Assuming there are M actor ranks and N rollout ranks, each actor rank
        will send weights to most ceil(N/M) rollout ranks according to the modulo rule.
        """
        rollout_world_size = self._component_placement.get_world_size("rollout")
        actor_world_size = self._world_size
        rank = self._rank
        self._weight_dst_rank_in_rollout = []
        rollout_ranks_per_actor = (
            rollout_world_size + actor_world_size - 1
        ) // actor_world_size
        for i in range(rollout_ranks_per_actor):
            if i * actor_world_size + rank < rollout_world_size:
                self._weight_dst_rank_in_rollout.append(i * actor_world_size + rank)

    def init_worker(self) -> None:
        """
        Initialize the actor worker. build the model and use corresponding training backend,
        if needed, offload model parameters and optimizer states to CPU.
        """
        self.setup_model_and_optimizer()

        # FlowIPO: config only (EMA ref model lives on rollout worker)
        self._is_flow_ipo = self.cfg.algorithm.loss_type == "flow_ipo"
        if self._is_flow_ipo:
            self._flow_ipo_cfg = {
                "alpha": self.cfg.algorithm.get("flow_ipo_alpha", 2.0),
            }

        # FlowSAR: config only (EMA ref model + self-annotation on rollout worker)
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

        # FlowFPI: config only (value head in model, EMA on rollout worker)
        self._is_flow_fpi = self.cfg.algorithm.loss_type == "flow_fpi"
        if self._is_flow_fpi:
            self._flow_fpi_cfg = {
                "lambda": self.cfg.algorithm.get("flow_fpi_lambda", 5.0),
                "gamma": self.cfg.algorithm.get("flow_fpi_gamma", 0.99),
                "w_min": self.cfg.algorithm.get("flow_fpi_w_min", 0.0),
                "w_max": self.cfg.algorithm.get("flow_fpi_w_max", 5.0),
                "value_coeff": self.cfg.algorithm.get("flow_fpi_value_coeff", 0.5),
                "kl_coeff": self.cfg.algorithm.get("flow_fpi_kl_coeff", 1.0),
                "t_min": self.cfg.algorithm.get("flow_fpi_t_min", 0.0),
                "t_max": self.cfg.algorithm.get("flow_fpi_t_max", 1.0),
                "adv_normalize": self.cfg.algorithm.get("flow_fpi_adv_normalize", True),
            }
            # Value warm-up: first N epochs use w=1 (pure BC), only train value head
            self._flow_fpi_warmup_total = self.cfg.algorithm.get("flow_fpi_value_warmup", 10)
            self._flow_fpi_warmup_remaining = self._flow_fpi_warmup_total

        # FlowAWM: Advantage Weighted Matching for VLA
        # Key difference from FPI: advantage weights can be NEGATIVE (bidirectional gradient)
        self._is_flow_awm = self.cfg.algorithm.loss_type == "flow_awm"
        if self._is_flow_awm:
            self._flow_awm_cfg = {
                "gamma": self.cfg.algorithm.get("flow_awm_gamma", 0.99),
                "gae_lambda": self.cfg.algorithm.get("flow_awm_gae_lambda", 0.95),
                "adv_clip": self.cfg.algorithm.get("flow_awm_adv_clip", 2.0),
                "value_coeff": self.cfg.algorithm.get("flow_awm_value_coeff", 0.5),
                "kl_coeff": self.cfg.algorithm.get("flow_awm_kl_coeff", 0.1),
                "t_min": self.cfg.algorithm.get("flow_awm_t_min", 0.0),
                "t_max": self.cfg.algorithm.get("flow_awm_t_max", 1.0),
                # Weight mode: "linear" (original AWM, can be negative) or "exp" (AWR-style, always positive)
                "weight_mode": self.cfg.algorithm.get("flow_awm_weight_mode", "linear"),
                "beta": self.cfg.algorithm.get("flow_awm_beta", 1.0),           # exp mode temperature
                "max_weight": self.cfg.algorithm.get("flow_awm_max_weight", 6.0),  # exp mode upper clamp
            }
            # AWM warm-up: first N epochs use Ã=0 (pure BC), only train value head
            self._flow_awm_warmup_total = self.cfg.algorithm.get("flow_awm_value_warmup", 10)
            self._flow_awm_warmup_remaining = self._flow_awm_warmup_total

        # GFN-Flow: GFlowNet-guided denoising for diversity-preserving RL
        # Key difference from FPI/AWM: SubTB loss on K-step denoising DAG instead of
        # advantage-weighted MSE. No value network, uses State Flow Network F_ψ.
        self._is_flow_gfn = self.cfg.algorithm.loss_type == "flow_gfn"
        if self._is_flow_gfn:
            self._flow_gfn_cfg = {
                "denoise_steps": self.cfg.algorithm.get("flow_gfn_denoise_steps", 4),
                "sigma_f": self.cfg.algorithm.get("flow_gfn_sigma_f", 0.1),
                "sigma_b": self.cfg.algorithm.get("flow_gfn_sigma_b", 0.1),
                "subtb_lambda": self.cfg.algorithm.get("flow_gfn_subtb_lambda", 1.0),
                "boundary_coeff": self.cfg.algorithm.get("flow_gfn_boundary_coeff", 1.0),
                "kl_coeff": self.cfg.algorithm.get("flow_gfn_kl_coeff", 0.5),
                "flow_loss_coeff": self.cfg.algorithm.get("flow_gfn_flow_loss_coeff", 1.0),
                "t_min": self.cfg.algorithm.get("flow_gfn_t_min", 0.0),
                "t_max": self.cfg.algorithm.get("flow_gfn_t_max", 1.0),
                "reward_floor": self.cfg.algorithm.get("flow_gfn_reward_floor", 0.01),
            }
            # GFN warm-up: first N epochs use pure BC (SubTB inactive), only train F_ψ
            self._flow_gfn_warmup_total = self.cfg.algorithm.get("flow_gfn_warmup", 5)
            self._flow_gfn_warmup_remaining = self._flow_gfn_warmup_total

        # FlowNFT: π-StepNFT contrastive mirror loss (critic-free)
        # Key difference: no EMA, no value network, data from rollout SDE chain
        self._is_flow_nft = self.cfg.algorithm.loss_type == "flow_nft"
        if self._is_flow_nft:
            self._flow_nft_cfg = {
                "beta": self.cfg.algorithm.get("flow_nft_beta", 1.0),
                "kl_beta": self.cfg.algorithm.get("flow_nft_kl_beta", 0.0001),
                "adv_clip_max": self.cfg.algorithm.get("flow_nft_adv_clip_max", 1.0),
                "max_drift": self.cfg.algorithm.get("flow_nft_max_drift", 0.5),
                "dpo_beta": self.cfg.algorithm.get("flow_nft_dpo_beta", 1.0),
                "noise_level": self.cfg.algorithm.get("flow_nft_noise_level", 0.2),
                "adv_type": self.cfg.algorithm.get("flow_nft_adv_type", "terminal_binary"),
                "fea_gamma": self.cfg.algorithm.get("flow_nft_fea_gamma", 0.99),
                "fea_ridge_lambda": self.cfg.algorithm.get("flow_nft_fea_ridge_lambda", 1.0),
                # AG-NFT: GAE-based per-step credit assignment
                "value_coeff": self.cfg.algorithm.get("flow_nft_value_coeff", 0.5),
                "gae_gamma": self.cfg.algorithm.get("gamma", 0.99),
                "gae_lambda": self.cfg.algorithm.get("gae_lambda", 0.95),
            }

        # Hinge-NFT: symmetric hinge margin loss variant of NFT
        # Replaces softplus with max(0, margin + logit) for noise robustness.
        # Uses FEA (Frozen Embedding Advantage) for step-level credit via VLM embeddings.
        self._is_flow_hinge_nft = self.cfg.algorithm.loss_type == "flow_hinge_nft"
        if self._is_flow_hinge_nft:
            self._flow_hinge_nft_cfg = {
                "margin": self.cfg.algorithm.get("flow_hinge_nft_margin", 1.0),
                "beta": self.cfg.algorithm.get("flow_hinge_nft_beta", 1.0),
                "kl_beta": self.cfg.algorithm.get("flow_hinge_nft_kl_beta", 0.0001),
                "adv_clip_max": self.cfg.algorithm.get("flow_hinge_nft_adv_clip_max", 1.0),
                "max_drift": self.cfg.algorithm.get("flow_hinge_nft_max_drift", 0.5),
                "noise_level": self.cfg.algorithm.get("flow_hinge_nft_noise_level", 0.2),
                "adv_type": self.cfg.algorithm.get("flow_hinge_nft_adv_type", "frozen_embedding"),
                # FEA settings
                "fea_gamma": self.cfg.algorithm.get("flow_hinge_nft_fea_gamma", 0.99),
                "fea_ridge_lambda": self.cfg.algorithm.get("flow_hinge_nft_fea_ridge_lambda", 1.0),
            }

        # VP-PPO: VLM-Potential Reward Shaping (works WITH standard PPO, not a separate loss_type)
        self._use_vp_ppo = self.cfg.algorithm.get("use_vp_ppo", False)
        if self._use_vp_ppo:
            self._vp_cfg = {
                "alpha": self.cfg.algorithm.get("vp_ppo_alpha", 0.2),
                "alpha_min": self.cfg.algorithm.get("vp_ppo_alpha_min", 0.05),
                "alpha_decay": self.cfg.algorithm.get("vp_ppo_alpha_decay", True),
                "alpha_warmup_iters": self.cfg.algorithm.get("vp_ppo_alpha_warmup_iters", 200),
                "buffer_min_count": self.cfg.algorithm.get("vp_ppo_buffer_min_count", 5),
            }
            self._vp_buffer = SuccessFeatureBuffer(
                ema_rate=self.cfg.algorithm.get("vp_ppo_buffer_ema_rate", 0.99),
                min_count=self._vp_cfg["buffer_min_count"],
            )
            self._vp_metrics: dict[str, float] = {}

        if self.enable_offload:
            self.offload_param_and_grad()
            self.offload_optimizer()
        self._setup_rollout_weight_dst_ranks()

    # ==================== End FlowIPO Methods ====================

    def model_provider_func(self) -> nn.Module:
        model = get_model(self.cfg.actor.model)
        if model is None:
            model = super().model_provider_func()

        if self.cfg.runner.get("ckpt_path", None):
            model_dict = torch.load(self.cfg.runner.ckpt_path)
            model.load_state_dict(model_dict)

        return model

    def sync_model_to_rollout(self) -> None:
        """
        Sync the model's full state dict to the rollout worker.
        """
        if self.enable_offload and not self.is_optimizer_offloaded:
            self.offload_optimizer()

        if self.enable_offload and self.is_weight_offloaded:
            self.load_param_and_grad(self.device)

        state_dict = self.get_model_state_dict(cpu_offload=False, full_state_dict=True)
        for rank in self._weight_dst_rank_in_rollout:
            self.send(
                state_dict,
                self._rollout_group_name,
                rank,
                async_op=True,
                options=self._sync_weight_comm_options,
            )
        if self.enable_offload and not self.is_weight_offloaded:
            self.offload_param_and_grad()

    async def recv_rollout_trajectories(self, input_channel: Channel) -> None:
        """
        Receive rollout trajectories from rollout workers.

        Args:
            input_channel: The input channel to read from.
        """
        send_num = self._component_placement.get_world_size("rollout") * self.stage_num
        recv_num = self._component_placement.get_world_size("actor")
        split_num = compute_split_num(send_num, recv_num)

        recv_list = []
        for _ in range(split_num):
            trajectory: Trajectory = await input_channel.get(async_op=True).async_wait()
            recv_list.append(trajectory)

        self.rollout_batch = convert_trajectories_to_batch(recv_list)

        self.rollout_batch = self._process_received_rollout_batch(self.rollout_batch)

    def _process_received_rollout_batch(
        self, rollout_batch: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """
        original shape: [rollout_epoch x n_chunk_steps, bsz, num_action_chunks, ...]
        target shape: [n_chunk_steps, rollout_epoch x bsz, num_action_chunks, ...]
        """
        rollout_epoch = self.cfg.algorithm.rollout_epoch
        rollout_batch = process_nested_dict_for_adv(rollout_batch, rollout_epoch)

        if (
            not self.cfg.env.train.auto_reset
            and not self.cfg.env.train.ignore_terminations
        ):
            dones = rollout_batch[
                "dones"
            ]  # [n_chunk_step, rollout_epoch x bsz, num_action_chunks]
            loss_mask, loss_mask_sum = compute_loss_mask(dones)

            if self.cfg.algorithm.reward_type == "chunk_level":
                loss_mask = loss_mask.any(dim=-1, keepdim=True)
                loss_mask_sum = loss_mask_sum[..., -1:]

            rollout_batch["loss_mask"] = loss_mask
            rollout_batch["loss_mask_sum"] = loss_mask_sum

        # filter data by rewards
        if self.cfg.algorithm.get("filter_rewards", False):
            rewards = rollout_batch[
                "rewards"
            ]  # [n_chunk_step, batch, num_action_chunks]
            if rollout_batch.get("loss_mask", None) is not None:
                rewards = rewards * rollout_batch["loss_mask"]
            n_chunk_step, batch_size, num_action_chunks = rewards.shape

            group_size = self.cfg.algorithm.group_size
            assert batch_size % group_size == 0, (
                f"batch {batch_size} not divisible by group_size {group_size}"
            )
            n_prompts = batch_size // group_size

            # calculate rewards by prompt
            rewards = rewards.transpose(
                0, 1
            )  # [batch, n_chunk_step, num_action_chunks]
            rewards = rewards.reshape(rewards.shape[0], -1)  # [batch, n_step]
            reward_matrix = rewards.reshape(
                n_prompts, group_size, rewards.shape[-1]
            )  # [n_prompts, group_size, n_step]
            reward_matrix = reward_matrix.sum(dim=-1)  # [n_prompts, group_size]
            mean_reward_in_group = reward_matrix.mean(dim=1)  # [n_prompts]

            # mask
            reward_filter_mask = (
                mean_reward_in_group >= self.cfg.algorithm.rewards_lower_bound
            ) & (
                mean_reward_in_group <= self.cfg.algorithm.rewards_upper_bound
            )  # [n_prompts]

            # extend mask dimension
            reward_filter_mask = reward_filter_mask.repeat_interleave(
                group_size
            )  # [batch]
            reward_filter_mask = (
                reward_filter_mask.unsqueeze(0).expand(n_chunk_step, -1).unsqueeze(-1)
            )  # [n_chunk_step, batch, 1]

            # update loss_mask
            if rollout_batch.get("loss_mask", None) is not None:
                rollout_batch["loss_mask"] = (
                    reward_filter_mask & rollout_batch["loss_mask"]
                )
            else:
                rollout_batch["loss_mask"] = reward_filter_mask

        return rollout_batch

    def compute_advantages_and_returns(self) -> dict[str, torch.Tensor]:
        """
        Compute the advantages and returns.
        For FlowIPO: compute credit assignment weights instead of GAE/GRPO.
        """
        if self._is_flow_ipo:
            return self._compute_flow_ipo_credit_assignment()
        if self._is_flow_sar:
            return self._compute_flow_sar_credit_assignment()
        if self._is_flow_fpi:
            return self._compute_flow_fpi_credit_assignment()
        if self._is_flow_awm:
            return self._compute_flow_awm_credit_assignment()
        if self._is_flow_gfn:
            return self._compute_flow_gfn_credit_assignment()
        if self._is_flow_nft:
            return self._compute_flow_nft_credit_assignment()
        if self._is_flow_hinge_nft:
            return self._compute_flow_hinge_nft_credit_assignment()

        # VP-PPO: apply PBRS reward shaping before GAE (modifies rewards in-place)
        if self._use_vp_ppo:
            self._apply_vlm_potential_shaping()

        kwargs = {
            "task_type": self.cfg.runner.task_type,
            "adv_type": self.cfg.algorithm.adv_type,
            "rewards": self.rollout_batch["rewards"],
            "dones": self.rollout_batch["dones"],
            "values": self.rollout_batch.get("prev_values", None),
            "gamma": self.cfg.algorithm.get("gamma", 1),
            "gae_lambda": self.cfg.algorithm.get("gae_lambda", 1),
            "group_size": self.cfg.algorithm.get("group_size", 8),
            "reward_type": self.cfg.algorithm.reward_type,
            "loss_mask": self.rollout_batch.get("loss_mask", None),
            "loss_mask_sum": self.rollout_batch.get("loss_mask_sum", None),
        }

        advantages_and_returns = calculate_adv_and_returns(**kwargs)

        self.rollout_batch.update(advantages_and_returns)
        if kwargs["loss_mask"] is not None:
            self.rollout_batch.update({"loss_mask": kwargs["loss_mask"]})
        if kwargs["loss_mask_sum"] is not None:
            self.rollout_batch.update({"loss_mask_sum": kwargs["loss_mask_sum"]})

        rollout_metrics = compute_rollout_metrics(self.rollout_batch)
        # VP-PPO: append shaping metrics to rollout_metrics
        if self._use_vp_ppo and self._vp_metrics:
            rollout_metrics.update(self._vp_metrics)
        return rollout_metrics

    def _apply_vlm_potential_shaping(self):
        """VP-PPO: PBRS reward shaping using frozen VLM embedding as potential function.

        Modifies self.rollout_batch["rewards"] in-place before GAE computation.
        r'_t = r_t + alpha * (gamma * Phi(s_{t+1}) - Phi(s_t))
        where Phi(s) = cos_sim(vlm_embedding(s), EMA_success_target).
        """
        rewards = self.rollout_batch["rewards"]  # [n_steps, batch, chunk]
        forward_inputs = self.rollout_batch.get("forward_inputs", {})
        vlm_embs = forward_inputs.get("vlm_embedding", None)

        if vlm_embs is None:
            self._vp_metrics = {"vp_ppo/status": 0.0}  # no embedding available
            return

        n_steps = rewards.shape[0]
        batch_size = rewards.shape[1]

        # Reshape vlm_embs to [n_steps, batch, hidden_dim]
        if vlm_embs.dim() == 2:
            hidden_dim = vlm_embs.shape[-1]
            vlm_embs = vlm_embs.reshape(n_steps, batch_size, hidden_dim)

        # Episode-level reward for success/failure detection
        loss_mask = self.rollout_batch.get("loss_mask", None)
        if loss_mask is not None:
            episode_rewards = (rewards * loss_mask).sum(dim=(0, 2))
        else:
            episode_rewards = rewards.sum(dim=(0, 2))
        episode_rewards = episode_rewards.clamp(0, 1)  # [batch]
        success_mask = episode_rewards > 0.5  # [batch]

        # --- Update SuccessFeatureBuffer ---
        task_hash = 0  # single-task default; multi-task would hash language tokens
        if success_mask.any():
            success_final_embs = vlm_embs[-1, success_mask, :]  # [n_success, hidden]
            self._vp_buffer.update(task_hash, success_final_embs)

        # --- Get target; skip shaping if cold-starting ---
        target = self._vp_buffer.get_target(task_hash)
        if target is None or not self._vp_buffer.has_enough(task_hash):
            self._vp_metrics = {
                "vp_ppo/status": 1.0,  # buffer collecting
                "vp_ppo/success_in_buffer": self._vp_buffer.count.get(task_hash, 0),
            }
            return

        target = target.to(vlm_embs.device)

        # --- Compute potential Phi ---
        phi = compute_vlm_potential(vlm_embs, target)  # [n_steps, batch]

        # --- Compute alpha (with optional linear decay) ---
        alpha = self._vp_cfg["alpha"]
        if self._vp_cfg["alpha_decay"]:
            warmup_iters = self._vp_cfg["alpha_warmup_iters"]
            alpha_min = self._vp_cfg["alpha_min"]
            progress = min(self.optimizer_steps / max(warmup_iters, 1), 1.0)
            alpha = alpha * (1.0 - progress) + alpha_min * progress

        # --- Apply PBRS to step-level rewards ---
        gamma = self.cfg.algorithm.get("gamma", 0.99)
        step_rewards = rewards.sum(dim=-1)  # [n_steps, batch] (sum over chunk dim)
        shaped_step_rewards = compute_pbrs_reward(step_rewards, phi, gamma=gamma, alpha=alpha)

        # Write shaped rewards back (distribute PBRS into first chunk slot)
        shaped_rewards = rewards.clone()
        pbrs_delta = shaped_step_rewards - step_rewards  # [n_steps, batch]
        shaped_rewards[:, :, 0] = shaped_rewards[:, :, 0] + pbrs_delta
        self.rollout_batch["rewards"] = shaped_rewards

        # --- Metrics ---
        self._vp_metrics = {
            "vp_ppo/status": 2.0,  # active shaping
            "vp_ppo/phi_mean": phi.mean().item(),
            "vp_ppo/phi_std": phi.std().item(),
            "vp_ppo/pbrs_mean": pbrs_delta.mean().item(),
            "vp_ppo/pbrs_abs_mean": pbrs_delta.abs().mean().item(),
            "vp_ppo/alpha": alpha,
            "vp_ppo/success_in_buffer": self._vp_buffer.count.get(task_hash, 0),
            "vp_ppo/phi_success_final": phi[-1, success_mask].mean().item() if success_mask.any() else 0.0,
            "vp_ppo/phi_fail_final": phi[-1, ~success_mask].mean().item() if (~success_mask).any() else 0.0,
        }

    def _compute_flow_ipo_credit_assignment(self) -> dict:
        """
        FlowIPO credit assignment (ref_actions pre-computed by rollout worker):
        1. Read pre-computed reference actions from forward_inputs
        2. Compute per-step policy divergence δ_i
        3. Compute interpolation weights w_i = sigmoid(α * (2R-1) * normalize(δ_i))
        """
        actions = self.rollout_batch["actions"]  # [n_steps, batch, chunk, action_dim]
        rewards = self.rollout_batch["rewards"]  # [n_steps, batch, chunk] or similar
        forward_inputs = self.rollout_batch.get("forward_inputs", {})

        # Compute episode-level reward: sum across steps and chunks
        loss_mask = self.rollout_batch.get("loss_mask", None)
        if loss_mask is not None:
            episode_rewards = (rewards * loss_mask).sum(dim=(0, 2))  # [batch]
        else:
            episode_rewards = rewards.sum(dim=(0, 2))  # [batch]
        episode_rewards = episode_rewards.clamp(0, 1)

        # Reference actions pre-computed by rollout worker (distributed, no ODE here)
        ref_actions = forward_inputs.get("ref_action", None)
        if ref_actions is None:
            ref_actions = actions.clone()

        # Compute FlowIPO weights
        flow_ipo_weights = compute_flow_ipo_weights(
            actions=actions,
            ref_actions=ref_actions,
            rewards=episode_rewards,
            alpha=self._flow_ipo_cfg["alpha"],
        )  # [n_steps, batch]

        # Store in rollout_batch for training loop
        self.rollout_batch["flow_ipo_weights"] = flow_ipo_weights
        self.rollout_batch["advantages"] = torch.zeros_like(
            self.rollout_batch["prev_logprobs"]
        )

        rollout_metrics = compute_rollout_metrics(self.rollout_batch)
        rollout_metrics["flow_ipo/mean_weight"] = flow_ipo_weights.mean().item()
        rollout_metrics["flow_ipo/episode_reward"] = episode_rewards.mean().item()
        return rollout_metrics

    def _compute_flow_sar_credit_assignment(self) -> dict:
        """
        FlowSAR Phase 3: Credit assignment based on self-annotation reconstruction errors.

        Success trajectories: uncertain-but-correct steps get high weight (softmax(e/T))
        Failure trajectories: confident-but-wrong steps get high weight (softmax(-e/T))
        """
        rewards = self.rollout_batch["rewards"]  # [n_steps, batch, chunk] or similar
        forward_inputs = self.rollout_batch.get("forward_inputs", {})

        # Compute episode-level reward: sum across steps and chunks
        loss_mask = self.rollout_batch.get("loss_mask", None)
        if loss_mask is not None:
            episode_rewards = (rewards * loss_mask).sum(dim=(0, 2))  # [batch]
        else:
            episode_rewards = rewards.sum(dim=(0, 2))  # [batch]
        episode_rewards = episode_rewards.clamp(0, 1)

        # Extract pre-computed reconstruction errors from forward_inputs
        recon_errors = forward_inputs.get("recon_error", None)
        if recon_errors is None:
            # Fallback: uniform weights if self-annotation was not computed
            n_steps = rewards.shape[0]
            batch_size = rewards.shape[1]
            recon_errors = torch.ones(n_steps, batch_size)

        # Compute FlowSAR weights and contrastive labels
        flow_sar_weights, flow_sar_labels = compute_flow_sar_weights(
            recon_errors=recon_errors,
            rewards=episode_rewards,
            temperature=self._flow_sar_cfg["temperature"],
            w_min=self._flow_sar_cfg["w_min"],
            w_max=self._flow_sar_cfg["w_max"],
        )  # weights: [n_steps, batch], labels: [batch]

        # Store in rollout_batch for training loop
        # flow_sar_weights: already [n_steps, batch]
        self.rollout_batch["flow_sar_weights"] = flow_sar_weights
        # flow_sar_labels: [batch] -> expand to [n_steps, batch] for data pipeline compatibility
        # (each step in same episode gets the same label y_i = 2R - 1)
        n_steps = rewards.shape[0]
        self.rollout_batch["flow_sar_labels"] = flow_sar_labels.unsqueeze(0).expand(n_steps, -1)
        self.rollout_batch["advantages"] = torch.zeros_like(
            self.rollout_batch["prev_logprobs"]
        )

        rollout_metrics = compute_rollout_metrics(self.rollout_batch)
        rollout_metrics["flow_sar/mean_weight"] = flow_sar_weights.mean().item()
        rollout_metrics["flow_sar/weight_std"] = flow_sar_weights.std().item()
        rollout_metrics["flow_sar/episode_reward"] = episode_rewards.mean().item()
        rollout_metrics["flow_sar/success_rate"] = (episode_rewards > 0.5).float().mean().item()
        rollout_metrics["flow_sar/mean_recon_error"] = recon_errors.mean().item()
        return rollout_metrics

    def _compute_flow_fpi_credit_assignment(self) -> dict:
        """
        FPI Phase: Compute per-step advantages and importance weights.

        1. Compute advantages A_t = V(o_{t+1}) - V(o_t) from rollout-time values
        2. Compute self-normalized exponential weights w_t = exp(A_t / λ) / Z
        3. Compute TD targets for value training: y_t = r_t + γ * V_target(o_{t+1})
        4. Compute progress targets: t/T for success episodes (progress warm-start)
        """
        prev_values = self.rollout_batch.get("prev_values", None)
        rewards = self.rollout_batch["rewards"]  # [n_steps, batch, chunk]
        forward_inputs = self.rollout_batch.get("forward_inputs", {})
        v_target_all = forward_inputs.get("fpi_v_target", None)

        n_steps = rewards.shape[0]
        batch_size = rewards.shape[1]
        gamma = self._flow_fpi_cfg["gamma"]
        lambda_ = self._flow_fpi_cfg["lambda"]

        # Episode-level reward (binary success/failure)
        loss_mask = self.rollout_batch.get("loss_mask", None)
        if loss_mask is not None:
            episode_rewards = (rewards * loss_mask).sum(dim=(0, 2))
        else:
            episode_rewards = rewards.sum(dim=(0, 2))
        episode_rewards = episode_rewards.clamp(0, 1)  # [batch]

        # Per-step reward (sum over action chunk dim)
        step_rewards = rewards.sum(dim=-1)  # [n_steps, batch]

        # Value predictions from rollout
        if prev_values is not None and prev_values.numel() > 0:
            V = prev_values[:n_steps].squeeze(-1).float()  # [n_steps, batch]
        else:
            # No value predictions available: uniform weights (equivalent to BC)
            V = torch.zeros(n_steps, batch_size)

        # Advantages: A_t = V(o_{t+1}) - V(o_t)
        advantages = torch.zeros(n_steps, batch_size, device=V.device)
        if prev_values is not None and prev_values.shape[0] > n_steps:
            # Extra value for V(o_{T+1}) available
            V_next = prev_values[1:n_steps + 1].squeeze(-1).float()
            advantages = V_next - V
        else:
            # Use consecutive values within trajectory
            advantages[:-1] = V[1:] - V[:-1]
            # Terminal step: A_T = r_T - V(o_T)
            advantages[-1] = step_rewards[-1] - V[-1]

        # Value warm-up: first N epochs use uniform weights (pure BC for policy)
        in_warmup = self._flow_fpi_warmup_remaining > 0
        if in_warmup:
            fpi_weights = torch.ones(n_steps, batch_size, device=V.device)
            self._flow_fpi_warmup_remaining -= 1
        else:
            # FPI weights: w = exp(A/λ), self-normalized + advantage normalization
            fpi_weights = compute_flow_fpi_weights(
                advantages, lambda_,
                self._flow_fpi_cfg["w_min"], self._flow_fpi_cfg["w_max"],
                normalize_adv=self._flow_fpi_cfg["adv_normalize"],
            )  # [n_steps, batch]

        # TD targets for value training: y_t = r_t + γ * V_target(o_{t+1})
        td_targets = torch.zeros_like(V)
        if v_target_all is not None:
            V_tgt = v_target_all.float()  # [n_steps, batch]
            td_targets[:-1] = step_rewards[:-1] + gamma * V_tgt[1:]
            td_targets[-1] = step_rewards[-1]  # terminal: no bootstrap
        else:
            # Fallback: use current model values as target
            td_targets[:-1] = step_rewards[:-1] + gamma * V[1:]
            td_targets[-1] = step_rewards[-1]

        # Progress targets (π-RL style): reward-scaled progress prior
        #   Success (R=1): target = t/T  → V learns temporal progress structure
        #   Failure (R=0): target = 0    → V learns failure states have low value
        # This gives dense supervision for ALL trajectories (not just ~15% success),
        # approximating π-RL's demonstration pre-training effect.
        raw_progress = torch.zeros_like(V)
        for t in range(n_steps):
            raw_progress[t] = (t + 1.0) / n_steps
        progress_targets = raw_progress * episode_rewards.unsqueeze(0)  # [n_steps, batch]

        success_mask = (episode_rewards > 0.5)  # [batch]

        # Store in rollout_batch for training loop
        self.rollout_batch["fpi_weights"] = fpi_weights
        self.rollout_batch["fpi_td_targets"] = td_targets
        self.rollout_batch["fpi_progress_targets"] = progress_targets
        self.rollout_batch["fpi_success_mask"] = success_mask.unsqueeze(0).expand(n_steps, -1)
        self.rollout_batch["advantages"] = torch.zeros_like(
            self.rollout_batch["prev_logprobs"]
        )

        rollout_metrics = compute_rollout_metrics(self.rollout_batch)
        rollout_metrics["fpi/mean_weight"] = fpi_weights.mean().item()
        rollout_metrics["fpi/weight_std"] = fpi_weights.std().item()
        rollout_metrics["fpi/weight_max"] = fpi_weights.max().item()
        rollout_metrics["fpi/advantage_mean"] = advantages.mean().item()
        rollout_metrics["fpi/advantage_std"] = advantages.std().item()
        rollout_metrics["fpi/episode_reward"] = episode_rewards.mean().item()
        rollout_metrics["fpi/success_rate"] = success_mask.float().mean().item()
        rollout_metrics["fpi/v_mean"] = V.mean().item()
        rollout_metrics["fpi/warmup_remaining"] = self._flow_fpi_warmup_remaining
        rollout_metrics["fpi/in_warmup"] = float(in_warmup)
        return rollout_metrics

    def _compute_flow_awm_credit_assignment(self) -> dict:
        """
        AWM-VLA Phase: Compute per-step advantages (can be negative).

        1. Compute GAE advantages from rewards, values, and dones
        2. Normalize and clip: Ã = clip((A - μ) / (σ + ε), -c, c)
        3. Compute TD targets for value training: y_t = r_t + γ * V_target(o_{t+1})
        4. Compute progress targets: (t/T) * R for all episodes

        Key difference from FPI: advantages can be NEGATIVE, enabling
        bidirectional gradient signal (push away from bad actions).
        """
        prev_values = self.rollout_batch.get("prev_values", None)
        rewards = self.rollout_batch["rewards"]  # [n_steps, batch, chunk]
        dones = self.rollout_batch.get("dones", None)  # [n_steps+1, batch, chunk]
        forward_inputs = self.rollout_batch.get("forward_inputs", {})
        v_target_all = forward_inputs.get("awm_v_target", None)

        n_steps = rewards.shape[0]
        batch_size = rewards.shape[1]
        gamma = self._flow_awm_cfg["gamma"]
        gae_lambda = self._flow_awm_cfg["gae_lambda"]
        adv_clip = self._flow_awm_cfg["adv_clip"]

        # Episode-level reward (binary success/failure)
        loss_mask = self.rollout_batch.get("loss_mask", None)
        if loss_mask is not None:
            episode_rewards = (rewards * loss_mask).sum(dim=(0, 2))
        else:
            episode_rewards = rewards.sum(dim=(0, 2))
        episode_rewards = episode_rewards.clamp(0, 1)  # [batch]

        # Per-step reward (sum over action chunk dim)
        step_rewards = rewards.sum(dim=-1)  # [n_steps, batch]

        # Prepare values for GAE: need [n_steps+1, batch] (extra step for bootstrap)
        if prev_values is not None and prev_values.numel() > 0:
            # prev_values: [n_steps+1, batch, chunk] or [n_steps+1, batch, 1] or [n_steps, ...]
            if prev_values.dim() == 3:
                V_squeezed = prev_values.squeeze(-1).float()  # [T, batch]
            else:
                V_squeezed = prev_values.float()
            if V_squeezed.shape[0] > n_steps:
                # Has bootstrap value V(o_{T+1})
                V_full = V_squeezed[:n_steps + 1]  # [n_steps+1, batch]
            else:
                # No bootstrap value: pad with zeros for V(o_{T+1})
                V_full = torch.cat([
                    V_squeezed[:n_steps],
                    torch.zeros(1, batch_size, device=V_squeezed.device),
                ], dim=0)  # [n_steps+1, batch]
            V = V_full[:n_steps]  # [n_steps, batch] for metrics/TD targets
        else:
            V_full = torch.zeros(n_steps + 1, batch_size, device=step_rewards.device)
            V = V_full[:n_steps]

        # Prepare dones for GAE: [n_steps+1, batch], boolean
        if dones is not None:
            if dones.dim() == 3:
                # [n_steps+1, batch, chunk] -> [n_steps+1, batch]
                dones_flat = dones.max(dim=-1)[0].bool()
            else:
                dones_flat = dones.bool()
        else:
            dones_flat = torch.zeros(n_steps + 1, batch_size, dtype=torch.bool, device=step_rewards.device)

        # Compute GAE advantages (reusing RLinf's implementation)
        # normalize_advantages=False because AWM does its own normalization + clipping
        gae_advantages, gae_returns = compute_gae_advantages_and_returns(
            rewards=step_rewards,        # [n_steps, batch]
            values=V_full,               # [n_steps+1, batch]
            dones=dones_flat,            # [n_steps+1, batch]
            gamma=gamma,
            gae_lambda=gae_lambda,
            normalize_advantages=False,  # AWM normalizes separately
            normalize_returns=False,
        )  # gae_advantages: [n_steps, batch]

        # AWM warm-up: first N epochs use Ã=0 or w=1 (pure BC for policy)
        weight_mode = self._flow_awm_cfg["weight_mode"]
        in_warmup = self._flow_awm_warmup_remaining > 0
        if in_warmup:
            if weight_mode == "exp":
                awm_advantages = torch.ones(n_steps, batch_size, device=V.device)
            else:
                awm_advantages = torch.zeros(n_steps, batch_size, device=V.device)
            self._flow_awm_warmup_remaining -= 1
        else:
            if weight_mode == "exp":
                # AWR-style: w = clamp(exp(A/β), max=w_max), always positive, no gradient cancellation
                awm_advantages = compute_flow_awm_exp_weights(
                    gae_advantages,
                    beta=self._flow_awm_cfg["beta"],
                    max_weight=self._flow_awm_cfg["max_weight"],
                )  # [n_steps, batch], all > 0
            else:
                # Original AWM: Ã = clip(normalize(A), -c, c), can be negative
                awm_advantages = compute_flow_awm_advantages(
                    gae_advantages, clip_range=adv_clip,
                )  # [n_steps, batch]

        # TD targets for value training: y_t = r_t + γ * V_target(o_{t+1})
        td_targets = torch.zeros_like(V)
        if v_target_all is not None:
            V_tgt = v_target_all.float()  # [n_steps, batch]
            td_targets[:-1] = step_rewards[:-1] + gamma * V_tgt[1:]
            td_targets[-1] = step_rewards[-1]
        else:
            td_targets[:-1] = step_rewards[:-1] + gamma * V[1:]
            td_targets[-1] = step_rewards[-1]

        # Progress targets (π-RL style): reward-scaled progress prior
        raw_progress = torch.zeros_like(V)
        for t in range(n_steps):
            raw_progress[t] = (t + 1.0) / n_steps
        progress_targets = raw_progress * episode_rewards.unsqueeze(0)  # [n_steps, batch]

        success_mask = (episode_rewards > 0.5)  # [batch]

        # Store in rollout_batch for training loop
        self.rollout_batch["awm_advantages"] = awm_advantages
        self.rollout_batch["awm_td_targets"] = td_targets
        self.rollout_batch["awm_progress_targets"] = progress_targets
        self.rollout_batch["awm_success_mask"] = success_mask.unsqueeze(0).expand(n_steps, -1)
        self.rollout_batch["advantages"] = torch.zeros_like(
            self.rollout_batch["prev_logprobs"]
        )

        rollout_metrics = compute_rollout_metrics(self.rollout_batch)
        rollout_metrics["awm/weight_mean"] = awm_advantages.mean().item()
        rollout_metrics["awm/weight_std"] = awm_advantages.std().item()
        rollout_metrics["awm/weight_min"] = awm_advantages.min().item()
        rollout_metrics["awm/weight_max"] = awm_advantages.max().item()
        rollout_metrics["awm/raw_gae_adv_mean"] = gae_advantages.mean().item()
        rollout_metrics["awm/raw_gae_adv_std"] = gae_advantages.std().item()
        rollout_metrics["awm/pos_frac"] = (awm_advantages > 0).float().mean().item()
        rollout_metrics["awm/neg_frac"] = (awm_advantages < 0).float().mean().item()
        rollout_metrics["awm/episode_reward"] = episode_rewards.mean().item()
        rollout_metrics["awm/success_rate"] = success_mask.float().mean().item()
        rollout_metrics["awm/v_mean"] = V.mean().item()
        rollout_metrics["awm/warmup_remaining"] = self._flow_awm_warmup_remaining
        rollout_metrics["awm/in_warmup"] = float(in_warmup)
        rollout_metrics["awm/weight_mode"] = 1.0 if weight_mode == "exp" else 0.0
        return rollout_metrics

    def _compute_flow_gfn_credit_assignment(self) -> dict:
        """
        GFN-Flow credit assignment: compute log rewards for boundary condition.

        GFN requires R > 0 for all trajectories. Failed episodes get a floor
        reward ε (default 0.01) so that log R is well-defined.

        No per-step advantage computation needed (SubTB handles credit assignment
        implicitly through the flow balance constraint).
        """
        rewards = self.rollout_batch["rewards"]  # [n_steps, batch, chunk]
        n_steps = rewards.shape[0]

        # Episode-level reward
        loss_mask = self.rollout_batch.get("loss_mask", None)
        if loss_mask is not None:
            episode_rewards = (rewards * loss_mask).sum(dim=(0, 2))
        else:
            episode_rewards = rewards.sum(dim=(0, 2))
        episode_rewards = episode_rewards.clamp(0, 1)  # [batch]

        # GFN requires R > 0: floor reward for failed episodes
        reward_floor = self._flow_gfn_cfg["reward_floor"]
        floored_rewards = episode_rewards.clamp(min=reward_floor)
        log_rewards = torch.log(floored_rewards)  # [batch]

        # Store in rollout_batch
        self.rollout_batch["gfn_log_rewards"] = log_rewards.unsqueeze(0).expand(n_steps, -1)
        self.rollout_batch["advantages"] = torch.zeros_like(
            self.rollout_batch["prev_logprobs"]
        )

        success_mask = (episode_rewards > 0.5)

        rollout_metrics = compute_rollout_metrics(self.rollout_batch)
        rollout_metrics["gfn/episode_reward"] = episode_rewards.mean().item()
        rollout_metrics["gfn/success_rate"] = success_mask.float().mean().item()
        rollout_metrics["gfn/log_reward_mean"] = log_rewards.mean().item()
        rollout_metrics["gfn/warmup_remaining"] = self._flow_gfn_warmup_remaining
        return rollout_metrics

    def _compute_flow_nft_credit_assignment(self) -> dict:
        """
        NFT credit assignment: terminal binary or frozen embedding advantage.

        Terminal binary: +adv_clip_max for success, -adv_clip_max for failure,
        broadcast to all env steps. Simple but effective.

        Frozen embedding (FEA): Ridge regression on frozen VLM embeddings for
        per-step temporal credit assignment without a value network.
        """
        rewards = self.rollout_batch["rewards"]  # [n_steps, batch, chunk]
        forward_inputs = self.rollout_batch.get("forward_inputs", {})
        n_steps = rewards.shape[0]
        batch_size = rewards.shape[1]

        # Episode-level reward
        loss_mask = self.rollout_batch.get("loss_mask", None)
        if loss_mask is not None:
            episode_rewards = (rewards * loss_mask).sum(dim=(0, 2))
        else:
            episode_rewards = rewards.sum(dim=(0, 2))
        episode_rewards = episode_rewards.clamp(0, 1)  # [batch]

        adv_clip = self._flow_nft_cfg["adv_clip_max"]
        adv_type = self._flow_nft_cfg["adv_type"]

        if adv_type == "frozen_embedding":
            # FEA: ridge regression on frozen VLM embeddings
            vlm_embs = forward_inputs.get("vlm_embedding", None)
            if vlm_embs is not None and vlm_embs.dim() >= 2:
                # vlm_embs shape after trajectory processing:
                #   3D: [n_steps, batch, hidden_dim] (before process_nested_dict_for_train)
                #   2D: [n_steps*batch, hidden_dim] (after flatten)
                if vlm_embs.dim() == 3:
                    vlm_embs_3d = vlm_embs  # Already [n_steps, batch, hidden_dim]
                else:
                    hidden_dim = vlm_embs.shape[-1]
                    vlm_embs_3d = vlm_embs.reshape(n_steps, batch_size, hidden_dim)
                advantages = compute_frozen_embedding_advantages(
                    embeddings=vlm_embs_3d,
                    episode_rewards=episode_rewards,
                    gamma=self._flow_nft_cfg["fea_gamma"],
                    ridge_lambda=self._flow_nft_cfg["fea_ridge_lambda"],
                    adv_clip_max=adv_clip,
                )  # [n_steps, batch]
            else:
                # Fallback to terminal binary if embeddings not available
                advantages = compute_terminal_binary_advantages(
                    rewards=episode_rewards,
                    n_steps=n_steps,
                    adv_clip_max=adv_clip,
                )
        elif adv_type == "gae":
            # AG-NFT: GAE-based per-step credit assignment via value network
            # Uses standard GAE with V(o,l) from VLM value head.
            # Softplus naturally gates noisy labels: |y| ≈ 0 → gradient ≈ 0.
            prev_values = self.rollout_batch.get("prev_values", None)
            dones = self.rollout_batch.get("dones", None)
            gamma = self._flow_nft_cfg["gae_gamma"]
            gae_lambda = self._flow_nft_cfg["gae_lambda"]

            # Per-step reward (sum over action chunk dim)
            step_rewards = rewards.sum(dim=-1)  # [n_steps, batch]

            # Prepare values: [n_steps+1, batch] (following AWM pattern)
            if prev_values is not None and prev_values.numel() > 0:
                if prev_values.dim() == 3:
                    V_squeezed = prev_values.squeeze(-1).float()
                else:
                    V_squeezed = prev_values.float()
                if V_squeezed.shape[0] > n_steps:
                    V_full = V_squeezed[:n_steps + 1]
                else:
                    V_full = torch.cat([
                        V_squeezed[:n_steps],
                        torch.zeros(1, batch_size, device=V_squeezed.device),
                    ], dim=0)
            else:
                V_full = torch.zeros(n_steps + 1, batch_size, device=step_rewards.device)

            # Prepare dones: [n_steps+1, batch]
            if dones is not None:
                if dones.dim() == 3:
                    dones_flat = dones.max(dim=-1)[0].bool()
                else:
                    dones_flat = dones.bool()
            else:
                dones_flat = torch.zeros(n_steps + 1, batch_size,
                                         dtype=torch.bool, device=step_rewards.device)

            # Compute GAE
            gae_advantages, gae_returns = compute_gae_advantages_and_returns(
                rewards=step_rewards,
                values=V_full,
                dones=dones_flat,
                gamma=gamma,
                gae_lambda=gae_lambda,
                normalize_advantages=False,  # AG-NFT uses adv_clip_max mapping
                normalize_returns=False,
            )  # gae_advantages: [n_steps, batch], gae_returns: [n_steps, batch]

            # Clip to [-adv_clip, adv_clip] for NFT label mapping
            advantages = gae_advantages.clamp(-adv_clip, adv_clip)  # [n_steps, batch]
            # Store returns for value training during actor update
            self.rollout_batch["agnft_returns"] = gae_returns
        else:
            # Terminal binary: +1 success, -1 failure
            advantages = compute_terminal_binary_advantages(
                rewards=episode_rewards,
                n_steps=n_steps,
                adv_clip_max=adv_clip,
            )  # [n_steps, batch]

        self.rollout_batch["advantages"] = advantages
        # Dummy prev_logprobs-compatible field if needed
        if "prev_logprobs" not in self.rollout_batch:
            self.rollout_batch["prev_logprobs"] = torch.zeros(n_steps, batch_size, 1)

        rollout_metrics = compute_rollout_metrics(self.rollout_batch)
        rollout_metrics["nft/episode_reward"] = episode_rewards.mean().item()
        rollout_metrics["nft/success_rate"] = (episode_rewards > 0.5).float().mean().item()
        rollout_metrics["nft/adv_type"] = (
            2.0 if adv_type == "gae"
            else 1.0 if adv_type == "frozen_embedding"
            else 0.0
        )
        rollout_metrics["nft/adv_mean"] = advantages.mean().item()
        rollout_metrics["nft/adv_std"] = advantages.std().item()
        rollout_metrics["nft/pos_frac"] = (advantages > 0).float().mean().item()

        # AG-NFT diagnostics
        if adv_type == "gae":
            rollout_metrics["agnft/raw_gae_adv_mean"] = gae_advantages.mean().item()
            rollout_metrics["agnft/raw_gae_adv_std"] = gae_advantages.std().item()
            rollout_metrics["agnft/returns_mean"] = gae_returns.mean().item()
            rollout_metrics["agnft/v_mean"] = V_full[:n_steps].mean().item()
            rollout_metrics["agnft/v_std"] = V_full[:n_steps].std().item()
            # Self-calibration diagnostic: y = advantages / adv_clip maps to [-1, 1]
            y_labels = advantages / adv_clip
            rollout_metrics["agnft/y_near_zero_frac"] = (y_labels.abs() < 0.1).float().mean().item()
            rollout_metrics["agnft/y_strong_frac"] = (y_labels.abs() > 0.5).float().mean().item()

        return rollout_metrics

    def _compute_flow_hinge_nft_credit_assignment(self) -> dict:
        """
        Hinge-NFT credit assignment: terminal binary or self-annotation.

        Terminal binary: +adv_clip_max for success, -adv_clip_max for failure,
        broadcast to all env steps. Same as NFT.

        Self-annotation: use FlowSAR-style reconstruction error for per-step
        temporal credit (recon_error comes from rollout worker annotation pass).
        """
        rewards = self.rollout_batch["rewards"]  # [n_steps, batch, chunk]
        forward_inputs = self.rollout_batch.get("forward_inputs", {})
        n_steps = rewards.shape[0]
        batch_size = rewards.shape[1]

        # Episode-level reward
        loss_mask = self.rollout_batch.get("loss_mask", None)
        if loss_mask is not None:
            episode_rewards = (rewards * loss_mask).sum(dim=(0, 2))
        else:
            episode_rewards = rewards.sum(dim=(0, 2))
        episode_rewards = episode_rewards.clamp(0, 1)  # [batch]

        adv_clip = self._flow_hinge_nft_cfg["adv_clip_max"]
        adv_type = self._flow_hinge_nft_cfg["adv_type"]

        if adv_type == "frozen_embedding":
            # FEA: ridge regression on frozen VLM embeddings → per-step importance
            vlm_embs = forward_inputs.get("vlm_embedding", None)
            if vlm_embs is not None and vlm_embs.dim() >= 2:
                if vlm_embs.dim() == 3:
                    vlm_embs_3d = vlm_embs  # [n_steps, batch, hidden_dim]
                else:
                    hidden_dim = vlm_embs.shape[-1]
                    vlm_embs_3d = vlm_embs.reshape(n_steps, batch_size, hidden_dim)
                advantages = compute_frozen_embedding_advantages(
                    embeddings=vlm_embs_3d,
                    episode_rewards=episode_rewards,
                    gamma=self._flow_hinge_nft_cfg["fea_gamma"],
                    ridge_lambda=self._flow_hinge_nft_cfg["fea_ridge_lambda"],
                    adv_clip_max=adv_clip,
                )  # [n_steps, batch]
            else:
                # Fallback to terminal binary if embeddings not available
                advantages = compute_terminal_binary_advantages(
                    rewards=episode_rewards,
                    n_steps=n_steps,
                    adv_clip_max=adv_clip,
                )
        else:
            # Terminal binary: +1 success, -1 failure
            advantages = compute_terminal_binary_advantages(
                rewards=episode_rewards,
                n_steps=n_steps,
                adv_clip_max=adv_clip,
            )  # [n_steps, batch]

        self.rollout_batch["advantages"] = advantages
        if "prev_logprobs" not in self.rollout_batch:
            self.rollout_batch["prev_logprobs"] = torch.zeros(n_steps, batch_size, 1)

        rollout_metrics = compute_rollout_metrics(self.rollout_batch)
        rollout_metrics["hinge_nft/episode_reward"] = episode_rewards.mean().item()
        rollout_metrics["hinge_nft/success_rate"] = (episode_rewards > 0.5).float().mean().item()
        rollout_metrics["hinge_nft/adv_type"] = 1.0 if adv_type == "frozen_embedding" else 0.0
        rollout_metrics["hinge_nft/adv_mean"] = advantages.mean().item()
        rollout_metrics["hinge_nft/adv_std"] = advantages.std().item()
        rollout_metrics["hinge_nft/pos_frac"] = (advantages > 0).float().mean().item()
        if adv_type == "frozen_embedding":
            rollout_metrics["hinge_nft/y_abs_mean"] = advantages.abs().mean().item() / max(adv_clip, 1e-8)
        return rollout_metrics

    @Worker.timer("run_training")
    def run_training(self) -> None:
        """
        Run the training process using the received rollout batch.
        """
        if self.is_weight_offloaded:
            self.load_param_and_grad(self.device)
        if self.is_optimizer_offloaded:
            self.load_optimizer(self.device)

        self.model.train()
        rollout_size = (
            self.rollout_batch["prev_logprobs"].shape[0]
            * self.rollout_batch["prev_logprobs"].shape[1]
        )
        g = torch.Generator()
        g.manual_seed(self.cfg.actor.seed + self._rank)
        shuffle_id = torch.randperm(rollout_size, generator=g)

        with torch.no_grad():
            self.rollout_batch = process_nested_dict_for_train(
                self.rollout_batch, shuffle_id
            )

        assert (
            self.cfg.actor.global_batch_size
            % (self.cfg.actor.micro_batch_size * self._world_size)
            == 0
        ), "global_batch_size is not divisible by micro_batch_size * world_size"

        self.gradient_accumulation = (
            self.cfg.actor.global_batch_size
            // self.cfg.actor.micro_batch_size
            // self._world_size
        )

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        rollout_size = self.rollout_batch["prev_logprobs"].size(0)
        batch_size_per_rank = self.cfg.actor.global_batch_size // self._world_size
        assert rollout_size % batch_size_per_rank == 0, (
            f"{rollout_size} is not divisible by {batch_size_per_rank}"
        )
        metrics = {}
        update_epoch = self.cfg.algorithm.get("update_epoch", 1)
        for _ in range(update_epoch):
            rollout_dataloader_iter = split_dict_to_chunk(
                self.rollout_batch,
                rollout_size // batch_size_per_rank,
            )
            for train_global_batch in rollout_dataloader_iter:
                # split batch into micro_batches
                train_global_batch_size = train_global_batch["prev_logprobs"].shape[0]
                assert (
                    train_global_batch_size
                    == self.cfg.actor.global_batch_size
                    // torch.distributed.get_world_size()
                )
                assert train_global_batch_size % self.cfg.actor.micro_batch_size == 0, (
                    f"{train_global_batch_size=}, {self.cfg.actor.micro_batch_size}"
                )

                train_micro_batch = split_dict_to_chunk(
                    train_global_batch,
                    train_global_batch_size // self.cfg.actor.micro_batch_size,
                )

                self.optimizer.zero_grad()
                for idx, batch in enumerate(train_micro_batch):
                    batch = put_tensor_device(
                        batch, f"cuda:{int(os.environ['LOCAL_RANK'])}"
                    )
                    backward_ctx = self.before_micro_batch(
                        self.model,
                        is_last_micro_batch=(idx + 1) == self.gradient_accumulation,
                    )

                    loss_mask = batch.get("loss_mask", None)
                    forward_inputs = batch.get("forward_inputs", None)

                    if self._is_flow_ipo:
                        # ============ FlowIPO Training Branch ============
                        actions = batch["actions"]              # [batch, chunk, action_dim]
                        flow_ipo_weights = batch["flow_ipo_weights"]  # [batch]

                        action_chunk = self.cfg.actor.model.get("action_chunk", 5)
                        action_dim = self.cfg.actor.model.get("action_dim", 7)
                        bsz = actions.shape[0]

                        # Read pre-computed t, ε, v_ref from rollout worker
                        t = forward_inputs["flow_t"]       # [batch, 1, 1]
                        epsilon = forward_inputs["flow_epsilon"]  # [batch, horizon, dim]
                        v_ref = forward_inputs["ref_v"]     # [batch, horizon, dim]

                        # Recompute x_t and u_i from stored t, ε
                        x_t = (1 - t) * actions + t * epsilon
                        u_i = epsilon - actions

                        # Interpolated target ṽ_i = w_i * u_i + (1 - w_i) * v_ref
                        w = flow_ipo_weights
                        while w.dim() < u_i.dim():
                            w = w.unsqueeze(-1)
                        interpolated_target = (w * u_i + (1 - w) * v_ref).detach()

                        # Model forward v_θ(x_t, t, s_i) — with grad (via FSDP forward())
                        timestep = t.reshape(bsz)
                        with self.amp_context:
                            v_theta = self.model(
                                forward_type=ForwardType.VELOCITY,
                                forward_inputs=forward_inputs,
                                x_t=x_t, timestep=timestep,
                            )

                        # Step 7: Compute FlowIPO loss
                        # Only use the action_chunk portion for loss
                        v_theta_chunk = v_theta[:, :action_chunk, :action_dim]
                        target_chunk = interpolated_target[:, :action_chunk, :action_dim]

                        loss_kwargs = {
                            "loss_type": "flow_ipo",
                            "v_theta": v_theta_chunk,
                            "interpolated_target": target_chunk,
                            "loss_mask": loss_mask,
                        }
                        loss, metrics_data = policy_loss(**loss_kwargs)
                        # ============ End FlowIPO Branch ============
                    elif self._is_flow_sar:
                        # ============ FlowSAR Training Branch ============
                        actions = batch["actions"]  # [batch, chunk, action_dim]
                        flow_sar_weights = batch["flow_sar_weights"]  # [batch]
                        flow_sar_labels = batch["flow_sar_labels"]    # [batch]

                        action_chunk = self.cfg.actor.model.get("action_chunk", 5)
                        action_dim = self.cfg.actor.model.get("action_dim", 7)
                        bsz = actions.shape[0]

                        # Read pre-computed t, ε, v_old from rollout worker
                        t = forward_inputs["flow_t"]         # [batch, 1, 1]
                        epsilon = forward_inputs["flow_epsilon"]  # [batch, horizon, dim]
                        v_old = forward_inputs["v_old"]       # [batch, horizon, dim]

                        # Recompute x_t and u_i from stored t, ε
                        x_t = (1 - t) * actions + t * epsilon
                        u_i = epsilon - actions  # flow matching target velocity

                        # Model forward v_θ(x_t, t, s_i) — with grad (via FSDP forward())
                        timestep = t.reshape(bsz)
                        with self.amp_context:
                            v_theta = self.model(
                                forward_type=ForwardType.VELOCITY,
                                forward_inputs=forward_inputs,
                                x_t=x_t, timestep=timestep,
                            )

                        # Crop to action_chunk portion for loss
                        v_theta_chunk = v_theta[:, :action_chunk, :action_dim]
                        v_old_chunk = v_old[:, :action_chunk, :action_dim].detach()
                        u_i_chunk = u_i[:, :action_chunk, :action_dim]

                        # Compute FlowSAR loss
                        loss_kwargs = {
                            "loss_type": "flow_sar",
                            "v_theta": v_theta_chunk,
                            "v_old": v_old_chunk,
                            "u_target": u_i_chunk,
                            "weights": flow_sar_weights,
                            "labels": flow_sar_labels,
                            "beta": self._flow_sar_cfg["beta"],
                            "energy_type": self._flow_sar_cfg["energy_type"],
                            "loss_variant": self._flow_sar_cfg["loss_variant"],
                            "kl_coeff": self._flow_sar_cfg["kl_coeff"],
                            "flow_t": t,
                            "loss_mask": loss_mask,
                        }
                        loss, metrics_data = policy_loss(**loss_kwargs)
                        # ============ End FlowSAR Branch ============
                    elif self._is_flow_fpi:
                        # ============ FlowFPI Training Branch ============
                        actions = batch["actions"]  # [batch, chunk, action_dim]
                        fpi_weights = batch["fpi_weights"]          # [batch]
                        td_targets = batch["fpi_td_targets"]        # [batch]
                        progress_targets = batch["fpi_progress_targets"]  # [batch]
                        success_mask = batch["fpi_success_mask"]    # [batch]

                        action_chunk = self.cfg.actor.model.get("action_chunk", 5)
                        action_dim = self.cfg.actor.model.get("action_dim", 7)
                        bsz = actions.shape[0]

                        # --- 1. Value Training (ForwardType.VALUE) ---
                        with self.amp_context:
                            value_output = self.model(
                                forward_type=ForwardType.VALUE,
                                forward_inputs=forward_inputs,
                            )
                        v_current = value_output["values"]  # [batch]

                        # TD loss: ||V(o_t) - y_t||²
                        td_loss = (v_current - td_targets.detach()).pow(2).mean()
                        # Progress loss (π-RL style): ||V(o_t) - target||²
                        # Targets are reward-scaled: success→t/T, failure→0
                        # Applied to ALL trajectories for dense supervision
                        prog_loss = (v_current - progress_targets.detach()).pow(2).mean()
                        value_loss = td_loss + prog_loss

                        # --- 2. Policy Training (ForwardType.VELOCITY) ---
                        # Use pre-computed (t, ε, v_old) from rollout worker for
                        # stable training + trust region anchor
                        t = forward_inputs["flow_t"]             # [batch, 1, 1]
                        epsilon = forward_inputs["flow_epsilon"] # [batch, horizon, dim]
                        v_old = forward_inputs["v_old"]          # [batch, horizon, dim]

                        x_t = (1 - t) * actions + t * epsilon
                        u_i = epsilon - actions  # flow matching target (t-convention: t=0→clean)

                        timestep = t.reshape(bsz)
                        with self.amp_context:
                            v_theta = self.model(
                                forward_type=ForwardType.VELOCITY,
                                forward_inputs=forward_inputs,
                                x_t=x_t, timestep=timestep,
                            )

                        v_theta_chunk = v_theta[:, :action_chunk, :action_dim]
                        v_old_chunk = v_old[:, :action_chunk, :action_dim].detach()
                        u_i_chunk = u_i[:, :action_chunk, :action_dim]

                        loss_kwargs = {
                            "loss_type": "flow_fpi",
                            "v_theta": v_theta_chunk,
                            "u_target": u_i_chunk,
                            "weights": fpi_weights,
                            "loss_mask": loss_mask,
                        }
                        policy_loss_val, metrics_data = policy_loss(**loss_kwargs)

                        # Trust region: KL penalty ||v_θ - v_old||²
                        # Prevents velocity field from drifting across update epochs
                        kl_coeff = self._flow_fpi_cfg["kl_coeff"]
                        kl_loss = (v_theta_chunk - v_old_chunk).pow(2).mean()

                        # --- 3. Combined Loss ---
                        value_coeff = self._flow_fpi_cfg["value_coeff"]
                        loss = policy_loss_val + value_coeff * value_loss + kl_coeff * kl_loss

                        metrics_data["actor/fpi_value_loss"] = value_loss.detach().item()
                        metrics_data["actor/fpi_td_loss"] = td_loss.detach().item()
                        metrics_data["actor/fpi_prog_loss"] = prog_loss.detach().item()
                        metrics_data["actor/fpi_kl_loss"] = kl_loss.detach().item()
                        metrics_data["actor/fpi_v_mean"] = v_current.detach().mean().item()
                        # ============ End FlowFPI Branch ============
                    elif self._is_flow_gfn:
                        # ============ GFN-Flow Training Branch ============
                        # GFlowNet-guided denoising for diversity-preserving RL
                        # SubTB loss on K-step denoising DAG + boundary condition
                        import math as _math

                        actions = batch["actions"]  # [batch, chunk, action_dim]
                        gfn_log_rewards = batch["gfn_log_rewards"]  # [batch]

                        action_chunk = self.cfg.actor.model.get("action_chunk", 5)
                        action_dim = self.cfg.actor.model.get("action_dim", 7)
                        bsz = actions.shape[0]
                        total_action_dim = action_chunk * action_dim

                        K = self._flow_gfn_cfg["denoise_steps"]
                        sigma_f = self._flow_gfn_cfg["sigma_f"]
                        sigma_b = self._flow_gfn_cfg["sigma_b"]

                        # Read pre-computed chain, timesteps, v_old from rollout worker
                        gfn_chain = forward_inputs["gfn_chain"]        # [batch, K+1, horizon, dim]
                        gfn_timesteps = forward_inputs["gfn_timesteps"]  # [K+1]
                        v_old = forward_inputs["v_old"]                 # [batch, horizon, dim]
                        t = forward_inputs["flow_t"]                    # [batch, 1, 1]
                        epsilon = forward_inputs["flow_epsilon"]        # [batch, horizon, dim]

                        # Compute x_t for trust region
                        x_t_reg = (1 - t) * actions + t * epsilon
                        timestep_reg = t.reshape(bsz)

                        # --- Single combined forward: 1 VLM + (K+2) suffix ---
                        with self.amp_context:
                            gfn_output = self.model(
                                forward_type=ForwardType.GFN_CHAIN,
                                forward_inputs=forward_inputs,
                                gfn_chain=gfn_chain,
                                gfn_timesteps=gfn_timesteps,
                                x_t_reg=x_t_reg,
                                timestep_reg=timestep_reg,
                            )

                        velocities = gfn_output["velocities"]   # [batch, K, horizon, dim]
                        log_flows = gfn_output["log_flows"]      # [batch, K+1]
                        v_reg = gfn_output["v_reg"]              # [batch, horizon, dim]

                        # Crop to action_chunk portion
                        chain_chunk = gfn_chain[:, :, :action_chunk, :action_dim]
                        vel_chunk = velocities[:, :, :action_chunk, :action_dim]

                        # --- Log P_F (forward policy, uses v_θ, has gradients to θ) ---
                        log_pf = compute_gfn_log_pf(
                            chain=chain_chunk,
                            velocities=vel_chunk,
                            timesteps=gfn_timesteps,
                            sigma_f=sigma_f,
                            action_dim=total_action_dim,
                        )  # [batch, K]

                        # --- Log P_B (backward policy, fixed, no gradients) ---
                        log_pb = compute_gfn_log_pb(
                            chain=chain_chunk,
                            timesteps=gfn_timesteps,
                            sigma_b=sigma_b,
                            action_dim=total_action_dim,
                        )  # [batch, K]

                        # --- SubTB + boundary loss ---
                        in_warmup = self._flow_gfn_warmup_remaining > 0
                        if in_warmup:
                            self._flow_gfn_warmup_remaining -= 1

                        loss_kwargs = {
                            "loss_type": "flow_gfn",
                            "log_flows": log_flows,
                            "log_pf": log_pf,
                            "log_pb": log_pb,
                            "log_rewards": gfn_log_rewards,
                            "subtb_lambda": self._flow_gfn_cfg["subtb_lambda"],
                            "boundary_coeff": self._flow_gfn_cfg["boundary_coeff"],
                            "loss_mask": loss_mask,
                        }
                        flow_loss_val, metrics_data = policy_loss(**loss_kwargs)

                        # --- Trust region: ||v_θ - v_old||² ---
                        v_reg_chunk = v_reg[:, :action_chunk, :action_dim]
                        v_old_chunk = v_old[:, :action_chunk, :action_dim].detach()
                        kl_coeff = self._flow_gfn_cfg["kl_coeff"]
                        kl_loss = (v_reg_chunk - v_old_chunk).pow(2).mean()

                        # --- Combined Loss ---
                        flow_coeff = self._flow_gfn_cfg["flow_loss_coeff"]
                        if in_warmup:
                            # During warmup: only train F_ψ via boundary loss, zero SubTB
                            loss = flow_coeff * flow_loss_val
                        else:
                            loss = flow_coeff * flow_loss_val + kl_coeff * kl_loss

                        metrics_data["actor/gfn_kl_loss"] = kl_loss.detach().item()
                        metrics_data["actor/gfn_total_loss"] = loss.detach().item()
                        metrics_data["actor/gfn_warmup_remaining"] = float(self._flow_gfn_warmup_remaining)
                        metrics_data["actor/gfn_in_warmup"] = float(in_warmup)
                        # ============ End GFN-Flow Branch ============
                    elif self._is_flow_awm:
                        # ============ FlowAWM Training Branch ============
                        # AWM-VLA: Advantage Weighted Matching for VLA
                        # L = Ã * ||v_θ - u||² + β * ||v_θ - v_old||² + α * L_V
                        # Key: Ã can be NEGATIVE → pushes velocity AWAY from bad actions
                        actions = batch["actions"]  # [batch, chunk, action_dim]
                        awm_advantages = batch["awm_advantages"]       # [batch]
                        td_targets = batch["awm_td_targets"]           # [batch]
                        progress_targets = batch["awm_progress_targets"]  # [batch]

                        action_chunk = self.cfg.actor.model.get("action_chunk", 5)
                        action_dim = self.cfg.actor.model.get("action_dim", 7)
                        bsz = actions.shape[0]

                        # Use pre-computed (t, ε, v_old) from rollout worker
                        t = forward_inputs["flow_t"]             # [batch, 1, 1]
                        epsilon = forward_inputs["flow_epsilon"] # [batch, horizon, dim]
                        v_old = forward_inputs["v_old"]          # [batch, horizon, dim]

                        x_t = (1 - t) * actions + t * epsilon
                        u_i = epsilon - actions  # flow matching target (t-convention)

                        # --- Single VLM forward: velocity + value together ---
                        # return_value=True reuses the same VLM prefix_output for
                        # both velocity head and value head (saves one full VLM pass)
                        timestep = t.reshape(bsz)
                        with self.amp_context:
                            vel_val_out = self.model(
                                forward_type=ForwardType.VELOCITY,
                                forward_inputs=forward_inputs,
                                x_t=x_t, timestep=timestep,
                                return_value=True,
                            )
                        v_theta, v_current = vel_val_out  # unpack (velocity, values)

                        # --- Value loss (detached from VLM by get_value_from_vlm) ---
                        td_loss = (v_current - td_targets.detach()).pow(2).mean()
                        prog_loss = (v_current - progress_targets.detach()).pow(2).mean()
                        value_loss = td_loss + prog_loss

                        # --- Policy loss ---
                        v_theta_chunk = v_theta[:, :action_chunk, :action_dim]
                        v_old_chunk = v_old[:, :action_chunk, :action_dim].detach()
                        u_i_chunk = u_i[:, :action_chunk, :action_dim]

                        # AWM loss: Ã * ||v_θ - u||² (Ã can be negative!)
                        loss_kwargs = {
                            "loss_type": "flow_awm",
                            "v_theta": v_theta_chunk,
                            "u_target": u_i_chunk,
                            "advantages": awm_advantages,
                            "loss_mask": loss_mask,
                        }
                        policy_loss_val, metrics_data = policy_loss(**loss_kwargs)

                        # Trust region: KL penalty ||v_θ - v_old||²
                        kl_coeff = self._flow_awm_cfg["kl_coeff"]
                        kl_loss = (v_theta_chunk - v_old_chunk).pow(2).mean()

                        # --- Combined Loss ---
                        value_coeff = self._flow_awm_cfg["value_coeff"]
                        loss = policy_loss_val + value_coeff * value_loss + kl_coeff * kl_loss

                        metrics_data["actor/awm_value_loss"] = value_loss.detach().item()
                        metrics_data["actor/awm_td_loss"] = td_loss.detach().item()
                        metrics_data["actor/awm_prog_loss"] = prog_loss.detach().item()
                        metrics_data["actor/awm_kl_loss"] = kl_loss.detach().item()
                        metrics_data["actor/awm_v_mean"] = v_current.detach().mean().item()
                        metrics_data["actor/awm_total_loss"] = loss.detach().item()
                        # ============ End FlowAWM Branch ============
                    elif self._is_flow_nft:
                        # ============ FlowNFT / AG-NFT Training Branch ============
                        # FlowNFT: contrastive mirror loss with SDE chain snapshots
                        # AG-NFT: adds value loss for GAE-based per-step credit assignment
                        actions = batch["actions"]  # [batch, chunk, action_dim]

                        action_chunk = self.cfg.actor.model.get("action_chunk", 5)
                        action_dim = self.cfg.actor.model.get("action_dim", 7)
                        bsz = actions.shape[0]

                        # Retrieve NFT snapshots from rollout (stored in forward_inputs)
                        v_old = forward_inputs["nft_v"]          # [batch, chunk, dim]
                        x_t = forward_inputs["nft_xt"]           # [batch, horizon, dim]
                        x_next = forward_inputs["nft_xnext"]     # [batch, horizon, dim]
                        step_indices = forward_inputs["nft_step_index"]  # [batch]
                        noise_level = forward_inputs["nft_noise_level"]  # [batch]

                        # Build denoising schedule
                        num_steps = self.cfg.actor.model.get("num_steps", 10)
                        schedule = torch.linspace(1, 0, num_steps + 1, device=self.device,
                                                  dtype=x_t.dtype)
                        t = schedule[step_indices.long()]

                        # AG-NFT: request value from same VLM forward (zero extra cost)
                        is_agnft = self._flow_nft_cfg["adv_type"] == "gae"
                        use_return_value = (
                            is_agnft
                            and hasattr(self.model, 'use_vlm_value')
                            and getattr(self.model, 'use_vlm_value', False)
                        )

                        # Actor forward: compute v_theta (and optionally value) at (x_t, t)
                        timestep = t.reshape(bsz)
                        with self.amp_context:
                            vel_out = self.model(
                                forward_type=ForwardType.VELOCITY,
                                forward_inputs=forward_inputs,
                                x_t=x_t, timestep=timestep,
                                return_value=use_return_value,
                            )

                        if use_return_value:
                            v_theta, v_current = vel_out
                        else:
                            v_theta = vel_out
                            v_current = None

                        # Crop all tensors to action chunk (matching reference)
                        v_theta_c = v_theta[:, :action_chunk, :action_dim]
                        v_old_c = v_old[:, :action_chunk, :action_dim].detach()
                        x_t_c = x_t[:, :action_chunk, :action_dim]
                        x_next_c = x_next[:, :action_chunk, :action_dim]

                        # NFT loss (unchanged for both FlowNFT and AG-NFT)
                        loss_kwargs = {
                            "loss_type": "flow_nft",
                            "v_theta": v_theta_c,
                            "v_old": v_old_c,
                            "x_t": x_t_c,
                            "x_next": x_next_c,
                            "schedule": schedule,
                            "step_indices": step_indices,
                            "total_denoise_steps": num_steps,
                            "noise_level": noise_level,
                            "advantages": batch["advantages"],
                            "loss_mask": loss_mask,
                            "beta": self._flow_nft_cfg["beta"],
                            "kl_beta": self._flow_nft_cfg["kl_beta"],
                            "adv_clip_max": self._flow_nft_cfg["adv_clip_max"],
                            "max_drift": self._flow_nft_cfg["max_drift"],
                            "dpo_beta": self._flow_nft_cfg["dpo_beta"],
                            "critic_warmup": self.optimizer_steps < self.critic_warmup_steps,
                        }
                        nft_loss, metrics_data = policy_loss(**loss_kwargs)

                        # AG-NFT: add value loss for training the value network
                        if is_agnft and v_current is not None:
                            agnft_returns = batch.get("agnft_returns", None)
                            if agnft_returns is not None:
                                value_targets = agnft_returns.detach()
                                v_pred = v_current.squeeze(-1) if v_current.dim() > 1 else v_current
                                value_loss = (v_pred - value_targets).pow(2).mean()
                                value_coeff = self._flow_nft_cfg["value_coeff"]
                                loss = nft_loss + value_coeff * value_loss
                                metrics_data["actor/agnft_value_loss"] = value_loss.detach().item()
                                metrics_data["actor/agnft_v_mean"] = v_pred.detach().mean().item()
                                metrics_data["actor/agnft_v_std"] = v_pred.detach().std().item()
                                metrics_data["actor/agnft_total_loss"] = loss.detach().item()
                            else:
                                loss = nft_loss
                        else:
                            loss = nft_loss
                        # ============ End FlowNFT / AG-NFT Branch ============
                    elif self._is_flow_hinge_nft:
                        # ============ FlowHingeNFT Training Branch ============
                        # Hinge-NFT: symmetric hinge margin loss variant of NFT
                        # L = max(0, margin + 0.5*y*ΔE) + kl_beta * ||v_θ - v_old||²
                        # Reuses NFT's SDE snapshot data, only loss function differs
                        actions = batch["actions"]  # [batch, chunk, action_dim]

                        action_chunk = self.cfg.actor.model.get("action_chunk", 5)
                        action_dim = self.cfg.actor.model.get("action_dim", 7)
                        bsz = actions.shape[0]

                        # Retrieve NFT snapshots from rollout (same as FlowNFT)
                        v_old = forward_inputs["nft_v"]          # [batch, chunk, dim]
                        x_t = forward_inputs["nft_xt"]           # [batch, horizon, dim]
                        x_next = forward_inputs["nft_xnext"]     # [batch, horizon, dim]
                        step_indices = forward_inputs["nft_step_index"]  # [batch]
                        noise_level = forward_inputs["nft_noise_level"]  # [batch]

                        # Build denoising schedule
                        num_steps = self.cfg.actor.model.get("num_steps", 10)
                        schedule = torch.linspace(1, 0, num_steps + 1, device=self.device,
                                                  dtype=x_t.dtype)
                        t = schedule[step_indices.long()]

                        # Actor forward: compute v_theta at (x_t, t)
                        timestep = t.reshape(bsz)
                        with self.amp_context:
                            v_theta = self.model(
                                forward_type=ForwardType.VELOCITY,
                                forward_inputs=forward_inputs,
                                x_t=x_t, timestep=timestep,
                            )

                        # Crop all tensors to action chunk
                        v_theta_c = v_theta[:, :action_chunk, :action_dim]
                        v_old_c = v_old[:, :action_chunk, :action_dim].detach()
                        x_t_c = x_t[:, :action_chunk, :action_dim]
                        x_next_c = x_next[:, :action_chunk, :action_dim]

                        # Hinge-NFT loss
                        loss_kwargs = {
                            "loss_type": "flow_hinge_nft",
                            "v_theta": v_theta_c,
                            "v_old": v_old_c,
                            "x_t": x_t_c,
                            "x_next": x_next_c,
                            "schedule": schedule,
                            "step_indices": step_indices,
                            "total_denoise_steps": num_steps,
                            "noise_level": noise_level,
                            "advantages": batch["advantages"],
                            "loss_mask": loss_mask,
                            "margin": self._flow_hinge_nft_cfg["margin"],
                            "beta": self._flow_hinge_nft_cfg["beta"],
                            "kl_beta": self._flow_hinge_nft_cfg["kl_beta"],
                            "adv_clip_max": self._flow_hinge_nft_cfg["adv_clip_max"],
                            "max_drift": self._flow_hinge_nft_cfg["max_drift"],
                            "critic_warmup": self.optimizer_steps < self.critic_warmup_steps,
                        }
                        loss, metrics_data = policy_loss(**loss_kwargs)
                        # ============ End FlowHingeNFT Branch ============
                    else:
                        # ============ Original PPO/GRPO Branch ============
                        advantages = batch["advantages"]
                        prev_logprobs = batch["prev_logprobs"]
                        returns = batch.get("returns", None)
                        prev_values = batch.get("prev_values", None)
                        loss_mask_sum = batch.get("loss_mask_sum", None)

                        kwargs = {}
                        if SupportedModel(self.cfg.actor.model.model_type) in [
                            SupportedModel.OPENVLA,
                            SupportedModel.OPENVLA_OFT,
                        ]:
                            kwargs["temperature"] = (
                                self.cfg.algorithm.sampling_params.temperature_train
                            )
                            kwargs["top_k"] = self.cfg.algorithm.sampling_params.top_k
                        elif (
                            SupportedModel(self.cfg.actor.model.model_type)
                            == SupportedModel.GR00T
                        ):
                            kwargs["prev_logprobs"] = prev_logprobs

                        compute_values = (
                            True if self.cfg.algorithm.adv_type == "gae" else False
                        )

                        with self.amp_context:
                            output_dict = self.model(
                                forward_inputs=forward_inputs,
                                compute_logprobs=True,
                                compute_entropy=self.cfg.algorithm.entropy_bonus > 0,
                                compute_values=compute_values,
                                use_cache=False,
                                **kwargs,
                            )

                        if (
                            SupportedModel(self.cfg.actor.model.model_type)
                            == SupportedModel.GR00T
                        ):
                            prev_logprobs = output_dict["prev_logprobs"]

                        kwargs = {
                            "loss_type": self.cfg.algorithm.loss_type,
                            "logprob_type": self.cfg.algorithm.logprob_type,
                            "reward_type": self.cfg.algorithm.reward_type,
                            "single_action_dim": self.cfg.actor.model.get("action_dim", 7),
                            "logprobs": output_dict["logprobs"],
                            "values": output_dict.get("values", None),
                            "old_logprobs": prev_logprobs,
                            "advantages": advantages,
                            "returns": returns,
                            "prev_values": prev_values,
                            "clip_ratio_high": self.cfg.algorithm.clip_ratio_high,
                            "clip_ratio_low": self.cfg.algorithm.clip_ratio_low,
                            "value_clip": self.cfg.algorithm.get("value_clip", None),
                            "huber_delta": self.cfg.algorithm.get("huber_delta", None),
                            "loss_mask": loss_mask,
                            "loss_mask_sum": loss_mask_sum,
                            "max_episode_steps": self.cfg.env.train.max_episode_steps,
                            "task_type": self.cfg.runner.task_type,
                            "critic_warmup": self.optimizer_steps
                            < self.critic_warmup_steps,
                        }
                        loss, metrics_data = policy_loss(**kwargs)

                        entropy_loss = torch.tensor(0.0, device=torch.cuda.current_device())
                        if (
                            self.cfg.algorithm.entropy_bonus > 0
                            and not kwargs["critic_warmup"]
                        ):
                            entropy = output_dict["entropy"]
                            entropy = reshape_entropy(
                                entropy,
                                entropy_type=self.cfg.algorithm.entropy_type,
                                action_dim=self.cfg.actor.model.get("action_dim", 7),
                                batch_size=output_dict["logprobs"].shape[0],
                            )
                            entropy_loss = masked_mean(entropy, mask=loss_mask)
                            loss -= self.cfg.algorithm.entropy_bonus * entropy_loss
                        metrics_data["actor/entropy_loss"] = entropy_loss.detach().item()
                        # ============ End Original Branch ============

                    loss /= self.gradient_accumulation
                    with backward_ctx:
                        self.grad_scaler.scale(loss).backward()

                    metrics_data["actor/total_loss"] = loss.detach().item()
                    append_to_dict(metrics, metrics_data)

                torch.cuda.empty_cache()

                grad_norm, lr_list = self.optimizer_step()
                data = {
                    "actor/grad_norm": grad_norm,
                    "actor/lr": lr_list[0],
                }
                if len(lr_list) > 1:
                    data["critic/lr"] = lr_list[1]
                append_to_dict(metrics, data)
        # put LR scheduler step here
        self.lr_scheduler.step()
        self.optimizer.zero_grad()
        clear_memory()
        mean_metric_dict = {key: np.mean(value) for key, value in metrics.items()}
        mean_metric_dict = all_reduce_dict(
            mean_metric_dict, op=torch.distributed.ReduceOp.AVG
        )

        return mean_metric_dict

    def set_global_step(self, global_step) -> None:
        """
        Set the global step for the model, if needed.
        """
        if hasattr(self.model, "set_global_step"):
            self.model.set_global_step(global_step)
