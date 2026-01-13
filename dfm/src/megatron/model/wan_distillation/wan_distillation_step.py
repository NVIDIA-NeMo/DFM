# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Add fastgen to the path so that internal fastgen imports work correctly
# _fastgen_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../fastgen"))
# if _fastgen_path not in sys.path:
#     sys.path.insert(0, _fastgen_path)

import gc
import logging
from functools import partial
from typing import Any, Iterable

import torch
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from easydict import EasyDict
from megatron.bridge.training.losses import masked_next_token_loss
from megatron.bridge.training.state import GlobalState
from megatron.core import parallel_state
from megatron.core.models.common.vision_module.vision_module import VisionModule
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.utils import get_model_config, unwrap_model
from tqdm import tqdm

import wandb
from dfm.src.fastgen.fastgen.methods.model import FastGenModel
from dfm.src.fastgen.fastgen.networks.Wan.network import _decode
from dfm.src.megatron.model.wan.flow_matching.flow_inference_pipeline import FlowInferencePipeline
from dfm.src.megatron.model.wan.inference import SIZE_CONFIGS
from dfm.src.megatron.model.wan.utils import unpatchify


logger = logging.getLogger(__name__)


class MegatronModelWrapper:
    """
    Wrapper that adapts Megatron WanModel interface to FastGen network interface.
    This allows using FastGenModel.generator_fn with Megatron models.
    """

    def __init__(self, model, batch: dict):
        self.model = model
        self.grid_sizes = batch["grid_sizes"]
        self.packed_seq_params = batch.get("packed_seq_params", None)
        # Forward noise_scheduler from wrapped model
        self.noise_scheduler = model.noise_scheduler

    @property
    def training(self):
        return self.model.training

    def eval(self):
        self.model.eval()
        return self

    def train(self, mode=True):
        self.model.train(mode)
        return self

    def parameters(self):
        return self.model.parameters()

    def __call__(self, x, t, condition=None, fwd_pred_type=None, **kwargs):
        """Adapt FastGen call signature to Megatron WanModel signature."""
        return self.model(
            x,
            t=t,
            context=condition,  # FastGen uses 'condition', WanModel uses 'context'
            grid_sizes=self.grid_sizes,
            packed_seq_params=self.packed_seq_params,
            scale_t=True,  # Needed when using noise scheduler timesteps
            fwd_pred_type=fwd_pred_type,
            unpatchify_features=True,
            **kwargs,
        )


def wan_data_step(qkv_format, dataloader_iter):
    batch = next(dataloader_iter)
    batch = {k: v.to(device="cuda", non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
    # Construct packed sequence parameters
    if ("seq_len_q" in batch) and ("seq_len_kv" in batch):
        zero = torch.zeros(1, dtype=torch.int32, device="cuda")

        cu_seqlens = batch["seq_len_q"].cumsum(dim=0).to(torch.int32)
        cu_seqlens = torch.cat((zero, cu_seqlens))

        cu_seqlens_padded = batch["seq_len_q_padded"].cumsum(dim=0).to(torch.int32)
        cu_seqlens_padded = torch.cat((zero, cu_seqlens_padded))

        cu_seqlens_kv = batch["seq_len_kv"].cumsum(dim=0).to(torch.int32)
        cu_seqlens_kv = torch.cat((zero, cu_seqlens_kv))

        cu_seqlens_kv_padded = batch["seq_len_kv_padded"].cumsum(dim=0).to(torch.int32)
        cu_seqlens_kv_padded = torch.cat((zero, cu_seqlens_kv_padded))

        batch["packed_seq_params"] = {
            "self_attention": PackedSeqParams(
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_q_padded=cu_seqlens_padded,
                cu_seqlens_kv=cu_seqlens,
                cu_seqlens_kv_padded=cu_seqlens_padded,
                qkv_format=qkv_format,
            ),
            "cross_attention": PackedSeqParams(
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_q_padded=cu_seqlens_padded,
                cu_seqlens_kv=cu_seqlens_kv,
                cu_seqlens_kv_padded=cu_seqlens_kv_padded,
                qkv_format=qkv_format,
            ),
        }

    return batch


class WanDistillationStep:
    def __init__(
        self,
        config: dict = None,
        model_id="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        cache_dir="/opt/artifacts",
    ):
        self.model_id = model_id
        self.cache_dir = cache_dir
        self.valid = False
        self.train = False
        self.training_trigered = False
        self._inference_pipeline = None

        # Cached negative condition embedding (computed lazily on first forward)
        self._neg_condition = None

        # Inference config (matching FlowInferencePipeline defaults)
        self.inference_cfg = EasyDict(
            {
                "t5_dtype": torch.bfloat16,
                "text_len": 512,
                "vae_stride": (4, 8, 8),
                "param_dtype": torch.bfloat16,
                "patch_size": (1, 2, 2),
                "num_train_timesteps": 1000,
                "sample_fps": 16,
                "english_sample_neg_prompt": (
                    "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, "
                    "images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, "
                    "incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, "
                    "misshapen limbs, fused fingers, still picture, messy background, three legs, many people "
                    "in the background, walking backwards"
                ),
            }
        )

        # Lazy-loaded inference pipeline (loaded on first validation)
        self._inference_pipeline = None

    def _get_neg_condition(self, unwrapped_model):
        """
        Get the negative condition embedding, computing and caching it on first call.
        The negative condition is the embedding of an empty string "".
        """
        if self._neg_condition is None:
            logger.info("Computing and caching negative condition embedding...")
            neg_prompt = [""]
            neg_condition = unwrapped_model.get_text_encoder().encode(neg_prompt, precision=torch.bfloat16)
            self._neg_condition = neg_condition.transpose(0, 1).contiguous()
            logger.info(f"Negative condition cached with shape: {self._neg_condition.shape}")
        return self._neg_condition

    def _ensure_inference_pipeline_loaded(self, model):
        """Lazily create inference pipeline with the given model."""
        if self._inference_pipeline is not None:
            return

        logger.info(f"Creating FlowInferencePipeline with model_id={self.model_id}...")

        # Get parallelism settings from the current parallel state to match the model's configuration
        tensor_parallel_size = parallel_state.get_tensor_model_parallel_world_size()
        pipeline_parallel_size = parallel_state.get_pipeline_model_parallel_world_size()
        context_parallel_size = parallel_state.get_context_parallel_world_size()
        # Check if sequence parallel is enabled by inspecting model config
        sequence_parallel = getattr(model.config, "sequence_parallel", False) if hasattr(model, "config") else False

        # Get device_id from torch (set by torchrun)
        device_id = torch.cuda.current_device()
        global_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

        self._inference_pipeline = FlowInferencePipeline(
            inference_cfg=self.inference_cfg,
            model_id=self.model_id,
            model=model,  # Pass model directly, skip checkpoint loading
            cache_dir=self.cache_dir,
            device_id=device_id,
            rank=0,
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            context_parallel_size=context_parallel_size,
            sequence_parallel=sequence_parallel,
        )
        logger.info(
            f"FlowInferencePipeline created successfully with device_id={device_id}, TP={tensor_parallel_size}, PP={pipeline_parallel_size}, CP={context_parallel_size}, SP={sequence_parallel}."
        )

    def on_train_start(self, student, teacher, fake_score, state: GlobalState):
        pass

    def on_validation_start(self, single_step_outputs, batch, student, teacher, state: GlobalState):
        """
        Generate validation videos from teacher (50 steps) and student (1 step).
        Logs videos to Weights & Biases.
        """
        if self._inference_pipeline is None:
            self._ensure_inference_pipeline_loaded(teacher)

        # Clear GPU memory before video generation to avoid OOM on single GPU
        gc.collect()
        torch.cuda.empty_cache()

        # Create pipeline with teacher model (we'll swap for student later)

        gen_latent = single_step_outputs["gen_rand"]
        grid_sizes = batch["grid_sizes"]
        patch_size = self.inference_cfg.patch_size
        z_dim = self._inference_pipeline.vae.config.z_dim

        with torch.no_grad():
            # gen_latent = gen_latent.transpose(0, 1)  # [batch, seq, hidden_dim]
            # unpatchified_latents = unpatchify(gen_latent, grid_sizes, z_dim, patch_size)
            # gen_latent_video = torch.stack(unpatchified_latents, dim=0)
            gen_videos = _decode(gen_latent, self._inference_pipeline.vae)
            fps = self.inference_cfg.sample_fps

        # Extract prompt from batch video_metadata
        video_metadata = batch.get("video_metadata", {})
        if isinstance(video_metadata, dict):
            prompt = video_metadata.get("caption", "no caption")
        elif isinstance(video_metadata, list) and len(video_metadata) > 0:
            prompt = (
                video_metadata[0].get("caption", "no caption") if isinstance(video_metadata[0], dict) else "no caption"
            )
        else:
            prompt = "The video captures a series of images showing a group of children seated in an outdoor setting, possibly at a sports event. The children are dressed in casual attire, with one wearing a red top and another in a white top with a rainbow design. The background is filled with other spectators, some of whom are wearing baseball caps. The lighting suggests it's either late afternoon or early evening, and the atmosphere appears to be casual and relaxed."

        print("prompt", prompt)
        self._log_videos_to_wandb(
            videos=gen_videos,
            video_name="student_prediction",
            caption=f"Student (1 step): {prompt}",
            fps=fps,
            state=state,
        )

        # Free memory from student video generation before teacher generation
        del gen_videos
        gc.collect()
        torch.cuda.empty_cache()

        student_steps = 4
        input_rand = single_step_outputs.get("input_rand", None)
        logger.info(f"Generating validation video from student with {student_steps} steps using generator_fn...")

        # Get condition from batch
        condition = batch.get("context_embeddings", None)
        # Extract prompt for caption

        with torch.no_grad():
            # Wrap student to adapt interface for FastGenModel.generator_fn
            wrapped_student = MegatronModelWrapper(student, batch)

            # Use FastGenModel.generator_fn directly
            student_4step_latents = FastGenModel.generator_fn(
                net=wrapped_student,
                latents=input_rand,  # [B, C, T, H, W] unit Gaussian
                condition=condition,
                student_sample_steps=student_steps,
                student_sample_type="sde",  # stochastic sampling
            )

            # Decode latents to video
            student_4step_videos = _decode(student_4step_latents, self._inference_pipeline.vae)
            self._log_videos_to_wandb(
                videos=student_4step_videos,
                video_name="student_4step_prediction",
                caption=f"Student ({student_steps} steps): {prompt}",
                fps=fps,
                state=state,
            )

            del student_4step_videos, student_4step_latents
            gc.collect()
            torch.cuda.empty_cache()

        # Generation parameters
        size_key = "832*480"
        size = SIZE_CONFIGS[size_key]
        frame_num = 81
        shift = 5.0
        guide_scale = 5.0

        seed = parallel_state.get_data_parallel_rank()

        # Get the same initial noise that was used by the student
        # input_rand is the unit Gaussian noise (input_student / max_sigma)
        input_rand = single_step_outputs.get("input_rand", None)
        if input_rand is not None:
            # Take first sample only, squeeze batch dimension
            # generate() expects (C, T, H, W), not (B, C, T, H, W)
            initial_latents = input_rand[0]  # Shape: (C, T, H, W)
            logger.info(f"Using student's input noise for teacher generation (shape: {initial_latents.shape})")
        else:
            initial_latents = None
            logger.warning("input_rand not found in outputs, teacher will use random noise")

        # Generate from teacher with multiple step counts only on first iteration
        if state.train_state.step == 0:
            teacher_step_counts = [4, 10, 50]
        else:
            teacher_step_counts = []  # Skip teacher generation after first iteration
        self._inference_pipeline.model = teacher

        for step in teacher_step_counts:
            logger.info(f"Generating validation video from teacher with {step} steps (seed={seed})...")

            with torch.no_grad():
                teacher_videos = self._inference_pipeline.generate(
                    prompts=[prompt],
                    sizes=[size],
                    frame_nums=[frame_num],
                    shift=shift,
                    sampling_steps=step,
                    # t5_cpu=True,
                    seed=seed,
                    guide_scale=guide_scale,
                    offload_model=False,
                    initial_latents=initial_latents,
                )

                self._log_videos_to_wandb(
                    videos=teacher_videos,
                    video_name=f"teacher_{step}step_prediction",
                    caption=f"Teacher ({step} steps): {prompt}",
                    fps=fps,
                    state=state,
                )

                # Free memory from teacher video generation
                del teacher_videos
                gc.collect()
                torch.cuda.empty_cache()

        # logger.info("Generating video from teacher using FastGen sampling procedure...")
        # condition = batch.get("context_embeddings", None)
        # with torch.no_grad():
        #     initial_noise = single_step_outputs["input_rand"]
        #     neg_condition = torch.zeros_like(condition) if condition is not None else None
        #     fastgen_output = self._sample_with_fastgen_procedure(
        #         model=teacher,
        #         latents=initial_noise,
        #         condition=condition,
        #         neg_condition=neg_condition,
        #         batch=batch,
        #         num_steps=step,
        #         guidance_scale=guide_scale,
        #         shift=shift,
        #     )

        #     # Unpatchify and decode
        #     fastgen_videos = _decode(fastgen_output, self._inference_pipeline.vae)

        # self._log_videos_to_wandb(
        #     videos=fastgen_videos,
        #     video_name="teacher_fastgen_sample",
        #     caption=f"Teacher FastGen ({step} steps): {prompt}",
        #     fps=fps,
        #     state=state,
        # )
        # logger.info("Successfully generated video using FastGen sampling procedure")

    def _sample_with_fastgen_procedure(
        self,
        model,
        latents: torch.Tensor,
        condition: torch.Tensor,
        neg_condition: torch.Tensor,
        batch: dict,
        num_steps: int = 10,
        guidance_scale: float = 5.0,
        shift: float = 5.0,
    ) -> torch.Tensor:
        """
        Sample from the teacher model using FastGen-style sampling procedure.

        This replicates the sampling logic from fastgen/networks/Wan/network.py Wan.sample()
        using the batch data directly (packed_seq_params, grid_sizes, etc.).

        Args:
            model: The teacher model (Megatron WanModel)
            latents: Initial noise tensor in patchified format [seq, batch, hidden]
            condition: Text condition embeddings
            neg_condition: Negative condition embeddings
            batch: The data batch containing packed_seq_params, grid_sizes, etc.
            num_steps: Number of sampling steps
            guidance_scale: Classifier-free guidance scale
            shift: Noise schedule shift parameter

        Returns:
            Generated latents in patchified format [seq, batch, hidden]
        """
        # Setup scheduler (same as FastGen's Wan.sample)
        unipc_scheduler = UniPCMultistepScheduler.from_pretrained(
            self.model_id,
            subfolder="scheduler",
            cache_dir=self.cache_dir,
        )
        unipc_scheduler.config.flow_shift = shift
        unipc_scheduler.set_timesteps(num_inference_steps=num_steps, device=latents.device)
        timesteps = unipc_scheduler.timesteps

        # Store original training state
        was_training = model.training
        model.eval()

        # Get batch info
        grid_sizes = batch["grid_sizes"]
        packed_seq_params = batch.get("packed_seq_params", None)
        batch_size = latents.shape[1]  # latents is [seq, batch, hidden]

        # Get model dtype and cast inputs to match
        model_dtype = next(model.parameters()).dtype
        latents = latents.to(model_dtype)
        condition = condition.to(model_dtype)
        if neg_condition is not None:
            neg_condition = neg_condition.to(model_dtype)

        with torch.no_grad():
            # Main sampling loop (matching FastGen's Wan.sample)
            for idx, timestep in tqdm(enumerate(timesteps), total=len(timesteps), desc="FastGen sampling"):
                # timestep
                # Forward pass with condition (flow prediction)
                print("timestep", timestep)
                t = (timestep / unipc_scheduler.config.num_train_timesteps).expand(latents.shape[1])

                flow_pred = model(
                    latents,
                    grid_sizes=grid_sizes,
                    t=t.expand(batch_size).to(latents.device),
                    context=condition,
                    packed_seq_params=packed_seq_params,
                    scale_t=True,
                )
                # if guidance_scale > 1.0:
                #     flow_uncond = model(
                #         latents,
                #         grid_sizes=grid_sizes,
                #         t=t.expand(batch_size).to(latents.device),
                #         context=neg_condition,
                #         packed_seq_params=packed_seq_params,
                #         scale_t=True,
                #     )
                #     print('negative output fastgen', flow_uncond)
                #     flow_pred = flow_uncond + guidance_scale * (flow_pred - flow_uncond)
                #     print('combine noise pred fastgen', flow_pred)

            patch_size = self.inference_cfg.patch_size
            z_dim = self._inference_pipeline.vae.config.z_dim

            flow_pred = flow_pred.transpose(0, 1)  # bring sbhd -> bshd
            flow_pred = unpatchify(flow_pred, grid_sizes, z_dim, patch_size)
            flow_pred = torch.stack(flow_pred, dim=0)  # pyright: ignore[reportUnusedVariable]

            latents = latents.transpose(0, 1)
            latents = unpatchify(latents, grid_sizes, z_dim, patch_size)
            latents = torch.stack(latents, dim=0)

            latents = unipc_scheduler.step(flow_pred, timestep, latents, return_dict=False)[0]

        # Restore model state (outside the loop)
        model.train(was_training)

        return latents

    def _log_videos_to_wandb(self, videos, video_name: str, caption: str, fps: int, state: GlobalState):
        """
        Log generated videos to Weights & Biases.
        Gathers videos from all DP ranks and logs on the last DP rank.

        Args:
            videos: List of video tensors from generation
            video_name: Name for the wandb log key (e.g., "teacher_prediction", "student_prediction")
            caption: Caption to display with the video
            fps: Frames per second for video playback
            state: Global training state containing wandb_logger
        """
        if videos is None:
            return

        # Determine DP rank info
        is_last_dp_rank = parallel_state.get_data_parallel_rank() == (
            parallel_state.get_data_parallel_world_size() - 1
        )

        last_dp_local_rank = parallel_state.get_data_parallel_world_size() - 1
        dp_group = parallel_state.get_data_parallel_group()
        dp_ranks = torch.distributed.get_process_group_ranks(dp_group)
        wandb_rank = dp_ranks[last_dp_local_rank]

        # Prepare video data for gathering
        # Convert from (C, T, H, W) to (T, H, W, C) then to uint8 numpy
        video_tensor = videos[0] if videos is not None else None
        if video_tensor is not None:
            video = video_tensor.clamp(-1, 1)
            video = ((video + 1) / 2 * 255).to(torch.uint8)
            video_np = video.permute(1, 2, 3, 0).cpu().numpy()
        else:
            video_np = None

        # Prepare gather list
        if is_last_dp_rank:
            gather_list = [None for _ in range(parallel_state.get_data_parallel_world_size())]
        else:
            gather_list = None

        # Gather videos from all DP ranks
        torch.distributed.gather_object(
            obj=video_np,
            object_gather_list=gather_list,
            dst=wandb_rank,
            group=parallel_state.get_data_parallel_group(),
        )

        # Log to wandb on the last DP rank
        if is_last_dp_rank and state.wandb_logger is not None:
            if gather_list is not None:
                wandb_videos = []
                for vid in gather_list:
                    if vid is not None:
                        # wandb.Video expects (T, C, H, W) format
                        vid_transposed = vid.transpose(0, 3, 1, 2)
                        wandb_videos.append(wandb.Video(vid_transposed, fps=fps, format="mp4", caption=caption))

                if wandb_videos:
                    step = state.train_state.step
                    state.wandb_logger.log({video_name: wandb_videos}, step=step)
                    logger.info(f"Logged {video_name} to Weights & Biases at step {step}.")

    def __call__(
        self, state: GlobalState, data_iterator: Iterable, model: VisionModule
    ) -> tuple[torch.Tensor, partial]:
        """
        Forward training step.
        """
        model_config = get_model_config(model)
        output_tensor, loss_function, outputs_dict, batch = self.forward_step(
            state, data_iterator, model, model_config
        )

        if model.training and not self.train:
            self.train = True
            unwrapped_model = unwrap_model(model)
            # self.on_validation_start(outputs_dict, batch, student=unwrapped_model.net, teacher=unwrapped_model.teacher, state=state)
        elif model.training and self.valid:
            self.train = True
            self.valid = False
        elif (not model.training) and self.train:
            unwrapped_model = unwrap_model(model)
            self.on_validation_start(
                outputs_dict, batch, student=unwrapped_model.net, teacher=unwrapped_model.teacher, state=state
            )
            self.train = False
            self.valid = True
        return output_tensor, loss_function

    def forward_step(
        self, state: GlobalState, data_iterator: Iterable, model, model_config
    ) -> tuple[torch.Tensor, partial]:
        """
        Forward training step.
        """

        print("current iteration", state.train_state.step + 1)

        print("*********** MODEL TYPE ***********", type(model))

        if hasattr(model, "optimizer"):
            print("=" * 50)
            print("OPTIMIZER PARAM GROUPS:")
            for i, param_group in enumerate(model.optimizer.param_groups):
                print(f"\nParam Group {i}:")
                print(f"  lr: {param_group.get('lr', 'N/A')}")
                print(f"  weight_decay: {param_group.get('weight_decay', 'N/A')}")
                print(f"  num params: {len(param_group['params'])}")
                # Print first few param shapes
                for j, p in enumerate(param_group["params"][:5]):
                    print(f"    param {j}: shape={p.shape}, requires_grad={p.requires_grad}")
        print("=" * 50)

        unwrapped_model = unwrap_model(model)

        from dfm.src.fastgen.fastgen.utils.basic_utils import print_param_values

        print("FAKE SCORE PARAM VALUES")
        print_param_values(unwrapped_model.fake_score)
        print("NET PARAM VALUES")
        print_param_values(unwrapped_model.net)
        # unwrapped_model = unwrap_model(model)
        # unwrapped_model.net.train()
        # unwrapped_model.fake_score.train()
        timers = state.timers
        straggler_timer = state.straggler_timer
        timers("batch-generator", log_level=2).start()

        qkv_format = getattr(model_config, "qkv_format", "sbhd")
        with straggler_timer(bdata=True):
            batch = wan_data_step(qkv_format, data_iterator)
        timers("batch-generator").stop()

        # Use fixed prompt for deterministic comparison mode (fastgen vs megatron)
        # Enable by setting model.config.deterministic_comparison = True
        if getattr(unwrapped_model.config, "constant_prompts", False):
            print("constant prompts")
            from fastgen.fixed_prompts import prompts

            rank = parallel_state.get_data_parallel_rank()
            # prompt = [batch['video_metadata'][0]['caption']]
            prompt = [prompts[rank]]

            print("prompt", prompt)
            condition = unwrapped_model.get_text_encoder().encode(prompt, precision=torch.bfloat16)
            batch["context_embeddings"] = condition.transpose(0, 1).contiguous()
            batch["video_metadata"][0]["caption"] = prompt[0]
        else:
            print("no constant prompts")

        # Always add negative condition to batch (cached, computed once on first forward)
        batch["neg_condition"] = self._get_neg_condition(unwrapped_model)

        check_for_nan_in_loss = state.cfg.rerun_state_machine.check_for_nan_in_loss
        check_for_spiky_loss = state.cfg.rerun_state_machine.check_for_spiky_loss

        with straggler_timer:
            # Debug: check pipeline parallel state
            if parallel_state.is_pipeline_last_stage():
                loss_map, outputs_dict = model.forward(
                    data=batch,
                    iteration=state.train_state.step + 1,
                )
                output_tensor = loss_map["total_loss"]
                print("final total loss", output_tensor)
            else:
                output_tensor = model.forward(
                    data=batch,
                    iteration=state.train_state.step + 1,
                )
                outputs_dict = None  # Ensure outputs_dict is defined for non-last stages

        # TODO: do we need to gather output with sequence or context parallelism here
        #       especially when we have pipeline parallelism

        loss = output_tensor
        loss_mask = torch.ones_like(loss)
        print("loss", loss)
        print("loss_mask", loss_mask)

        # NOTE: Gradient debugging is now done in megatron/bridge/training/train.py
        # after forward_backward_func returns and before optimizer.step()

        loss_function = self._create_loss_function(loss_mask, check_for_nan_in_loss, check_for_spiky_loss)

        return output_tensor, loss_function, outputs_dict, batch

    def _create_loss_function(
        self, loss_mask: torch.Tensor, check_for_nan_in_loss: bool, check_for_spiky_loss: bool
    ) -> partial:
        """Create a partial loss function with the specified configuration.

        Args:
            loss_mask: Used to mask out some portions of the loss
            check_for_nan_in_loss: Whether to check for NaN values in the loss
            check_for_spiky_loss: Whether to check for spiky loss values

        Returns:
            A partial function that can be called with output_tensor to compute the loss
        """
        return partial[Any](
            masked_next_token_loss,
            loss_mask,
            check_for_nan_in_loss=check_for_nan_in_loss,
            check_for_spiky_loss=check_for_spiky_loss,
        )
