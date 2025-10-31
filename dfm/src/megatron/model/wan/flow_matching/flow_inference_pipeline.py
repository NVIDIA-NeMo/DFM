import gc
import logging
import math
import os
import random
import sys
import types
import re
from contextlib import contextmanager
from functools import partial

import torch
import torch.cuda.amp as amp
import torch.distributed as dist
from tqdm import tqdm

from dfm.src.megatron.model.wan.wan_model import WanModel
from megatron.bridge.models.wan.wan_provider import WanModelProvider
from dfm.src.megatron.model.wan.modules.t5 import T5EncoderModel
from dfm.src.megatron.model.wan.modules import WanVAE
from dfm.src.megatron.model.wan.inference.utils.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from dfm.src.megatron.model.wan.inference.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from dfm.src.megatron.model.wan.utils.utils import grid_sizes_calculation, patchify
from megatron.core import parallel_state
from torch.nn import functional as F
from dfm.src.megatron.model.wan.utils.utils import split_inputs_cp, cat_outputs_cp

import math
from typing import Tuple, Union

class FlowInferencePipeline:

    def __init__(
        self,
        config,
        checkpoint_dir,
        checkpoint_step=None,
        t5_checkpoint_dir=None,
        vae_checkpoint_dir=None,
        device_id=0,
        rank=0,
        t5_cpu=False,

        tensor_parallel_size=1,
        context_parallel_size=1,
        pipeline_parallel_size=1,
        sequence_parallel=False,
        pipeline_dtype=torch.float32,
    ):
        r"""
        Initializes the FlowInferencePipeline with the given parameters.

        Args:
            config (EasyDict):
                Object containing model parameters initialized from config.py
            checkpoint_dir (`str`):
                Path to directory containing model checkpoints
            t5_checkpoint_dir (`str`, *optional*, defaults to None):
                Optional directory containing T5 checkpoint and tokenizer; falls back to `checkpoint_dir` if None.
            vae_checkpoint_dir (`str`, *optional*, defaults to None):
                Optional directory containing VAE checkpoint; falls back to `checkpoint_dir` if None.
            device_id (`int`,  *optional*, defaults to 0):
                Id of target GPU device
            rank (`int`,  *optional*, defaults to 0):
                Process rank for distributed training
            t5_cpu (`bool`, *optional*, defaults to False):
                Whether to place T5 model on CPU. Only works without t5_fsdp.
        """
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.t5_cpu = t5_cpu
        self.tensor_parallel_size = tensor_parallel_size
        self.context_parallel_size = context_parallel_size
        self.pipeline_parallel_size = pipeline_parallel_size
        self.sequence_parallel = sequence_parallel
        self.pipeline_dtype = pipeline_dtype
        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = config.param_dtype

        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=os.path.join(t5_checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(t5_checkpoint_dir, config.t5_tokenizer),
            shard_fn=None)

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size        
        self.vae = WanVAE(
            vae_pth=os.path.join(vae_checkpoint_dir, config.vae_checkpoint),
            device=self.device)

        wan_checkpoint_dir = self._select_checkpoint_dir(checkpoint_dir, checkpoint_step)
        self.model = self.setup_model_from_checkpoint(wan_checkpoint_dir)
        
        # DEBUGGING thd
        # set qkv_format to to "thd" for context parallelism
        # self.model.config.qkv_format = "sbhd"
        self.model.config.qkv_format = "thd"

        # set self.sp_size=1 for later use, just to respect the original Wan inference code
        self.sp_size = 1

        if dist.is_initialized():
            dist.barrier()
        self.model.to(self.device)

        self.sample_neg_prompt = config.sample_neg_prompt
        

    def unpatchify(self, x: torch.Tensor, grid_sizes: torch.Tensor, out_dim: int) -> list[torch.Tensor]:
        r"""
        Reconstruct video tensors from patch embeddings into a list of videotensors.

        Args:
            x (torch.Tensor):
                Tensor of patchified features, with shape [seq_len, c * pF * pH * pW]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            list[torch.Tensor]: list of tensors, each with shape [c, F_latents, H_latents, W_latents]
        """

        c = out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out


    def setup_model_from_checkpoint(self, checkpoint_dir):
        provider = WanModelProvider()
        provider.tensor_model_parallel_size = self.tensor_parallel_size
        provider.pipeline_model_parallel_size = self.pipeline_parallel_size
        provider.context_parallel_size = self.context_parallel_size
        provider.sequence_parallel = self.sequence_parallel
        provider.pipeline_dtype = self.pipeline_dtype
        # Once all overrides are set, finalize the model provider to ensure the post initialization logic is run
        provider.finalize()
        provider.initialize_model_parallel(seed=0)
        
        ## Read from megatron checkpoint
        from megatron.bridge.training.model_load_save import load_megatron_model as _load_megatron_model
        model = _load_megatron_model(
            checkpoint_dir,
            mp_overrides={
                "tensor_model_parallel_size": self.tensor_parallel_size,
                "pipeline_model_parallel_size": self.pipeline_parallel_size,
                "context_parallel_size": self.context_parallel_size,
                "sequence_parallel": self.sequence_parallel,
                "pipeline_dtype": self.pipeline_dtype,
            },
        )
        if isinstance(model, list):
            model = model[0]
        if hasattr(model, "module"):
            model = model.module

        return model

    def _select_checkpoint_dir(self, base_dir: str, checkpoint_step) -> str:
        """
        Resolve checkpoint directory:
        - If checkpoint_step is provided, use base_dir/iter_{step:07d}
        - Otherwise, pick the largest iter_######## subdirectory under base_dir
        """
        if checkpoint_step is not None:
            path = os.path.join(base_dir, f"iter_{int(checkpoint_step):07d}")
            if os.path.isdir(path):
                logging.info(f"Using specified checkpoint: {path}")
                return path
            raise FileNotFoundError(f"Specified checkpoint step {checkpoint_step} not found at {path}")

        if not os.path.isdir(base_dir):
            raise FileNotFoundError(f"Checkpoint base directory does not exist: {base_dir}")

        pattern = re.compile(r"^iter_(\d+)$")
        try:
            _, latest_path = max(
                ((int(pattern.match(e.name).group(1)), e.path)
                 for e in os.scandir(base_dir)
                 if e.is_dir() and pattern.match(e.name)),
                key=lambda x: x[0],
            )
        except ValueError:
            raise FileNotFoundError(
                f"No checkpoints found under {base_dir}. Expected subdirectories named like 'iter_0001800'.")

        logging.info(f"Auto-selected latest checkpoint: {latest_path}")
        return latest_path


    def forward_pp_step(
        self,
        latent_model_input: torch.Tensor,
        grid_sizes: list[Tuple[int, int, int]],
        max_video_seq_len: int,
        timestep: torch.Tensor,
        arg_c: dict,        
    ) -> torch.Tensor:
        """
        Forward pass supporting pipeline parallelism.
        """

        from megatron.core import parallel_state
        from megatron.core.inference.communication_utils import broadcast_from_last_pipeline_stage, recv_from_prev_pipeline_rank_, send_to_next_pipeline_rank

        pp_world_size = parallel_state.get_pipeline_model_parallel_world_size()
        is_pp_first = parallel_state.is_pipeline_first_stage(ignore_virtual=True)
        is_pp_last = parallel_state.is_pipeline_last_stage(ignore_virtual=True)

        # PP=1: no pipeline parallelism
        if pp_world_size == 1:
            noise_pred_pp = self.model(
                latent_model_input,
                grid_sizes=grid_sizes,
                t=timestep,
                **arg_c)
            return noise_pred_pp

        # PP>1: pipeline parallelism
        hidden_size = self.model.config.hidden_size
        batch_size = latent_model_input.shape[1]
        # noise prediction shape for communication between first and last pipeline stages
        noise_pred_pp_shape = list(latent_model_input.shape)

        if is_pp_first:
            # First stage: compute multimodal + first PP slice, send activations, then receive sampled token
            hidden_states = self.model(
                latent_model_input,
                grid_sizes=grid_sizes,
                t=timestep,
                **arg_c)
            send_to_next_pipeline_rank(hidden_states)

            noise_pred_pp = broadcast_from_last_pipeline_stage(noise_pred_pp_shape, dtype=torch.float32)
            return noise_pred_pp

        if is_pp_last:
            # Last stage: recv activations, run final slice + output, sample, broadcast
            recv_buffer = torch.empty(
                (max_video_seq_len, batch_size, hidden_size),
                dtype=next(self.model.parameters()).dtype,
                device=latent_model_input[0].device,
            )
            recv_from_prev_pipeline_rank_(recv_buffer)
            recv_buffer = recv_buffer.to(torch.bfloat16) # ????
            self.model.set_input_tensor(recv_buffer)
            noise_pred_pp = self.model(
                latent_model_input,
                grid_sizes=grid_sizes,
                t=timestep,
                **arg_c)

            noise_pred_pp = broadcast_from_last_pipeline_stage(noise_pred_pp_shape, dtype=noise_pred_pp.dtype, tensor=noise_pred_pp.contiguous())
            return noise_pred_pp

        # Intermediate stages: recv -> run local slice -> send -> receive broadcast token
        recv_buffer = torch.empty(
            (max_video_seq_len, batch_size, hidden_size),
            dtype=next(self.model.parameters()).dtype,
            device=latent_model_input[0].device,
        )
        recv_from_prev_pipeline_rank_(recv_buffer)
        recv_buffer = recv_buffer.to(torch.bfloat16) # ????
        self.model.set_input_tensor(recv_buffer)
        hidden_states = self.model(
            latent_model_input,
            grid_sizes=grid_sizes,
            t=timestep,
            **arg_c)
        send_to_next_pipeline_rank(hidden_states)

        noise_pred_pp = broadcast_from_last_pipeline_stage(noise_pred_pp_shape, dtype=torch.float32)
        return noise_pred_pp


    def generate(self,
                 prompts,
                 sizes,
                 frame_nums,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=50,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True):
        r"""
        Generates video frames from text prompt using diffusion process.

        Args:
            prompts (`list[str]`):
                Text prompt for content generation
            sizes (list[tuple[int, int]]):
                Controls video resolution, (width,height).
            frame_nums (`list[int]`):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 40):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float`, *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed.
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from size)
                - W: Frame width from size)
        """
    
        # preprocess
        target_shapes = []
        for size, frame_num in zip(sizes, frame_nums):
            target_shapes.append((self.vae.model.z_dim, (frame_num - 1) // self.vae_stride[0] + 1,
                                size[1] // self.vae_stride[1],
                                size[0] // self.vae_stride[2]))

        max_video_seq_len = 0
        seq_lens = []
        for target_shape in target_shapes:
            seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                                (self.patch_size[1] * self.patch_size[2]) *
                                target_shape[1] / self.sp_size) * self.sp_size
            seq_lens.append(seq_len)
        max_video_seq_len = max(seq_lens)

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)


        ## process context
        context_max_len = 512
        context_lens = []
        contexts = []
        contexts_null = []
        for prompt in prompts:
            if not self.t5_cpu:
                self.text_encoder.model.to(self.device)
                context = self.text_encoder([prompt], self.device)[0]
                context_null = self.text_encoder([n_prompt], self.device)[0]
                if offload_model:
                    self.text_encoder.model.cpu()
            else:
                context = self.text_encoder([prompt], torch.device('cpu'))[0].to(self.device)
                context_null = self.text_encoder([n_prompt], torch.device('cpu'))[0].to(self.device)
            context_lens.append(context_max_len) # all samples have the same context_max_len
            contexts.append(context)
            contexts_null.append(context_null)
        # pad to context_max_len tokens, and stack to a tensor of shape [s, b, hidden]
        contexts = [F.pad(context, (0, 0, 0, context_max_len - context.shape[0])) for context in contexts]
        contexts_null = [F.pad(context_null, (0, 0, 0, context_max_len - context_null.shape[0])) for context_null in contexts_null]
        contexts = torch.stack(contexts, dim=1)
        contexts_null = torch.stack(contexts_null, dim=1)


        ## setup noise
        noises = []
        for target_shape in target_shapes:
            noises.append(
                torch.randn(
                    target_shape[0],
                    target_shape[1],
                    target_shape[2],
                    target_shape[3],
                    dtype=torch.float32,
                    device=self.device,
                    generator=seed_g)
            )


        # calculate grid_sizes
        grid_sizes = [grid_sizes_calculation(
            input_shape =u.shape[1:], 
            patch_size=self.model.patch_size,
            ) for u in noises]
        grid_sizes = torch.tensor(grid_sizes, dtype=torch.long)


        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)

        # evaluation mode
        with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():

            if sample_solver == 'unipc':
                # Create a prototype scheduler to compute shared timesteps
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(
                    sampling_steps, device=self.device, shift=shift)
                timesteps = sample_scheduler.timesteps

                # Instantiate per-sample schedulers so each sample maintains its own state
                batch_size_for_schedulers = len(noises)
                schedulers = []
                for _ in range(batch_size_for_schedulers):
                    s = FlowUniPCMultistepScheduler(
                        num_train_timesteps=self.num_train_timesteps,
                        shift=1,
                        use_dynamic_shifting=False)
                    s.set_timesteps(sampling_steps, device=self.device, shift=shift)
                    schedulers.append(s)
            elif sample_solver == 'dpm++':
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(
                    sample_scheduler,
                    device=self.device,
                    sigmas=sampling_sigmas)
            else:
                raise NotImplementedError("Unsupported solver.")

            # sample videos
            latents = noises

            from megatron.core.packed_seq_params import PackedSeqParams
            cu_q = torch.cat([torch.tensor([0]), torch.cumsum(torch.tensor(seq_lens), dim=0)])
            cu_q = cu_q.to(torch.int32).to(self.device)
            cu_kv_self = cu_q
            cu_kv_cross = torch.cat([torch.tensor([0]), torch.cumsum(torch.tensor(context_lens), dim=0)])
            cu_kv_cross = cu_kv_cross.to(torch.int32).to(self.device)
            packed_seq_params = {
                "self_attention": PackedSeqParams(
                    cu_seqlens_q=cu_q,
                    cu_seqlens_kv=cu_kv_self,
                    qkv_format=self.model.config.qkv_format,
                ),
                "cross_attention": PackedSeqParams(
                    cu_seqlens_q=cu_q,
                    cu_seqlens_kv=cu_kv_cross,
                    qkv_format=self.model.config.qkv_format,
                ),
            }
            

            arg_c = {'context': contexts, 'max_seq_len': max_video_seq_len, 'packed_seq_params': packed_seq_params}
            arg_null = {'context': contexts_null, 'max_seq_len': max_video_seq_len, 'packed_seq_params': packed_seq_params}

            for _, t in enumerate(tqdm(timesteps)):

                batch_size = len(latents)

                # patchify latents
                unpatchified_latents = latents
                latents = patchify(latents, self.patch_size)
                # pad to have same length
                for i in range(batch_size):
                    latents[i] = F.pad(latents[i], (0, 0, 0, max_video_seq_len - latents[i].shape[0]))
                latents = torch.stack(latents, dim=1)


                latent_model_input = latents
                timestep = [t] * batch_size
                timestep = torch.stack(timestep)

                # DEBUGGING thd
                # # run context parallelism slitting
                # if parallel_state.get_context_parallel_world_size() > 1:
                #     latent_model_input = split_inputs_cp(latent_model_input, 0)
                #     arg_c['context'] = split_inputs_cp(arg_c['context'], 0)
                #     arg_null['context'] = split_inputs_cp(arg_null['context'], 0)

                self.model.to(self.device)
                noise_pred_cond = self.forward_pp_step(
                    latent_model_input, grid_sizes=grid_sizes, max_video_seq_len=max_video_seq_len, timestep=timestep, arg_c=arg_c)

                noise_pred_uncond = self.forward_pp_step(
                    latent_model_input, grid_sizes=grid_sizes, max_video_seq_len=max_video_seq_len, timestep=timestep, arg_c=arg_null)

                # DEBUGGING thd
                # # run context parallelism gathering
                # if parallel_state.get_context_parallel_world_size() > 1:
                #     arg_c['context'] = cat_outputs_cp(arg_c['context'], 0) # we need to cat the context back together for the next timestep
                #     arg_null['context'] = cat_outputs_cp(arg_null['context'], 0) # we need to cat the context back together for the next timestep
                #     # TODO: does this step slow down speed???
                #     noise_pred_cond = noise_pred_cond.contiguous()
                #     noise_pred_uncond = noise_pred_uncond.contiguous()
                #     noise_pred_cond = cat_outputs_cp(noise_pred_cond, 0)
                #     noise_pred_uncond = cat_outputs_cp(noise_pred_uncond, 0)

                # run unpatchify
                unpatchified_noise_pred_cond = noise_pred_cond
                unpatchified_noise_pred_cond = unpatchified_noise_pred_cond.transpose(0, 1) # bring sbhd -> bshd
                # when unpatchifying, the code will truncate the padded videos into the original video shape, based on the grid_sizes.
                unpatchified_noise_pred_cond = self.unpatchify(unpatchified_noise_pred_cond, grid_sizes, self.vae.model.z_dim)
                unpatchified_noise_pred_uncond = noise_pred_uncond
                unpatchified_noise_pred_uncond = unpatchified_noise_pred_uncond.transpose(0, 1) # bring sbhd -> bshd
                # when unpatchifying, the code will truncate the padded videos into the original video shape, based on the grid_sizes.
                unpatchified_noise_pred_uncond = self.unpatchify(unpatchified_noise_pred_uncond, grid_sizes, self.vae.model.z_dim)

                noise_preds = []
                for i in range(batch_size):
                    noise_pred = unpatchified_noise_pred_uncond[i] + guide_scale * (
                        unpatchified_noise_pred_cond[i] - unpatchified_noise_pred_uncond[i])
                    noise_preds.append(noise_pred)

                # step and update latents
                latents = []
                for i in range(batch_size):

                    if sample_solver == 'unipc':
                        temp_x0 = schedulers[i].step(
                            noise_preds[i].unsqueeze(0),
                            t,
                            unpatchified_latents[i].unsqueeze(0),
                            return_dict=False,
                            generator=seed_g)[0]
                    else:
                        temp_x0 = sample_scheduler.step(
                            noise_preds[i].unsqueeze(0),
                            t,
                            unpatchified_latents[i].unsqueeze(0),
                            return_dict=False,
                            generator=seed_g)[0]
                    latents.append(temp_x0.squeeze(0))

            x0 = latents
            if offload_model:
                self.model.cpu()
                torch.cuda.empty_cache()
            if self.rank == 0:
                videos = self.vae.decode(x0)
            else:
                videos = None

        del noises, latents
        if sample_solver == 'unipc':
            del schedulers
        else:
            del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        return videos if self.rank == 0 else None
