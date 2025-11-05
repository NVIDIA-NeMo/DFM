# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from nemo.lightning.megatron_parallel import MegatronParallel
from transformers import T5EncoderModel, T5TokenizerFast
from dfm.src.megatron.model.dit.edm.edm_pipeline import EDMPipeline
from nemo.collections.common.video_tokenizers.cosmos_tokenizer import CausalVideoTokenizer
from dfm.src.megatron.model.dit.dit_model_provider import DiTModelProvider
from dfm.src.common.utils.save_video import save_video
from nemo_vfm.diffusion.utils.mcore_parallel_utils import Utils
from einops import rearrange
import argparse
import numpy as np
import torch


MegatronParallel.init_ddp = lambda self: None

EXAMPLE_PROMPT = (
    "The teal robot is cooking food in a kitchen. Steam rises from a simmering pot "
    "as the robot chops vegetables on a worn wooden cutting board. Copper pans hang "
    "from an overhead rack, catching glints of afternoon light, while a well-loved "
    "cast iron skillet sits on the stovetop next to scattered measuring spoons and "
    "a half-empty bottle of olive oil."
)


def parse_args():
    parser = argparse.ArgumentParser(description="Video foundation model inference")
    parser.add_argument(
        "--prompt",
        type=str,
        default=EXAMPLE_PROMPT,
        help="Prompt which the sampled video condition on",
    )
    # We turn on negative prompt by default. set to "" to turn it off.
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=None,
        help="Negative prompt which the sampled video condition on",
    )
    parser.add_argument("--subject_name", type=str, default="", help="Name of fine-tuned subject")
    parser.add_argument("--guidance", type=float, default=7, help="Classifier-free guidance scale")
    parser.add_argument("--sampler", type=str, default="RES", help="Currently only supports RES sampler.")
    parser.add_argument("--video_save_path", type=str, default="outputs", help="Path to save the video")
    parser.add_argument("--fps", type=int, default=24, help="FPS of the sampled video")
    parser.add_argument("--height", type=int, default=704, help="Height of image to sample")
    parser.add_argument("--width", type=int, default=1280, help="Width of image to sample")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--num_devices", type=int, default=1, help="Number of devices for inference")
    parser.add_argument("--cp_size", type=int, default=1, help="Number of cp ranks for multi-gpu inference.")
    parser.add_argument("--num_steps", type=float, default=35, help="Number of diffusion sampling steps")
    parser.add_argument("--num_video_frames", type=int, default=121, help="Number of video frames to sample")
    parser.add_argument("--tokenizer_model", type=str, default="nvidia/Cosmos-1.0-Tokenizer-CV8x8x8", help="Mode of video tokenizer")
    parser.add_argument("--tokenizer_dir", type=str, default="", help="Directory for video tokenizer")
    parser.add_argument("--cosmos_assets_dir", type=str, default="", help="Directory containing cosmos assets")
    parser.add_argument("--guardrail_dir", type=str, default="", help="Guardrails weights directory")
    parser.add_argument("--nemo_checkpoint", type=str, default="", help="Video diffusion model nemo weights")
    parser.add_argument("--t5_cache_dir", type=str, default=None, help="Path to T5 model")
    args = parser.parse_args()
    return args


def print_rank_0(string: str):
    rank = torch.distributed.get_rank()
    if rank == 0:
        print(string)


@torch.no_grad()
def encode_for_batch(tokenizer: T5TokenizerFast, encoder: T5EncoderModel, prompts: list[str], max_length: int = 512):
    """
    Encode a batch of text prompts to a batch of T5 embeddings.
    Parameters:
        tokenizer: T5 embedding tokenizer.
        encoder: T5 embedding text encoder.
        prompts: A batch of text prompts.
        max_length: Sequence length of text embedding (defaults to 512).
    """

    batch_encoding = tokenizer.batch_encode_plus(
        prompts,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_length=True,
        return_offsets_mapping=False,
    )

    # We expect all the processing is done on GPU.
    input_ids = batch_encoding.input_ids.cuda()
    attn_mask = batch_encoding.attention_mask.cuda()

    outputs = encoder(input_ids=input_ids, attention_mask=attn_mask)
    encoded_text = outputs.last_hidden_state

    lengths = attn_mask.sum(dim=1).cpu()
    for batch_id in range(encoded_text.shape[0]):
        encoded_text[batch_id][lengths[batch_id] :] = 0

    return encoded_text


class PosID3D:
    def __init__(self, *, max_t=32, max_h=128, max_w=128):
        self.max_t = max_t
        self.max_h = max_h
        self.max_w = max_w
        self.generate_pos_id()

    def generate_pos_id(self):
        self.grid = torch.stack(
            torch.meshgrid(
                torch.arange(self.max_t, device="cpu"),
                torch.arange(self.max_h, device="cpu"),
                torch.arange(self.max_w, device="cpu"),
            ),
            dim=-1,
        )

    def get_pos_id_3d(self, *, t, h, w):
        if t > self.max_t or h > self.max_h or w > self.max_w:
            self.max_t = max(self.max_t, t)
            self.max_h = max(self.max_h, h)
            self.max_w = max(self.max_w, w)
            self.generate_pos_id()
        return self.grid[:t, :h, :w]


def prepare_data_batch(args, t5_embeding_max_length=512):
    tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-11b", cache_dir=args.t5_cache_dir)
    text_encoder = T5EncoderModel.from_pretrained("google-t5/t5-11b", cache_dir=args.t5_cache_dir)
    text_encoder.to("cuda")
    text_encoder.eval()

    print('[args.prompt]: ', args.prompt)
    # Encode text to T5 embedding
    out = encode_for_batch(tokenizer, text_encoder, args.prompt.split(','))
    encoded_text = torch.tensor(out, dtype=torch.bfloat16)
    B, L, C = encoded_text.shape
    t5_embed = torch.zeros(B, t5_embeding_max_length, C, dtype=torch.bfloat16)
    t5_embed[:, :L, :] = encoded_text
    neg_t5_embed = None
    t, h, w = args.num_video_frames, args.height, args.width
    pt, ph, pw = 1, 2, 2
    state_shape = [
        B, # batch dimension
        ((h // 8) // ph) * ((w // 8) // pw) * 1, # number of tokens: (h //8) * (w // 8) * 1 -> ((h // 8) // ph) * ((w // 8) // pw) * 1
        16 * (ph*pw*pt) # token hidden size (channel * patch_spatial * patch_spatial * patch_temporal)
    ]
    # prepare pos_emb
    pos_id_3d = PosID3D()
    pt, ph, pw = 1, 2, 2
    pos_ids = rearrange(
        # pos_id_3d.get_pos_id_3d(t=t // 4, h=h // 8, w=w // 8),
        pos_id_3d.get_pos_id_3d(t=1, h=(h // 8) // ph, w=(w // 8) // pw),
        "T H W d -> T (H W) d",
    )
    data_batch = {
        "video": torch.zeros((B, 3, t, h, w), dtype=torch.uint8).cuda(),
        "t5_text_embeddings": t5_embed,
        "t5_text_mask": torch.ones(B, t5_embeding_max_length, dtype=torch.bfloat16).cuda(),
        "image_size": torch.tensor(
            [[args.height, args.width, args.height, args.width]] * B, dtype=torch.bfloat16
        ).cuda(),
        "fps": torch.tensor([[args.fps]] * B, dtype=torch.bfloat16).cuda(),
        "num_frames": torch.tensor([[args.num_video_frames]] * B, dtype=torch.bfloat16).cuda(),
        "padding_mask": torch.zeros((B, 1, args.height, args.width), dtype=torch.bfloat16).cuda(),
        "pos_ids": pos_ids,
        'latent_shape': [16, t//pt, h//8//ph, w//8//pw],
    }
    return data_batch, state_shape


def setup_diffusion_pipeline(args):
    """
    Initialize DiT model, parallel strategy, and diffusion pipeline for inference.
    """
    model_config = DiTModelProvider()
    model = model_config.provide()
    model = model.cuda().to(torch.bfloat16)
    diffusion_pipeline = EDMPipeline(seed=args.seed)
    diffusion_pipeline.net = model
    return model, diffusion_pipeline, model_config
    

def data_preprocess(data_batch, state_shape):
    from dfm.src.megatron.model.dit.dit_data_process import encode_seq_length
    data_batch = {k: v.cuda() if torch.is_tensor(v) else v for k, v in data_batch.items()}
    data_batch["inference_fwd"] = True

    data_batch['seq_len_q'] = torch.tensor([state_shape[1]] * state_shape[0]).cuda()
    data_batch['seq_len_kv'] = torch.tensor([data_batch['t5_text_embeddings'].shape[1]] * state_shape[0]).cuda()
    data_batch = encode_seq_length(data_batch, format="sbhd")
    return data_batch


def main(args):
    # Initialize distributed environment and model parallel groups
    Utils.initialize_distributed(1, 1, context_parallel_size=args.cp_size)
    model_parallel_cuda_manual_seed(args.seed)

    # Setup model / diffusion pipeline
    print_rank_0("setting up diffusion pipeline...")
    import random

    model_parallel_cuda_manual_seed(args.seed)
    def set_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        # For deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    set_seed(42)
    model, diffusion_pipeline, model_config = setup_diffusion_pipeline(args)

    new_state = {}
    print("loading model....")
    state = torch.load('model.pth')
    for key, value in state.items():
        if 'extra_state' in key:
            continue
        new_state[key.replace('0.module.', '')] = value
    model.load_state_dict(new_state, strict=False)

    print_rank_0("preparing data batch...")
    data_batch, state_shape = prepare_data_batch(args)
    vae = CausalVideoTokenizer.from_pretrained("Cosmos-0.1-Tokenizer-CV4x8x8")
    vae.to("cuda")

    print_rank_0("generating video...")
    data_batch = data_preprocess(data_batch, state_shape)
    C, T, H, W = data_batch["latent_shape"]
    latent = diffusion_pipeline.generate_samples_from_batch(
        data_batch,
        guidance=args.guidance,
        state_shape=state_shape,
        num_steps=args.num_steps,
        is_negative_prompt=True if "neg_t5_text_embeddings" in data_batch else False,
    )
    rank = torch.distributed.get_rank()
    latent = latent[0, None, :state_shape[1]]
    latent = rearrange(
            latent,
            'b (T H W) (ph pw pt c) -> b c (T pt) (H ph) (W pw)',
            ph=model_config.patch_spatial, pw=model_config.patch_spatial,
            pt=model_config.patch_temporal,
            c=C, T=T, H=H, W=W,
    )
    decoded_video = (1.0 + vae.decode(latent / model_config.sigma_data)).clamp(0, 2) / 2
    decoded_video = (decoded_video * 255).to(torch.uint8).permute(0, 2, 3, 4, 1).cpu().numpy()
    for i in range(len(decoded_video)):
        save_video(
            grid=decoded_video[i],
            fps=args.fps,
            H=args.height,
            W=args.width,
            video_save_quality=5,
            video_save_path=f'idx={i}_rank={rank}_' + args.video_save_path,
        )
        print_rank_0(f"saved video to idx={i}_rank={rank}_{args.video_save_path}")

if __name__ == "__main__":
    args = parse_args()
    main(args)

