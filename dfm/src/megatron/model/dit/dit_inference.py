import torch
from dfm.src.common.utils.decode_cosmos_latent import decode_cosmos_latent
from dfm.src.common.utils.save_video import save_video

def print_rank_0(string: str):
    rank = torch.distributed.get_rank()
    if rank == 0:
        print(string)

def run_diffusion_inference(diffusion_pipeline, args, data_batch, state_shape, vae):
    latent = diffusion_pipeline.generate_samples_from_batch(
        data_batch,
        guidance=args.guidance,
        state_shape=state_shape,
        num_steps=args.num_steps,
        is_negative_prompt=True if "neg_t5_text_embeddings" in data_batch else False,
    )
    rank = torch.distributed.get_rank()
    latent = latent[:, :1536]
    decoded_video = decode_cosmos_latent(latent, args.height, args.width, args.num_video_frames, vae)
    for i in range(len(decoded_video)):
        save_video(
            grid=decoded_video[i],
            fps=args.fps,
            H=args.height,
            W=args.width,
            video_save_quality=5,
            video_save_path=args.video_save_path + f'_{i}_rank_{rank}.mp4',
        )
        print_rank_0(f"saved video to {args.video_save_path}_{i}_rank_{rank}.mp4!")