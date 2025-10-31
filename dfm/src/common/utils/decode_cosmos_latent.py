import torch

def decode_cosmos_latent(latent, height, width, num_video_frames, vae):
    print("[nemo_vfm/diffusion/inference.py] sample.shape before rearrange: ", latent.shape)
    ph, pw, pt, C = 2, 2, 1, 16  # D must be ph*pw*pt*C == 64
    # Target latent grid from your desired output video spec
    H_img, W_img, F = height, width, num_video_frames
    H = (H_img // 8) // ph
    W = (W_img // 8) // pw
    T = 1

    print('latent shape: ', latent.shape)
    from einops import rearrange
    latent = rearrange(
        latent,
        'b (T H W) (ph pw pt c) -> b c (T pt) (H ph) (W pw)',
        ph=ph, pw=pw, pt=pt, c=C,
        T=T, H=H, W=W,
    )
    print("[nemo_vfm/diffusion/inference.py] sample.shape after rearrange: ", latent.shape)
    sigma_data = 0.5
    decoded_video = (1.0 + vae.decode(latent / sigma_data)).clamp(0, 2) / 2
    decoded_video = (decoded_video * 255).to(torch.uint8).permute(0, 2, 3, 4, 1).cpu().numpy()
    return decoded_video

   