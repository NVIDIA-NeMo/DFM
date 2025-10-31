import torch
from torch.cuda import amp
from dfm.src.megatron.model.wan.utils.utils import split_inputs_cp
from megatron.core import parallel_state

class Wan3DRopeEmbeddings(torch.nn.Module):
    """
    Wan 3D RoPE embeddings implementation.
    Implements Wan's 3D RoPE embeddings for Mcore Attention based on Wan's implementation at https://github.com/Wan-Video/Wan2.1/blob/main/wan/modules/model.py.
    """

    def __init__(self, dim_head, max_position_len):
        super().__init__()
        self.freqs = torch.cat([
            self.rope_params(max_position_len, dim_head - 4 * (dim_head // 6)),
            self.rope_params(max_position_len, 2 * (dim_head // 6)),
            self.rope_params(max_position_len, 2 * (dim_head // 6))
        ], dim=1)

    def rope_params(self, max_position_len, dim_head, theta=10000):
        assert dim_head % 2 == 0
        freqs = torch.outer(
            torch.arange(max_position_len),
            1.0 / torch.pow(theta,
                            torch.arange(0, dim_head, 2).div(dim_head)))
        return freqs

    def forward(self, n_head, dim_head, max_seq_len, grid_sizes, device):
        self.freqs = self.freqs.to(device) # ??? do we need to put this here, or the when we move WanModel to device, it also move freqs to device?

        n, c = n_head, dim_head // 2

        # split freqs
        freqs = self.freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

        freqs_real = []
        for i, (f, h, w) in enumerate(grid_sizes.tolist()):
            seq_len = f * h * w
            freqs_real_i = torch.cat([
                freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
            ], dim=-1).reshape(seq_len, 1, 1, -1)  # <-- add 1,1 for batch/head broadcasting

            # Double dimension from c -> 2c with rotating angles as (x0, x0, x1, x1, ...), for interleaving RoPE
            freqs_real_i = freqs_real_i.unsqueeze(-1).expand(-1, -1, -1, -1, 2).reshape(seq_len, 1, 1, dim_head)

            # Pad freqs_real_i to (max_seq_len, 1, 1, dim_head) with 0s
            if freqs_real_i.shape[0] < max_seq_len:
                pad_shape = (max_seq_len - freqs_real_i.shape[0], 1, 1, dim_head)
                freqs_real_i = torch.cat(
                    [freqs_real_i, torch.zeros(pad_shape, dtype=freqs_real_i.dtype, device=freqs_real_i.device)]
                )
            freqs_real.append(freqs_real_i)

        # Each freqs_real[i] is (max_seq_len, 1, 1, dim_head)
        # We concatenate them along dim=1 to get (max_seq_len, batch_size, 1, dim_head)
        freqs_real = torch.cat(freqs_real, dim=1)

        # DEBUGGING thd
        # # TODO: if run context/sequence related parallel, then we need to scatter 
        # # the freqs_real to the context parallel region, using specific cp_rank split method
        # if parallel_state.get_context_parallel_world_size() > 1:
        #     freqs_real = split_inputs_cp(freqs_real, 0)

        return freqs_real