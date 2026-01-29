"""Core layer implementations."""

import torch
from beartype import beartype
from jaxtyping import Bool, Float, jaxtyped
from torch import nn
import time
import collections

class LatencyTracker:
    def __init__(self, interval=5000):
        self.buffer = []
        self.last_report = time.time()
        self.interval = interval

    def update(self, msg, now=None):
        self.buffer.append(msg)
        
        if now is None:
            now = time.time()
            
        if now - self.last_report > self.interval:
            self.report(now)

    def report(self, now):
        # Only print on rank 0
        if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
            self.buffer = []
            return
            
        if self.buffer:
            print("\n" + "\n".join(self.buffer) + "\n")
        
        self.buffer = []
        self.last_report = now

latency_tracker = LatencyTracker()

def l2_normalize(
    x: Float[torch.Tensor, "... dims"], eps: float = 1e-6
) -> Float[torch.Tensor, "... dims"]:
    dtype = x.dtype
    compute_dtype = torch.float32
    norm = torch.norm(x, p=2, dim=-1, keepdim=True, dtype=compute_dtype)
    normalized = x / (norm + eps)
    return normalized.to(dtype)


class RMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.dims = dims

    def forward(
        self, x: Float[torch.Tensor, "... dims"]
    ) -> Float[torch.Tensor, "... dims"]:
        input_dtype = x.dtype
        compute_dtype = torch.float32
        x = x.to(compute_dtype)
        x = torch.nn.functional.rms_norm(x, (self.dims,), eps=self.eps)
        return x.to(input_dtype)


class Modulation(nn.Module):
    def __init__(self, dims: int):
        super().__init__()
        self.lin = nn.Linear(dims, dims, bias=True)
        nn.init.constant_(self.lin.bias, 0)
        nn.init.constant_(self.lin.weight, 0)
        self.act = nn.SiLU()

    def forward(
        self, vec: Float[torch.Tensor, "... dims"]
    ) -> Float[torch.Tensor, "... dims"]:
        scale = self.lin(nn.functional.silu(vec))
        return scale + 1.0


class GateResiduals(nn.Module):
    def __init__(self, dims: int, do_modulation: bool, epsilon: float = 2e-2) -> None:
        super().__init__()
        if do_modulation:
            self.modulation = Modulation(dims)
        else:
            self.gate = nn.Parameter(torch.empty(dims))
            nn.init.constant_(self.gate, 0.0)
        self.do_modulation = do_modulation
        self.norm = RMSNorm(dims)
        self.epsilon = epsilon

    def forward(
        self,
        backbone: Float[torch.Tensor, "bs seq_len dims"],
        residual: Float[torch.Tensor, "bs seq_len dims"],
        vec: Float[torch.Tensor, "bs dims"] | None,
    ) -> Float[torch.Tensor, "bs seq_len dims"]:

        if self.do_modulation:
            gate = self.modulation(vec) + 2.9
        else:
            gate = self.gate + 3.9
            gate = gate.unsqueeze(0)

        gate = torch.sigmoid(gate)
        gate = gate * (1 - 2 * self.epsilon) + self.epsilon

        normalized_residual = self.norm(residual)

        if gate.ndim == 2:
            gate = gate.unsqueeze(1)  # to broadcast with the sequence

        return backbone * gate + (1 - gate) * normalized_residual


class Attention(nn.Module):
    def __init__(
        self,
        dims_per_head: int,
        num_heads: int,
        scale: float,
        do_cross_attn: bool,
        do_modulation: bool,
        use_residual: bool = True,
    ):
        super().__init__()

        dims = dims_per_head * num_heads
        if do_modulation:
            self.mod = Modulation(dims)
        self.q = nn.Linear(dims, dims, bias=False)
        self.kv = nn.Linear(dims, dims * 2, bias=False)
        self.proj = nn.Linear(dims, dims, bias=False)
        self.scale = scale
        self.num_heads = num_heads
        self.do_modulation = do_modulation
        self.do_cross_attn = do_cross_attn

        from einops.layers.torch import Rearrange

        self.split_q = Rearrange("B L (H D) -> B H L D", H=self.num_heads)
        self.split_kv = Rearrange("B L (K H D) -> K B H L D", K=2, H=self.num_heads)
        self.combine_heads = Rearrange("B H L D -> B L (H D)")

        self.gate_residual = None
        if use_residual:
            self.gate_residual = GateResiduals(dims, do_modulation)

    def forward(
        self,
        x: Float[torch.Tensor, "bs seq_len dims"],
        cross: Float[torch.Tensor, "bs cross_seq_len dims"] | None,
        vec: Float[torch.Tensor, "bs dims"] | None,
        rope_cis: tuple[
            Float[torch.Tensor, "bs seq_len head_dims"],
            Float[torch.Tensor, "bs seq_len head_dims"],
        ]
        | None,
        mask: Bool[torch.Tensor, "bs seq_len"]
        | Bool[torch.Tensor, "bs cross_seq_len"]
        | Float[torch.Tensor, "bs seq_len cross_seq_len"]
        | None = None,
        cross_rope_cis: tuple[
            Float[torch.Tensor, "bs cross_seq_len head_dims"],
            Float[torch.Tensor, "bs cross_seq_len head_dims"],
        ]
        | None = None,
    ) -> Float[torch.Tensor, "bs seq_len dims"]:
        if mask is not None:
            # we want a mask with shape [bs, num_heads, seq_len, cross_seq_len]
            if mask.ndim == 2:
                # add a dimension for the input sequence
                mask = mask.unsqueeze(1)
            assert mask.ndim == 3, "mask should have shape [bs, seq_len, cross_seq_len]"
            # add a dimension for the multi heads
            mask = mask.unsqueeze(1)

        # DEBUGGING (benchmarking)
        attention_modulation_start_time = time.time()

        input = x
        if self.do_modulation:
            assert vec is not None, "vec should not be None if `do_modulation` is True"
            scale = self.mod(vec)
            if scale.ndim == 2:
                scale = scale.unsqueeze(1)  # to broadcast with the sequence
            x = x * scale
        else:
            assert vec is None, "vec should be None if `do_modulation` is False"

        # DEBUGGING (benchmarking)
        attention_modulation_end_time = time.time()
        latency_tracker.update(f"[DEBUG]             (reve_pytorch/layers.py) - attention modulation time: {attention_modulation_end_time - attention_modulation_start_time} seconds", now=attention_modulation_end_time)

        # DEBUGGING (benchmarking)
        attention_core_attention_start_time = time.time()

        q = self.q(x)
        q = self.split_q(q)
        q = l2_normalize(q)

        # Get key and values
        assert (cross is not None) == self.do_cross_attn, (
            "cross should be None if `do_cross_attn` is False"
        )

        kv = self.kv(cross) if self.do_cross_attn else self.kv(x)
        k, v = self.split_kv(kv)
        k = l2_normalize(k)

        # Rotary positional embedding
        if rope_cis is not None:
            q = apply_rotary_pos_emb(q, rope_cis[0], rope_cis[1])
            if self.do_cross_attn:
                assert cross_rope_cis is not None, (
                    "cross_rope_cis should not be None when using RoPE with `do_cross_attn == True`."
                )
                # cross attention case, use the cross rope_cis
                k = apply_rotary_pos_emb(k, cross_rope_cis[0], cross_rope_cis[1])
            else:
                # self attention case, use the same rope_cis as the query
                k = apply_rotary_pos_emb(k, rope_cis[0], rope_cis[1])
        else:
            assert cross_rope_cis is None, (
                "cross_rope_cis should be None when not using RoPE"
            )

        # # DEBUGGING (benchmarking)
        # # torch.cuda.synchronize()
        # # sdpa_start_time = time.time()
        # from torch.backends.cuda import sdp_kernel
        # start_event = torch.cuda.Event(enable_timing=True)
        # end_event = torch.cuda.Event(enable_timing=True)
        # torch.cuda.synchronize() # Clear the deck
        # start_event.record()


        # Use PyTorch's scaled_dot_product_attention
        # DEBUGGING (benchmarking)
        # force use of flash attention
        from torch.backends.cuda import sdp_kernel
        with sdp_kernel(
            enable_flash=True,
            enable_mem_efficient=False,
            enable_math=False,
        ):
            x = torch.nn.functional.scaled_dot_product_attention(
                query=q,
                key=k,
                value=v,
                attn_mask=mask,
                scale=self.scale,
            )

        # # DEBUGGING (benchmarking)
        # # torch.cuda.synchronize()
        # # sdpa_end_time = time.time()
        # # print(f"[DEBUG] (layers.py) SDPA time: {sdpa_end_time - sdpa_start_time} seconds")
        # end_event.record()
        # torch.cuda.synchronize()
        # sdpa_time_ms = start_event.elapsed_time(end_event)
        # print(f"[DEBUG] (layers.py) SDPA time: {sdpa_time_ms} milliseconds")

        x = self.combine_heads(x)
        x = self.proj(x)

        # DEBUGGING (benchmarking)
        attention_core_attention_end_time = time.time()
        latency_tracker.update(f"[DEBUG]             (reve_pytorch/layers.py) - attention core attention time: {attention_core_attention_end_time - attention_core_attention_start_time} seconds", now=attention_core_attention_end_time)

        # DEBUGGING (benchmarking)
        attention_gate_residuals_start_time = time.time()

        if self.gate_residual is not None:
            x = self.gate_residual(input, x, vec)

        # DEBUGGING (benchmarking)
        attention_gate_residuals_end_time = time.time()
        latency_tracker.update(f"[DEBUG]             (reve_pytorch/layers.py) - attention gate residuals time: {attention_gate_residuals_end_time - attention_gate_residuals_start_time} seconds", now=attention_gate_residuals_end_time)

        return x


class MLP(nn.Module):
    def __init__(self, dims: int, mlp_ratio: float, do_modulation: bool):
        super().__init__()
        mlp_hidden_dim = int(dims * mlp_ratio)
        self.lin1 = nn.Linear(dims, mlp_hidden_dim, bias=False)
        self.act = nn.SiLU()
        self.lin2 = nn.Linear(mlp_hidden_dim, dims, bias=False)

        if do_modulation:
            self.mod = Modulation(dims)

        self.gate_residual = GateResiduals(dims, do_modulation)
        self.do_modulation = do_modulation

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        x: Float[torch.Tensor, "... seq_len dims"],
        vec: Float[torch.Tensor, "... dims"] | None,
    ) -> Float[torch.Tensor, "... seq_len dims"]:

        # DEBUGGING (benchmarking)
        mlp_modulation_start_time = time.time()

        input = x
        if self.do_modulation:
            assert vec is not None
            scale = self.mod(vec)
            if scale.ndim == 2:
                scale = scale.unsqueeze(1)  # to broadcast with the sequence
            x = x * scale

        # DEBUGGING (benchmarking)
        mlp_modulation_end_time = time.time()
        latency_tracker.update(f"[DEBUG]                 (reve_pytorch/layers.py) - mlp modulation time: {mlp_modulation_end_time - mlp_modulation_start_time} seconds", now=mlp_modulation_end_time)

        # DEBUGGING (benchmarking)
        mlp_core_computation_start_time = time.time()

        x = self.lin1(x)
        x = self.act(x)
        x = self.lin2(x)

        # DEBUGGING (benchmarking)
        mlp_core_computation_end_time = time.time()
        latency_tracker.update(f"[DEBUG]                 (reve_pytorch/layers.py) - mlp core computation time: {mlp_core_computation_end_time - mlp_core_computation_start_time} seconds", now=mlp_core_computation_end_time)

        # DEBUGGING (benchmarking)
        mlp_gate_residuals_start_time = time.time()

        x = self.gate_residual(input, x, vec)

        # DEBUGGING (benchmarking)
        mlp_gate_residuals_end_time = time.time()
        latency_tracker.update(f"[DEBUG]                 (reve_pytorch/layers.py) - mlp gate residuals time: {mlp_gate_residuals_end_time - mlp_gate_residuals_start_time} seconds", now=mlp_gate_residuals_end_time)

        return x


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dims_per_head: int,
        num_heads: int,
        scale: float,
        do_cross_attn: bool,
        do_modulation: bool,
        mlp_ratio: float,
    ):
        super().__init__()
        self.self_attn = Attention(
            dims_per_head=dims_per_head,
            num_heads=num_heads,
            scale=scale,
            do_cross_attn=False,
            do_modulation=do_modulation,
        )

        if do_cross_attn:
            self.cross_attn = Attention(
                dims_per_head=dims_per_head,
                num_heads=num_heads,
                scale=scale,
                do_cross_attn=True,
                do_modulation=do_modulation,
            )
        self.do_cross_attn = do_cross_attn
        dims = dims_per_head * num_heads
        self.mlp = MLP(dims=dims, mlp_ratio=mlp_ratio, do_modulation=do_modulation)

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        x: Float[torch.Tensor, "... seq_len dims"],
        cross: Float[torch.Tensor, "bs cross_seq_len dims"] | None,
        vec: Float[torch.Tensor, "bs dims"] | None,
        rope_cis: tuple[
            Float[torch.Tensor, "bs seq_len head_dims"],
            Float[torch.Tensor, "bs seq_len head_dims"],
        ],
        mask: Bool[torch.Tensor, "bs seq_len"] | None = None,
        cross_mask: Bool[torch.Tensor, "bs cross_seq_len"]
        | Float[torch.Tensor, "bs seq_len cross_seq_len"]
        | None = None,
        cross_q_rope_cis: tuple[
            Float[torch.Tensor, "bs seq_len head_dims"],
            Float[torch.Tensor, "bs seq_len head_dims"],
        ]
        | None = None,
        cross_k_rope_cis: tuple[
            Float[torch.Tensor, "bs cross_seq_len head_dims"],
            Float[torch.Tensor, "bs cross_seq_len head_dims"],
        ]
        | None = None,
    ) -> Float[torch.Tensor, "bs seq_len dims"]:

        # DEBUGGING (benchmarking)
        full_layer_forward_start_time = time.time()

        # DEBUGGING (benchmarking)
        self_attention_start_time = time.time()

        x = self.self_attn(
            x=x,
            cross=None,
            vec=vec,
            rope_cis=rope_cis,
            mask=mask,
        )

        # DEBUGGING (benchmarking)
        self_attention_end_time = time.time()
        latency_tracker.update(f"[DEBUG]         (reve_pytorch/layers.py) - self attention time: {self_attention_end_time - self_attention_start_time} seconds", now=self_attention_end_time)

        # DEBUGGING (benchmarking)
        cross_attention_start_time = time.time()

        if self.do_cross_attn:
            x = self.cross_attn(
                x=x,
                cross=cross,
                vec=vec,
                rope_cis=cross_q_rope_cis,
                cross_rope_cis=cross_k_rope_cis,
                mask=cross_mask,
            )

        # DEBUGGING (benchmarking)
        cross_attention_end_time = time.time()
        latency_tracker.update(f"[DEBUG]         (reve_pytorch/layers.py) - cross attention time: {cross_attention_end_time - cross_attention_start_time} seconds", now=cross_attention_end_time)

        # DEBUGGING (benchmarking)
        mlp_start_time = time.time()

        x = self.mlp(x, vec)

        # DEBUGGING (benchmarking)
        mlp_end_time = time.time()
        latency_tracker.update(f"[DEBUG]         (reve_pytorch/layers.py) - mlp time: {mlp_end_time - mlp_start_time} seconds", now=mlp_end_time)

        # DEBUGGING (benchmarking)
        full_layer_forward_end_time = time.time()
        latency_tracker.update(f"[DEBUG]     (reve_pytorch/layers.py) - full layer forward time: {full_layer_forward_end_time - full_layer_forward_start_time} seconds", now=full_layer_forward_end_time)

        return x



# ============ rope embeddings ============

def cis_embed(
    x: Float[torch.Tensor, "..."], dims: int, max_wavelength: float, scale: float
) -> tuple[Float[torch.Tensor, "... dims"], Float[torch.Tensor, "... dims"]]:
    compute_dtype = torch.float64
    output_dtype = torch.float32

    a = torch.div(
        torch.tensor(1.0, device=x.device, dtype=compute_dtype),
        torch.tensor(max_wavelength, device=x.device, dtype=compute_dtype),
    )
    exp = torch.div(
        torch.arange(dims, dtype=compute_dtype, device=x.device),
        torch.tensor(dims, device=x.device, dtype=compute_dtype),
    )
    freqs = torch.pow(a, exp)
    x = torch.mul(
        x.to(dtype=compute_dtype),
        torch.tensor(scale, dtype=compute_dtype, device=x.device),
    )
    out = torch.einsum("...,d->...d", x, freqs)
    return torch.cos(out).to(dtype=output_dtype), torch.sin(out).to(dtype=output_dtype)


def _rotate_half(
    x: Float[torch.Tensor, "... dims"],
) -> Float[torch.Tensor, "... dims"]:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    x: Float[torch.Tensor, "bs heads seq_len dims"],
    cos: Float[torch.Tensor, "bs seq_len dims"],
    sin: Float[torch.Tensor, "bs seq_len dims"],
) -> Float[torch.Tensor, "bs heads seq_len dims"]:
    cos = cos[:, None]
    sin = sin[:, None]
    with torch.autocast(device_type="cuda", enabled=False):
        dtype = x.dtype
        x = x.float()
        x_embed = (x * cos) + (_rotate_half(x) * sin)
        return x_embed.to(dtype)


class RoPEEmbed(nn.Module):
    def __init__(
        self, dims_per_axis: list[int], max_wavelength: float, scale: float = 1.0
    ):
        super().__init__()

        for dims in dims_per_axis:
            assert dims % 2 == 0, "dims must be even for RoPE cos/sin embeddings"
        self.dims_per_axis = dims_per_axis

        assert max_wavelength > 0, "max_wavelength must be positive"
        self.max_wavelength = max_wavelength

        self.cis_embed = cis_embed
        self.scale = scale

    @jaxtyped(typechecker=beartype)
    def forward(
        self, pos_indices: Float[torch.Tensor, "... axis"]
    ) -> tuple[Float[torch.Tensor, "... dims"], Float[torch.Tensor, "... dims"]]:
        assert pos_indices.shape[-1] == len(self.dims_per_axis), (
            f"number of dims ({pos_indices.shape[-1]}) must match `dims_per_axis` ({len(self.dims_per_axis)})"
        )
        all_cos = []
        all_sin = []
        for axis, dims in enumerate(self.dims_per_axis):
            assert dims % 2 == 0, "dims must be even"
            pos_idx = pos_indices[..., axis]
            cos, sin = self.cis_embed(
                pos_idx, dims // 2, self.max_wavelength, self.scale
            )
            all_cos.append(cos)
            all_sin.append(sin)

        return torch.concat(all_cos * 2, dim=-1), torch.concat(all_sin * 2, dim=-1)

    @jaxtyped(typechecker=beartype)
    def forward_interleaved(
        self, pos_indices: Float[torch.Tensor, "... axis"]
    ) -> Float[torch.Tensor, "... dims"]:
        assert pos_indices.shape[-1] == len(self.dims_per_axis), (
            f"number of dims ({pos_indices.shape[-1]}) must match `dims_per_axis` ({len(self.dims_per_axis)})"
        )
        all_freqs = []
        compute_dtype = torch.float64
        output_dtype = torch.float32

        for axis, dims in enumerate(self.dims_per_axis):
            assert dims % 2 == 0, "dims must be even"
            pos_idx = pos_indices[..., axis]
            
            # Reimplementing frequency calculation from cis_embed to get angles directly
            half_dims = dims // 2
            a = torch.div(
                torch.tensor(1.0, device=pos_idx.device, dtype=compute_dtype),
                torch.tensor(self.max_wavelength, device=pos_idx.device, dtype=compute_dtype),
            )
            exp = torch.div(
                torch.arange(half_dims, dtype=compute_dtype, device=pos_idx.device),
                torch.tensor(half_dims, device=pos_idx.device, dtype=compute_dtype),
            )
            base_freqs = torch.pow(a, exp)
            
            # Apply scale to pos_idx before multiplying
            scaled_pos = torch.mul(
                pos_idx.to(dtype=compute_dtype),
                torch.tensor(self.scale, dtype=compute_dtype, device=pos_idx.device),
            )
            
            # Calculate angles: pos * base_freqs
            angles = torch.einsum("...,d->...d", scaled_pos, base_freqs)
            all_freqs.append(angles.to(dtype=output_dtype))

        full_freqs = torch.concat(all_freqs, dim=-1)
        # Interleave frequencies: [f1, f1, f2, f2, ...]
        return torch.repeat_interleave(full_freqs, 2, dim=-1)

# ============ others ============

class CosineEmbed(nn.Module):
    def __init__(self, dims: int, max_wavelength: float, scale: float):
        super().__init__()
        assert dims % 2 == 0
        self.dims = dims
        self.max_wavelength = max_wavelength
        self.scale = scale

    @jaxtyped(typechecker=beartype)
    def forward(self, x: Float[torch.Tensor, "..."]) -> Float[torch.Tensor, "... dims"]:
        cos, sin = cis_embed(x, self.dims // 2, self.max_wavelength, self.scale)
        return torch.concat([cos, sin], dim=-1)


class MLPEmbed(nn.Module):
    def __init__(self, in_dims: int, out_dims: int, apply_rms_norm: bool = False):
        super().__init__()
        self.in_layer = nn.Linear(in_dims, out_dims, bias=True)
        self.act = nn.SiLU()
        self.out_layer = nn.Linear(out_dims, out_dims, bias=True)
        self.apply_rms_norm = apply_rms_norm
        if self.apply_rms_norm:
            self.norm = RMSNorm(out_dims)

    @jaxtyped(typechecker=beartype)
    def forward(
        self, x: Float[torch.Tensor, "... in_dims"]
    ) -> Float[torch.Tensor, "... out_dims"]:
        with torch.autocast(device_type="cuda", enabled=False):
            out = self.out_layer(self.act(self.in_layer(x)))
            if self.apply_rms_norm:
                out = self.norm(out)
        return out


class PadToken(nn.Module):
    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        x: Float[torch.Tensor, "bs seq_len dims"],
        mask: Bool[torch.Tensor, "bs seq_len"] | None,
    ) -> tuple[
        Float[torch.Tensor, "bs seq_len dims"], Bool[torch.Tensor, "bs seq_len"] | None
    ]:
        if mask is None:
            return x, None

        bs, length = mask.shape
        if length == 0:
            raise ValueError("sequence length must be greater than 0")

        # Replace padded tokens with -1
        x = x.where(mask[..., None], -1)

        return x, mask

