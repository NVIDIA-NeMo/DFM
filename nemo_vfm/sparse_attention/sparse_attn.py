import flash_attn.flash_attn_interface
import torch.nn.functional as F


original_flash_attn_varlen_forward = flash_attn.flash_attn_interface._flash_attn_varlen_forward
original_flash_attn_forward = flash_attn.flash_attn_interface._flash_attn_forward
original_sdpa = F.scaled_dot_product_attention


def flash_attn_func(q, k, v, *args, **kwargs):
    if q.shape != k.shape or q.shape != v.shape:
        return original_flash_attn_forward(q, k, v, *args, **kwargs)
    kwargs["should_rearrange"] = True
    o = invoke_sparse_layer(q, k, v, *args, **kwargs)
    if o is None:
        return original_flash_attn_forward(q, k, v, *args, **kwargs)
    return (o, None, None, None, o, q[:, :, :, 0], None, None)


def flash_attn_varlen_func(q, k, v, *args, **kwargs):
    if q.shape != k.shape or q.shape != v.shape:
        return original_flash_attn_varlen_forward(q, k, v, *args, **kwargs)
    kwargs["should_rearrange"] = True
    o = invoke_sparse_layer(q, k, v, *args, should_rearrange=True, **kwargs)
    if o is None:
        return original_flash_attn_varlen_forward(q, k, v, *args, **kwargs)
    return (o, None, None, None, o, q[:, :, :, 0], None, None)


def sdpa_func(q, k, v, *args, **kwargs):
    if q.shape != k.shape or q.shape != v.shape:
        return original_sdpa(q, k, v, *args, **kwargs)
    o = invoke_sparse_layer(q, k, v, *args, should_rearrange=False, **kwargs)
    if o is None:
        return original_sdpa(q, k, v, *args, **kwargs)
    return o


flash_attn.flash_attn_interface._flash_attn_varlen_forward = flash_attn_varlen_func
flash_attn.flash_attn_interface._flash_attn_forward = flash_attn_func
F.scaled_dot_product_attention = sdpa_func

from chipmunk.modules import SparseDiffAttn
from chipmunk.util import GLOBAL_CONFIG, LayerCounter
from einops import rearrange


_, layer_counter = LayerCounter.build_for_layer(is_mlp_sparse=False, is_attn_sparse=True)
sparse_attn_layers = None
num_heads = None
tensor_parallel_size = None


def setup_sparse_attn(file_path):
    import chipmunk.util

    chipmunk.util.config.load_from_file(file_path)
    global sparse_attn_layers
    global num_heads
    global tensor_parallel_size

    layer_counter.num_layers = GLOBAL_CONFIG["model_config"]["num_layers"]
    layer_counter.num_submodules_per_layer = GLOBAL_CONFIG["model_config"]["context_parallel_size"]
    tensor_parallel_size = GLOBAL_CONFIG["model_config"]["tensor_parallel_size"]

    GLOBAL_CONFIG["num_steps"] = GLOBAL_CONFIG["model_config"]["num_steps"]
    num_heads = GLOBAL_CONFIG["model_config"]["num_heads"]
    sparse_attn_layers = [
        [SparseDiffAttn(i, layer_counter) for j in range(layer_counter.num_submodules_per_layer)]
        for i in range(layer_counter.num_layers)
    ]


rearrange_forward = lambda o: rearrange(o, "b n h d -> b h n d").contiguous()
rearrange_reverse = lambda o: rearrange(o, "b h n d -> b n h d").contiguous()
has_initialized = False


def invoke_sparse_layer(q, k, v, *args, should_rearrange, **kwargs):
    if should_rearrange:
        q, k, v = rearrange_forward(q), rearrange_forward(k), rearrange_forward(v)
    w, h, d = (
        GLOBAL_CONFIG["model_config"]["latent_vector_shape"]["width"],
        GLOBAL_CONFIG["model_config"]["latent_vector_shape"]["height"],
        GLOBAL_CONFIG["model_config"]["latent_vector_shape"]["depth"],
    )
    expected_sequence_length = w * h * d
    print(f"expected_sequence_length: {expected_sequence_length}, q.shape: {q.shape}")
    if q.shape[2] < expected_sequence_length:
        # Don't process sparse attention for short sequences - this means that we're running a second model in the background that we don't care about
        return None
    print("Sparse attention", q.shape, k.shape, v.shape)
    global has_initialized
    if not has_initialized:
        sparse_attn_layers[0][0].initialize_static_mask(
            seq_shape=(w, h, d), txt_len=0, local_heads_num=num_heads // tensor_parallel_size, device="cuda"
        )
        has_initialized = True

    cur_state = layer_counter.cur_inference_step, layer_counter.cur_layer, layer_counter.cur_layer_submodule
    step, layer_idx, submodule_idx = cur_state
    o = sparse_attn_layers[layer_idx][submodule_idx](q, k, v)

    if should_rearrange:
        o = rearrange_reverse(o).contiguous()
    return o
