import torch
import math
import triton
import triton.language as tl
from einops import rearrange

# from tl_base_attention import attention as tl_dense

DEVICE = 'cuda'

cdiv = lambda a, b: (a + b - 1) // b
configs = [
    triton.Config({'BLOCK_M': BM, 'BLOCK_N': BN}, num_stages=s, num_warps=w) \
    for BM in [64]\
    for BN in [64]\
    for s in [3, 4, 7]\
    for w in [4, 8]\
]
def keep(conf):
    BLOCK_M = conf.kwargs["BLOCK_M"]
    BLOCK_N = conf.kwargs["BLOCK_N"]
    if BLOCK_M * BLOCK_N < 128 * 128 and conf.num_warps == 8:
        return False
    return True

@triton.jit
def _sparse_attn_fwd_inner(acc, l_i, m_i, q,  #
                    K_block_ptr_orig, V_block_ptr_orig,  #
                    start_m, qk_scale,  #
                    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,  #
                    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,  #
                    N_CTX: tl.constexpr, fp8_v: tl.constexpr, #
                    stride_k_seqlen, stride_v_seqlen,  #
                    sparsity_indices_ptr, sparsity_counts_ptr, #
                    ):
    sparsity_count = tl.load(sparsity_counts_ptr + start_m)
    sparsity_offsets = tl.arange(0, BLOCK_N)
    sparsity_indices_ptr += start_m * N_CTX + sparsity_offsets
    # sparsity_indices = tl.load(sparsity_indices_ptr)
    n_iters = tl.cdiv(sparsity_count, BLOCK_N)
    cur_iter = 0
    # loop over k, v and update accumulator
    for start_n in range(0, sparsity_count, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        sparsity_indices = tl.load(sparsity_indices_ptr)
        K_block_ptr = K_block_ptr_orig + (sparsity_indices[None, :]) * stride_k_seqlen
        V_block_ptr = V_block_ptr_orig + (sparsity_indices[:, None]) * stride_v_seqlen
        # Commented out lines are for when we use random sparsity counts, in production it's always a multiple of BLOCK_N = 64
        # K_block_ptr = K_block_ptr_orig + (sparsity_indices[None, :] % N_CTX) * stride_k_seqlen
        # V_block_ptr = V_block_ptr_orig + (sparsity_indices[:, None] % N_CTX) * stride_v_seqlen
        # is_valid_mask = sparsity_offsets < sparsity_count - start_n # shape (BLOCK_N,)
        k = tl.load(K_block_ptr)
        qk = tl.dot(q, k)
        m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
        # qk = qk * qk_scale + tl.where(is_valid_mask[None, :], 0, -1.0e6)
        # qk -= m_ij[:, None]
        qk = qk * qk_scale - m_ij[:, None] # use fused multiply add!
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        # -- update output accumulator --
        acc = acc * alpha[:, None]
        # update acc
        v = tl.load(V_block_ptr)
        p = p.to(tl.bfloat16)
        acc = tl.dot(p, v, acc)
        m_i = m_ij
        sparsity_indices_ptr += BLOCK_N
        cur_iter += 1

    return acc, l_i, m_i

@triton.autotune(list(filter(keep, configs)), key=["N_CTX", "HEAD_DIM"])
@triton.jit
def _sparse_attn_fwd(Q, K, V, sm_scale, M, L, Out, Out_accum, Out_scale: tl.constexpr, #
              sparsity_indices, sparsity_counts, #
              stride_qz, stride_qh, stride_qm, stride_qk,  #
              stride_kz, stride_kh, stride_kn, stride_kk,  #
              stride_vz, stride_vh, stride_vk, stride_vn,  #
              stride_oz, stride_oh, stride_om, stride_on,  #
              stride_spiz, stride_spih,  #
              stride_spcz, stride_spch,  #
              Z, H, N_CTX,  #
              HEAD_DIM: tl.constexpr,  #
              BLOCK_M: tl.constexpr,  #
              BLOCK_N: tl.constexpr,  #
              STAGE: tl.constexpr  #
              ):
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh

    spi_offset = off_z.to(tl.int64) * stride_spiz + off_h.to(tl.int64) * stride_spih
    spi_ptr = sparsity_indices + spi_offset
    spc_offset = off_z.to(tl.int64) * stride_spcz + off_h.to(tl.int64) * stride_spch
    spc_ptr = sparsity_counts + spc_offset

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_headsize = tl.arange(0, HEAD_DIM)

    # block pointers
    Q_block_ptr = (
        Q
        + qvk_offset
        + offs_m[:, None] * stride_qm
        + offs_headsize[None, :] * stride_qk
    )
    K_block_ptr = (
        K
        + qvk_offset
        + (offs_n[None, :] // BLOCK_N) * stride_kn
        + offs_headsize[:, None] * stride_kk
    )
    V_block_ptr = (
        V
        + qvk_offset
        + (offs_n[:, None] // BLOCK_N) * stride_vk
        + offs_headsize[None, :] * stride_vn
    )
    O_block_ptr = (
        Out
        + qvk_offset
        + offs_m[:, None] * stride_om
        + offs_headsize[None, :] * stride_on
    )
    O_accum_block_ptr = (
        Out_accum
        + qvk_offset
        + offs_m[:, None] * stride_om
        + offs_headsize[None, :] * stride_on
    )
    # initialize offsets
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    # load scales
    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2)
    # load q: it will stay in SRAM throughout
    # qo_mask = (offs_m < N_CTX)[:, None]
    q = tl.load(Q_block_ptr)
    # stage 1: off-band
    # For causal = True, STAGE = 3 and _sparse_attn_fwd_inner gets 1 as its STAGE
    # For causal = False, STAGE = 1, and _sparse_attn_fwd_inner gets 3 as its STAGE
    acc, l_i, m_i = _sparse_attn_fwd_inner(acc, l_i, m_i, q, K_block_ptr, V_block_ptr,  #
                                    start_m, qk_scale,  #
                                    BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                    4 - STAGE, offs_m, offs_n, N_CTX, V.dtype.element_ty == tl.float8e5,  #
                                    stride_kn, stride_vk, #
                                    spi_ptr, spc_ptr, #
                                    )
    # epilogue
    # m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * N_CTX + offs_m
    l_ptrs = L + off_hz * N_CTX + offs_m
    tl.store(m_ptrs, m_i)
    tl.store(l_ptrs, l_i)
    acc *= Out_scale # will get optimized out when Out_scale is 1.0 since it's tl.constexpr
    acc += tl.load(O_accum_block_ptr)
    tl.store(O_block_ptr, acc.to(Out.type.element_ty))

class _sparse_attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, o_accum, sm_scale, sparsity_indices, sparsity_counts, O_scale = 1.0):
        # shape constraints
        HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
        # when v is in float8_e5m2 it is transposed.
        HEAD_DIM_V = v.shape[-1]
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}
        o = torch.empty_like(q)
        stage = 1
        extra_kern_args = {}

        grid = lambda args: (triton.cdiv(q.shape[2], args["BLOCK_M"]), q.shape[0] * q.shape[1], 1)
        M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        L = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        _sparse_attn_fwd[grid](
            q, k, v, sm_scale, M, L, o, o_accum, O_scale,  #
            sparsity_indices, sparsity_counts, #
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),  #
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),  #
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),  #
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),  #
            sparsity_indices.stride(0), sparsity_indices.stride(1), #
            sparsity_counts.stride(0), sparsity_counts.stride(1), #
            q.shape[0], q.shape[1],  #
            N_CTX=q.shape[2],  #
            HEAD_DIM=HEAD_DIM_K,  #
            STAGE=stage,  #
            **extra_kern_args)

        return o, M, L

sparse_attention = _sparse_attention.apply

def get_sparsity_data(b, h, seqlen, sparsity_amount = 0.5):
    import random
    random.seed(0)
    
    assert sparsity_amount < 1, "sparsity_amount must be less than 1"
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    num_m_blocks = cdiv(seqlen, BLOCK_SIZE_M)

    dtype = torch.int32
    device = 'cuda'
    sparsity_indices = torch.full((b, h, num_m_blocks, seqlen), fill_value=0, device=device, dtype=dtype)
    sparsity_indices_counts = torch.full((b, h, num_m_blocks), fill_value=0, device=device, dtype=dtype)
    blockmask = torch.zeros((b, h, num_m_blocks, seqlen), device=device, dtype=torch.bool)

    for i in range(b):
        for j in range(h):
            for k in range(num_m_blocks):
                # make it a multiple of BLOCK_SIZE_N since this is a constraint of the kernel
                sparsity_count = int(seqlen * (1 - sparsity_amount) / BLOCK_SIZE_N) * BLOCK_SIZE_N
                indices = torch.randperm(seqlen, device=device, dtype=dtype)[:sparsity_count]
                # print(f'len(indices): {len(indices)}')
                sparsity_indices[i, j, k, :len(indices)] = indices
                sparsity_indices_counts[i, j, k] = sparsity_count
                blockmask[i, j, k, indices] = True
            
    blockmask = rearrange(blockmask.unsqueeze(3).expand(-1, -1, -1, BLOCK_SIZE_M, -1), 'b h mb bm n -> b h (mb bm) n')[:, :, :seqlen]

    # # add a few extra values to make sure that our masking works properly
    # sparsity_difference_range = 15
    # for i in range(num_m_blocks):
    #     offset = random.randint(-sparsity_difference_range, 0)
    #     num_indices = min(max(int(seqlen * (1 - sparsity_amount)), 1) - 0, seqlen)
    #     indices = list(range(0, seqlen))
    #     random.shuffle(indices)
    #     indices = indices[:num_indices]
    #     sparsity_indices[i, :num_indices] = torch.tensor(indices, device=device, dtype=dtype)
    #     sparsity_indices_counts[i] = num_indices

    assert sparsity_indices.shape[-1] == seqlen, "stride of sparsity_indices must be seqlen"
    # print(f'sparsity_indices: {sparsity_indices}')
    # print(f'sparsity_indices_counts: {sparsity_indices_counts}')
    # print(f'sparsity_indices shape: {sparsity_indices.shape}')
    # print(f'sparsity_indices_counts shape: {sparsity_indices_counts.shape}')
    # print(f'blockmask shape: {blockmask.shape}')
    # raise Exception('stop')
    return sparsity_indices, sparsity_indices_counts, blockmask


def test_sparse_op(b, h, n, d, sp_inds, sp_counts, blockmask, dtype=torch.bfloat16):
    torch.manual_seed(20)
    q = torch.randn((b, h, n, d), device='cuda', dtype=dtype)
    k = torch.randn((b, h, n, d), device='cuda', dtype=dtype)
    v = torch.randn((b, h, n, d), device='cuda', dtype=dtype)
    o_accum = torch.randn((b, h, n, d), device='cuda', dtype=dtype)
    sparsity_amount = 0.5
    kt = k.transpose(2, 3).contiguous()
    sm_scale = 1 / math.sqrt(d)

    # qkt_rows = []
    # for i in range(sp_counts.shape[0]):
    #     sp_count = sp_counts[0,0, i].item()
    #     sp_inds_i = sp_inds[0,0, i, :sp_count]
    #     sparsity_mask = torch.zeros((d, n), device='cuda', dtype=torch.bool)
    #     sparsity_mask[:, sp_inds_i] = True
    #     kt_block_sparse = torch.where(sparsity_mask, kt, 0)
    #     q_block_dense = q[:, :, i * 64:(i + 1) * 64, :]
    #     scaled = (q_block_dense @ kt_block_sparse) * sm_scale
    #     scaled.masked_fill_(~sparsity_mask[0].unsqueeze(0), float("-inf"))
    #     qkt_rows.append(scaled)

    # qkt = torch.cat(qkt_rows, dim=2)
    # breakpoint()
    # sparse_attn_ref = torch.softmax(qkt, dim=-1) @ v + o_accum

    logits = torch.matmul(q, kt) * sm_scale + torch.where(blockmask, 0, float("-inf"))
    sparse_attn_ref = torch.softmax(logits.float(), dim=-1).to(dtype) @ v + o_accum

    print(f'sparse ref shape: {sparse_attn_ref.shape}')
    tri_out, M, L = sparse_attention(q, k, v, o_accum, sm_scale, sp_inds, sp_counts)
    print(f'sparse tri shape: {tri_out.shape}')
    return sparse_attn_ref, tri_out


gpu_name = torch.cuda.get_device_name(0).replace("NVIDIA", "").strip()

baseline_results = {}
@triton.testing.perf_report([triton.testing.Benchmark(
    args={},
	x_names=["sparsity_amt"],
	# x_vals=[x/10.0 for x in list(range(0, 10, 1))],
	x_vals=[0.835],
	styles=[('gray', 'dashed'), ("black", "dashed"), ("green", "-"), ("blue", "-")],
    line_arg="provider",
    line_vals=["y=1-x", "flash", "tl-dense", "sparse"],
    line_names=["y=1-x", "FlashAttention", "TL Dense", "SparseAttention"],
	xlabel="Sparsity amount",
	ylabel=f"Kernel Duration (% of FlashAttention) {gpu_name}",
	plot_name=f"FlashAttention Performance {gpu_name}",
)])
def benchmark_sparse_attn(sparsity_amt, provider):
    from torch.nn import functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel
    global baseline_results
    b, h, n, d = 1, 24, 4352, 128
    sp_ind, sp_counts, blockmask = get_sparsity_data(b, h, n, sparsity_amt)
    key = f"{b}_{h}_{n}_{d}_{gpu_name}"

    q = torch.randn((b, h, n, d), device='cuda', dtype=torch.bfloat16)
    k = torch.randn((b, h, n, d), device='cuda', dtype=torch.bfloat16)
    v = torch.randn((b, h, n, d), device='cuda', dtype=torch.bfloat16)
    o_accum = torch.randn((b, h, n, d), device='cuda', dtype=torch.bfloat16)
    # q = torch.load(f'../sp-attn-q.pt')
    # k = torch.load(f'../sp-attn-k.pt')
    # v = torch.load(f'../sp-attn-v.pt')
    # o_accum = torch.load(f'../sp-attn-oc.pt')
    # sp_ind = torch.load(f'../sp-attn-inds.pt')
    # sp_counts = torch.load(f'../sp-attn-counts.pt')

    quantiles = [0.5, 0.2, 0.8]
    rescale = lambda tuple: (tuple[0] / baseline_results[key][0], tuple[1] / baseline_results[key][1], tuple[2] / baseline_results[key][2])
    # rescale = lambda tuple: tuple

    if provider == "flash":
        if key not in baseline_results:
            # with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            #     ans = triton.testing.do_bench(lambda: F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False), quantiles=quantiles)
            ans = triton.testing.do_bench(lambda: F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False), quantiles=quantiles)
            baseline_results[key] = ans
            flops = 4 * b * h * n * n * d
            print(f'tflops fa2: {flops * 1e-12 / (ans[1] * 1e-3)}')
        # return rescale(baseline_results[key])
        return baseline_results[key]
    elif provider == "tl-dense":
        results = triton.testing.do_bench(lambda: tl_dense(q, k, v, False, 0.5), quantiles=quantiles, warmup=100, rep=1000)
        minms, meanms, maxms = results
        # TFLOPs calculation
        flops = 4 * b * h * n * n * d
        print(f'tflops tl-dense: {flops * 1e-12 / (meanms * 1e-3)}')
        # return rescale(results)
        return results
    elif provider == "sparse":
        results = triton.testing.do_bench(lambda: sparse_attention(q, k, v, o_accum, 0.5, sp_ind, sp_counts), quantiles=quantiles, warmup=100, rep=1000)
        minms, meanms, maxms = results
        # TFLOPs calculation
        flops = 4 * b * h * n * n * d * (1 - sparsity_amt)
        print(f'tflops sparse: {flops * 1e-12 / (meanms * 1e-3)}')
        # return rescale(results)
        return results
    elif provider == "y=1-x":
        return (1 - sparsity_amt, 1 - sparsity_amt, 1 - sparsity_amt)
    else:
        raise ValueError(f"Invalid provider: {provider}")


if __name__ == "__main__":
    b, h, n, d = 1, 24, 34 * 128 + 0, 128
    print(f'b: {b}, h: {h}, n: {n}, d: {d}')
    sparsity_amount = 0.5
    sp_inds, sp_counts, blockmask = get_sparsity_data(b, h, n, sparsity_amount)
    ref_out, tri_out = test_sparse_op(b, h, n, d, sp_inds, sp_counts, blockmask)
    print(f'ref_out: {ref_out[0, -1, :10, :10]}')
    print(f'tri_out: {tri_out[0, -1, :10, :10]}')
    print(f'distance: {torch.dist(ref_out, tri_out)}')
    print(f'max diff: {torch.max(torch.abs(ref_out - tri_out))}')
    if not torch.allclose(ref_out, tri_out, atol=1e-2):
        print("❌ Sparse attention output does not match reference output")
    else:
        print("✅ All correctness checks passed.")

    benchmark_sparse_attn.run(show_plots=True, print_data=True)