import torch
import math
import triton
import triton.language as tl

from torch.nn import functional as F

DEVICE = 'cuda'


cdiv = lambda a, b: (a + b - 1) // b

# We don't run auto-tuning every time to keep the tutorial fast. Keeping
# the code below and commenting out the equivalent parameters is convenient for
# re-tuning.
configs = [
    triton.Config({'BLOCK_M': BM, 'BLOCK_N': BN}, num_stages=s, num_warps=w) \
    for BM in [64]\
    for BN in [64]\
    for s in ([3, 4, 7])\
    for w in [4, 8]\
]
def keep(conf):
    BLOCK_M = conf.kwargs["BLOCK_M"]
    BLOCK_N = conf.kwargs["BLOCK_N"]
    if BLOCK_M * BLOCK_N < 128 * 128 and conf.num_warps == 8:
        return False
    return True

@triton.jit
def _full_attn_fwd_inner(acc, l_i, m_i, q,  #
                    prev_maxes_final_ptrs, #
                    prev_normalization_final_ptrs, #
                    blocksums_ptrs,
                    softmax_stride_b, softmax_stride_h, softmax_stride_n, #
                    blocksums_stride_b, blocksums_stride_h, blocksums_stride_m, blocksums_stride_n,
                    K_block_ptr, V_block_ptr,  #
                    start_m, qk_scale, seqlen,  #
                    H, #
                    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,  #
                    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,  #
                    N_CTX: tl.constexpr, fp8_v: tl.constexpr,
                    ):
    # non-causal full attention
    lo, hi = 0, N_CTX
    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))
    off_hb = tl.program_id(1)
    off_b = off_hb // H
    off_h = off_hb % H
    softmax_data_offset = off_b.to(tl.int64) * softmax_stride_b + off_h.to(tl.int64) * softmax_stride_h + (start_m * BLOCK_M + tl.arange(0, BLOCK_M)) * softmax_stride_n
    # blocksums_ptrs += off_b.to(tl.int64) * blocksums_stride_b + off_h.to(tl.int64) * blocksums_stride_h + start_m * blocksums_stride_m + tl.arange(0, BLOCK_N) * blocksums_stride_n
    bsp = blocksums_ptrs + off_b.to(tl.int64) * blocksums_stride_b + off_h.to(tl.int64) * blocksums_stride_h + start_m * blocksums_stride_m + tl.arange(0, BLOCK_N) * blocksums_stride_n
    
    # previous m and l values
    prev_maxes_final = tl.load(prev_maxes_final_ptrs + softmax_data_offset)
    prev_normalization_final = tl.load(prev_normalization_final_ptrs + softmax_data_offset, mask=(offs_m < seqlen), other=1.0e6)

    # blocksums = tl.zeros([N // BLOCK_N], dtype=tl.float32)
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k = tl.load(K_block_ptr)
        # q_dot_k = tl.dot(q, k)
        q_dot_k = tl.dot(q, k)
        # q_dot_k = tl.where(start_n + offs_n[None, :] < 4592, q_dot_k, -1.0e6)
        qk = q_dot_k
        m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
        # qk = qk * qk_scale - m_ij[:, None]
        qk = qk * qk_scale - m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        # -- update output accumulator --
        acc = acc * alpha[:, None]
        # update acc
        v = tl.load(V_block_ptr)
        # v = tl.where(start_n + offs_n[:, None] < 4592, v, 0).to(tl.bfloat16)
        p = p.to(tl.bfloat16)
        acc = tl.dot(p, v, acc)
        m_i = m_ij

        # ---------------- PREVIOUS PHASE OF SOFTMAX -------------------
        qk_prev = q_dot_k * qk_scale - prev_maxes_final[:, None]
        p_prev = tl.math.exp2(qk_prev)
        p_prev = (p_prev / prev_normalization_final[:, None])
        # p_prev = tl.where(offs_m[:, None] < seqlen, p_prev, 0)
        blocksums = tl.sum(p_prev, 0)
        tl.store(bsp, blocksums, mask=(start_n + offs_n) < seqlen)

        # ----------------- UPDATE POINTERS -----------------
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        bsp += BLOCK_N * blocksums_stride_n
    
    return acc, l_i, m_i

@triton.autotune(list(filter(keep, configs)), key=["N_CTX", "HEAD_DIM"])
@triton.jit
def _full_attn_fwd(Q, K, V, sm_scale, M, L, Out, seqlen,  #
              prev_maxes_ptr, #
              prev_normalization_final_ptrs, #
              blocksums_ptrs, #
              softmax_stride_b, softmax_stride_h, softmax_stride_n, #
              blocksums_stride_b, blocksums_stride_h, blocksums_stride_m, blocksums_stride_n, #
              stride_qz, stride_qh, stride_qm, stride_qk,  #
              stride_kz, stride_kh, stride_kn, stride_kk,  #
              stride_vz, stride_vh, stride_vk, stride_vn,  #
              stride_oz, stride_oh, stride_om, stride_on,  #
              Z, H, N_CTX,  #
              HEAD_DIM: tl.constexpr,  #
              BLOCK_M: tl.constexpr,  #
              BLOCK_N: tl.constexpr,  #
              STAGE: tl.constexpr,  #
              ):
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh

    # block pointers
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    v_order: tl.constexpr = (0, 1) if V.dtype.element_ty == tl.float8e5 else (1, 0)
    V_block_ptr = tl.make_block_ptr(
        base=V + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=v_order,
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset,
        shape=(HEAD_DIM, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_N),
        order=(0, 1),
    )
    # O_block_ptr = tl.make_block_ptr(
    #     base=Out + qvk_offset,
    #     shape=(N_CTX, HEAD_DIM),
    #     strides=(stride_om, stride_on),
    #     offsets=(start_m * BLOCK_M, 0),
    #     block_shape=(BLOCK_M, HEAD_DIM),
    #     order=(1, 0),
    # )
    offs_o = (start_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]) * stride_om + tl.arange(0, HEAD_DIM)[None, :] * stride_on
    O_ptrs = Out + qvk_offset + offs_o

    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    # load scales
    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2)
    # load q: it will stay in SRAM throughout
    q = tl.load(Q_block_ptr)
    # stage 1: off-band
    # For causal = True, STAGE = 3 and _attn_fwd_inner gets 1 as its STAGE
    # For causal = False, STAGE = 1, and _attn_fwd_inner gets 3 as its STAGE
    acc, l_i, m_i = _full_attn_fwd_inner(
        acc, l_i, m_i, q, 
        prev_maxes_ptr,  
        prev_normalization_final_ptrs, 
        blocksums_ptrs,
        softmax_stride_b, softmax_stride_h, softmax_stride_n,
        blocksums_stride_b, blocksums_stride_h, blocksums_stride_m, blocksums_stride_n,
        K_block_ptr, V_block_ptr,  #
        start_m, qk_scale, seqlen,  #
        H, #
        BLOCK_M, HEAD_DIM, BLOCK_N,  #
        4 - STAGE, offs_m, offs_n, N_CTX, V.dtype.element_ty == tl.float8e5,  #
    )
    # epilogue
    # m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * N_CTX + offs_m
    l_ptrs = L + off_hz * N_CTX + offs_m
    tl.store(m_ptrs, m_i, mask=offs_m < seqlen)
    tl.store(l_ptrs, l_i, mask=offs_m < seqlen)
    # tl.store(O_block_ptr, acc.to(Out.type.element_ty))
    tl.store(O_ptrs, acc.to(Out.type.element_ty), mask=offs_m[:, None] < seqlen)
    # tl.store(O_ptrs, acc.to(Out.type.element_ty))

class _full_attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, sm_scale, prev_maxes, prev_normalization):
        # shape constraints
        HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
        # when v is in float8_e5m2 it is transposed.
        HEAD_DIM_V = v.shape[-1]
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}
        stage = 1

        # mb = q.shape[2] // 128 if q.shape[2] % 128 == 0 else q.shape[2] // 128 + 1
        mb = triton.cdiv(q.shape[2], 64)
        # print(f'mb: {mb}')
        # print(f'q.shape[2] // 128: {q.shape[2] // 128}')
        # print(f'triton cdiv: {triton.cdiv(q.shape[2], 128)}')

        grid = lambda args: (triton.cdiv(q.shape[2], args["BLOCK_M"]), q.shape[0] * q.shape[1], 1)
        # grid = lambda args: (mb, q.shape[0] * q.shape[1], 1)

        # print(f'grid: {grid({})}')
        
        M = torch.zeros((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        L = torch.zeros_like(M, dtype=torch.float32)
        o = torch.empty_like(q)
        
        blocksums = torch.zeros((q.shape[0], q.shape[1], mb, q.shape[2]), device=q.device, dtype=torch.float32)
        # print(f'blocksums: {blocksums.shape}')
        seqlen = q.shape[2]
        # seqlen = 4592
        _full_attn_fwd[grid](
            q, k, v, sm_scale, M, L, o, seqlen,  #
            prev_maxes, #
            prev_normalization, #
            blocksums,
            prev_maxes.stride(0), prev_maxes.stride(1), prev_maxes.stride(2), #
            blocksums.stride(0), blocksums.stride(1), blocksums.stride(2), blocksums.stride(3), #
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),  #
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),  #
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),  #
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),  #
            q.shape[0], q.shape[1],  #
            N_CTX=q.shape[2],  #
            HEAD_DIM=HEAD_DIM_K,  #
            STAGE=stage
        )

        return o, M, L, blocksums

full_attention = _full_attention.apply

def partial_col_sum(tensor: torch.Tensor, group_size: int) -> torch.Tensor:
    # assert tensor.shape[2] % group_size == 0, "The number of rows must be divisible by group_size."
    padding = group_size - tensor.shape[2] % group_size if tensor.shape[2] % group_size != 0 else 0
    tp = F.pad(tensor, (0, 0, 0, padding))
    # print(f'tp: {tp}')
    # print(f'tp.shape: {tp.shape}')
    # raise ValueError('stop')
    return tp.view(tp.shape[0], tp.shape[1], tp.shape[2] // group_size, group_size, tp.shape[3]).sum(dim=3)

def reference_impl():
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    p_prev_exp = torch.exp2(p.float() * 1.44269504 - prev_m.unsqueeze(-1)) / prev_l.unsqueeze(-1)
    ref_blocksums = partial_col_sum(p_prev_exp, 64)
    
    ref_m = torch.max(p.float(), dim=-1).values * 1.44269504
    p_softmax = torch.softmax(p.float(), dim=-1).to(q.dtype)
    ref_out = torch.matmul(p_softmax, v)
    l = torch.exp(p.float() - ref_m.unsqueeze(-1)).sum(dim=-1, keepdim=True)
    return ref_out, ref_m, l, ref_blocksums

baseline_results = {}
@triton.testing.perf_report([triton.testing.Benchmark(
    args={},
	x_names=["sparsity_amt"],
	x_vals=[x/10.0 for x in list(range(0, 10, 1))],
	styles=[('gray', 'dashed'), ("black", "dashed"), ("green", "-")],
    line_arg="provider",
    line_vals=["y=1-x", "flash", "blocksum"],
    line_names=["y=1-x", "FlashAttention", "BlocksumAttention"],
	xlabel="Sparsity amount",
	ylabel=f"Kernel Duration (% of FlashAttention)",
	plot_name=f"FlashAttention Performance",
)])
def benchmark_blocksum_attn(sparsity_amt, provider):
    from torch.nn import functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel
    global baseline_results
    b, h, n, d = 1, 24, 4592, 128
    key = f"{b}_{h}_{n}_{d}"

    q = torch.randn((b, h, n, d), device='cuda', dtype=torch.bfloat16)
    k = torch.randn((b, h, n, d), device='cuda', dtype=torch.bfloat16)
    v = torch.randn((b, h, n, d), device='cuda', dtype=torch.bfloat16)
    o_accum = torch.randn((b, h, n, d), device='cuda', dtype=torch.bfloat16)
    prev_maxes = torch.randn((b, h, n), device='cuda', dtype=torch.float32)
    prev_normalization = torch.randn((b, h, n), device='cuda', dtype=torch.float32)

    quantiles = [0.5, 0.2, 0.8]
    rescale = lambda tuple: (tuple[0] / baseline_results[key][0], tuple[1] / baseline_results[key][1], tuple[2] / baseline_results[key][2])
    # rescale = lambda tuple: tuple

    if provider == "flash":
        if key not in baseline_results:
            # with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            #     ans = triton.testing.do_bench(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=False), quantiles=quantiles)
            ans = triton.testing.do_bench(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=False), quantiles=quantiles)
            baseline_results[key] = ans
        return rescale(baseline_results[key])
    elif provider == "blocksum":
        results = triton.testing.do_bench(lambda: full_attention(q, k, v, 0.5, prev_maxes, prev_normalization), quantiles=quantiles, warmup=100, rep=1000)
        minms, meanms, maxms = results
        # TFLOPs calculation
        flops = 4 * b * h * n * n * d
        print(f'tflops: {flops * 1e-12 / (meanms * 1e-3)}')
        return rescale(results)
    elif provider == "y=1-x":
        return (1 - sparsity_amt, 1 - sparsity_amt, 1 - sparsity_amt)
    else:
        raise ValueError(f"Invalid provider: {provider}")

if __name__ == "__main__":
    # Z, H, N_CTX, HEAD_DIM = 1, 2, 1024, 64
    Z, H, N_CTX, HEAD_DIM = 1, 24, 34 * 128 + 0, 128
    BLOCK_M = 64
    dtype = torch.bfloat16
    torch.manual_seed(20)
    q = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_(True))
    k = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_(True))
    v = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_(True))
    prev_m = (torch.empty((Z, H, N_CTX), dtype=torch.float32, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_(False))
    prev_l = (torch.empty((Z, H, N_CTX), dtype=torch.float32, device=DEVICE).normal_(mean=0.0, std=0.5).abs().requires_grad_(False))
    sm_scale = 1 / HEAD_DIM ** 0.5


    ref_out, ref_m, ref_l, ref_blocksums = reference_impl()
    # prev_m = ref_m
    # prev_l = ref_l.squeeze(-1)
    # print(f'prev_l: {prev_l[0, -1, -128:]}')
    print(f'prev_m shape: {prev_m.shape}')
    print(f'prev_l shape: {prev_l.shape}')
    tri_out, tri_m, tri_l, tri_blocksums = full_attention(q, k, v, sm_scale, prev_m, prev_l)
    print('Output difference (max):', (ref_out - tri_out).abs().max().item())
    assert (ref_out - tri_out).abs().max().item() < 0.01, "The output of the reference attention implementation and the tri implementation are not close enough."
    print('Max m difference:', (tri_m - ref_m).abs().max().item())
    assert (tri_m - ref_m).abs().max().item() < 0.02, "The m values of the reference attention implementation and the tri implementation are not close enough."
    print('Blocksums difference (mean):', (tri_blocksums - ref_blocksums).abs().mean().item())
    # breakpoint()
    print(f'tri_blocksums: {tri_blocksums[0, -1, -1]}')
    print(f'ref_blocksums: {ref_blocksums[0, -1, -1]}')
    # breakpoint()
    assert (tri_blocksums - ref_blocksums).abs().mean().item() < 0.7, "The blocksums of the reference attention implementation and the tri implementation are not close enough."
    print("✅ All correctness checks passed.")
    
    benchmark_blocksum_attn.run(show_plots=False, print_data=True)
    