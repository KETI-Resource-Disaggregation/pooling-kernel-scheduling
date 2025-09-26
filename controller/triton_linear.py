# smrmdoql dkswogns dmz dkakwek wognsdl duwkdjqtdl 45sus ahthfdlwl aldks b
import torch
import triton
import triton.language as tl
import torch.nn as nn

import ctypes
try:
    _lib = ctypes.CDLL(None)
    _bind = getattr(_lib, "bless_bind_thread")
    _bind.argtypes = [ctypes.c_int]
    _bind.restype = None
    _curr = getattr(_lib, "bless_current_route")
    _curr.restype = ctypes.c_int
except Exception:
    _bind = None
    _curr = None

def _bind_to_current_route():
    """libbless가 들고있는 route(LIMITED/UNLIMITED)에 맞춰 현재 스레드 컨텍스트 바인딩"""
    if _bind and _curr:
        r = int(_curr())  # 0=LIMITED, 1=UNLIMITED
        _bind(1 if r == 1 else 0)

@triton.jit
def _matmul_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    group_id = pid // (GROUP_M * num_pid_n)
    first_pid_m = group_id * GROUP_M
    pid_m = first_pid_m + (pid % GROUP_M)
    pid_n = (pid // GROUP_M) % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a_mask = (offs_m[:, None] < M) & (k + offs_k[None, :] < K)
        b_mask = (k + offs_k[:, None] < K) & (offs_n[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = C + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)

def _triton_matmul_2d(a2d: torch.Tensor, b2d: torch.Tensor) -> torch.Tensor:
    """a2d: [M,K], b2d: [K,N] -> c2d: [M,N] (acc fp32; half/bf16 입력일 때는 원 dtype으로 캐스트)"""
    _bind_to_current_route()
    assert a2d.dim() == 2 and b2d.dim() == 2
    assert a2d.is_cuda and b2d.is_cuda
    assert a2d.dtype in (torch.float16, torch.bfloat16, torch.float32)
    assert b2d.dtype == a2d.dtype
    M, K = a2d.shape
    K2, N = b2d.shape
    assert K == K2

    a_ = a2d.contiguous()
    b_ = b2d.contiguous()
    c = torch.empty((M, N), device=a_.device, dtype=torch.float32)

    BLOCK_M = 128; BLOCK_N = 128; BLOCK_K = 32; GROUP_M = 8
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)

    _matmul_kernel[grid](
        a_, b_, c,
        M, N, K,
        a_.stride(0), a_.stride(1),
        b_.stride(0), b_.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, GROUP_M=GROUP_M,
        num_warps=4, num_stages=2,
    )

    if a2d.dtype in (torch.float16, torch.bfloat16):
        return c.to(a2d.dtype)
    return c  # fp32

def triton_warmup_both_contexts(d_model: int = 128, batch: int = 2):
    # 작은 dummy 연산으로 컴파일만 유도
    x = torch.randn(batch, d_model, device="cuda", dtype=torch.float16)
    w = torch.randn(d_model, d_model, device="cuda", dtype=torch.float16)

    # 현재 route에서 1번
    _ = _triton_matmul_2d(x, w)
    # route를 바꾸고 1번 (외부 스케줄러가 바꾸는 구조면, 그 route에 맞게 호출만 해도 됨)
    # 여기서는 bless_current_route() 결과를 토대로 반대편 컨텍스트로 임시 바인딩:
    if _bind and _curr:
        cur = int(_curr())
        _bind(1 - cur)    # 반대편으로 바인딩
        _ = _triton_matmul_2d(x, w)
        _bind(cur)        # 원래로 복귀

def _flatten_2d(x: torch.Tensor):
    """[*, K] -> (x2d:[M,K], prefix_sizes: tuple)"""
    assert x.dim() >= 2
    K = x.size(-1)
    prefix = x.shape[:-1]
    M = 1
    for s in prefix: M *= s
    return x.reshape(M, K), prefix

def _view_from_2d(y2d: torch.Tensor, prefix_sizes: tuple):
    """[M,N] + prefix_sizes -> [*, N]"""
    N = y2d.size(-1)
    return y2d.reshape(*prefix_sizes, N)

class TritonLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, b):
        # x: [*, in], w: [out, in], b: [out] or None
        x2d, prefix = _flatten_2d(x)
        y2d = _triton_matmul_2d(x2d, w.t())
        if b is not None:
            y2d = y2d + b.unsqueeze(0)  # [M,N] + [1,N]
        y = _view_from_2d(y2d, prefix)
        ctx.save_for_backward(x, w, b)
        return y

    @staticmethod
    def backward(ctx, dy):
        x, w, b = ctx.saved_tensors
        # dy: [*, out]
        dy2d, _ = _flatten_2d(dy)
        x2d, _  = _flatten_2d(x)

        # dx = dy @ w  => [M,out] @ [out,in] = [M,in]
        dx2d = _triton_matmul_2d(dy2d, w)
        dx = dx2d.reshape_as(x)

        # dw = dy^T @ x => [out,M] @ [M,in] = [out,in]
        dw2d = _triton_matmul_2d(dy2d.transpose(0,1).contiguous(), x2d)
        # cast to weight dtype if needed
        if dw2d.dtype != w.dtype:
            dw2d = dw2d.to(w.dtype)
        dw = dw2d.reshape_as(w)

        db = dy.sum(dim=tuple(range(dy.dim()-1))) if b is not None else None
        return dx, dw, db

class MyLinear(nn.Linear):
    def forward(self, x):
        return TritonLinearFn.apply(x, self.weight, self.bias)

