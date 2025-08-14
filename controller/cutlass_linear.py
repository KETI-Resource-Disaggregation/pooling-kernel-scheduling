# controller/cutlass_linear.py
import os
import torch
import torch.nn as nn

# 안전 경로 스위치: 기본 안전(on). 나중에 고속 경로 붙이면 여기서 분기하면 됨.
SAFE = os.environ.get("BLESS_SAFE_GEMM", "1") == "1"

def _matmul2d_safe(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """항상 2D, contiguous, float32 보장 후 torch.matmul 사용."""
    if A.dim() != 2:
        A = A.reshape(A.size(0), -1)
    if B.dim() != 2:
        B = B.reshape(B.size(0), -1)
    A = A.contiguous()
    B = B.contiguous()
    if A.dtype != torch.float32:
        A = A.float()
    if B.dtype != torch.float32:
        B = B.float()
    return torch.matmul(A, B)

def matmul2d(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    # 지금은 안정성이 최우선 → 항상 safe 경로
    # (추후 CUTLASS/Triton 붙일 때 SAFE==False 분기 추가)
    return _matmul2d_safe(A, B)

class MyLinear(nn.Module):
    """x[... , in_f] * W[out_f, in_f]^T + b"""
    def __init__(self, in_f: int, out_f: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_f, in_f))
        self.bias   = nn.Parameter(torch.zeros(out_f)) if bias else None
        nn.init.trunc_normal_(self.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_f  = self.weight.size(1)
        out_f = self.weight.size(0)
        x2d   = x.reshape(-1, in_f).contiguous()      # [M, in_f]
        Wt    = self.weight.t().contiguous()          # [in_f, out_f]
        y2d   = matmul2d(x2d, Wt)                     # [M, out_f]
        if self.bias is not None:
            y2d = y2d + self.bias.unsqueeze(0)        # 안전 브로드캐스트
        return y2d.reshape(*x.shape[:-1], out_f)
