import os
import torch
import torch.nn as nn
import ctypes

# bless bind  dnfl wognsdl doalenlwlstorl enlwufk wpqkf
def _load_bless_symbols():
    try:
        lib = ctypes.CDLL(None)
        _bind = getattr(lib, "bless_bind_thread")
        _bind.argtypes = [ctypes.c_int]
        _bind.restype = None
        _curr = getattr(lib, "bless_current_route")
        _curr.restype = ctypes.c_int  # 0=LIMITED, 1=UNLIMITED
        return _bind, _curr
    except Exception:
        return None, None

_BLESS_BIND, _BLESS_CURR = _load_bless_symbols()
def _bind_to_route_if_possible():
    if _BLESS_BIND and _BLESS_CURR:
        try:
            r = int(_BLESS_CURR())
            _BLESS_BIND(1 if r == 1 else 0)
        except Exception:
            pass

SAFE = os.environ.get("BLESS_SAFE_GEMM", "1") == "1"

def _matmul2d_safe(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    _bind_to_route_if_possible()  # cuBLAS 실행 전 현재 route 바인딩

    if A.dim() != 2: A = A.reshape(A.size(0), -1)
    if B.dim() != 2: B = B.reshape(B.size(0), -1)

    A = A.contiguous()
    B = B.contiguous()

    if A.dtype != torch.float32: A = A.float()
    if B.dtype != torch.float32: B = B.float()

    return torch.matmul(A, B)

def matmul2d(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return _matmul2d_safe(A, B)

class MyLinear(nn.Module):
    """y = x[... , in_f] @ W[out_f, in_f]^T + b"""
    def __init__(self, in_f: int, out_f: int, bias: bool = True):
        super().__init__()
        # CPU에서 파라미터 생성(컨텍스트 충돌 방지)
        self.weight = nn.Parameter(torch.empty(out_f, in_f, device="cpu"))
        self.bias = nn.Parameter(torch.zeros(out_f, device="cpu")) if bias else None
        nn.init.trunc_normal_(self.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _bind_to_route_if_possible()
        dev = x.device
        W = self.weight.to(dev, non_blocking=True)
        b = self.bias.to(dev, non_blocking=True) if self.bias is not None else None

        in_f  = W.size(1)
        out_f = W.size(0)

        x2d = x.reshape(-1, in_f).contiguous()
        Wt  = W.t().contiguous()

        y2d = matmul2d(x2d, Wt)
        if b is not None:
            y2d = y2d + b.unsqueeze(0)

        return y2d.reshape(*x.shape[:-1], out_f)
