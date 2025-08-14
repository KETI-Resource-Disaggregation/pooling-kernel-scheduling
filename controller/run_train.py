
import os, sys, time, csv, argparse, warnings, math, random
_HERE = os.path.dirname(os.path.abspath(__file__))
_SYSROOT = os.path.dirname(_HERE)
if _HERE not in sys.path: sys.path.insert(0, _HERE)
if _SYSROOT not in sys.path: sys.path.insert(0, _SYSROOT)

try:
    from cutlass_linear import MyLinear, matmul2d
except Exception:
    from controller.cutlass_linear import MyLinear, matmul2d

import torch
import torch.nn as nn
import torch.nn.functional as F

MASTER = os.environ.get("BLESS_MASTER", "/tmp/bless-master.sock")
TENANT = os.environ.get("BLESS_TENANT", "")

class MHA(nn.Module):
    def __init__(self, d_model, n_head, causal=True):
        super().__init__()
        assert d_model % n_head == 0, f"d_model({d_model}) must be divisible by n_head({n_head})"
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.scale = self.head_dim ** -0.5
        self.qkv = MyLinear(d_model, 3 * d_model, bias=False)
        self.proj = MyLinear(d_model, d_model, bias=False)
        self.causal = causal

    def forward(self, x):
        # x: [B,S,D]
        B, S, D = x.shape
        H, Hd = self.n_head, self.head_dim

        qkv = self.qkv(x)                          # [B,S,3D]
        q, k, v = qkv.split(D, dim=-1)             # 각 [B,S,D]

        # [B,H,S,Hd] -> [B,H,S,Hd] contiguous 보장
        def split_heads(t):
            return t.contiguous().view(B, S, H, Hd).permute(0, 2, 1, 3).contiguous()

        q = split_heads(q)                         # [B,H,S,Hd]
        k = split_heads(k)                         # [B,H,S,Hd]
        v = split_heads(v)                         # [B,H,S,Hd]

        # batched matmul: (B,H,S,Hd) x (B,H,Hd,S) -> (B,H,S,S)
        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        if self.causal:
            mask = torch.ones((S, S), dtype=torch.bool, device=x.device).triu(1)
            scores = scores.masked_fill(mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)

        # (B,H,S,S) x (B,H,S,Hd) -> (B,H,S,Hd)
        y = torch.matmul(attn, v)

        # [B,H,S,Hd] -> [B,S,D]
        y = y.permute(0, 2, 1, 3).contiguous().view(B, S, D)
        return self.proj(y)

def send_master(msg: str):
    if not MASTER: return
    try:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        s.sendto(msg.encode(), MASTER)
    except Exception:
        pass
    finally:
        try: s.close()
        except: pass

# ---- sys.path 보정: 프로젝트 루트 추가 ----
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import nullcontext
# ... 상단에 이어서 ...
import ctypes
try:
    _lib = ctypes.CDLL(None)
    _bind = getattr(_lib, "bless_bind_thread")
    _bind.argtypes = [ctypes.c_int]; _bind.restype = None
    _curr = getattr(_lib, "bless_current_route"); _curr.restype = ctypes.c_int
except Exception:
    _bind = None; _curr = None

def _load_c_symbol(name, restype=ctypes.c_int):
    try:
        lib = ctypes.CDLL(None); f = getattr(lib, name); f.restype = restype; return f
    except Exception: return None
_bless_sprog = _load_c_symbol("bless_squad_progress")
def query_sprog(): return int(_bless_sprog()) if _bless_sprog else -1

_hb_stop = [False]
def start_hb_thread():
    def _loop():
        while not _hb_stop[0]:
            sp = query_sprog()
            send_master(f"HB pid={os.getpid()} t={TENANT} sp={sp}")
            time.sleep(0.02)  # 20ms
    th = threading.Thread(target=_loop, daemon=True); th.start()
    return th

def bind_to_route():
    if _bind and _curr:
        r = int(_curr())  # 0=LIMITED, 1=UNLIMITED
        _bind(1 if r == 1 else 0)

# 백엔드 선택: cutlass 고정 권장
BACKEND = os.environ.get("BLESS_LINEAR", "cutlass").lower()
if BACKEND in ("cutlass", "2"):
    from controller.cutlass_linear import MyLinear, matmul2d
    print("[patch] Using CUTLASS(MyLinear) & bless_gemm for matmul", flush=True)
else:
    # 안전하게 cutlass로 강제 (지금은 Triton가 컨텍스트 캐시 이슈)
    from controller.cutlass_linear import MyLinear, matmul2d
    print("[patch] Forcing CUTLASS backend (Triton disabled for stability)", flush=True)


# ===== bless helpers =====
def _load_bind():
    try:
        lib = ctypes.CDLL(None)
        f = getattr(lib, "bless_bind_thread")
        f.argtypes = [ctypes.c_int]; f.restype = None
        return f
    except Exception:
        return None
_bless_bind = _load_bind()
def bind_thread(route=0):
    if _bless_bind: _bless_bind(ctypes.c_int(route))

def bless_send(cmd, pid=None):
    pid = pid or os.getpid()
    path = f"/tmp/bless-{pid}.sock"
    try:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM); s.settimeout(0.1)
        s.sendto(cmd.encode(), path); s.close(); return True
    except Exception:
        return False

# ===== 안전 워밍업(라이브러리 커널 회피) =====
@torch.no_grad()
def _aten_warm_ops():
    x = torch.randn(1024, device="cuda")
    y = torch.tanh(x) + torch.sin(x)
    _ = y.sum().item()  # sync

# ===== 백엔드 선택: triton / cutlass / torch =====
BACKEND = os.environ.get("BLESS_LINEAR", os.environ.get("BLESS_USE_TRITON", "triton"))
LinearImpl = nn.Linear
if str(BACKEND).lower() in ("1", "true", "triton"):
    try:
        from controller.triton_linear import MyLinear as LinearImpl
        print("[patch] Using Triton MyLinear for GEMM", flush=True)
    except Exception as e:
        print(f"[patch] Triton not available -> fallback to torch Linear: {e}", flush=True)
        LinearImpl = nn.Linear
elif str(BACKEND).lower() in ("cutlass", "2"):
    try:
        from controller.cutlass_linear import MyLinear as LinearImpl
        print("[patch] Using CUTLASS(MyLinear) for GEMM", flush=True)
    except Exception as e:
        print(f"[patch] CUTLASS extension not available -> fallback: {e}", flush=True)
        LinearImpl = nn.Linear
else:
    print("[patch] Using vanilla torch.nn.Linear", flush=True)

# ===== 모델 =====
try:
    from torch.nn.attention import sdpa_kernel, SDPBackend
    def _sdpa_ctx(): return sdpa_kernel(backends=[SDPBackend.MATH])
except Exception:
    def _sdpa_ctx(): return nullcontext()

class SelfAttention(nn.Module):
    def __init__(self, d_model, n_head, causal: bool):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.qkv = MyLinear(d_model, 3*d_model, bias=False)
        self.proj = MyLinear(d_model, d_model, bias=False)
        self.causal = causal

    def forward(self, x):  # x: [B,S,D]
        bind_to_route()  # 스케줄러 라우트와 현재 스레드 컨텍스트 동기화

        B,S,D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        def pack(t):  # [B,S,D] -> [B,nH,S,Hd]
            return t.view(B, S, self.n_head, self.head_dim).transpose(1, 2).contiguous()
        q, k, v = pack(q), pack(k), pack(v)

        # (batched) q @ k^T  →  2D GEMM로 통일
        q2 = q.reshape(B*self.n_head, S, self.head_dim)
        k2 = k.reshape(B*self.n_head, S, self.head_dim)
        scores = torch.empty(B*self.n_head, S, S, device=x.device, dtype=torch.float32)
        for i in range(B*self.n_head):
            scores[i] = matmul2d(q2[i], k2[i].transpose(0,1))  # [S,S]
        scores = scores / (self.head_dim ** 0.5)

        if self.causal:
            m = torch.ones(S, S, device=x.device, dtype=torch.bool).triu(1)
            scores = scores.masked_fill(m.unsqueeze(0), float('-inf'))

        attn = torch.softmax(scores, dim=-1)  # [B*nH,S,S]

        # y = attn @ v
        v2 = v.reshape(B*self.n_head, S, self.head_dim)
        y2 = torch.empty(B*self.n_head, S, self.head_dim, device=x.device, dtype=v2.dtype)
        for i in range(B*self.n_head):
            y2[i] = matmul2d(attn[i], v2[i])  # [S,Hd]

        y = y2.reshape(B, self.n_head, S, self.head_dim).transpose(1,2).contiguous().view(B,S,D)
        return self.proj(y)

class GPTBlock(nn.Module):
    def __init__(self, d_model, n_head, ff_mult=4):
        super().__init__()
        self.attn = SelfAttention(d_model, n_head, causal=True)
        self.ln1  = nn.LayerNorm(d_model)
        self.ff   = nn.Sequential(
            LinearImpl(d_model, ff_mult*d_model),
            nn.GELU(),
            LinearImpl(ff_mult*d_model, d_model)
        )
        self.ln2  = nn.LayerNorm(d_model)
    def forward(self, x):
        x = self.ln1(x + self.attn(x))
        x = self.ln2(x + self.ff(x))
        return x

class EncBlock(nn.Module):
    def __init__(self, d_model, n_head, ff_mult=4):
        super().__init__()
        self.attn = SelfAttention(d_model, n_head, causal=False)
        self.ln1  = nn.LayerNorm(d_model)
        self.ff   = nn.Sequential(
            LinearImpl(d_model, ff_mult*d_model),
            nn.GELU(),
            LinearImpl(ff_mult*d_model, d_model)
        )
        self.ln2  = nn.LayerNorm(d_model)
    def forward(self, x):
        x = self.ln1(x + self.attn(x))
        x = self.ln2(x + self.ff(x))
        return x

class TinyGPT2(nn.Module):
    def __init__(self, vocab=4096, d_model=768, n_layer=6, n_head=12, seq=512, ff_mult=4):
        super().__init__()
        self.seq = seq
        self.tok = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(seq, d_model)
        self.blocks = nn.ModuleList([GPTBlock(d_model, n_head, ff_mult) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm   = LinearImpl(d_model, vocab, bias=False)
    def forward(self, idx):
        B,S = idx.shape
        x = self.tok(idx) + self.pos.weight[:S]
        for blk in self.blocks: x = blk(x)
        x = self.ln_f(x)
        return self.lm(x)

class TinyBERT(nn.Module):
    def __init__(self, vocab=4096, d_model=768, n_layer=6, n_head=12, seq=512, ff_mult=4):
        super().__init__()
        self.seq = seq
        self.tok = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(seq, d_model)
        self.blocks = nn.ModuleList([EncBlock(d_model, n_head, ff_mult) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm   = LinearImpl(d_model, vocab, bias=False)
    def forward(self, idx):
        B,S = idx.shape
        x = self.tok(idx) + self.pos.weight[:S]
        for blk in self.blocks: x = blk(x)
        x = self.ln_f(x)
        return self.lm(x)

def build_model(model_name, vocab, d_model, n_layer, n_head, seq, ff_mult):
    if model_name == "gpt2":
        m = TinyGPT2(vocab, d_model, n_layer, n_head, seq, ff_mult)
    elif model_name == "bert":
        m = TinyBERT(vocab, d_model, n_layer, n_head, seq, ff_mult)
    else:
        raise ValueError(model_name)
    return m.train()

# ===== 학습 =====
def run(args):
    import time, os as _os
    # (1) CUDA 컨텍스트 먼저 생성해서 cuBLAS 경고 방지
    import torch
    torch.cuda.set_device(0)
    torch.rand(1, device="cuda"); torch.cuda.synchronize()

    # (2) run_name 먼저 정하고 CSV 오픈
    tenant = os.environ.get("BLESS_TENANT")
    run_name = (
        os.environ.get("RUNNER_NAME")
        or (tenant if tenant else f"{args.model}-{_os.getpid()}-{time.strftime('%m%d-%H%M%S')}")
    )
    csv_path = f"{run_name}.csv"
    print(f"[{run_name}] csv opened -> {csv_path}", flush=True)

    import csv as _csv
    f = open(csv_path, "w", newline="", buffering=1)
    w = _csv.writer(f)
    w.writerow(["t_s","step","loss","tok_per_s"]); f.flush(); os.fsync(f.fileno())
    print(f"[{run_name}] csv opened -> {csv_path}", flush=True)

    # 1) bless 컨텍스트 고정(LIMITED) + 가벼운 워밍업(라이브러리 호출 회피)
    bind_thread(0)
    _aten_warm_ops()
    torch.cuda.set_device(0) # primary context attach(안정화)
    
    # 2) 모델
    model = build_model(args.model, args.vocab, args.d_model, args.n_layer, args.n_head, args.seq, args.ff_mult)
    def _init_small(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)  # CPU에서 실행
            if m.bias is not None: nn.init.zeros_(m.bias)
        if isinstance(m, nn.Embedding):
            nn.init.trunc_normal_(m.weight, std=0.02)  # CPU에서 실행
    model.apply(_init_small)
    model = model.cuda().train()
    opt = torch.optim.AdamW(
    model.parameters(), lr=1e-4, betas=(0.9,0.95), weight_decay=0.01,
    foreach=False, fused=False
)
    torch.cuda.synchronize(); torch.cuda._sleep(1000)
    t0 = time.time()
    torch.autograd.set_detect_anomaly(True)

    for step in range(1, args.steps+1):
        # 입력 배치 생성 (기존 그대로)
        x = torch.randint(0, args.vocab, (args.batch, args.seq), device="cuda", dtype=torch.long)
        y = torch.randint(0, args.vocab, (args.batch, args.seq), device="cuda", dtype=torch.long)

        opt.zero_grad(set_to_none=True)
        tic = time.time()  # <-- 여기서 측정 시작

        # fwd/bwd (기존 그대로)
        logits = model(x)
        loss = F.cross_entropy(
            logits.float().clamp_(-30,30).reshape(-1, logits.size(-1)),
            y.reshape(-1), label_smoothing=0.05
        )
        loss.backward()
        torch.cuda.synchronize()
        if not torch.isfinite(loss):
            print("[guard] non-finite loss, skip step")
            opt.zero_grad(set_to_none=True)
            continue
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        torch.cuda.synchronize()
        dt = max(1e-6, time.time() - tic)  # <-- 여기서 경과시간 계산
        tps = (args.batch * args.seq) / dt

        w.writerow([f"{time.time()-t0:.3f}", step, f"{loss.item():.4f}", f"{tps:.1f}"])
        f.flush(); os.fsync(f.fileno())

        if step % 10 == 0:
            print(f"[{run_name}] step={step} tok/s={tps:.0f}", flush=True)
        torch.cuda._sleep(1000000)

    f.close()
    print(f"[train] {tenant} done steps={args.steps}", flush=True)

    _hb_stop[0] = True

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["gpt2","bert"], required=True)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--seq",   type=int, default=384)
    ap.add_argument("--vocab", type=int, default=4096)
    ap.add_argument("--d_model", type=int, default=1280)
    ap.add_argument("--n_layer", type=int, default=12)
    ap.add_argument("--n_head",  type=int, default=20)
    ap.add_argument("--ff_mult", type=int, default=8)
    args = ap.parse_args()
    run(args)
