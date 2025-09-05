#!/usr/bin/env python3
import os, sys, time, csv, argparse, warnings, math, random, socket
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

# bless bind helpers
import ctypes
try:
    _lib = ctypes.CDLL(None)
    _bind = getattr(_lib, "bless_bind_thread"); _bind.argtypes=[ctypes.c_int]; _bind.restype=None
    _curr = getattr(_lib, "bless_current_route"); _curr.restype=ctypes.c_int
except Exception:
    _bind = None; _curr = None

def bind_to_current_route():
    """현재 설정된 route(LIMITED/UNLIMITED)에 스레드를 바인딩."""
    if _bind and _curr:
        try:
            r = int(_curr())
            _bind(1 if r==1 else 0)
        except Exception:
            pass

# ===== 안전 워밍업 & 전역 가드 =====
@torch.no_grad()
def _aten_warm_ops():
    x = torch.randn(1024, device="cuda")
    y = torch.tanh(x) + torch.sin(x)
    _ = y.sum().item()

# 안전 옵션 (환경변수로 덮을 수 있음)
os.environ.setdefault("BLESS_LINEAR", "cutlass")
os.environ.setdefault("BLESS_SAFE_GEMM", "1")
if os.environ.get("ANOMALY", "0") == "1":
    torch.autograd.set_detect_anomaly(True)

# ===== 모델 정의 =====
from contextlib import nullcontext
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
    def forward(self, x):
        B,S,D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        def pack(t):
            return t.view(B, S, self.n_head, self.head_dim).transpose(1, 2).contiguous()
        q, k, v = pack(q), pack(k), pack(v)
        q2 = q.reshape(B*self.n_head, S, self.head_dim)
        k2 = k.reshape(B*self.n_head, S, self.head_dim)
        scores = torch.empty(B*self.n_head, S, S, device=x.device, dtype=torch.float32)
        for i in range(B*self.n_head):
            scores[i] = matmul2d(q2[i], k2[i].transpose(0,1))
        scores = scores / (self.head_dim ** 0.5)
        if self.causal:
            m = torch.ones(S, S, device=x.device, dtype=torch.bool).triu(1)
            scores = scores.masked_fill(m.unsqueeze(0), float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        v2 = v.reshape(B*self.n_head, S, self.head_dim)
        y2 = torch.empty(B*self.n_head, S, self.head_dim, device=x.device, dtype=v2.dtype)
        for i in range(B*self.n_head):
            y2[i] = matmul2d(attn[i], v2[i])
        y = y2.reshape(B, self.n_head, S, self.head_dim).transpose(1,2).contiguous().view(B,S,D)
        return self.proj(y)

class GPTBlock(nn.Module):
    def __init__(self, d_model, n_head, ff_mult=4):
        super().__init__()
        self.attn = SelfAttention(d_model, n_head, causal=True)
        self.ln1  = nn.LayerNorm(d_model)
        self.ff   = nn.Sequential(
            nn.Linear(d_model, ff_mult*d_model),
            nn.GELU(),
            nn.Linear(ff_mult*d_model, d_model)
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
            nn.Linear(d_model, ff_mult*d_model),
            nn.GELU(),
            nn.Linear(ff_mult*d_model, d_model)
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
        self.lm   = nn.Linear(d_model, vocab, bias=False)
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
        self.lm   = nn.Linear(d_model, vocab, bias=False)
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

def _one_step(model, opt, args):
    """한 스텝을 수행하고 (loss, tok_per_s) 반환. 실패 시 None."""
    bind_to_current_route()
    # 입력
    x = torch.randint(0, args.vocab, (args.batch, args.seq), device="cuda", dtype=torch.long)
    y = torch.randint(0, args.vocab, (args.batch, args.seq), device="cuda", dtype=torch.long)

    opt.zero_grad(set_to_none=True)
    tic = time.time()

    logits = model(x)
    logits = torch.nan_to_num(logits.float(), nan=0.0, posinf=30.0, neginf=-30.0).clamp_(-30, 30)

    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        y.reshape(-1),
        label_smoothing=0.05
    )

    if not torch.isfinite(loss):
        print(f"[guard] non-finite loss, skip", flush=True)
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        return None

    try:
        loss.backward()
    except Exception as e:
        print(f"[guard] backward failed: {e}. skip", flush=True)
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        return None

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    torch.cuda.synchronize()

    dt = max(1e-6, time.time() - tic)
    tps = (args.batch * args.seq) / dt
    return (float(loss.item()), float(tps))

def run(args):
    torch.cuda.set_device(0)
    torch.rand(1, device="cuda"); torch.cuda.synchronize()

    tenant = os.environ.get("BLESS_TENANT", "")
    run_name = os.environ.get("RUNNER_NAME") or (tenant if tenant else f"{args.model}-{os.getpid()}-{time.strftime('%m%d-%H%M%S')}")
    csv_path = f"{run_name}.csv"
    print(f"[{run_name}] csv opened -> {csv_path}", flush=True)
    f = open(csv_path, "w", newline="", buffering=1)
    w = csv.writer(f); w.writerow(["t_s","step","loss","tok_per_s"]); f.flush(); os.fsync(f.fileno())

    # bless route 바인딩 & ATen 워밍업
    bind_to_current_route()
    _aten_warm_ops()
    torch.cuda.set_device(0)

    # 모델 초기화 (CPU -> GPU 이동)
    model = build_model(args.model, args.vocab, args.d_model, args.n_layer, args.n_head, args.seq, args.ff_mult).cpu()
    def _init_small(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        if isinstance(m, nn.Embedding):
            nn.init.trunc_normal_(m.weight, std=0.02)
    model.apply(_init_small)
    model = model.cuda().train()

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9,0.95), weight_decay=0.01, foreach=False, fused=False)
    torch.cuda.synchronize(); torch.cuda._sleep(1000)

    # ==== 워밍업(시간 기반) ====
    if args.warmup_s > 0:
        print(f"[{run_name}] warmup_s={args.warmup_s}", flush=True)
        t_warm_end = time.time() + args.warmup_s
        while time.time() < t_warm_end:
            _ = _one_step(model, opt, args)

    # ==== 메인 루프: duration_s > 0 이면 시간 기반, 아니면 steps 기반 ====
    t0 = time.time()
    step = 0
    if args.duration_s > 0:
        t_end = t0 + args.duration_s
        while time.time() < t_end:
            step += 1
            out = _one_step(model, opt, args)
            if out is None: 
                continue
            loss, tps = out
            w.writerow([f"{time.time()-t0:.3f}", step, f"{loss:.4f}", f"{tps:.1f}"])
            f.flush(); os.fsync(f.fileno())
            if step % 10 == 0:
                print(f"[{run_name}] step={step} tok/s={tps:.0f}", flush=True)
    else:
        for step in range(1, args.steps+1):
            out = _one_step(model, opt, args)
            if out is None:
                continue
            loss, tps = out
            w.writerow([f"{time.time()-t0:.3f}", step, f"{loss:.4f}", f"{tps:.1f}"])
            f.flush(); os.fsync(f.fileno())
            if step % 10 == 0:
                print(f"[{run_name}] step={step} tok/s={tps:.0f}", flush=True)

    f.close()
    print(f"[train] {tenant or run_name} done steps={step}", flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["gpt2","bert"], required=True)
    # 시간 기반 실행(초). 0이면 스텝 기반.
    ap.add_argument("--duration_s", type=float, default=0.0, help=">0이면 시간 기반 실행 (초)")
    ap.add_argument("--warmup_s",   type=float, default=0.0, help=">0이면 워밍업 (초)")

    # 스텝 기반 기본값
    ap.add_argument("--steps", type=int, default=200)

    # 모델/데이터 스펙
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--seq",   type=int, default=384)
    ap.add_argument("--vocab", type=int, default=4096)
    ap.add_argument("--d_model", type=int, default=1280)
    ap.add_argument("--n_layer", type=int, default=12)
    ap.add_argument("--n_head",  type=int, default=20)
    ap.add_argument("--ff_mult", type=int, default=8)
    args = ap.parse_args()
    run(args)
