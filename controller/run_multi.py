#!/usr/bin/env python3 
# -*- coding: utf-8 -*-
"""
다종 모델 통합 러너:
- CV: resnet*, vgg*, mobilenet_v2/v3 (torchvision 있으면 사용, 없으면 간단 CNN fallback)
- NLP: gpt2(tiny), bert(tiny) (내장 구현: 외부 다운로드 불필요)
- 합성 데이터로 빠르게 학습 루프를 돌리며 CSV 로그 기록 (A.csv/B.csv/...)
- libbless 크레딧 게이트와 충돌 없이 안전하게 동작(컨텍스트 고정+CPU init→GPU move)

wognsdk wlsWK enlwlrhtlvsl? dj?
"""

import os, sys, time, csv, argparse, socket, ctypes, math, random
import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import nullcontext

# ---------------- bless helpers ----------------
try:
    _lib = ctypes.CDLL(None)
    _bind = getattr(_lib, "bless_bind_thread"); _bind.argtypes=[ctypes.c_int]; _bind.restype=None
    _curr = getattr(_lib, "bless_current_route"); _curr.restype=ctypes.c_int
except Exception:
    _bind = None; _curr = None

def bind_to_current_route():
    if _bind and _curr:
        try:
            r = int(_curr())
            _bind(1 if r==1 else 0)
        except Exception:
            pass

@torch.no_grad()
def warm_cuda():
    torch.cuda.set_device(0)
    x = torch.randn(1024, device="cuda")
    y = torch.tanh(x) + torch.sin(x)
    _ = y.sum().item()
    torch.cuda.synchronize()

# ---------------- (optional) CUTLASS linear ----------------
try:
    # 있으면 사용, 없으면 torch.matmul 경로
    from controller.cutlass_linear import MyLinear as _CUTLASS_Linear
    from controller.cutlass_linear import matmul2d as _matmul2d
    HAVE_CUTLASS = True
except Exception:
    HAVE_CUTLASS = False
    class _CUTLASS_Linear(nn.Linear): pass
    def _matmul2d(A, B): return torch.matmul(A, B)

# ---------------- Tiny GPT/BERT (내장) ----------------
def _sdpa_ctx():
    try:
        from torch.nn.attention import sdpa_kernel, SDPBackend
        return sdpa_kernel(backends=[SDPBackend.MATH])
    except Exception:
        return nullcontext()

class _SelfAttn(nn.Module):
    def __init__(self, d_model, n_head, causal: bool):
        super().__init__()
        assert d_model % n_head == 0
        self.nh = n_head
        self.hd = d_model // n_head
        # QKV/PROJ는 CUTLASS가 있으면 그쪽으로
        self.qkv  = _CUTLASS_Linear(d_model, 3*d_model, bias=False)
        self.proj = _CUTLASS_Linear(d_model, d_model, bias=False)
        self.causal = causal

    def forward(self, x):  # x: [B,S,D]
        B,S,D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        def pack(t): return t.view(B, S, self.nh, self.hd).transpose(1,2).contiguous()
        q, k, v = pack(q), pack(k), pack(v)

        q2 = q.reshape(B*self.nh, S, self.hd)
        k2 = k.reshape(B*self.nh, S, self.hd)
        scores = torch.empty(B*self.nh, S, S, device=x.device, dtype=torch.float32)
        for i in range(B*self.nh):
            scores[i] = _matmul2d(q2[i], k2[i].transpose(0,1))
        scores = scores / (self.hd ** 0.5)
        if self.causal:
            m = torch.ones(S, S, device=x.device, dtype=torch.bool).triu(1)
            scores = scores.masked_fill(m.unsqueeze(0), float('-inf'))
        attn = torch.softmax(scores, dim=-1)

        v2 = v.reshape(B*self.nh, S, self.hd)
        y2 = torch.empty(B*self.nh, S, self.hd, device=x.device, dtype=v2.dtype)
        for i in range(B*self.nh):
            y2[i] = _matmul2d(attn[i], v2[i])
        y = y2.reshape(B, self.nh, S, self.hd).transpose(1,2).contiguous().view(B,S,D)
        return self.proj(y)

class _GPTBlock(nn.Module):
    def __init__(self, d_model, n_head, ff_mult=4):
        super().__init__()
        self.attn = _SelfAttn(d_model, n_head, causal=True)
        self.ln1  = nn.LayerNorm(d_model)
        self.ff   = nn.Sequential(
            nn.Linear(d_model, ff_mult*d_model),
            nn.GELU(),
            nn.Linear(ff_mult*d_model, d_model),
        )
        self.ln2  = nn.LayerNorm(d_model)
    def forward(self, x):
        x = self.ln1(x + self.attn(x))
        x = self.ln2(x + self.ff(x))
        return x

class _EncBlock(nn.Module):
    def __init__(self, d_model, n_head, ff_mult=4):
        super().__init__()
        self.attn = _SelfAttn(d_model, n_head, causal=False)
        self.ln1  = nn.LayerNorm(d_model)
        self.ff   = nn.Sequential(
            nn.Linear(d_model, ff_mult*d_model),
            nn.GELU(),
            nn.Linear(ff_mult*d_model, d_model),
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
        self.blocks = nn.ModuleList([_GPTBlock(d_model, n_head, ff_mult) for _ in range(n_layer)])
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
        self.blocks = nn.ModuleList([_EncBlock(d_model, n_head, ff_mult) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm   = nn.Linear(d_model, vocab, bias=False)
    def forward(self, idx):
        B,S = idx.shape
        x = self.tok(idx) + self.pos.weight[:S]
        for blk in self.blocks: x = blk(x)
        x = self.ln_f(x)
        return self.lm(x)

def build_nlp(model: str, vocab, d_model, n_layer, n_head, seq, ff_mult):
    model = model.lower()
    if model in ("gpt","gpt2","tinygpt","tinygpt2"):
        return TinyGPT2(vocab, d_model, n_layer, n_head, seq, ff_mult)
    if model in ("bert","tinybert"):
        return TinyBERT(vocab, d_model, n_layer, n_head, seq, ff_mult)
    raise ValueError(f"Unknown NLP model: {model}")

# ---------------- CV models (torchvision, with fallback) ----------------
def try_build_torchvision(model: str, num_classes: int):
    import torchvision.models as tvm
    model = model.lower()
    def ctor(name):
        if not hasattr(tvm, name): raise ValueError
        fn = getattr(tvm, name)
        try:
            return fn(weights=None)
        except TypeError:
            return fn(pretrained=False)
    if model in ("resnet18","resnet34","resnet50","resnet101"):
        m = ctor(model); m.fc = nn.Linear(m.fc.in_features, num_classes); return m
    if model in ("vgg11","vgg16","vgg19"):
        m = ctor(model); m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes); return m
    if model in ("mobilenet_v2","mobilenet_v3_small","mobilenet_v3_large"):
        m = ctor(model)
        last = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(last, num_classes)
        return m
    if model in ("efficientnet_b0",):
        m = ctor(model); m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes); return m
    raise ValueError

class TinyCNN(nn.Module):
    """torchvision이 없을 때 쓰는 간단한 CNN fallback"""
    def __init__(self, num_classes=1000):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1), nn.ReLU(), nn.Conv2d(128,128,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128,256,3,padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((1,1))
        )
        self.fc = nn.Linear(256, num_classes)
    def forward(self, x):
        x = self.net(x)
        x = x.flatten(1)
        return self.fc(x)

def build_cv(model: str, num_classes: int):
    try:
        return try_build_torchvision(model, num_classes)
    except Exception:
        print(f"[warn] torchvision '{model}' 사용 실패 → TinyCNN fallback", flush=True)
        return TinyCNN(num_classes)

# ---------------- Train loops ----------------
def run_nlp(args):
    torch.cuda.set_device(0); torch.rand(1, device="cuda"); torch.cuda.synchronize()
    tenant = os.environ.get("BLESS_TENANT","")
    run_name = os.environ.get("RUNNER_NAME") or (tenant if tenant else f"{args.model}-{os.getpid()}-{time.strftime('%m%d-%H%M%S')}")
    csv_path = f"{run_name}.csv"
    print(f"[{run_name}] csv -> {csv_path}", flush=True)
    f = open(csv_path, "w", newline="", buffering=1); w = csv.writer(f)
    w.writerow(["t_s","step","loss","tok_per_s"])

    bind_to_current_route()
    warm_cuda()
    torch.cuda.set_device(0)

    # CPU init → GPU (컨텍스트 안전)
    m = build_nlp(args.model, args.vocab, args.d_model, args.n_layer, args.n_head, args.seq, args.ff_mult).cpu()
    for mod in m.modules():
        if isinstance(mod, (nn.Linear, nn.Embedding)):
            if hasattr(mod, "weight"): nn.init.trunc_normal_(mod.weight, std=0.02)
            if hasattr(mod, "bias") and mod.bias is not None: nn.init.zeros_(mod.bias)
    m = m.cuda().train()

    opt = torch.optim.AdamW(m.parameters(), lr=1e-4, betas=(0.9,0.95), weight_decay=0.01, foreach=False, fused=False)
    t0 = time.time()

    for step in range(1, args.steps+1):
        x = torch.randint(0, args.vocab, (args.batch, args.seq), device="cuda", dtype=torch.long)
        y = torch.randint(0, args.vocab, (args.batch, args.seq), device="cuda", dtype=torch.long)

        tic = time.time()
        opt.zero_grad(set_to_none=True)
        out = m(x)
        loss = F.cross_entropy(out.float().clamp_(-30,30).reshape(-1, out.size(-1)),
                               y.reshape(-1), label_smoothing=0.05)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        torch.cuda.synchronize()

        dt = max(1e-6, time.time()-tic)
        tps = (args.batch * args.seq) / dt
        w.writerow([f"{time.time()-t0:.3f}", step, f"{loss.item():.4f}", f"{tps:.1f}"]); f.flush(); os.fsync(f.fileno())

        if step % 10 == 0:
            print(f"[{run_name}] step={step} tok/s={tps:.0f}", flush=True)
        torch.cuda._sleep(600000)  # 약간의 간격

    f.close()
    print(f"[train-nlp] {tenant} done steps={args.steps}", flush=True)

def run_cv(args):
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(0); torch.rand(1, device="cuda"); torch.cuda.synchronize()
    tenant = os.environ.get("BLESS_TENANT","")
    run_name = os.environ.get("RUNNER_NAME") or (tenant if tenant else f"{args.model}-{os.getpid()}-{time.strftime('%m%d-%H%M%S')}")
    csv_path = f"{run_name}.csv"
    print(f"[{run_name}] csv -> {csv_path}", flush=True)
    f = open(csv_path, "w", newline="", buffering=1); w = csv.writer(f)
    w.writerow(["t_s","step","loss","img_per_s"])

    bind_to_current_route()
    warm_cuda()
    torch.cuda.set_device(0)

    m = build_cv(args.model, args.classes).cpu()
    for mod in m.modules():
        if isinstance(mod, (nn.Conv2d, nn.Linear)):
            if hasattr(mod, "weight"): nn.init.kaiming_normal_(mod.weight, nonlinearity="relu")
            if hasattr(mod, "bias") and mod.bias is not None: nn.init.zeros_(mod.bias)
    m = m.cuda().train()

    if args.optim == "sgd":
        opt = torch.optim.SGD(m.parameters(), lr=0.4, momentum=0.9, weight_decay=1e-4)
    else:
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3, betas=(0.9,0.95), weight_decay=0.01)

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    B, C, H, W = args.batch, 3, args.imgsz, args.imgsz
    num_cls = args.classes
    t0 = time.time()

    for step in range(1, args.steps+1):
        images = torch.randn(B, C, H, W, device="cuda")
        labels = torch.randint(0, num_cls, (B,), dtype=torch.long, device="cuda")

        tic = time.time()
        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=args.amp):
            logits = m(images)
            loss = F.cross_entropy(logits.float().clamp_(-30,30), labels, label_smoothing=0.05)

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        scaler.step(opt); scaler.update()
        torch.cuda.synchronize()

        dt = max(1e-6, time.time()-tic)
        ips = B / dt
        w.writerow([f"{time.time()-t0:.3f}", step, f"{loss.item():.4f}", f"{ips:.1f}"]); f.flush(); os.fsync(f.fileno())

        if step % 10 == 0:
            print(f"[{run_name}] step={step} img/s={ips:.0f}", flush=True)
        torch.cuda._sleep(500000)

    f.close()
    print(f"[train-cv] {tenant} done steps={args.steps}", flush=True)

# ---------------- CLI ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=["nlp","cv"], required=True, help="모델 도메인 선택")
    ap.add_argument("--model", required=True,
                    help="NLP: gpt2/tinygpt, bert/tinybert | CV: resnet18/34/50/101, vgg11/16/19, mobilenet_v2/…")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--amp", action="store_true", help="(CV) mixed precision")

    # NLP
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--seq",   type=int, default=384)
    ap.add_argument("--vocab", type=int, default=4096)
    ap.add_argument("--d_model", type=int, default=768)
    ap.add_argument("--n_layer", type=int, default=6)
    ap.add_argument("--n_head",  type=int, default=12)
    ap.add_argument("--ff_mult", type=int, default=4)

    # CV
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--classes", type=int, default=1000)
    ap.add_argument("--optim", choices=["sgd","adamw"], default="sgd")

    args = ap.parse_args()

    # 안전 기본값(있으면 사용)
    os.environ.setdefault("BLESS_LINEAR", "cutlass")
    os.environ.setdefault("BLESS_SAFE_GEMM", "1")

    if args.domain == "nlp":
        run_nlp(args)
    else:
        run_cv(args)

if __name__ == "__main__":
    main()
