#!/usr/bin/env python3
# controller/run_cnn.py
import os, sys, time, csv, argparse, ctypes, socket, torch, torch.nn as nn, torch.nn.functional as F
from contextlib import nullcontext

MASTER = os.environ.get("BLESS_MASTER", "/tmp/bless-master.sock")
TENANT = os.environ.get("BLESS_TENANT", "")

def send_master(msg: str):
    try:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM); s.sendto(msg.encode(), MASTER); s.close()
    except Exception: pass

def _load_c_symbol(name, restype=ctypes.c_int):
    try:
        lib = ctypes.CDLL(None); f = getattr(lib, name); f.restype = restype; return f
    except Exception: return None

_bless_bind = _load_c_symbol("bless_bind_thread")  # void (int route)

def bind_thread(route=0):
    if _bless_bind: 
        try: _bless_bind(ctypes.c_int(route))
        except Exception: pass

@torch.no_grad()
def _aten_warm_ops():
    x = torch.randn(1024, device="cuda"); y = torch.tanh(x) + torch.sin(x); _ = y.sum().item()

def build_cnn(name, num_classes=1000):
    import torchvision.models as tm
    name = name.lower()
    if name == "resnet50":  m = tm.resnet50(weights=None, num_classes=num_classes)
    elif name == "resnet101": m = tm.resnet101(weights=None, num_classes=num_classes)
    elif name == "vgg16":   m = tm.vgg16(num_classes=num_classes)
    elif name == "vgg19":   m = tm.vgg19(num_classes=num_classes)
    elif name == "inception_v3": m = tm.inception_v3(weights=None, aux_logits=False, num_classes=num_classes)
    else: raise ValueError(f"unknown cnn: {name}")
    return m

@torch.no_grad()
def rand_batch(bs, shape, num_classes, device):
    x = torch.randn((bs, *shape), device=device)
    y = torch.randint(0, num_classes, (bs,), device=device)
    return x, y

def run(args):
    torch.cuda.set_device(0); torch.rand(1, device="cuda"); torch.cuda.synchronize()
    tenant = os.environ.get("BLESS_TENANT") or f"cnn-{os.getpid()}"
    run_name = os.environ.get("RUNNER_NAME") or tenant
    csv_path = f"{run_name}.csv"
    print(f"[{tenant}] csv -> {csv_path}", flush=True)
    f = open(csv_path, "w", newline="", buffering=1); w = csv.writer(f)
    w.writerow(["t_s","step","loss","imgs_per_s"]); f.flush(); os.fsync(f.fileno())

    bind_thread(0); _aten_warm_ops()

    model = build_cnn(args.model, num_classes=args.num_classes).cuda().train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9,0.95), weight_decay=0.01, foreach=False, fused=False)
    torch.cuda.synchronize(); t0 = time.time()
    C=3; H,W = args.imgsz, args.imgsz

    for step in range(1, args.steps+1):
        bind_thread(0)  # 스텝마다 바인딩 보강
        x, y = rand_batch(args.batch, (C,H,W), args.num_classes, "cuda")
        opt.zero_grad(set_to_none=True)
        tic = time.time()
        logits = model(x)
        # NaN/Inf 방지
        logits = torch.nan_to_num(logits.float(), nan=0.0, posinf=30.0, neginf=-30.0).clamp_(-30, 30)
        loss = F.cross_entropy(logits, y)

        if not torch.isfinite(loss):
            print(f"[guard] non-finite loss at step={step}, skip", flush=True)
            opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            continue

        try:
            loss.backward()
        except Exception as e:
            print(f"[guard] backward failed at step={step}: {e}. skip", flush=True)
            opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            continue

        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        torch.cuda.synchronize()
        dt = max(1e-6, time.time()-tic); ips = args.batch / dt
        w.writerow([f"{time.time()-t0:.3f}", step, f"{loss.item():.4f}", f"{ips:.1f}"]); f.flush(); os.fsync(f.fileno())
        if step % 10 == 0: print(f"[{tenant}] step={step} imgs/s={ips:.0f}", flush=True)
        torch.cuda._sleep(1000000)
    f.close(); print(f"[train] {tenant} done steps={args.steps}", flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["resnet50","resnet101","vgg16","vgg19","inception_v3"], required=True)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--num_classes", type=int, default=1000)
    args = ap.parse_args()
    run(args)
