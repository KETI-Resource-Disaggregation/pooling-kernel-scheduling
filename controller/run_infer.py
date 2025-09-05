import os, time, csv, argparse
import torch
import torch.nn as nn
import torchvision.models as tvm

def assert_heads(d_model: int, n_head: int):
    assert d_model % n_head == 0, f"d_model({d_model}) must be divisible by n_head({n_head})"

def resolve_vocab(model: str, vocab_arg: int, reserve_arg: int):
    """
    Returns (num_embeddings, reserve_effective).
    - GPT2 : num_embeddings=vocab(default 50257), reserve=0
    - BERT : num_embeddings=vocab+reserve(default 30522+3), reserve>=2([CLS],[SEP])
    """
    if model == "gpt2":
        vocab = vocab_arg if vocab_arg > 0 else 50257
        return vocab, 0
    if model == "bert":
        vocab = vocab_arg if vocab_arg > 0 else 30522
        reserve = max(2, reserve_arg)
        return vocab + reserve, reserve
    return 0, 0 

def build_model(args):
    if args.model == "gpt2":
        assert_heads(args.d_model, args.n_head)
        class TinyGPT2(nn.Module):
            def __init__(self, d=768, n_layer=12, num_embeddings=50257):
                super().__init__()
                self.emb = nn.Embedding(num_embeddings, d)
                self.blocks = nn.ModuleList([
                    nn.Sequential(
                        nn.LayerNorm(d),
                        nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d)
                    ) for _ in range(n_layer)
                ])
                self.ln = nn.LayerNorm(d)
                self.lm = nn.Linear(d, num_embeddings)
            def forward(self, x):
                h = self.emb(x)  
                for blk in self.blocks:
                    h = h + blk(h)
                h = self.ln(h)
                return self.lm(h[:, -1, :])  
        return TinyGPT2(d=args.d_model, n_layer=args.n_layer, num_embeddings=args.num_embeddings).cuda().eval()

    if args.model == "bert":
        assert_heads(args.d_model, args.n_head)
        class TinyBERT(nn.Module):
            def __init__(self, d=768, n_layer=12, num_embeddings=30522):
                super().__init__()
                self.emb = nn.Embedding(num_embeddings, d, padding_idx=0)
                self.blocks = nn.ModuleList([
                    nn.Sequential(
                        nn.LayerNorm(d),
                        nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d)
                    ) for _ in range(n_layer)
                ])
                self.ln = nn.LayerNorm(d)
                self.cls = nn.Linear(d, 2)
            def forward(self, x):
                h = self.emb(x)
                for blk in self.blocks:
                    h = h + blk(h)
                h = self.ln(h)
                return self.cls(h[:, 0, :])  # CLS
        return TinyBERT(d=args.d_model, n_layer=args.n_layer, num_embeddings=args.num_embeddings).cuda().eval()

    if args.model in ("resnet50","resnet101"):
        net = tvm.resnet50(weights=None) if args.model == "resnet50" else tvm.resnet101(weights=None)
        return net.cuda().eval()

    raise ValueError(f"unknown model {args.model}")

def make_inputs(args):
    if args.model == "gpt2":
        x = torch.randint(0, args.num_embeddings, (args.batch, args.seq), device="cuda", dtype=torch.long)
        assert x.dtype == torch.long
        mx = int(x.max().item()); assert mx < args.num_embeddings, f"GPT2 token OOB: max={mx} vs {args.num_embeddings}"
        return (x,)
    if args.model == "bert":
        B, T = args.batch, args.seq
        reserve = args.reserve_effective
        num_emb = args.num_embeddings
        assert reserve >= 2, "BERT needs at least CLS/SEP reserved"
        body_len = max(1, T-2)
        body = torch.randint(low=reserve, high=num_emb, size=(B, body_len), device="cuda", dtype=torch.long)
        CLS = torch.full((B,1), 1, device="cuda", dtype=torch.long)
        SEP = torch.full((B,1), 2, device="cuda", dtype=torch.long)
        x = torch.cat([CLS, body, SEP], dim=1)
        assert x.dtype == torch.long
        mn = int(x.min().item()); mx = int(x.max().item())
        assert 0 <= mn and mx < num_emb, f"BERT token OOB: [{mn},{mx}] vs {num_emb}"
        return (x,)
    x = torch.randn(args.batch, 3, args.img, args.img, device="cuda")
    return (x,)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["gpt2","bert","resnet50","resnet101"])
    ap.add_argument("--n_reqs", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=10, help="warmup iterations (ignored if --warmup_s>0)")
    ap.add_argument("--duration_s", type=float, default=0.0, help=">0이면 시간 기반 실행 (초)")
    ap.add_argument("--warmup_s", type=float, default=0.0, help=">0이면 초 단위 워밍업 (우선 적용)")
    ap.add_argument("--pace_ms", type=float, default=0.0, help="요청 간 최소 간격(ms); 0이면 최대속도")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--img", type=int, default=224)
    ap.add_argument("--n_layer", type=int, default=12)
    ap.add_argument("--n_head", type=int, default=12)
    ap.add_argument("--d_model", type=int, default=768)
    ap.add_argument("--vocab", type=int, default=-1, help="GPT2: default 50257, BERT: default 30522")
    ap.add_argument("--reserve_special", type=int, default=3, help="BERT에서 [PAD,CLS,SEP] 등 예약 개수 (>=2)")
    args = ap.parse_args()

    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        torch.cuda.set_device(0)

    num_embeddings, reserve_effective = resolve_vocab(args.model, args.vocab, args.reserve_special)
    args.num_embeddings = num_embeddings
    args.reserve_effective = reserve_effective

    model = build_model(args)
    inp = make_inputs(args)

    if args.warmup_s and args.warmup_s > 0:
        t0w = time.time()
        while time.time() - t0w < args.warmup_s:
            with torch.no_grad():
                _ = model(*inp); torch.cuda.synchronize()
    else:
        for _ in range(max(0, args.warmup)):
            with torch.no_grad():
                _ = model(*inp); torch.cuda.synchronize()

    runner = os.environ.get("RUNNER_NAME","inference")
    csv_path = runner + ".csv" if not runner.endswith(".csv") else runner
    dname = os.path.dirname(csv_path)
    if dname: os.makedirs(dname, exist_ok=True)
    f = open(csv_path, "w", newline="", buffering=1)
    w = csv.writer(f)
    w.writerow(["req_id","t_s","lat_ms","qps","tok_per_s","imgs_per_s"])

    use_time_mode = (args.duration_s and args.duration_s > 0)
    req_id = 0
    t_last = time.time()
    pace_s = args.pace_ms/1000.0 if args.pace_ms and args.pace_ms > 0 else 0.0
    next_deadline = time.time()

    if use_time_mode:
        t_end = time.time() + args.duration_s
        while time.time() < t_end:
            if pace_s > 0:
                now = time.time()
                if now < next_deadline: time.sleep(next_deadline - now)
            t0 = time.time()
            with torch.no_grad():
                _ = model(*inp); torch.cuda.synchronize()
            t1 = time.time()
            req_id += 1
            lat_ms = (t1 - t0) * 1000.0
            dt = max(1e-6, t1 - t_last); qps = 1.0/dt; t_last = t1
            tokps = (args.batch*args.seq) / (lat_ms/1000.0) if args.model in ("gpt2","bert") else 0.0
            imgps = (args.batch) / (lat_ms/1000.0) if args.model.startswith("resnet") else 0.0
            w.writerow([req_id, f"{t1:.6f}", f"{lat_ms:.3f}", f"{qps:.3f}", f"{tokps:.3f}", f"{imgps:.3f}"])
            if pace_s > 0: next_deadline = t0 + pace_s
    else:
        for i in range(1, args.n_reqs+1):
            if pace_s > 0:
                now = time.time()
                if now < next_deadline: time.sleep(next_deadline - now)
            t0 = time.time()
            with torch.no_grad():
                _ = model(*inp); torch.cuda.synchronize()
            t1 = time.time()
            lat_ms = (t1 - t0) * 1000.0
            dt = max(1e-6, t1 - t_last); qps = 1.0/dt; t_last = t1
            tokps = (args.batch*args.seq) / (lat_ms/1000.0) if args.model in ("gpt2","bert") else 0.0
            imgps = (args.batch) / (lat_ms/1000.0) if args.model.startswith("resnet") else 0.0
            w.writerow([i, f"{t1:.6f}", f"{lat_ms:.3f}", f"{qps:.3f}", f"{tokps:.3f}", f"{imgps:.3f}"])
            if pace_s > 0: next_deadline = t0 + pace_s

    f.close()

if __name__ == "__main__":
    main()
