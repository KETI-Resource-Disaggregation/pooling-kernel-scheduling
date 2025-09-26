#!/usr/bin/env python3
import os, time, csv, argparse, socket, json
import torch
import torch.nn as nn
import torchvision.models as tvm
import http.client
from typing import Optional

MASTER = os.environ.get("BLESS_MASTER", "/tmp/bless-master.sock")
TENANT = os.environ.get("BLESS_TENANT", "")
API_ADDR = os.environ.get("API_ADDR", "127.0.0.1")
API_PORT = int(os.environ.get("API_PORT", "6060"))
TELEM_PERIOD_S = float(os.environ.get("TELEM_PERIOD_S", "1.0"))
HTTP_TIMEOUT = float(os.environ.get("TELEM_HTTP_TIMEOUT", "0.05"))

# bless bind helpers 
import ctypes
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

def assert_heads(d_model: int, n_head: int):
    assert d_model % n_head == 0, f"d_model({d_model}) must be divisible by n_head({n_head})"

def resolve_vocab(model: str, vocab_arg: int, reserve_arg: int):
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

# ---- Telemetry Agent for inference ----
class TelemetryAgent:
    def __init__(self, tenant:str, period_s:float=1.0, host:str="127.0.0.1", port:int=6060, timeout:float=0.05):
        self.tenant = tenant or ""
        self.period = max(0.1, period_s)
        self.host = host; self.port = port; self.timeout = timeout
        self.last_send = time.time()
        self.lat_hist = []  # per-request latency (s)
        self.lat_hist_max = 40
        self.qps_ema = None
        self.alpha = 0.3
        self.last_req_ts = None
        self.last_tokps = 0.0
        self.last_imgps = 0.0
    def push_req(self, lat_s: float, tokps: float, imgps: float):
        self.lat_hist.append(lat_s); 
        if len(self.lat_hist) > self.lat_hist_max: self.lat_hist.pop(0)
        now = time.time()
        if self.last_req_ts is None: self.last_req_ts = now
        dt = max(1e-6, now - self.last_req_ts)
        inst_qps = 1.0/dt
        self.qps_ema = inst_qps if self.qps_ema is None else (self.alpha*inst_qps + (1-self.alpha)*self.qps_ema)
        self.last_req_ts = now
        self.last_tokps = tokps; self.last_imgps = imgps
        if now - self.last_send >= self.period:
            self.send()
            self.last_send = now
    def _p90_ms(self):
        if not self.lat_hist: return None
        s = sorted(self.lat_hist)
        i = int(round(0.9*(len(s)-1)))
        i = max(0, min(len(s)-1, i))
        return s[i]*1000.0
    def send(self):
        if not self.tenant: return
        body = {
            "tenant": self.tenant,
            "qps": float(self.qps_ema) if self.qps_ema is not None else None,
            "lat_p90": float(self._p90_ms()) if self._p90_ms() is not None else None,
            "tok_per_s": float(self.last_tokps),
            "imgs_per_s": float(self.last_imgps)
        }
        try:
            conn = http.client.HTTPConnection(API_ADDR, API_PORT, timeout=HTTP_TIMEOUT)
            conn.request("POST", "/telemetry", body=json.dumps(body), headers={"Content-Type": "application/json"})
            # >>> 응답을 반드시 소비! (안 하면 BrokenPipe)
            resp = conn.getresponse()
            try:
                _ = resp.read()
            finally:
                resp.close()
            conn.close()
        except Exception:
        # 텔레메트리는 베스트에포트: 조용히 무시
            pass

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
    bind_to_current_route()

    # Warmup
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

    ta = TelemetryAgent(tenant=TENANT, period_s=TELEM_PERIOD_S, host=API_ADDR, port=API_PORT, timeout=HTTP_TIMEOUT)

    def one_iter(req_id:int):
        nonlocal t_last, next_deadline
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
        w.writerow([req_id, f"{t1:.6f}", f"{lat_ms:.3f}", f"{qps:.3f}", f"{tokps:.3f}", f"{imgps:.3f}"])
        ta.push_req(lat_ms/1000.0, tokps, imgps)
        if pace_s > 0: next_deadline = t0 + pace_s

    if use_time_mode:
        t_end = time.time() + args.duration_s
        while time.time() < t_end:
            req_id += 1
            one_iter(req_id)
    else:
        for i in range(1, args.n_reqs+1):
            one_iter(i)

    f.close()

if __name__ == "__main__":
    main()
