#!/usr/bin/env python3
import os, sys, csv, json, argparse, math, statistics as st
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODES = ["UNBOUND","SALUS","BLESSISH","OURS"]

def read_csv_series(p):
    out = {"t":[], "lat_ms":[], "tok_per_s":[], "imgs_per_s":[]}
    if not os.path.exists(p): return out
    with open(p, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                out["t"].append(float(row["t_s"]))
                out["lat_ms"].append(float(row.get("lat_ms","nan")))
                out["tok_per_s"].append(float(row.get("tok_per_s","0")))
                out["imgs_per_s"].append(float(row.get("imgs_per_s","0")))
            except: pass
    return out

def bin_series_1s(series, trim_first=3.0, trim_last=5.0):
    if not series["t"]: return [], [], [], []
    t0 = series["t"][0]
    t_end = series["t"][-1]
    t0 += trim_first
    t_end -= trim_last
    if t_end <= t0: 
        return [], [], [], []
    # 활동 구간만 필터
    T, L, K, I = [], [], [], []
    for t,l,k,i in zip(series["t"], series["lat_ms"], series["tok_per_s"], series["imgs_per_s"]):
        if t < t0 or t > t_end: 
            continue
        T.append(t - t0); L.append(l); K.append(k); I.append(i)
    if not T: return [], [], [], []
    # 1초 bin
    maxs = int(math.floor(T[-1]))
    bins_lat = [[] for _ in range(maxs+1)]
    bins_tp  = [[] for _ in range(maxs+1)]
    for tt, l, k, ii in zip(T, L, K, I):
        bi = min(maxs, int(math.floor(tt)))
        if math.isfinite(l): bins_lat[bi].append(l)
        # 합산 처리량(토큰+이미지) – 단위 혼합이지만 전체 변동을 보기 위함
        bins_tp[bi].append(k + ii)
    xs = list(range(maxs+1))
    lat = [ (st.mean(b) if b else math.nan) for b in bins_lat ]
    tp  = [ (st.mean(b) if b else 0.0) for b in bins_tp ]
    return xs, lat, tp, (t0, t_end)

def collect(root):
    """
    root/CASE/SIZE/(quota|rep-*)/MODE/{A.csv,B.csv,...}
    → case_size_mode -> list of per-tenant series (merged across quotas/reps)
    """
    out = defaultdict(lambda: defaultdict(list))  # key -> tenant -> [series dict]
    for case in sorted(os.listdir(root)):
        cdir = os.path.join(root, case)
        if not os.path.isdir(cdir): continue
        for size in sorted(os.listdir(cdir)):
            sdir = os.path.join(cdir, size)
            if not os.path.isdir(sdir): continue
            for node in sorted(os.listdir(sdir)):
                ndir = os.path.join(sdir, node)
                if not os.path.isdir(ndir): continue
                # rep-* 또는 quota
                rep_nodes = []
                if node.startswith("rep-"):
                    rep_nodes = [ndir]
                else:
                    # quota 디렉터리
                    for rep in sorted(os.listdir(ndir)):
                        rep_nodes.append(os.path.join(ndir, rep))
                for rdir in rep_nodes:
                    for mode in MODES:
                        mdir = os.path.join(rdir, mode)
                        if not os.path.isdir(mdir): continue
                        key = f"{case}/{size}/{mode}"
                        for fn in os.listdir(mdir):
                            if not fn.endswith(".csv"): continue
                            tenant = os.path.splitext(fn)[0]
                            ser = read_csv_series(os.path.join(mdir, fn))
                            out[key][tenant].append(ser)
    return out

def plot_overlay(root, outdir, bin_s=1):
    os.makedirs(outdir, exist_ok=True)
    blob = collect(root)
    for key, per_t in blob.items():
        case, size, mode = key.split("/")
        # tenant별 평균 시계열을 만들고, 다시 전체 평균으로 합치기
        agg_tp = None; agg_lat = None; maxlen = 0
        for tname, ser_list in per_t.items():
            # rep/쿼터 합치기: 같은 길이로 맞춘 뒤 평균
            curves_tp = []; curves_lat = []
            common_x = None
            for ser in ser_list:
                xs, lat, tp, _ = bin_series_1s(ser, trim_first=3.0, trim_last=5.0)
                if not xs: continue
                if common_x is None:
                    common_x = xs
                else:
                    L = min(len(common_x), len(xs))
                    common_x = common_x[:L]; lat = lat[:L]; tp = tp[:L]
                curves_tp.append(tp[:len(common_x)])
                curves_lat.append(lat[:len(common_x)])
            if not curves_tp: continue
            mean_tp  = [st.mean([c[i] for c in curves_tp]) for i in range(len(common_x))]
            mean_lat = [st.mean([c[i] for c in curves_lat if math.isfinite(c[i])]) if any(math.isfinite(c[i]) for c in curves_lat) else math.nan for i in range(len(common_x))]
            maxlen = max(maxlen, len(common_x))
            if agg_tp is None:
                agg_tp = mean_tp
                agg_lat = mean_lat
                X = list(range(len(common_x)))
            else:
                L = min(len(agg_tp), len(mean_tp))
                agg_tp  = [ (agg_tp[i] + mean_tp[i]) / 2 for i in range(L) ]
                agg_lat = [ (agg_lat[i] + mean_lat[i]) / 2 for i in range(L) ]
        if agg_tp is None: 
            continue

        # 저장
        fig = plt.figure(figsize=(10,3.4), dpi=140)
        plt.plot(range(len(agg_tp)), agg_tp, linewidth=1.5)
        plt.title(f"{key} — total throughput (trim 3s/5s)")
        plt.xlabel("time (s)"); plt.ylabel("token/s + image/s")
        plt.grid(True, alpha=0.25); plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{case}_{size}_{mode}_tp.png")); plt.close(fig)

        fig = plt.figure(figsize=(10,3.4), dpi=140)
        plt.plot(range(len(agg_lat)), agg_lat, linewidth=1.5)
        plt.title(f"{key} — avg latency (trim 3s/5s)")
        plt.xlabel("time (s)"); plt.ylabel("latency (ms)")
        plt.grid(True, alpha=0.25); plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{case}_{size}_{mode}_lat.png")); plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="archive/logs/EVAL_INFER")
    ap.add_argument("--outdir", default="archive/logs/EVAL_INFER/paper_figs_ts")
    args = ap.parse_args()
    plot_overlay(args.root, args.outdir)
    print(f"[ok] timeseries -> {args.outdir}")

if __name__ == "__main__":
    main()
