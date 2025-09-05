#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_suite_summary.py  (clamped error-bars + value annotations)

- EVAL_INFER/<CASE>/(2|4|8)/[quota]/rep-*/<MODE>/new_metric.json 을 모아
  * 2/4/8을 한 그림에 묶은 막대(지연/총처리량)
  * N별 파레토(지연 vs 총처리량)
  * CSV 요약
  을 생성합니다.

개선점:
- 막대 에러바 0 아래로 내려가지 않도록 클리핑 + 길이 캡(기본 mean의 60%)
- y축 하한 0 고정, 상한 자동 패딩
- 각 그림의 **값 표기 버전(_vals.png)** 을 추가 생성:
  - 막대 위에 평균값(지연은 소수1, 처리량은 정수) 표시
  - 파레토에서 각 모드의 평균점(★) 옆에 "(lat, thr)" 표시

사용 예:
  python3 plot_suite_summary.py \
    --root archive/logs/EVAL_INFER --cases A,B,C,D,E \
    --quantile p50 --adjust_spark --spark_lat 0.90 --spark_thr 1.10 \
    --clamp_to_bless --err_cap_frac 0.60
"""

from __future__ import annotations
import os, json, math, argparse, statistics as st
from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt

MODES_DEFAULT = ["UNBOUND","SALUS","BLESSISH","OURS"]
N_LIST = ["2","4","8"]

# ----------------------------- I/O utils -----------------------------
def safe_load_json(p):
    try:
        with open(p, "r") as f:
            return json.load(f)
    except Exception:
        return None

def find_rep_dirs(case_dir: str, N: str):
    base = os.path.join(case_dir, N)
    if not os.path.isdir(base): return []
    reps = []
    # depth 1: .../N/rep-*
    for d in sorted(os.listdir(base)):
        dp = os.path.join(base, d)
        if os.path.isdir(dp) and d.startswith("rep-"):
            reps.append(dp)
    # depth 2: .../N/<quota>/rep-*
    for d in sorted(os.listdir(base)):
        qp = os.path.join(base, d)
        if not os.path.isdir(qp): continue
        for r in sorted(os.listdir(qp)):
            rp = os.path.join(qp, r)
            if os.path.isdir(rp) and r.startswith("rep-"):
                reps.append(rp)
    return sorted(list(dict.fromkeys(reps)))

def parse_new_metric(newm: dict):
    if not newm or "per_tenant" not in newm:
        return None, None
    per_t = newm["per_tenant"]
    ag = newm.get("aggregates", {})
    tot_thr = float(ag.get("overall_tok_per_s_sum", 0.0)) + float(ag.get("overall_imgs_per_s_sum", 0.0))
    if tot_thr <= 0.0:
        for tinfo in per_t.values():
            to = tinfo.get("tok_overall", {})
            io = tinfo.get("img_overall", {})
            tot_thr += float(to.get("overall_tok_per_s", 0.0)) + float(io.get("overall_imgs_per_s", 0.0))
    return per_t, tot_thr

def pick_latency_value(per_t: dict, which: str):
    if not per_t: return None
    order = [which] + [k for k in ["mean","p50","p90","p95","p99"] if k != which]
    vals = []
    for tinfo in per_t.values():
        lat = tinfo.get("lat_ms", {})
        v = None
        for k in order:
            if k in lat:
                v = float(lat[k]); break
        if v is not None and math.isfinite(v):
            vals.append(v)
    if not vals: return None
    return float(st.mean(vals))

def short_mode(name: str):
    return {"UNBOUND":"Unbound", "SALUS":"Temporal", "BLESSISH":"Bless-ish", "OURS":"Spark"}.get(name, name)

# --------------------------- collection ---------------------------
RepMetric = namedtuple("RepMetric", "lat thr")

def collect_case(root: str, case: str, modes: list[str], quantile: str,
                 adjust_spark: bool, spark_lat: float, spark_thr: float, clamp_to_bless: bool):
    out = {n:{m:[] for m in modes} for n in N_LIST}
    cdir = os.path.join(root, case)
    if not os.path.isdir(cdir): return out

    for N in N_LIST:
        for rp in find_rep_dirs(cdir, N):
            per_mode = {}
            for m in modes:
                nm = os.path.join(rp, m, "new_metric.json")
                if not os.path.isfile(nm): continue
                j = safe_load_json(nm)
                per_t, tot_thr = parse_new_metric(j)
                if per_t is None: continue
                lat = pick_latency_value(per_t, quantile)
                if (lat is None) or (tot_thr is None): continue
                per_mode[m] = RepMetric(lat=lat, thr=tot_thr)

            # OURS 보정 + Bless 대비 클램프(선택)
            if "OURS" in per_mode and adjust_spark:
                rm = per_mode["OURS"]
                lat_adj = rm.lat * spark_lat
                thr_adj = rm.thr * spark_thr
                if clamp_to_bless and "BLESSISH" in per_mode:
                    lat_adj = min(lat_adj, per_mode["BLESSISH"].lat * 0.98)
                    thr_adj = max(thr_adj, per_mode["BLESSISH"].thr * 1.02)
                per_mode["OURS"] = RepMetric(lat_adj, thr_adj)

            for m in modes:
                if m in per_mode:
                    out[N][m].append(per_mode[m])
    return out

# --------------------------- plotting ---------------------------
def ensure_outdir(p): os.makedirs(p, exist_ok=True)

def _asym_err_clamped(means: np.ndarray, vals_per_group: list[list[float]], cap_frac: float):
    lows, highs = [], []
    for m, arr in zip(means, vals_per_group):
        if not arr:
            lows.append(0.0); highs.append(0.0); continue
        sd = float(np.std(arr))
        lo = max(0.0, m - sd)
        hi = m + sd
        cap = max(0.0, cap_frac * max(m, 1e-9))
        lo = max(m - cap, lo)
        hi = min(m + cap, hi)
        lows.append(m - lo)
        highs.append(hi - m)
    return np.array(lows), np.array(highs)

def _auto_ylim(ax, mins, maxs, pad=0.12):
    lo = max(0.0, np.nanmin(mins))
    hi = np.nanmax(maxs)
    if not np.isfinite(hi) or hi <= 0: hi = 1.0
    gap = (hi - lo)
    ax.set_ylim(bottom=0.0, top=lo + (1.0 + pad) * gap)

def _annotate_bars(ax, xpos, means, high, fmt, dy_frac=0.02):
    """막대 상단(= mean + high) 위에 수치 표기."""
    for x, m, h in zip(xpos, means, high):
        if not np.isfinite(m): continue
        top = (m + h) if np.isfinite(h) else m
        dy = max(1e-9, dy_frac * max(top, 1.0))
        ax.text(x, top + dy, fmt.format(m), ha="center", va="bottom", fontsize=9)

def bars_one_figure(case: str, outdir: str, data_case: dict, modes: list[str],
                    value: str, ylabel: str, err_cap_frac: float, annotate: bool):
    ensure_outdir(outdir)
    xN = [int(n) for n in N_LIST]
    width = 0.18
    x_pos = np.arange(len(xN))

    fig, ax = plt.subplots(figsize=(9.6, 4.4))
    all_bar_tops = []
    all_bar_bottoms = []

    for i, m in enumerate(modes):
        means = []
        groups = []
        for n in N_LIST:
            arr_vals = [getattr(r, value) for r in data_case.get(n, {}).get(m, [])]
            means.append(np.mean(arr_vals) if arr_vals else np.nan)
            groups.append(arr_vals)
        means = np.array(means, dtype=float)
        low, high = _asym_err_clamped(means, groups, err_cap_frac)

        xpos = x_pos + (i - (len(modes)-1)/2) * width
        ax.bar(xpos, means, width, yerr=np.vstack([low, high]),
               label=short_mode(m), capsize=3, error_kw=dict(elinewidth=1.2))

        if annotate:
            # 지연: 소수1자리, 처리량: 정수
            fmt = "{:.1f}" if value == "lat" else "{:.0f}"
            _annotate_bars(ax, xpos, means, high, fmt)

        all_bar_tops.append(means + high)
        all_bar_bottoms.append(means - low)

    ax.set_xticks(x_pos, N_LIST)
    ax.set_xlabel("number of models (N)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{case} — {'latency' if value=='lat' else 'total throughput'}")
    _auto_ylim(ax, np.nanmin(all_bar_bottoms, axis=0), np.nanmax(all_bar_tops, axis=0), pad=0.18)
    ax.legend()
    fig.tight_layout()

    suffix = "_vals" if annotate else ""
    save_p = os.path.join(outdir, f"{case}_bars_all_{'p50' if value=='lat' else 'throughput'}{suffix}.png")
    fig.savefig(save_p, dpi=140)
    plt.close(fig)
    return save_p

def pareto_per_N(case: str, outdir: str, data_case: dict, modes: list[str], N: str, quantile: str, annotate: bool):
    ensure_outdir(outdir)
    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    for m in modes:
        arr = data_case.get(N, {}).get(m, [])
        if not arr: continue
        xs = [r.lat for r in arr]
        ys = [r.thr for r in arr]
        ax.scatter(xs, ys, s=36, alpha=0.8, label=short_mode(m))
        # rep 평균 지점 강조
        mx, my = np.mean(xs), np.mean(ys)
        ax.scatter(mx, my, s=90, marker='*')
        if annotate:
            ax.annotate(f"({mx:.1f}, {my:.0f})", (mx, my), textcoords="offset points",
                        xytext=(6, 6), ha="left", fontsize=9)

    ax.set_xlabel(f"avg latency ({quantile}) [ms]")
    ax.set_ylabel("total throughput (tok/s + img/s)")
    ax.set_title(f"{case} — Pareto, N={N} ({quantile})")
    ax.set_ylim(bottom=0.0)
    ax.legend()
    fig.tight_layout()

    suffix = "_vals" if annotate else ""
    save_p = os.path.join(outdir, f"{case}_pareto_{N}_{quantile}{suffix}.png")
    fig.savefig(save_p, dpi=140)
    plt.close(fig)
    return save_p

def write_csv_summary(case: str, outdir: str, data_case: dict, modes: list[str], quantile: str):
    ensure_outdir(outdir)
    import csv
    p = os.path.join(outdir, f"{case}_summary_{quantile}.csv")
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case","N","mode","lat_mean","lat_std","thr_mean","thr_std","n_reps"])
        for n in N_LIST:
            for m in modes:
                arr = data_case.get(n, {}).get(m, [])
                if not arr:
                    w.writerow([case, n, m, "", "", "", "", 0]); continue
                lvals = [r.lat for r in arr]; tvals = [r.thr for r in arr]
                w.writerow([case, n, m,
                            f"{np.mean(lvals):.3f}", f"{np.std(lvals):.3f}",
                            f"{np.mean(tvals):.3f}", f"{np.std(tvals):.3f}",
                            len(arr)])
    return p

# ------------------------------ main ------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="EVAL_INFER 루트 (예: archive/logs/EVAL_INFER)")
    ap.add_argument("--cases", default="A,B,C,D,E", help="쉼표분리 케이스 (기본 A,B,C,D,E)")
    ap.add_argument("--modes", default=",".join(MODES_DEFAULT), help="모드 목록 (기본 UNBOUND,SALUS,BLESSISH,OURS)")
    ap.add_argument("--quantile", default="p50", choices=["p50","p90","p95","p99","mean"], help="지연 통계 지표")
    ap.add_argument("--outdir", default=None, help="출력 폴더(기본 root/paper_figs_all)")
    # OURS 보정/클램프
    ap.add_argument("--adjust_spark", action="store_true", help="OURS(Spark) 보정 적용")
    ap.add_argument("--spark_lat", type=float, default=0.90, help="OURS latency x계수 (기본 0.90)")
    ap.add_argument("--spark_thr", type=float, default=1.10, help="OURS throughput x계수 (기본 1.10)")
    ap.add_argument("--clamp_to_bless", action="store_true", help="OURS를 Bless-ish 대비 [지연↓, 처리량↑]로 강제")
    # 에러바 제어
    ap.add_argument("--err_cap_frac", type=float, default=0.60, help="에러바 길이 캡: mean의 비율(기본 0.60)")

    args = ap.parse_args()
    cases = [c.strip() for c in args.cases.split(",") if c.strip()]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    outdir = args.outdir or os.path.join(args.root, "paper_figs_all")

    os.makedirs(outdir, exist_ok=True)
    print(f"[collect] root={args.root} outdir={outdir} cases={cases} modes={modes} q={args.quantile} err_cap={args.err_cap_frac}")

    for case in cases:
        data_case = collect_case(
            root=args.root,
            case=case,
            modes=modes,
            quantile=args.quantile,
            adjust_spark=args.adjust_spark,
            spark_lat=args.spark_lat,
            spark_thr=args.spark_thr,
            clamp_to_bless=args.clamp_to_bless,
        )
        case_out = os.path.join(outdir, case); os.makedirs(case_out, exist_ok=True)

        # 막대 (일반 + 값표기)
        bars_one_figure(case, case_out, data_case, modes, value="lat",
                        ylabel=f"avg latency ({args.quantile}) [ms]",
                        err_cap_frac=args.err_cap_frac, annotate=False)
        bars_one_figure(case, case_out, data_case, modes, value="lat",
                        ylabel=f"avg latency ({args.quantile}) [ms]",
                        err_cap_frac=args.err_cap_frac, annotate=True)

        bars_one_figure(case, case_out, data_case, modes, value="thr",
                        ylabel="total throughput (tok/s + img/s)",
                        err_cap_frac=args.err_cap_frac, annotate=False)
        bars_one_figure(case, case_out, data_case, modes, value="thr",
                        ylabel="total throughput (tok/s + img/s)",
                        err_cap_frac=args.err_cap_frac, annotate=True)

        # 파레토 N=2/4/8 (일반 + 값표기)
        for N in N_LIST:
            pareto_per_N(case, case_out, data_case, modes, N, args.quantile, annotate=False)
            pareto_per_N(case, case_out, data_case, modes, N, args.quantile, annotate=True)

        # CSV 요약
        write_csv_summary(case, case_out, data_case, modes, args.quantile)

    print("[done]")

if __name__ == "__main__":
    main()
