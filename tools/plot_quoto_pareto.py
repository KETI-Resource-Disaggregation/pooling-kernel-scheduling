#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quota-Pareto plotter
- N=2(기본)에서 쿼터 디렉토리별(A:33.333B:66.667 등) 포인트를 1개 점으로 그립니다.
- X: 평균 지연(선택 quantile, per-tenant 평균) / Y: 총 처리량(tok/s + img/s)
- 모드별(UNBOUND/SALUS/BLESSISH/OURS)로 색/마커를 구분합니다.

디렉토리 가정:
  <root>/<CASE>/<N>/<QUOTA>/rep-*/<MODE>/new_metric.json
또는 (쿼터 없는 경우)
  <root>/<CASE>/<N>/rep-*/<MODE>/new_metric.json  -> 쿼터 라벨은 "ALL"
"""

from __future__ import annotations
import os, re, json, argparse, statistics as st
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

MODES = ["UNBOUND", "SALUS", "BLESSISH", "OURS"]
MODE_STYLE = {
    "UNBOUND":  dict(color="#1f77b4", marker="o",  label="Unbound"),
    "SALUS":    dict(color="#ff7f0e", marker="s",  label="Temporal"),
    "BLESSISH": dict(color="#2ca02c", marker="^",  label="Bless-ish"),
    "OURS":     dict(color="#d62728", marker="*",  label="Spark"),
}

def safe_load_json(p):
    try:
        with open(p, "r") as f:
            return json.load(f)
    except Exception:
        return None

def quota_short(name: str) -> str:
    """'A:33.333B:66.667' -> '33/67' 형태로 축약."""
    if not name or ":" not in name:
        return name or "ALL"
    nums = re.findall(r":\s*([\d.]+)", name)
    if len(nums) == 2:
        try:
            a = float(nums[0]); b = float(nums[1])
            return f"{int(round(a))}/{int(round(b))}"
        except Exception:
            return name
    return name

def pick_latency_value(per_tenant: dict, which: str = "p50"):
    """per_tenant.{tenant}.lat_ms에서 which(p50/p90/p95/p99/mean)로 값을 뽑고, 테넌트 평균."""
    if not per_tenant:
        return None
    order = [which] + [k for k in ["mean", "p50", "p90", "p95", "p99"] if k != which]
    vals = []
    for tname, tinfo in per_tenant.items():
        lat = tinfo.get("lat_ms", {})
        v = None
        for k in order:
            if k in lat:
                try:
                    v = float(lat[k])
                    break
                except Exception:
                    pass
        if v is not None:
            vals.append(v)
    if not vals:
        return None
    return float(st.mean(vals))

def extract_tot_throughput(newm: dict) -> float | None:
    """new_metric.json 에서 총 처리량(tok/s + img/s)."""
    if not newm:
        return None
    ag = newm.get("aggregates", {})
    tot = 0.0
    if ag:
        tot = float(ag.get("overall_tok_per_s_sum", 0.0)) + float(ag.get("overall_imgs_per_s_sum", 0.0))
    if tot <= 0.0:
        # per-tenant fallback
        per_t = newm.get("per_tenant", {})
        for _, tinfo in per_t.items():
            to = tinfo.get("tok_overall", {})
            io = tinfo.get("img_overall", {})
            tot += float(to.get("overall_tok_per_s", 0.0)) + float(io.get("overall_imgs_per_s", 0.0))
    return tot if tot > 0.0 else None

def find_quota_rep_dirs(case_dir: str, N: str = "2"):
    """case_dir/N/ 밑의 (quota/rep-*) 혹은 직접 rep-* 구조를 탐색."""
    base = os.path.join(case_dir, N)
    if not os.path.isdir(base):
        return {}
    out = defaultdict(list)  # quota -> [rep_dir,...]

    # depth-1: rep-* 바로
    rep1 = [d for d in os.listdir(base) if d.startswith("rep-") and os.path.isdir(os.path.join(base, d))]
    for r in sorted(rep1):
        out["ALL"].append(os.path.join(base, r))

    # depth-2: <quota>/rep-*
    for d in sorted(os.listdir(base)):
        qp = os.path.join(base, d)
        if not os.path.isdir(qp) or d.startswith("rep-"):
            continue
        rs = [r for r in os.listdir(qp) if r.startswith("rep-") and os.path.isdir(os.path.join(qp, r))]
        for r in sorted(rs):
            out[d].append(os.path.join(qp, r))

    return out

def collect_points_for_case(root: str, case: str, quantile: str = "p50", N: str = "2"):
    """
    return: dict[mode] -> list of dict(lat=..., thr=..., quota=..., reps=...)
    """
    case_dir = os.path.join(root, case)
    quota_to_reps = find_quota_rep_dirs(case_dir, N)
    results = {m: [] for m in MODES}

    for quota, rep_dirs in quota_to_reps.items():
        # rep별 모드 포인트 수집
        per_mode_vals = {m: {"lat": [], "thr": []} for m in MODES}
        for rp in rep_dirs:
            for m in MODES:
                nm = os.path.join(rp, m, "new_metric.json")
                j = safe_load_json(nm)
                if not j:
                    continue
                per_t = j.get("per_tenant", {})
                lat = pick_latency_value(per_t, quantile)
                thr = extract_tot_throughput(j)
                if lat is None or thr is None:
                    continue
                per_mode_vals[m]["lat"].append(lat)
                per_mode_vals[m]["thr"].append(thr)

        # rep 평균 -> 한 점
        for m in MODES:
            lats = per_mode_vals[m]["lat"]
            thrs = per_mode_vals[m]["thr"]
            if not lats or not thrs:
                continue
            results[m].append({
                "lat": float(np.mean(lats)),
                "thr": float(np.mean(thrs)),
                "quota": quota,
                "reps": len(lats)
            })
    return results

def plot_case(quota_points: dict, case: str, outdir: str, quantile: str):
    os.makedirs(outdir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for m in MODES:
        pts = quota_points.get(m, [])
        if not pts:
            continue
        xs = [p["lat"] for p in pts]
        ys = [p["thr"] for p in pts]
        style = MODE_STYLE[m]
        ax.scatter(xs, ys, s=60, alpha=0.85, **style)

        # 각 점에 쿼터 라벨 표시
        for x, y, p in zip(xs, ys, pts):
            ax.text(x, y, quota_short(p["quota"]), fontsize=9, ha="left", va="bottom")

    ax.set_xlabel(f"avg latency ({quantile}) [ms]")
    ax.set_ylabel("total throughput (tok/s + img/s)")
    ax.set_title(f"{case} — quota Pareto (N=2, {quantile})")
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.5)
    ax.legend(loc="best", frameon=False)

    fig.tight_layout()
    save_p = os.path.join(outdir, f"{case}_quota_pareto_{quantile}.png")
    fig.savefig(save_p, dpi=140)
    plt.close(fig)
    return save_p

def plot_all_panel(all_data: dict, outdir: str, quantile: str):
    """모든 케이스를 2x3(최대 5개면 일부 빈칸) 멀티패널로."""
    os.makedirs(outdir, exist_ok=True)
    cases = list(all_data.keys())
    n = len(cases)
    rows, cols = 2, 3
    fig, axes = plt.subplots(rows, cols, figsize=(15, 8))
    axes = axes.ravel()

    for i, case in enumerate(cases):
        ax = axes[i]
        quota_points = all_data[case]
        for m in MODES:
            pts = quota_points.get(m, [])
            if not pts:
                continue
            xs = [p["lat"] for p in pts]
            ys = [p["thr"] for p in pts]
            style = MODE_STYLE[m]
            ax.scatter(xs, ys, s=45, alpha=0.85, **style)
            for x, y, p in zip(xs, ys, pts):
                ax.text(x, y, quota_short(p["quota"]), fontsize=8, ha="left", va="bottom")
        ax.set_title(case)
        ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.5)

    # 공통 레전드/라벨
    handles = []
    labels = []
    for m in MODES:
        style = MODE_STYLE[m]
        h = axes[0].scatter([], [], s=50, **style)
        handles.append(h); labels.append(style["label"])
    fig.legend(handles, labels, loc="upper center", ncol=len(MODES), frameon=False)
    for ax in axes:
        ax.set_xlabel(f"avg latency ({quantile}) [ms]")
        ax.set_ylabel("total throughput (tok/s + img/s)")

    # 남는 축 없애기
    for j in range(i+1, rows*cols):
        fig.delaxes(axes[j])

    fig.suptitle(f"Quota Pareto (N=2, {quantile})", y=0.98, fontsize=14)
    fig.tight_layout(rect=[0,0,1,0.95])
    save_p = os.path.join(outdir, f"quota_pareto_all_{quantile}.png")
    fig.savefig(save_p, dpi=150)
    plt.close(fig)
    return save_p

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="EVAL_INFER 루트 (예: archive/logs/EVAL_INFER)")
    ap.add_argument("--cases", default="A,B,C,D,E", help="쉼표분리 케이스 목록")
    ap.add_argument("--quantile", default="p50", choices=["p50","p90","p95","p99","mean"])
    ap.add_argument("--N", default="2", help="모델 수 디렉토리(기본 2; 본 스크립트는 N=2 가정)")
    ap.add_argument("--outdir", default=None, help="출력 폴더(기본 root/paper_figs_quota)")
    ap.add_argument("--per_case", action="store_true", help="케이스별 단일 그림도 저장")
    args = ap.parse_args()

    cases = [c.strip() for c in args.cases.split(",") if c.strip()]
    outdir = args.outdir or os.path.join(args.root, "paper_figs_quota")
    os.makedirs(outdir, exist_ok=True)

    all_data = {}
    for case in cases:
        data = collect_points_for_case(args.root, case, quantile=args.quantile, N=args.N)
        all_data[case] = data
        if args.per_case:
            p = plot_case(data, case, outdir, args.quantile)
            print(f"[{case}] -> {p}")

    p_all = plot_all_panel(all_data, outdir, args.quantile)
    print(f"[all] -> {p_all}")

if __name__ == "__main__":
    import numpy as np
    main()
