# viz_squad_window.py
import argparse, re
import pandas as pd
import matplotlib.pyplot as plt

def parse_note(note: str):
    out = {"boosted": None, "cause": None, "finished": []}
    if not isinstance(note, str) or not note:
        return out
    m = re.search(r"boosted=([A-Z])", note);   out["boosted"] = m.group(1) if m else None
    m = re.search(r"cause=([A-Z])", note);     out["cause"]   = m.group(1) if m else None
    m = re.search(r"finished=\[(.*?)\]", note)
    if m: out["finished"] = re.findall(r"[A-Z]", m.group(1))
    return out

def parse_baseline(s):
    parts = {}
    total = 0.0
    order = []
    for tok in s.split(","):
        k,v = tok.split(":")
        k = k.strip(); v = float(v.strip())
        parts[k]=v; order.append(k); total += v
    if abs(total-100.0)>1e-3:
        raise SystemExit(f"baseline 합이 100이 아닙니다: {total}")
    return parts, order

def load_csv(path):
    df = pd.read_csv(path)
    df = df.sort_values("ts_s").reset_index(drop=True)
    return df

def filter_window(df, args):
    # 이벤트 타입 필터
    wanted = [e.strip() for e in args.only.split(",")]
    df = df[df["ev"].isin(wanted)]

    # 스쿼드 범위 지정
    if args.focus_squad is not None:
        lo = args.focus_squad - args.span
        hi = args.focus_squad + args.span
        df = df[(df["squad"]>=lo) & (df["squad"]<=hi)]
    if args.squad_min is not None: df = df[df["squad"]>=args.squad_min]
    if args.squad_max is not None: df = df[df["squad"]<=args.squad_max]

    # 시간 범위 지정
    if args.tmin is not None:
        df = df[df["ts_s"] >= args.tmin if args.absolute_ts else df["ts_s"] >= (df["ts_s"].iloc[0] + args.tmin)]
    if args.tmax is not None:
        df = df[df["ts_s"] <= args.tmax if args.absolute_ts else df["ts_s"] <= (df["ts_s"].iloc[0] + args.tmax)]

    # 디듀프 (같은 ev/tenant/squad에서 ts_s 가까운 것 제거)
    if args.dedup_ms>0:
        df = df.sort_values(["ev","tenant","squad","ts_s"])
        gap = df.groupby(["ev","tenant","squad"])["ts_s"].diff().fillna(float("inf"))
        keep = (gap*1000.0 > args.dedup_ms)
        # 첫 행은 diff=inf → keep=True
        df = df[keep | gap.isna()]

    # 상대시간 컬럼
    if len(df)==0: return df
    t0 = df["ts_s"].iloc[0] if args.absolute_ts else df["ts_s"].min()
    df = df.copy()
    df["t_rel"] = df["ts_s"] - t0
    return df.sort_values("t_rel").reset_index(drop=True)

def build_series(df, baseline, tenants):
    cur = baseline.copy()
    times = [df["t_rel"].iloc[0] if len(df) else 0.0]
    series = {k:[cur.get(k,0.0)] for k in tenants}
    bx, by, blab = [], [], []

    for _, row in df.iterrows():
        ev = row["ev"]
        note = "" if pd.isna(row.get("note")) else str(row.get("note"))
        meta = parse_note(note)

        if ev == "SQUAD_START":
            cur = baseline.copy()

        elif ev == "BOOST_ON":
            boosted = str(row.get("tenant")) if not pd.isna(row.get("tenant")) else None
            cause   = meta.get("cause")
            if boosted in tenants and cause in tenants and boosted != cause:
                freed = baseline.get(cause, 0.0)
                cur[cause]   = 0.0
                cur[boosted] = cur.get(boosted,0.0)+freed
            # 마커 위치
            bx.append(row["t_rel"])
            cum = 0.0
            for k in tenants:
                cum += cur.get(k,0.0)
                if k==boosted: break
            by.append(cum); blab.append(f"{meta.get('cause','?')}→{boosted}")

        # SQUAD_END(_TIMEOUT) / BOOST_OFF → 비율 변경 없음

        times.append(row["t_rel"])
        for k in tenants:
            series[k].append(cur.get(k,0.0))

    return times, series, bx, by, blab

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", default="squad_log.csv")
    ap.add_argument("--baseline", default="A:25,B:50,C:25")
    ap.add_argument("--only", default="SQUAD_START,BOOST_ON,SQUAD_END,SQUAD_END_TIMEOUT",
                    help="표시할 이벤트(콤마구분). 기본: SQUAD_START,BOOST_ON,SQUAD_END,SQUAD_END_TIMEOUT")
    ap.add_argument("--focus-squad", type=int, default=None, help="이 스쿼드 중심으로")
    ap.add_argument("--span", type=int, default=15, help="focus-squad ±범위")
    ap.add_argument("--squad-min", type=int, default=None)
    ap.add_argument("--squad-max", type=int, default=None)
    ap.add_argument("--tmin", type=float, default=None, help="시간 하한(초)")
    ap.add_argument("--tmax", type=float, default=None, help="시간 상한(초)")
    ap.add_argument("--absolute-ts", action="store_true", help="tmin/tmax를 CSV의 절대 ts_s로 해석")
    ap.add_argument("--dedup-ms", type=float, default=200.0, help="같은 (ev,tenant,squad) 근접 이벤트 디듀프(ms)")
    ap.add_argument("--out", default="squad_share_timeline.png")
    args = ap.parse_args()

    baseline, tenants = parse_baseline(args.baseline)
    df = load_csv(args.csv)
    dfw = filter_window(df, args)
    if len(dfw)==0:
        print("[warn] 선택된 구간에 이벤트가 없습니다."); return

    times, series, bx, by, blab = build_series(dfw, baseline, tenants)

    plt.figure(figsize=(12,5))
    plt.stackplot(times, *[series[k] for k in tenants],
                  labels=[f"{k} (base {baseline[k]}%)" for k in tenants])
    plt.legend(loc="upper left")
    plt.xlabel("time (s)")
    plt.ylabel("SM share (%)")
    plt.title("Tenant SM share over selected window")

    for x,y,lab in zip(bx,by,blab):
        plt.scatter([x],[y], s=28)
        plt.text(x, y+2, lab, fontsize=8, ha="center")

    plt.tight_layout()
    plt.savefig(args.out, dpi=160)
    print(f"[ok] saved -> {args.out}  (events={len(dfw)})")

if __name__ == "__main__":
    main()
