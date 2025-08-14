#!/usr/bin/env python3
import csv, sys, argparse, math, collections
import matplotlib.pyplot as plt

def load(csv_path):
    rows=[]
    with open(csv_path) as f:
        r=csv.DictReader(f)
        for a in r: rows.append(a)
    return rows

def build_timeline(rows, total_sm=84, limited_sms_map=None):
    limited_sms_map = limited_sms_map or {}  # {"A":21,"B":42,"C":21}
    # 테넌트별 시계열: [(t, route('L'/'U'), sms)]
    per=collections.defaultdict(list)
    t0=None
    for a in rows:
        if a["ev"]!="EV": continue
        t=float(a["recv_s"])
        t0 = t0 or t
        name=a["tenant"]
        route=a["route"] or "L"
        sms=int(a["sms"]) if a["sms"] else (limited_sms_map.get(name, total_sm if route=="U" else total_sm//2))
        per[name].append((t-t0, route, sms))
    # 각 테넌트 타임라인을 step 구간으로 압축
    segs={}
    for name,evs in per.items():
        evs.sort()
        out=[]
        if not evs: continue
        cur_route, cur_sms = evs[0][1], evs[0][2]
        start=evs[0][0]
        for t,rt,s in evs[1:]:
            if rt!=cur_route or s!=cur_sms:
                out.append((start,t,cur_route,cur_sms))
                start=t; cur_route,cur_sms=rt,s
        out.append((start, evs[-1][0]+1e-3, cur_route, cur_sms))
        segs[name]=out
    # 시스템 활용률(추정)
    # 각 이벤트 시각을 정렬, 직전 상태로 선형 유지
    all_t=sorted(set([t for name,evs in per.items() for (t,_,_) in evs]))
    util=[]
    last_state={k:('L', limited_sms_map.get(k,0)) for k in per.keys()}
    for t in all_t:
        # 각 테넌트의 마지막 이벤트 <= t 의 상태
        for name,evs in per.items():
            le=[e for e in evs if e[0]<=t]
            if le: last_state[name]=(le[-1][1], le[-1][2])
        used=sum((s if r=='L' else total_sm) if s>0 else limited_sms_map.get(n,0) for n,(r,s) in last_state.items())
        util.append((t, min(1.0, used/total_sm)))
    return segs, util

def plot(segs, util, out="timeline.png"):
    tenants=sorted(segs.keys())
    fig,ax=plt.subplots(figsize=(12, 0.9+1.2*len(tenants)))
    ygap=0.6
    yt=[]
    for i,name in enumerate(tenants):
        y=i*ygap
        for (t0,t1,rt,sms) in segs[name]:
            color="#9b59b6" if rt=='L' else "#2ecc71"  # 보라=LIMITED, 초록=UNLIMITED
            ax.barh(y, width=(t1-t0), left=t0, height=0.45, color=color, alpha=0.8, edgecolor="k")
        ax.text(0, y+0.22, name, fontsize=10, weight="bold")
        yt.append(y)
    # 활용률 라인 (두번째 축)
    ax2=ax.twinx()
    if util:
        xs=[t for t,u in util]; ys=[u for t,u in util]
        ax2.plot(xs, ys, linewidth=2, alpha=0.9)
        ax2.set_ylim(0,1.05); ax2.set_ylabel("SM util (est.)")
    ax.set_yticks([]); ax.set_xlabel("time (s, relative)")
    ax.set_title("Kernel-squad context timeline (L=limited, U=unlimited)")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    print("saved:", out)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--csv", default="events_master.csv")
    ap.add_argument("--total-sm", type=int, default=84)
    ap.add_argument("--limited", default="A:21,B:42,C:21", help="tenant:limited_sms,...")
    ap.add_argument("--out", default="timeline.png")
    args=ap.parse_args()
    lim={}
    if args.limited:
        for kv in args.limited.split(","):
            if not kv: continue
            k,v=kv.split(":"); lim[k]=int(v)
    rows=load(args.csv)
    segs, util=build_timeline(rows, total_sm=args.total_sm, limited_sms_map=lim)
    plot(segs, util, out=args.out)
