#!/usr/bin/env python3
import os, sys, time, socket, re, json, csv, math
from typing import Dict, List

MASTER       = os.environ.get("BLESS_MASTER", "/tmp/bless-master.sock")

# ===== 스케줄링 파라미터 =====
SQUAD           = int(os.environ.get("SQUAD", "10000"))
SQUAD_TMO_S     = float(os.environ.get("SQUAD_TMO_S", "3.0"))
EQUAL_SHARE     = os.environ.get("EQUAL_SHARE", "1") == "1"
TICK_S          = float(os.environ.get("TICK_S", "0.006"))
CREDIT_PER_TICK = int(os.environ.get("CREDIT_PER_TICK", "524288"))

# ---- 부스트/솔로/램핑 정책 노브(환경변수로 조절) ----
BOOST_DEBOUNCE_S    = float(os.environ.get("BOOST_DEBOUNCE_S", "0.4"))
ROTATE_BOOST        = os.environ.get("ROTATE_BOOST", "1") == "1"   # 라운드로빈 부스트
SOLO_UNLIMITED      = os.environ.get("SOLO_UNLIMITED", "0") == "1" # True면 혼자 남으면 -1 허용
SOLO_MAX_FRACTION   = float(os.environ.get("SOLO_MAX_FRACTION", "0.6")) # 0~1, 혼자 남았을 때 캡
MIN_CREDIT_PER_TICK = int(os.environ.get("MIN_CREDIT_PER_TICK", "256")) # 0이면 비활성
FREED_RAMP_ALPHA    = float(os.environ.get("FREED_RAMP_ALPHA", "0.5"))   # 0~1, freed pct 램핑 알파

def parse_base_shares(txt: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for tok in txt.split(","):
        tok = tok.strip()
        if not tok or ":" not in tok: continue
        k, v = tok.split(":"); k = k.strip()
        try: out[k] = float(v)
        except: pass
    return out

# 글로벌 목표 가중치(활성 집합에 투영)
TARGET_W = parse_base_shares(os.environ.get("BASE_SHARES", ""))

# ===== CSV 로깅 =====
LOG_CSV = os.environ.get("SQUAD_LOG", "squad_log.csv")
csv_f = open(LOG_CSV, "w", newline="", buffering=1)
csv_w = csv.writer(csv_f)
csv_w.writerow(["ts_s","ev","squad","tenant","share","remain","note"])
def now_s(): return f"{time.time():.6f}"
def jdump(d):
    try: return json.dumps(d, ensure_ascii=False)
    except: return str(d)
def log(ev, squad_id, tenant="", share=None, remain=None, note=""):
    csv_w.writerow([now_s(), ev, squad_id, tenant, jdump(share or {}), jdump(remain or {}), note])

# ===== 텐넌트/컨트롤 프로토콜 =====
HELLO_RE = re.compile(r"HELLO pid=(\d+)\s+sock=([/\w\.\-]+)\s+tenant=(\w*)")
SD_RE    = re.compile(r"SD pid=(\d+)\s+t=(\w+)\s+sp=(\d+)\s+kseq=(\d+)")
BYE_RE   = re.compile(r"BYE pid=(\d+)\s+t=(\w+)")

CTRL_EQUAL_RE      = re.compile(r"\s*SET_EQUAL\s+([01])\s*$")
CTRL_WEIGHTS_RE    = re.compile(r"\s*SET_WEIGHTS\s+(.+?)\s*$")
CTRL_SET_SHARE_RE  = re.compile(r"\s*SET_SHARE\s+([A-Za-z0-9_\-]+)\s+([0-9]+(?:\.[0-9]+)?)\s*$")
CTRL_TICK_RE       = re.compile(r"\s*SET_TICK\s+([0-9]*\.?[0-9]+)\s*$")
CTRL_CRED_RE       = re.compile(r"\s*SET_CREDITS\s+([0-9]+)\s*$")
CTRL_BOOST_RE      = re.compile(r"\s*BOOST\s+([A-Za-z0-9_\-]+)\s*$")
CTRL_BOOST_ON_RE   = re.compile(r"\s*BOOST_ENABLE\s*$|^\s*BOOST_ON\s*$")
CTRL_BOOST_OFF_RE  = re.compile(r"\s*BOOST_DISABLE\s*$|^\s*BOOST_OFF\s*$")
CTRL_CRED_OFF_RE   = re.compile(r"\s*CREDIT_OFF\s*$")

def send(sock_path: str, msg: str, tries=30, delay=0.02):
    for _ in range(tries):
        try:
            s = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
            s.sendto(msg.encode(), sock_path)
            s.close()
            return True
        except OSError:
            time.sleep(delay)
    print(f"[warn] send fail -> {sock_path}: {msg}")
    return False

class Tenant:
    __slots__ = ("pid","sock","alive")
    def __init__(self, pid:int, sock:str):
        self.pid = pid; self.sock = sock; self.alive=True

tenants: Dict[str,Tenant] = {}
remain : Dict[str,int]   = {}
inbox_events: List[tuple] = []

# ===== 현재 스쿼드 상태 =====
CUR_BASE_PCT: Dict[str, float] = {}     # 현재 스쿼드 기본 비율(%), 합≈100
CUR_FINISHED: set[str] = set()          # SD 보낸 텐넌트
CUR_BOOST: str | None = None            # 부스트 타겟
BOOST_ALLOWED: bool = True              # SALUS: False, OURS: True

# 부스트 상태(디바운스/라운드로빈)
_boost_last_change = 0.0
_boost_last_tid    = None

# freed_pct 램핑(스파이크 감소용)
_freed_ramp = 0.0

def _norm_pct(d: Dict[str,float]) -> Dict[str,float]:
    s = sum(max(0.0, v) for v in d.values())
    if s <= 0:
        n = len(d) or 1
        return {k: (100.0/n) for k in d}
    return {k: 100.0 * max(0.0, v) / s for k,v in d.items()}

def _fill_weights_for_active(active: List[str], target_w: Dict[str,float]) -> Dict[str,float]:
    specified = {t: float(target_w[t]) for t in active if t in target_w}
    sum_spec  = sum(specified.values())
    unspec    = [t for t in active if t not in target_w]
    out: Dict[str,float] = {}
    if unspec:
        rest = max(0.0, 100.0 - sum_spec)
        each = rest/len(unspec) if unspec else 0.0
        for t in active:
            out[t] = specified.get(t, each)
    else:
        out = {t: specified.get(t, 0.0) for t in active}
    return _norm_pct(out)

def _compute_base_pct(active: List[str]) -> Dict[str,float]:
    if not active: return {}
    if EQUAL_SHARE or not TARGET_W:
        each = 100.0 / len(active)
        return {t: each for t in active}
    return _fill_weights_for_active(active, TARGET_W)

def _rebase_current_for_unfinished(active: List[str]):
    """끝난 텐넌트의 몫을 남은 텐넌트에 비례 재분배(합≈100 유지)."""
    global CUR_BASE_PCT
    if not active: return
    finished_sum = sum(CUR_BASE_PCT.get(t,0.0) for t in CUR_FINISHED)
    remain_budget = max(0.0, 100.0 - finished_sum)
    unfinished = [t for t in active if t not in CUR_FINISHED]
    if not unfinished: return
    desired_all = _compute_base_pct(active)
    denom = sum(desired_all[t] for t in unfinished) or 1.0
    new_map = {}
    for t in active:
        if t in CUR_FINISHED:
            new_map[t] = CUR_BASE_PCT.get(t, 0.0)
        else:
            new_map[t] = remain_budget * (desired_all[t] / denom)
    CUR_BASE_PCT = new_map

def weighted_split_int_by_pct(active: List[str], total: int, pct: Dict[str,float]) -> Dict[str,int]:
    if not active: return {}
    raw = [total * max(0.0, pct.get(t, 0.0)) / 100.0 for t in active]
    ints= [int(math.floor(x)) for x in raw]
    rem = total - sum(ints)
    frac= [(raw[i]-ints[i], i) for i in range(len(active))]
    frac.sort(reverse=True)
    for k in range(rem):
        ints[frac[k % len(active)][1]] += 1
    return {active[i]: ints[i] for i in range(len(active))}

def _freed_pct() -> float:
    return sum(CUR_BASE_PCT.get(t, 0.0) for t in CUR_FINISHED)

def _pick_boost_target(active: List[str]) -> str | None:
    unfinished = [t for t in active if t not in CUR_FINISHED]
    if not unfinished: return None
    if ROTATE_BOOST:
        uf = sorted(unfinished)
        global _boost_last_tid
        if _boost_last_tid not in uf:
            return uf[0]
        return uf[(uf.index(_boost_last_tid)+1) % len(uf)]
    # fallback: 원래 몫이 작은 쪽을 우선
    return min(unfinished, key=lambda k: CUR_BASE_PCT.get(k, 0.0))

def _recv_all():
    """소켓에서 수신 → 파싱 → inbox_events에 enqueue만 수행"""
    global EQUAL_SHARE, TICK_S, CREDIT_PER_TICK, BOOST_ALLOWED, CUR_BOOST
    try:
        while True:
            data,_ = srv.recvfrom(512)
            s = data.decode("utf-8","ignore").strip()

            # 텐넌트 이벤트
            m = HELLO_RE.match(s)
            if m:
                pid, sock, tid = m.groups()
                tenants[tid] = Tenant(int(pid), sock)
                if tid not in remain: remain[tid] = -1
                inbox_events.append(("HELLO", tid))
                print(f"[HELLO] {tid} -> {sock}")
                continue
            m = SD_RE.match(s)
            if m:
                _pid, tid, sp, kseq = m.groups()
                inbox_events.append(("SD", tid))
                continue
            m = BYE_RE.match(s)
            if m:
                _pid, tid = m.groups()
                if tid in tenants: tenants[tid].alive=False
                inbox_events.append(("BYE", tid))
                continue

            # 컨트롤(설정값 갱신 + 이벤트 enqueue)
            m = CTRL_CRED_OFF_RE.match(s)
            if m:
                inbox_events.append(("CTRL_CRED_OFF", True))
                continue

            m = CTRL_EQUAL_RE.match(s)
            if m:
                EQUAL_SHARE = (int(m.group(1))==1)
                inbox_events.append(("CTRL_EQUAL", EQUAL_SHARE))
                print(f"[ctrl] EQUAL_SHARE -> {1 if EQUAL_SHARE else 0}")
                continue

            m = CTRL_WEIGHTS_RE.match(s)
            if m:
                spec = m.group(1)
                new_w: Dict[str,float] = {}
                for tok in spec.split(","):
                    if ":" not in tok: continue
                    k,v = tok.split(":"); k = k.strip()
                    try: new_w[k] = float(v)
                    except: pass
                if new_w:
                    TARGET_W.clear(); TARGET_W.update(new_w)
                    EQUAL_SHARE = False
                    inbox_events.append(("CTRL_WEIGHTS", dict(TARGET_W)))
                    print(f"[ctrl] WEIGHTS -> {TARGET_W} (EQUAL=0)")
                continue

            m = CTRL_SET_SHARE_RE.match(s)
            if m:
                tgt = m.group(1); pct = float(m.group(2))
                alive = [tid for tid, info in tenants.items() if info.alive]
                if tgt not in alive:
                    alive = sorted(set(alive + [tgt]))
                others = [t for t in alive if t != tgt]
                tmp_w: Dict[str,float] = {}
                if not others:
                    tmp_w[tgt] = 100.0
                else:
                    rest = max(0.0, 100.0 - pct)
                    each = rest / len(others)
                    for t in alive:
                        tmp_w[t] = pct if t == tgt else each
                TARGET_W.clear(); TARGET_W.update(tmp_w)
                EQUAL_SHARE = False
                inbox_events.append(("CTRL_WEIGHTS", dict(TARGET_W)))
                print(f"[ctrl] SET_SHARE {tgt}={pct}% -> TARGET_W={TARGET_W}")
                continue

            m = CTRL_TICK_RE.match(s)
            if m:
                TICK_S = float(m.group(1))
                inbox_events.append(("CTRL_TICK", TICK_S))
                print(f"[ctrl] TICK_S -> {TICK_S}s")
                continue

            m = CTRL_CRED_RE.match(s)
            if m:
                CREDIT_PER_TICK = int(m.group(1))
                inbox_events.append(("CTRL_CRED", CREDIT_PER_TICK))
                print(f"[ctrl] CREDIT_PER_TICK -> {CREDIT_PER_TICK}")
                continue

            m = CTRL_BOOST_ON_RE.match(s)
            if m:
                BOOST_ALLOWED = True
                inbox_events.append(("CTRL_BOOST_ALLOWED", True))
                print(f"[ctrl] BOOST enable")
                continue
            m = CTRL_BOOST_OFF_RE.match(s)
            if m:
                BOOST_ALLOWED = False
                CUR_BOOST = None
                inbox_events.append(("CTRL_BOOST_ALLOWED", False))
                print(f"[ctrl] BOOST disable")
                continue

            m = CTRL_BOOST_RE.match(s)
            if m:
                tgt = m.group(1)
                if BOOST_ALLOWED:
                    CUR_BOOST = tgt
                    inbox_events.append(("CTRL_BOOST", tgt))
                    print(f"[ctrl] BOOST target -> {tgt}")
                else:
                    print(f"[ctrl] BOOST ignored (disabled)")
                continue

    except socket.timeout:
        pass

def push_credits(active: List[str]):
    """틱마다 크레딧 분배. 솔로 폭주 억제 + freed 램핑 + 부스트 보너스."""
    global _freed_ramp
    if not active: return {}

    alive_unfinished = [t for t in active if t not in CUR_FINISHED]
    total = max(1, CREDIT_PER_TICK)

    # ---- 혼자 남은 경우 처리 ----
    if len(alive_unfinished) == 1:
        only = alive_unfinished[0]
        if SOLO_UNLIMITED:
            for t in active:
                n = -1 if t == only else 0
                send(tenants[t].sock, f"credit_set {n}")
            return {only: -1, **{t:0 for t in active if t!=only}}
        # 캡 + 미니멈으로 페이싱(폭주 억제)
        cap = int(max(MIN_CREDIT_PER_TICK, SOLO_MAX_FRACTION * total))
        shares = {only: cap}
        for t in active:
            if t != only:
                shares[t] = 0
        for t in active:
            send(tenants[t].sock, f"credit_set {shares.get(t,0)}")
        return shares

    # ---- 2개 이상: freed 램핑, 부스트 보너스, 가중치 ----
    freed_target = _freed_pct()
    _freed_ramp = _freed_ramp + FREED_RAMP_ALPHA * (freed_target - _freed_ramp)  # 0~100
    bonus_pool = _freed_ramp if BOOST_ALLOWED and CUR_BOOST else 0.0

    pct_map: Dict[str, float] = {}
    for t in active:
        if t in CUR_FINISHED:
            pct_map[t] = 0.0
        else:
            base  = CUR_BASE_PCT.get(t, 0.0)
            bonus = (bonus_pool if (CUR_BOOST == t) else 0.0)
            pct_map[t] = base + bonus

    shares = weighted_split_int_by_pct(active, total, pct_map)

    # 최소 크레딧 보장(옵션) — starvation 방지
    if MIN_CREDIT_PER_TICK > 0:
        for t in active:
            if t not in CUR_FINISHED:
                shares[t] = max(shares.get(t,0), MIN_CREDIT_PER_TICK)

    for t in active:
        send(tenants[t].sock, f"credit_set {shares.get(t,0)}")
    return shares

def wait_tenants():
    deadline = time.time() + 30.0
    while True:
        _recv_all()
        if any(t.alive for t in tenants.values()): break
        if time.time() > deadline: break
        time.sleep(0.05)

def main():
    global srv, inbox_events, CUR_BASE_PCT, CUR_FINISHED, CUR_BOOST, BOOST_ALLOWED
    if os.path.exists(MASTER): os.unlink(MASTER)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    srv.bind(MASTER); srv.settimeout(0.1)
    print(f"[master] listen {MASTER}  tick={TICK_S}s credits/tick={CREDIT_PER_TICK} base={TARGET_W} equal={int(EQUAL_SHARE)}")

    wait_tenants()

    squad_id   = 0
    last_tick  = time.time()

    def all_done():
        return tenants and all(t.alive is False for t in tenants.values())

    while True:
        _recv_all()
        if all_done():
            print("[master] all done")
            break

        active = [tid for tid,info in tenants.items() if info.alive]
        if not active:
            time.sleep(0.02); continue

        # 주기적 분배
        now = time.time()
        if now - last_tick >= TICK_S:
            _ = push_credits(active)
            last_tick = now

        # 스쿼드 시작
        squad_id += 1
        CUR_FINISHED = set()
        CUR_BOOST    = None
        # freed 램프 초기화
        global _freed_ramp
        _freed_ramp = 0.0

        CUR_BASE_PCT = _compute_base_pct(active)      # 합≈100
        start_share = weighted_split_int_by_pct(active, SQUAD, CUR_BASE_PCT)
        for t in active:
            send(tenants[t].sock, f"set_squad {SQUAD}")
            send(tenants[t].sock, f"set_share {start_share[t]}")
            send(tenants[t].sock, "squad_reset")
        log("SQUAD_START", squad_id, "", start_share, remain, f"active={active}; base_pct={CUR_BASE_PCT}")

        t0 = time.time()

        while True:
            _recv_all()

            # 주기적 분배
            now2 = time.time()
            if now2 - last_tick >= TICK_S:
                _ = push_credits(active)
                last_tick = now2

            restart = False
            while inbox_events:
                ev, arg = inbox_events.pop(0)

                if ev == "SD" and arg in active and arg not in CUR_FINISHED:
                    CUR_FINISHED.add(arg)

                    # 끝난 뒤 base 재배분 + 부스트는 디바운스로 천천히 회전
                    _rebase_current_for_unfinished(active)
                    if BOOST_ALLOWED:
                        nowt = time.time()
                        if (nowt - _boost_last_change) >= BOOST_DEBOUNCE_S:
                            tgt = _pick_boost_target(active)
                            if tgt is not None:
                                CUR_BOOST = tgt
                                globals()['_boost_last_change'] = nowt
                                globals()['_boost_last_tid']    = tgt
                                log("BOOST_SET", squad_id, tgt, start_share, remain, f"cause=SD:{arg}")

                    log("SD", squad_id, arg, start_share, remain,
                        f"finished={len(CUR_FINISHED)}/{len(active)} freed={_freed_pct():.2f}% boost={CUR_BOOST}")

                    _ = push_credits(active); last_tick = time.time()

                elif ev == "BYE":
                    if arg in active: restart = True

                elif ev == "CTRL_CRED_OFF":
                    for t in active:
                        send(tenants[t].sock, "credit_off")
                    last_tick = time.time()
                    print("[ctrl] CREDIT_OFF broadcast")

                elif ev in ("CTRL_EQUAL","CTRL_WEIGHTS","CTRL_TICK","CTRL_CRED","CTRL_BOOST","CTRL_BOOST_ALLOWED"):
                    if ev == "CTRL_BOOST":
                        if BOOST_ALLOWED and isinstance(arg, str):
                            CUR_BOOST = arg if (arg in active) else None
                        else:
                            CUR_BOOST = None
                    if ev in ("CTRL_EQUAL","CTRL_WEIGHTS"):
                        _rebase_current_for_unfinished(active)
                    _ = push_credits(active); last_tick = time.time()
                    log(ev, squad_id, str(arg) if not isinstance(arg, dict) else "", start_share, remain,
                        f"TICK={TICK_S}, CRED={CREDIT_PER_TICK}, TARGET_W={TARGET_W}, base_pct={CUR_BASE_PCT}, boost_allowed={BOOST_ALLOWED}")

            if restart:
                log("SQUAD_RESTART", squad_id, "", start_share, remain, "topology change")
                break

            if len(CUR_FINISHED) == len(active):
                log("SQUAD_END", squad_id, "", start_share, remain, f"boosted={CUR_BOOST}")
                break

            if time.time() - t0 > SQUAD_TMO_S:
                log("SQUAD_END_TIMEOUT", squad_id, "", start_share, remain,
                    f"finished={sorted(list(CUR_FINISHED))},boosted={CUR_BOOST}")
                break

    csv_f.close()
    try:
        srv.close()
        if os.path.exists(MASTER): os.unlink(MASTER)
    except Exception:
        pass

if __name__ == "__main__":
    main()
