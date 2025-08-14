#!/usr/bin/env python3
import os, sys, time, socket, re, json, csv
from typing import Dict, List

# ---------------- config/env ----------------
MASTER      = os.environ.get("BLESS_MASTER", "/tmp/bless-master.sock")
SQUAD       = int(os.environ.get("SQUAD", "1000"))            # per-squad kernel quota (논리)
SQUAD_TMO_S = float(os.environ.get("SQUAD_TMO_S", "3.0"))     # timeout to end a squad
EXPECT      = os.environ.get("EXPECT_TENANTS", "")            # "A,B,C" 지정시 모두 올 때까지 대기
LOG_CSV     = os.environ.get("SQUAD_LOG", "squad_log.csv")    # 로그 파일명
EQUAL_SHARE = os.environ.get("EQUAL_SHARE", "1") == "1"       # 기본 균등할당

# remain/backlog는 종료조건에만 쓰고, 없으면 BYE까지 계속 도는 모드
BACKLOG_ENV = os.environ.get("BACKLOG", "")
BACKLOG: Dict[str,int] = {}
if BACKLOG_ENV:
    for tok in BACKLOG_ENV.split(","):
        t, v = tok.split(":")
        BACKLOG[t.strip()] = int(v)

# ---------------- CSV 준비 ----------------
csv_f = open(LOG_CSV, "w", newline="", buffering=1)
csv_w = csv.writer(csv_f)
csv_w.writerow(["ts_s","ev","squad","tenant","share","remain","note"])

def now_s(): return f"{time.time():.6f}"

def jdump(d): 
    try: return json.dumps(d, ensure_ascii=False)
    except: return str(d)

def log(ev, squad_id, tenant="", share=None, remain=None, note=""):
    csv_w.writerow([now_s(), ev, squad_id, tenant, jdump(share or {}), jdump(remain or {}), note])

# ---------------- 소켓/프로토콜 ----------------
HELLO_RE = re.compile(r"HELLO pid=(\d+)\s+sock=([/\w\.\-]+)\s+tenant=(\w*)")
SD_RE    = re.compile(r"SD pid=(\d+)\s+t=(\w+)\s+sp=(\d+)\s+kseq=(\d+)")
BE_RE    = re.compile(r"BE pid=(\d+)\s+t=(\w+)\s+what=(BOOST_ON|BOOST_OFF)")
BYE_RE   = re.compile(r"BYE pid=(\d+)\s+t=(\w+)")

def send(sock_path: str, msg: str, tries=30, delay=0.03):
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

# ---------------- 상태 ----------------
class Tenant:
    __slots__ = ("pid","sock","alive")
    def __init__(self, pid:int, sock:str):
        self.pid  = pid
        self.sock = sock
        self.alive= True

tenants: Dict[str, Tenant] = {}     # tid -> Tenant
remain : Dict[str,int]    = {}      # 남은 backlog (옵션)
alive_cnt = 0

def wait_tenants():
    """ EXPECT_TENANTS=A,B,C 면 그 셋 다 올 때까지 기다림. 없으면 최소 1명 올 때 시작 """
    expect = [t.strip() for t in EXPECT.split(",") if t.strip()] if EXPECT else []
    deadline = time.time() + 30.0  # 30초까지만 대기 (환경에 맞게)
    while True:
        if expect:
            if all(t in tenants for t in expect): break
        else:
            if len(tenants) >= 1: break
        if time.time() > deadline: break
        _recv_all()
        time.sleep(0.05)

def equal_split(active: List[str], total: int) -> Dict[str,int]:
    if not active: return {}
    base = total // len(active)
    rem  = total %  len(active)
    out = {t: base for t in active}
    # 순서 안정적 분배
    for t in active[:rem]:
        out[t] += 1
    return out

def _recv_all():
    """ 비동기 수신처리: HELLO / SD / BE / BYE """
    try:
        while True:
            data,_ = srv.recvfrom(512)
            s = data.decode("utf-8", "ignore")

            m = HELLO_RE.match(s)
            if m:
                pid, sock, tid = m.groups()
                tenants[tid] = Tenant(int(pid), sock)
                if tid not in remain:
                    # backlog가 있으면 초기화, 없으면 '모름' 표시(종료는 BYE로)
                    remain[tid] = BACKLOG.get(tid, -1)
                print(f"[HELLO] {tid} -> {sock}")
                continue

            m = SD_RE.match(s)
            if m:
                _pid, tid, sp, kseq = m.groups()
                # SD는 현 스쿼드에서 해당 tenant가 배정 share를 다 쓴 시점
                inbox_events.append(("SD", tid))
                continue

            m = BE_RE.match(s)
            if m:
                _pid, tid, what = m.groups()
                inbox_events.append((what, tid))  # BOOST_ON / BOOST_OFF
                continue

            m = BYE_RE.match(s)
            if m:
                _pid, tid = m.groups()
                if tid in tenants:
                    tenants[tid].alive = False
                inbox_events.append(("BYE", tid))
                continue

    except socket.timeout:
        pass

# ---------------- 메인 루프 ----------------
def main():
    global srv, inbox_events
    # 소켓 bind
    if os.path.exists(MASTER): os.unlink(MASTER)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    srv.bind(MASTER)
    srv.settimeout(0.1)
    print(f"[master] listen {MASTER}  SQUAD={SQUAD}")

    # 초기 기다림
    inbox_events = []  # (type, tenant)
    wait_tenants()

    squad_id = 0
    boosted  = None

    # 종료는 (a) EXPECT가 있으면 그들이 BYE 모두 보낼 때, (b) EXPECT 없으면 모든 등록자가 BYE일 때
    def all_done():
        if not tenants:
            return False
        if EXPECT:
            needs = [t.strip() for t in EXPECT.split(",") if t.strip()]
            if not needs: return False
            return all((t in tenants and tenants[t].alive==False) for t in needs)
        else:
            return all(t.alive==False for t in tenants.values())

    while True:
        _recv_all()
        if all_done():
            print("[master] all done"); break

        # 현재 살아있는 테넌트
        active = [tid for tid,info in tenants.items() if info.alive]
        if not active:
            time.sleep(0.05); continue

        # 스쿼드 share 계산(기본: 균등)
        share = equal_split(active, SQUAD) if EQUAL_SHARE else equal_split(active, SQUAD)
        # 각 테넌트에게 몫 알려주기
        for t in active:
            send(tenants[t].sock, f"set_squad {SQUAD}")
            send(tenants[t].sock, f"set_share {share[t]}")
            send(tenants[t].sock, "boost_off")
            send(tenants[t].sock, "squad_reset")
        boosted = None
        squad_id += 1
        log("SQUAD_START", squad_id, "", share, remain, "")

        finished = set()          # SD 보낸 테넌트들
        sd_cause_for_boost = ""   # 부스트 원인(처음 SD 보낸 테넌트)
        t0 = time.time()

        while True:
            _recv_all()

            # SD 처리
            while inbox_events:
                ev, who = inbox_events.pop(0)

                if ev == "SD":
                    if who in active and who not in finished:
                        finished.add(who)
                        # 아직 부스트 안했으면, 남은 대상 중 하나에게 boost_on
                        if boosted is None:
                            # 남아있는(아직 SD 안한) 테넌트 후보
                            candidates = [t for t in active if t not in finished]
                            if candidates:
                                target = candidates[0]   # 간단히 첫 후보
                                send(tenants[target].sock, "boost_on")
                                boosted = target
                                sd_cause_for_boost = who
                                note = f"cause={who}"
                                log("BOOST_ON", squad_id, target, share, remain, note)

                elif ev == "BOOST_ON":
                    # libbless가 자가 보고하는 경우도 로그 (선택)
                    if boosted is None:
                        boosted = who
                        log("BOOST_ON", squad_id, who, share, remain, "from_BE")

                elif ev == "BOOST_OFF":
                    log("BOOST_OFF", squad_id, who, share, remain, "from_BE")

                elif ev == "BYE":
                    # 텐넌트 종료(다음 스쿼드부터 active에서 빠짐)
                    pass

            # 스쿼드 종료 조건:
            # 1) active 전원이 SD를 보냄 → 정상 종료
            # 2) 타임아웃 → timeout 종료
            if len(finished) == len(active):
                # 정상 종료
                for t in active: send(tenants[t].sock, "boost_off")
                # remain 카운트가 있다면 차감
                if BACKLOG:
                    for t in active:
                        remain[t] = max(0, remain[t] - share.get(t,0))
                log("SQUAD_END", squad_id, "", share, remain, f"boosted={boosted},cause={sd_cause_for_boost}")
                break

            if time.time() - t0 > SQUAD_TMO_S:
                # 타임아웃 종료
                for t in active: send(tenants[t].sock, "boost_off")
                # SD 보낸 테넌트만 차감(선택), 혹은 차감하지 않음
                if BACKLOG:
                    for t in finished:
                        remain[t] = max(0, remain[t] - share.get(t,0))
                log("SQUAD_END_TIMEOUT", squad_id, "", share, remain, f"finished={sorted(list(finished))},boosted={boosted}")
                break

            time.sleep(0.002)

    csv_f.close()
    try:
        srv.close()
        if os.path.exists(MASTER): os.unlink(MASTER)
    except Exception:
        pass

if __name__ == "__main__":
    main()
