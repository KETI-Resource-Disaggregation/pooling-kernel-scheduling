"""NPU 요청 게이트 코어 — 프록시의 스케줄링·회계 로직 (Exp_44 확정본의 정본 승격).

runner 를 주입받으므로 furiosa 없는 환경에서 단위 테스트 가능 (test_16).

게이트 의미론 = libbless time_credit 계열 (feeder 계약 그대로, Exp_16/26):
  time_mode 1      : 게이트 무장 — credit>0 인 동안만 실행 자격
  time_credit <us> : 잔고 설정 (0=차단 시작, -1=unlimited 해제)
  time_add <us>    : 예산 주입 (feeder 가 tick 10ms 마다)
  time_stats       : 로그에 "total=<charged_us> kernels=<완료수>" 응답 발화

★Exp_44 오답 2건 재발 방지 (설계 확정):
  1) 선택은 min(charged/granted) 정규화 deficit — max-credit 선택 금지
     (credit 소비율 균등화 → 5:5 붕괴 실측). granted = 누적 주입 예산이므로
     feeder 가 어떤 비율로 주입하든 비례가 성립한다.
  2) 게이트 무장(armed) 테넌트는 credit≤0 이면 실행하지 않는다(strict —
     libbless 와 동일). 작업보존 폴백 없음. 클라이언트는 depth≥2 권장.

credit 상한: 잔고는 CREDIT_CAP_US 로 클램프 (유휴 후 폭주 방지 — Exp_16
언더플로 클램프와 대칭의 상한 클램프).
"""
import threading
import time
from collections import deque

CREDIT_CAP_US = 100_000   # 잔고 상한 (100ms 분)


class Tenant:
    __slots__ = ("name", "queue", "armed", "credit_us", "granted_us",
                 "charged_us", "done_n", "unlimited")

    def __init__(self, name):
        self.name = name
        self.queue = deque()      # 대기 요청 (req 객체 — dict)
        self.armed = False        # time_mode 1
        self.unlimited = True     # time_credit -1 상태 (기본: 게이트 없음)
        self.credit_us = 0.0
        self.granted_us = 0.0     # 누적 주입 예산 (deficit 정규화 분모)
        self.charged_us = 0.0     # 누적 실측 실행시간
        self.done_n = 0


class GateCore:
    """단일 PE 의 요청 스케줄러. run_fn(req)->None 은 호출 시간 회계를
    바깥에서 하지 않는다 — GateCore.step() 이 실측·차감한다."""

    def __init__(self, run_fn, clock=time.monotonic, work_conserving=True):
        self._run = run_fn
        self._clock = clock
        self._lock = threading.Lock()
        self._tenants = {}
        # [Exp_85] work_conserving: 크레딧 소진해도 일 있으면 후순위로 서비스(idle 안 함).
        #   False = 기존 strict(non-work-conserving, 크레딧 소진 시 유휴 — 회귀/대조용).
        self.work_conserving = work_conserving

    # ---- 테넌트 관리 ----
    def register(self, name):
        with self._lock:
            if name not in self._tenants:
                self._tenants[name] = Tenant(name)
            return self._tenants[name]

    def unregister(self, name):
        with self._lock:
            t = self._tenants.pop(name, None)
        return [] if t is None else list(t.queue)   # 미처리 요청 반환(호출자가 정리)

    def tenants(self):
        with self._lock:
            return list(self._tenants)

    # ---- 제어 명령 (libbless 계열) ----
    def cmd(self, name, cmd, arg=None):
        """time_mode/time_credit/time_add 처리. 반환: 응답 문자열 or None."""
        with self._lock:
            t = self._tenants.get(name)
            if t is None:
                return None
            if cmd == "time_mode":
                t.armed = (int(arg) == 1)
                if t.armed:
                    t.unlimited = False
                return None
            if cmd == "time_credit":
                v = int(arg)
                if v < 0:                    # -1 = unlimited (게이트 해제)
                    t.unlimited = True
                    t.credit_us = 0.0
                else:
                    t.unlimited = False
                    t.credit_us = float(min(v, CREDIT_CAP_US))
                return None
            if cmd == "time_add":
                v = float(arg)
                t.credit_us = min(t.credit_us + v, CREDIT_CAP_US)
                t.granted_us += v
                return None
            if cmd == "time_stats":
                # libbless 형식 유지 (feeder read_time_stats 정규식 호환)
                return (f"[npu-proxy] mode={1 if t.armed else 0} "
                        f"time_credit={int(t.credit_us)} "
                        f"total={int(t.charged_us)} kernels={t.done_n}")
        return None

    # ---- 제출 ----
    def submit(self, name, req):
        with self._lock:
            self._tenants[name].queue.append(req)

    # ---- 선택 + 실행 (스케줄러 루프가 반복 호출) ----
    def _eligible(self):
        ready = [t for t in self._tenants.values() if t.queue]
        if not ready:
            return None
        cand = [t for t in ready
                if t.unlimited or (not t.armed) or t.credit_us > 0]
        if not cand:
            # [Exp_85] work-conserving: 전원 크레딧 소진이어도 일이 있으면 유휴 대신
            #   후순위(deficit 최대)로 서비스한다. 크레딧은 "누가 먼저"만 정하고 "멈출지"는
            #   정하지 않는다 — Exp_84/85 지연 악화(non-work-conserving idle)의 처방.
            if self.work_conserving:
                cand = ready
            else:
                return None                   # 기존 strict: 전원 credit 소진 → 유휴
        # ★deficit 선택 (max-credit 금지 — Exp_44 오답 1). granted 미주입
        # 테넌트(unlimited)는 분모 1 로 두어 charged 최소 순.
        return min(cand, key=lambda t: t.charged_us / max(t.granted_us, 1.0))

    def step(self):
        """자격 있는 요청 1건 실행. 반환: (tenant_name, req, elapsed_us) or None."""
        with self._lock:
            t = self._eligible()
            if t is None:
                return None
            req = t.queue.popleft()
        s = self._clock()
        self._run(req)
        dt_us = (self._clock() - s) * 1e6
        with self._lock:
            t.charged_us += dt_us
            t.done_n += 1
            if t.armed and not t.unlimited:
                t.credit_us -= dt_us          # 실측시간 차감 (음수 허용 — 다음 add 로 상쇄)
        return (t.name, req, dt_us)

    # ---- 상태 ----
    def stats(self):
        with self._lock:
            return {n: {"armed": t.armed, "unlimited": t.unlimited,
                        "credit_us": round(t.credit_us),
                        "granted_us": round(t.granted_us),
                        "charged_us": round(t.charged_us),
                        "done": t.done_n, "queue": len(t.queue)}
                    for n, t in self._tenants.items()}
