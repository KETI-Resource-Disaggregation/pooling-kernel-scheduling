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
import os
import threading
import time
from collections import deque

# [Exp_99] 잔고 상한을 env 로 뺀다 — 기본값은 종전과 동일(100ms).
#   ★이 상한이 **주입 비율을 잘라먹는다**는 것이 Exp_99 의 발견이다.
#   feeder 가 틱마다 예산×비율을 넣는데(A 210ms, B 90ms), 상한이 100ms 면
#   A 는 210→100 으로 클램프되고 B 는 90 그대로다 → 실효 주입비 100:90 = 1.11:1
#   (목표 2.33 이 아니다). 예산을 4배로 올리면 A 840→100, B 360→100 으로
#   **둘 다** 클램프되어 비율 정보가 완전히 소멸한다(실측 1.00/0.98/1.00).
#   요청 1건의 실행시간이 상한보다 작을 때만 비율이 성립했다(mt=8, 82ms → 2.29).
CREDIT_CAP_US = float(os.environ.get("GATE_CREDIT_CAP_US", "100000"))   # 잔고 상한

# [Exp_124] 되갚음(구 동작) 대조 스위치. 1 이면 Exp_123 까지의 회계를 그대로 쓴다.
#   ★새 상수가 아니라 **대조 스위치**다 — 수정 전후를 같은 하네스로 재려면 필요하다.
LEGACY_REPAY = os.environ.get("GATE_LEGACY_REPAY", "0") == "1"


class Tenant:
    __slots__ = ("name", "queue", "armed", "credit_us", "granted_us",
                 "charged_us", "charged_extra_us", "done_n", "done_extra_n",
                 "unlimited", "last_run_t")

    def __init__(self, name):
        self.name = name
        self.queue = deque()      # 대기 요청 (req 객체 — dict)
        self.armed = False        # time_mode 1
        self.unlimited = True     # time_credit -1 상태 (기본: 게이트 없음)
        self.credit_us = 0.0
        self.granted_us = 0.0     # 누적 주입 예산 (deficit 정규화 분모)
        # [Exp_124] 계약 회계와 편승 회계를 분리한다.
        #   charged_us       = **계약분**. 크레딧으로 실행된 것. 몫 이행 판정의 분자다
        #   charged_extra_us = **편승분**. work-conserving 폴백으로 실행된 것.
        #     아무도 안 쓰던 시간을 쓴 것이므로 계약 이행 판정에 넣지 않는다.
        #     ★회계에서 빼는 것과 안 보이게 하는 것은 다르다 — stats() 로 그대로 노출한다
        self.charged_us = 0.0
        self.charged_extra_us = 0.0
        self.done_n = 0
        self.done_extra_n = 0
        # [Exp_124] 동률 깨기용. 계약비가 같을 때 **최근에 덜 돌린 쪽**을 먼저 준다.
        #   ★이것이 없으면 min() 이 항상 먼저 등록된 테넌트를 골라 조용히 편향된다
        #     (실측: 0.5:0.5 에서 지분 0.637). 0.7/0.3 에서는 계약비가 정확히 같아지는
        #     일이 드물어 드러나지 않았다 — 몫이 같을 때만 나타나는 편향이었다.
        self.last_run_t = 0.0


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
                # [Exp_124 1-B 분모] **쓰지 못한 몫은 준 것이 아니다.**
                #   잔고는 상한(CREDIT_CAP_US)이 있는데 분모만 무한히 쌓이면,
                #   쉬는 테넌트의 charged/granted 가 0 으로 내려가 복귀 시 독식한다
                #   (Exp_123 4-B: 편승 직후 상대가 0.07/s 로 굶었다).
                #   상한에 걸려 버려진 주입은 분모에 넣지 않는다. 새 상수 없음 —
                #   기존 상한을 재사용한다.
                before = t.credit_us
                t.credit_us = min(t.credit_us + v, CREDIT_CAP_US)
                if LEGACY_REPAY:
                    t.granted_us += v          # 대조용 구 동작
                else:
                    t.granted_us += (t.credit_us - before)
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

    @staticmethod
    def _key(t):
        """선택 키 = (계약비, 마지막 실행 시각).
        계약비만으로는 동률에서 dict 순서가 승자를 정한다 — 몫이 같은 테넌트끼리
        조용히 한쪽으로 기운다. 두 번째 항이 그 자리를 라운드로빈으로 메운다."""
        return (t.charged_us / max(t.granted_us, 1.0), t.last_run_t)

    # ---- 선택 + 실행 (스케줄러 루프가 반복 호출) ----
    def _eligible(self):
        ready = [t for t in self._tenants.values() if t.queue]
        if not ready:
            return None, False
        cand = [t for t in ready
                if t.unlimited or (not t.armed) or t.credit_us > 0]
        if cand:
            # ★계약 실행 — deficit 선택 (max-credit 금지, Exp_44 오답 1).
            #   분자는 **계약분만**(charged_us). 편승분은 여기 들어오지 않는다.
            #   granted 미주입 테넌트(unlimited)는 분모 1 로 두어 charged 최소 순.
            return min(cand, key=self._key), False
        # [Exp_85] work-conserving: 전원 크레딧 소진이어도 일이 있으면 유휴 대신
        #   서비스한다. 크레딧은 "누가 먼저"만 정하고 "멈출지"는 정하지 않는다 —
        #   Exp_84/85 지연 악화(non-work-conserving idle)의 처방.
        if not self.work_conserving:
            return None, False                # 기존 strict: 전원 credit 소진 → 유휴
        # [Exp_124 1-B 분자] 편승 실행의 **순서도 계약비(charged_us/granted_us)로 고른다.**
        #   ★한 번 틀렸던 자리다. 처음엔 총 사용량((base+extra)/granted)으로 골랐는데,
        #     그러면 **편승 이력이 순서에 남아** 오래 편승한 쪽이 복귀 후 편승 물량에서
        #     계속 밀린다(실측: 계약비는 양쪽 1.000 인데 done 지분이 0.363 로 굳었다).
        #     형태만 바뀐 되갚음이다.
        #   계약비로 고르면 편승은 순서에 흔적을 남기지 않는다 —
        #   **계약을 덜 받은 쪽에게 보너스를 먼저 준다**가 규칙이고, 계약비는 편승으로
        #   변하지 않으므로 과거 편승이 미래 배분을 벌하지 않는다.
        #   "계약비가 안 변해 독식한다"는 우려는 성립하지 않는다 — 양쪽이 활성이면
        #   credited 실행이 계약비를 계속 갱신해 교대가 생긴다(2-C 로 확인).
        return min(ready, key=self._key), True

    def step(self):
        """자격 있는 요청 1건 실행. 반환: (tenant_name, req, elapsed_us) or None."""
        with self._lock:
            t, extra = self._eligible()
            if t is None:
                return None
            req = t.queue.popleft()
        s = self._clock()
        self._run(req)
        dt_us = (self._clock() - s) * 1e6
        with self._lock:
            if extra and not LEGACY_REPAY:
                # [Exp_124] 편승 실행 — 계약 회계에 넣지 않고 크레딧도 깎지 않는다.
                #   ★차감까지 하면 잔고가 계속 음수로 눌려 이후 실행이 전부 편승이 되고,
                #     계약분(charged_us)이 자기 몫만큼 자라지 못해 비가 0 으로 내려간다
                #     (= 방향만 뒤집힌 되갚음). 아무도 안 쓰던 시간이므로 계약 밖이다.
                t.charged_extra_us += dt_us
                t.done_extra_n += 1
            else:
                t.charged_us += dt_us
                if t.armed and not t.unlimited:
                    t.credit_us -= dt_us      # 실측시간 차감 (음수 허용 — 다음 add 로 상쇄)
            t.done_n += 1
            t.last_run_t = self._clock()
        return (t.name, req, dt_us)

    # ---- 상태 ----
    def stats(self):
        with self._lock:
            return {n: {"armed": t.armed, "unlimited": t.unlimited,
                        "credit_us": round(t.credit_us),
                        "granted_us": round(t.granted_us),
                        # 계약분 / 편승분을 나눠 노출한다 (Exp_124 1-D:
                        # 회계에서 빼는 것과 안 보이게 하는 것은 다르다)
                        "charged_us": round(t.charged_us),
                        "charged_extra_us": round(t.charged_extra_us),
                        "done": t.done_n, "done_extra": t.done_extra_n,
                        "queue": len(t.queue)}
                    for n, t in self._tenants.items()}
