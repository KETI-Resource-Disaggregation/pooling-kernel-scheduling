#!/usr/bin/env python3
"""
Scheduling Coordinator - SM Manager + Time Credit + Work-conserving 통합

이 모듈은 SM 파티셔닝, 시간 크레딧, Work-conserving을 통합 관리합니다.

사용법:
    from scheduling_coordinator import SchedulingCoordinator

    coord = SchedulingCoordinator()

    # 워크로드 등록
    coord.register_workload("A", kernel_type="COMPUTE_BOUND", model="resnet50")

    # 크레딧 할당
    credits = coord.compute_credits(["A", "B"], total_credits=524288)

    # Work-conserving 처리
    coord.handle_workload_idle("A")
"""

import os
import time
import socket
import threading
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

from sm_manager import (
    SMManager, SMAllocation, KernelType, Priority,
    kernel_type_from_string, priority_from_string, get_sm_manager
)

try:
    from profiler_client import ProfilerClient, get_kernel_type as profiler_get_type
    from profiler_client import KernelType as ProfilerKernelType
    PROFILER_AVAILABLE = True
except ImportError:
    PROFILER_AVAILABLE = False
    ProfilerClient = None
    ProfilerKernelType = None


def convert_profiler_kernel_type(profiler_type) -> KernelType:
    """Profiler KernelType을 SM Manager KernelType으로 변환"""
    if profiler_type is None:
        return KernelType.UNKNOWN
    # Use name-based conversion to handle different enum instances
    name = profiler_type.name if hasattr(profiler_type, 'name') else str(profiler_type)
    return kernel_type_from_string(name)

# ==================== Configuration ====================
PROFILER_URL = os.environ.get("PROFILER_URL", "http://localhost:7070")

# ==================== Data Classes ====================
@dataclass
class WorkloadState:
    """워크로드 상태"""
    tenant_id: str
    model: str
    kernel_type: KernelType
    priority: Priority
    sm_allocation: Optional[SMAllocation] = None

    # 시간 크레딧 관련
    base_credit_pct: float = 0.0  # 기본 크레딧 비율
    time_compensation: float = 1.0  # SM 보상 계수
    wc_boost: float = 1.0  # Work-conserving 부스트

    # 상태
    is_active: bool = True
    is_idle: bool = False
    last_activity: float = field(default_factory=time.time)

    # 소켓 경로
    sock_path: str = ""
    pid: int = 0

@dataclass
class CreditAllocation:
    """크레딧 할당 결과"""
    tenant_id: str
    base_credits: int
    sm_compensated_credits: int
    wc_boosted_credits: int
    final_credits: int

# ==================== Scheduling Coordinator ====================
class SchedulingCoordinator:
    """
    Soft SM Partitioning + Time Credit + Work-conserving 통합 코디네이터
    """

    def __init__(self, total_sms: int = 84):
        self.sm_manager = SMManager(total_sms)
        self.workloads: Dict[str, WorkloadState] = {}
        self._lock = threading.RLock()

        # Profiler 클라이언트
        self.profiler: Optional[ProfilerClient] = None
        if PROFILER_AVAILABLE:
            try:
                self.profiler = ProfilerClient(PROFILER_URL)
                if self.profiler.health_check():
                    print(f"[coordinator] Connected to profiler: {PROFILER_URL}")
                else:
                    print(f"[coordinator] Profiler not available at {PROFILER_URL}")
                    self.profiler = None
            except Exception as e:
                print(f"[coordinator] Profiler connection failed: {e}")
                self.profiler = None

        # Work-conserving 설정
        self.wc_enabled = True
        self.idle_threshold_s = 1.0

    # ==================== Workload Management ====================

    def register_workload(
        self,
        tenant_id: str,
        kernel_type: str = "UNKNOWN",
        model: str = "",
        priority: str = "MED",
        sock_path: str = "",
        pid: int = 0
    ) -> WorkloadState:
        """
        워크로드 등록 및 SM 할당

        Args:
            tenant_id: 테넌트 ID
            kernel_type: 커널 타입 (COMPUTE_BOUND, MEMORY_BOUND, MIXED)
            model: 모델 이름 (resnet50, bert 등)
            priority: 우선순위 (HIGH, MED, LOW)
            sock_path: libbless 소켓 경로
            pid: 프로세스 ID
        """
        with self._lock:
            # 커널 타입 결정 (프로파일러 우선)
            ktype = kernel_type_from_string(kernel_type)
            if ktype == KernelType.UNKNOWN and model and self.profiler:
                profile = self.profiler.get_profile(model)
                if profile:
                    ktype = convert_profiler_kernel_type(profile.kernel_type)
                    print(f"[coordinator] Got profile for {model}: {ktype.name}")

            prio = priority_from_string(priority)

            # SM 할당
            active_ids = [w.tenant_id for w in self.workloads.values() if w.is_active]
            colocated = active_ids[0] if active_ids else None

            sm_alloc = self.sm_manager.allocate(
                tenant_id, ktype, prio, colocated_with=colocated
            )

            # 워크로드 상태 생성
            state = WorkloadState(
                tenant_id=tenant_id,
                model=model,
                kernel_type=ktype,
                priority=prio,
                sm_allocation=sm_alloc,
                time_compensation=sm_alloc.time_compensation,
                sock_path=sock_path,
                pid=pid
            )

            self.workloads[tenant_id] = state

            print(f"[coordinator] Registered {tenant_id}: type={ktype.name}, "
                  f"SM={sm_alloc.sm_count} ({sm_alloc.sm_pct:.1f}%), "
                  f"time_comp={sm_alloc.time_compensation:.2f}x")

            return state

    def unregister_workload(self, tenant_id: str) -> Optional[WorkloadState]:
        """워크로드 등록 해제"""
        with self._lock:
            if tenant_id not in self.workloads:
                return None

            state = self.workloads.pop(tenant_id)
            self.sm_manager.deallocate(tenant_id)

            # Work-conserving 트리거
            if self.wc_enabled:
                self._redistribute_to_active(tenant_id)

            print(f"[coordinator] Unregistered {tenant_id}")
            return state

    def get_workload(self, tenant_id: str) -> Optional[WorkloadState]:
        """워크로드 상태 조회"""
        return self.workloads.get(tenant_id)

    def get_all_workloads(self) -> Dict[str, WorkloadState]:
        """모든 워크로드 조회"""
        return dict(self.workloads)

    # ==================== Credit Allocation ====================

    def compute_credits(
        self,
        active_tenants: List[str],
        total_credits: int,
        equal_share: bool = False
    ) -> Dict[str, CreditAllocation]:
        """
        시간 크레딧 계산

        SM 보상과 Work-conserving 부스트를 적용한 크레딧 분배

        Args:
            active_tenants: 활성 테넌트 목록
            total_credits: 총 크레딧
            equal_share: 균등 분배 여부

        Returns:
            테넌트별 CreditAllocation
        """
        with self._lock:
            if not active_tenants:
                return {}

            results = {}

            # 1. 기본 크레딧 계산
            if equal_share:
                base_each = total_credits / len(active_tenants)
                base_credits = {t: base_each for t in active_tenants}
            else:
                # 우선순위 가중치 적용
                weights = {}
                for t in active_tenants:
                    state = self.workloads.get(t)
                    if state:
                        w = 1.0
                        if state.priority == Priority.HIGH:
                            w = 1.5
                        elif state.priority == Priority.LOW:
                            w = 0.7
                        weights[t] = w
                    else:
                        weights[t] = 1.0

                total_weight = sum(weights.values())
                base_credits = {
                    t: total_credits * (weights[t] / total_weight)
                    for t in active_tenants
                }

            # 2. SM 보상 적용
            sm_compensated = {}
            for t in active_tenants:
                state = self.workloads.get(t)
                comp = state.time_compensation if state else 1.0
                sm_compensated[t] = base_credits[t] * comp

            # 3. Work-conserving 부스트 적용
            wc_boosted = {}
            for t in active_tenants:
                boost = self.sm_manager.get_work_conserving_boost(t)
                state = self.workloads.get(t)
                if state:
                    state.wc_boost = boost
                wc_boosted[t] = sm_compensated[t] * boost

            # 4. 정규화 (총합 = total_credits)
            total_boosted = sum(wc_boosted.values())
            if total_boosted > 0:
                scale = total_credits / total_boosted
            else:
                scale = 1.0

            for t in active_tenants:
                final = int(wc_boosted[t] * scale)
                results[t] = CreditAllocation(
                    tenant_id=t,
                    base_credits=int(base_credits[t]),
                    sm_compensated_credits=int(sm_compensated[t]),
                    wc_boosted_credits=int(wc_boosted[t]),
                    final_credits=final
                )

            return results

    # ==================== Work-conserving ====================

    def handle_workload_idle(self, tenant_id: str):
        """워크로드 유휴 처리"""
        with self._lock:
            if tenant_id not in self.workloads:
                return

            state = self.workloads[tenant_id]
            state.is_idle = True
            state.is_active = False

            self.sm_manager.mark_idle(tenant_id)

            if self.wc_enabled:
                self._redistribute_to_active(tenant_id)

            print(f"[coordinator] {tenant_id} marked idle, triggered work-conserving")

    def handle_workload_active(self, tenant_id: str):
        """워크로드 활성 처리"""
        with self._lock:
            if tenant_id not in self.workloads:
                return

            state = self.workloads[tenant_id]
            state.is_idle = False
            state.is_active = True
            state.last_activity = time.time()

            self.sm_manager.update_activity(tenant_id)

    def check_idle_workloads(self) -> List[str]:
        """유휴 워크로드 검사"""
        now = time.time()
        idle = []
        with self._lock:
            for tid, state in self.workloads.items():
                if not state.is_idle and (now - state.last_activity) > self.idle_threshold_s:
                    self.handle_workload_idle(tid)
                    idle.append(tid)
        return idle

    def _redistribute_to_active(self, idle_tenant: str):
        """유휴 워크로드의 자원을 활성 워크로드에 재분배"""
        active = [
            tid for tid, state in self.workloads.items()
            if state.is_active and tid != idle_tenant
        ]

        if not active:
            return

        # SM Manager의 work-conserving이 자동으로 처리
        # 여기서는 추가 로깅만
        for tid in active:
            boost = self.sm_manager.get_work_conserving_boost(tid)
            print(f"[coordinator] {tid} boost: {boost:.2f}x")

    # ==================== Co-location ====================

    def setup_colocation(
        self,
        tenant_a: str,
        tenant_b: str,
        model_a: str = "",
        model_b: str = ""
    ) -> Tuple[WorkloadState, WorkloadState]:
        """
        두 워크로드의 Co-location 설정

        프로파일러에서 커널 타입을 조회하고 최적의 SM 분배 결정
        """
        with self._lock:
            # 커널 타입 조회
            type_a = KernelType.UNKNOWN
            type_b = KernelType.UNKNOWN

            if self.profiler:
                if model_a:
                    profile = self.profiler.get_profile(model_a)
                    if profile:
                        type_a = convert_profiler_kernel_type(profile.kernel_type)
                if model_b:
                    profile = self.profiler.get_profile(model_b)
                    if profile:
                        type_b = convert_profiler_kernel_type(profile.kernel_type)

            # SM 할당
            alloc_a, alloc_b = self.sm_manager.compute_colocation_allocation(
                tenant_a, type_a,
                tenant_b, type_b
            )

            # 워크로드 상태 생성/업데이트
            state_a = WorkloadState(
                tenant_id=tenant_a,
                model=model_a,
                kernel_type=type_a,
                priority=Priority.MED,
                sm_allocation=alloc_a,
                time_compensation=alloc_a.time_compensation
            )

            state_b = WorkloadState(
                tenant_id=tenant_b,
                model=model_b,
                kernel_type=type_b,
                priority=Priority.MED,
                sm_allocation=alloc_b,
                time_compensation=alloc_b.time_compensation
            )

            self.workloads[tenant_a] = state_a
            self.workloads[tenant_b] = state_b

            print(f"[coordinator] Co-location setup:")
            print(f"  {tenant_a} ({model_a}): {type_a.name}, SM={alloc_a.sm_count}")
            print(f"  {tenant_b} ({model_b}): {type_b.name}, SM={alloc_b.sm_count}")

            return state_a, state_b

    # ==================== libbless Communication ====================

    def send_sm_config(self, tenant_id: str) -> bool:
        """
        libbless에 SM 설정 전송

        주의: 메모리 할당 전에만 유효
        """
        state = self.workloads.get(tenant_id)
        if not state or not state.sock_path:
            return False

        alloc = state.sm_allocation
        if not alloc:
            return False

        msg = f"set_limit_pct {int(alloc.sm_pct)}"
        return self._send_to_tenant(state.sock_path, msg)

    def send_time_credit(self, tenant_id: str, credits: int) -> bool:
        """libbless에 시간 크레딧 전송"""
        state = self.workloads.get(tenant_id)
        if not state or not state.sock_path:
            return False

        msg = f"time_credit {credits}"
        return self._send_to_tenant(state.sock_path, msg)

    def _send_to_tenant(self, sock_path: str, msg: str) -> bool:
        """유닉스 소켓으로 메시지 전송"""
        try:
            s = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
            s.sendto(msg.encode(), sock_path)
            s.close()
            return True
        except Exception as e:
            print(f"[coordinator] Send failed to {sock_path}: {e}")
            return False

    # ==================== Status ====================

    def get_status(self) -> Dict[str, Any]:
        """전체 상태 조회"""
        with self._lock:
            return {
                "workloads": {
                    tid: {
                        "model": s.model,
                        "kernel_type": s.kernel_type.name,
                        "priority": s.priority.name,
                        "sm_count": s.sm_allocation.sm_count if s.sm_allocation else 0,
                        "sm_pct": s.sm_allocation.sm_pct if s.sm_allocation else 0,
                        "time_compensation": s.time_compensation,
                        "wc_boost": s.wc_boost,
                        "is_active": s.is_active,
                        "is_idle": s.is_idle
                    }
                    for tid, s in self.workloads.items()
                },
                "sm_manager": {
                    "total_sms": self.sm_manager.total_sms,
                    "allocations": len(self.sm_manager.allocations)
                },
                "work_conserving": {
                    "enabled": self.wc_enabled,
                    "idle_tenants": self.sm_manager.wc_state.idle_tenants,
                    "boosted_tenants": dict(self.sm_manager.wc_state.boosted_tenants)
                },
                "profiler_connected": self.profiler is not None
            }


# ==================== CLI ====================
def main():
    """CLI 테스트"""
    coord = SchedulingCoordinator(total_sms=84)

    print("=== Scheduling Coordinator Test ===\n")

    # 워크로드 등록
    print("1. Register workloads")
    coord.register_workload("A", model="resnet50", priority="HIGH")
    coord.register_workload("B", model="bert", priority="MED")
    print()

    # 크레딧 계산
    print("2. Compute credits (total=524288)")
    credits = coord.compute_credits(["A", "B"], 524288)
    for tid, alloc in credits.items():
        print(f"   {tid}: base={alloc.base_credits}, "
              f"sm_comp={alloc.sm_compensated_credits}, "
              f"final={alloc.final_credits}")
    print()

    # Work-conserving 테스트
    print("3. Work-conserving (A goes idle)")
    coord.handle_workload_idle("A")
    credits = coord.compute_credits(["A", "B"], 524288)
    for tid, alloc in credits.items():
        print(f"   {tid}: final={alloc.final_credits}")
    print()

    # 상태 조회
    print("4. Status")
    status = coord.get_status()
    for tid, info in status["workloads"].items():
        print(f"   {tid}: SM={info['sm_count']}, active={info['is_active']}, "
              f"wc_boost={info['wc_boost']:.2f}x")


if __name__ == "__main__":
    main()
