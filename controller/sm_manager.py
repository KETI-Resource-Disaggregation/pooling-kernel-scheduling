#!/usr/bin/env python3
"""
SM Manager - Soft SM Partitioning + Work-conserving

워크로드별 SM 할당 및 시간 보상 관리

주요 기능:
1. 커널 타입 기반 초기 SM 할당
2. 시간 보상 계수 계산 (SM이 적으면 시간 크레딧 증가)
3. Work-conserving: 유휴 워크로드의 자원 재분배
"""

import os
import time
import threading
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

# ==================== Configuration ====================
DEFAULT_TOTAL_SMS = int(os.environ.get("TOTAL_SMS", "84"))

class KernelType(Enum):
    UNKNOWN = 0
    COMPUTE_BOUND = 1
    MEMORY_BOUND = 2
    MIXED = 3

class Priority(Enum):
    LOW = 0
    MED = 1
    HIGH = 2

# SM 할당 기본 비율 (커널 타입별)
SM_RATIO_BY_TYPE = {
    # COMPUTE_BOUND는 SM을 더 많이 활용
    (KernelType.COMPUTE_BOUND, KernelType.MEMORY_BOUND): (0.60, 0.40),
    (KernelType.MEMORY_BOUND, KernelType.COMPUTE_BOUND): (0.40, 0.60),
    (KernelType.COMPUTE_BOUND, KernelType.MIXED): (0.55, 0.45),
    (KernelType.MIXED, KernelType.COMPUTE_BOUND): (0.45, 0.55),
    (KernelType.MEMORY_BOUND, KernelType.MIXED): (0.45, 0.55),
    (KernelType.MIXED, KernelType.MEMORY_BOUND): (0.55, 0.45),
    # 같은 타입이면 균등
    (KernelType.COMPUTE_BOUND, KernelType.COMPUTE_BOUND): (0.50, 0.50),
    (KernelType.MEMORY_BOUND, KernelType.MEMORY_BOUND): (0.50, 0.50),
    (KernelType.MIXED, KernelType.MIXED): (0.50, 0.50),
}

# 우선순위별 SM 보너스
PRIORITY_SM_BONUS = {
    Priority.HIGH: 0.10,  # +10%
    Priority.MED: 0.0,
    Priority.LOW: -0.05,  # -5%
}

# ==================== Data Classes ====================
@dataclass
class SMAllocation:
    """워크로드별 SM 할당 정보"""
    tenant_id: str
    kernel_type: KernelType
    priority: Priority
    sm_count: int
    sm_pct: float
    time_compensation: float  # SM이 적으면 > 1.0
    allocated_at: float = field(default_factory=time.time)
    is_active: bool = True
    last_activity: float = field(default_factory=time.time)

@dataclass
class WorkConservingState:
    """Work-conserving 상태"""
    idle_tenants: List[str] = field(default_factory=list)
    boosted_tenants: Dict[str, float] = field(default_factory=dict)  # tenant -> boost_factor
    freed_credits: float = 0.0
    last_update: float = field(default_factory=time.time)

# ==================== SM Manager ====================
class SMManager:
    """
    Soft SM Partitioning + Work-conserving Manager

    워크로드별 SM 할당과 시간 보상을 관리합니다.
    """

    def __init__(self, total_sms: int = DEFAULT_TOTAL_SMS):
        self.total_sms = total_sms
        self.allocations: Dict[str, SMAllocation] = {}
        self.wc_state = WorkConservingState()
        self._lock = threading.RLock()

        # 최소/최대 SM 비율
        self.min_sm_pct = 10.0  # 최소 10%
        self.max_sm_pct = 90.0  # 최대 90%

        # 유휴 판정 임계값
        self.idle_threshold_s = 1.0  # 1초 이상 활동 없으면 유휴

    def allocate(
        self,
        tenant_id: str,
        kernel_type: KernelType,
        priority: Priority = Priority.MED,
        colocated_with: Optional[str] = None
    ) -> SMAllocation:
        """
        새 워크로드에 SM 할당

        Args:
            tenant_id: 테넌트 ID
            kernel_type: 커널 타입
            priority: 우선순위
            colocated_with: 함께 실행 중인 다른 테넌트 ID

        Returns:
            SMAllocation 객체
        """
        with self._lock:
            # 이미 할당되어 있으면 업데이트
            if tenant_id in self.allocations:
                alloc = self.allocations[tenant_id]
                alloc.kernel_type = kernel_type
                alloc.priority = priority
                alloc.last_activity = time.time()
                return alloc

            # SM 비율 계산
            sm_pct = self._compute_sm_pct(kernel_type, priority, colocated_with)
            sm_count = int(self.total_sms * (sm_pct / 100.0))
            sm_count = max(1, min(self.total_sms, sm_count))

            # 시간 보상 계수 계산
            time_comp = self._compute_time_compensation(sm_pct)

            alloc = SMAllocation(
                tenant_id=tenant_id,
                kernel_type=kernel_type,
                priority=priority,
                sm_count=sm_count,
                sm_pct=sm_pct,
                time_compensation=time_comp
            )

            self.allocations[tenant_id] = alloc
            return alloc

    def deallocate(self, tenant_id: str) -> Optional[SMAllocation]:
        """워크로드 SM 할당 해제"""
        with self._lock:
            if tenant_id in self.allocations:
                alloc = self.allocations.pop(tenant_id)
                # Work-conserving: 해제된 SM 크레딧 추적
                self._trigger_work_conserving(tenant_id, alloc)
                return alloc
            return None

    def get_allocation(self, tenant_id: str) -> Optional[SMAllocation]:
        """특정 워크로드의 SM 할당 조회"""
        return self.allocations.get(tenant_id)

    def get_all_allocations(self) -> Dict[str, SMAllocation]:
        """모든 SM 할당 조회"""
        return dict(self.allocations)

    def update_activity(self, tenant_id: str):
        """워크로드 활동 업데이트 (유휴 판정용)"""
        with self._lock:
            if tenant_id in self.allocations:
                self.allocations[tenant_id].last_activity = time.time()
                self.allocations[tenant_id].is_active = True

                # 유휴 목록에서 제거
                if tenant_id in self.wc_state.idle_tenants:
                    self.wc_state.idle_tenants.remove(tenant_id)

    def mark_idle(self, tenant_id: str):
        """워크로드를 유휴 상태로 표시"""
        with self._lock:
            if tenant_id in self.allocations:
                self.allocations[tenant_id].is_active = False
                if tenant_id not in self.wc_state.idle_tenants:
                    self.wc_state.idle_tenants.append(tenant_id)
                    self._trigger_work_conserving(tenant_id, self.allocations[tenant_id])

    def check_idle_tenants(self) -> List[str]:
        """유휴 워크로드 검사"""
        now = time.time()
        idle = []
        with self._lock:
            for tid, alloc in self.allocations.items():
                if now - alloc.last_activity > self.idle_threshold_s:
                    if tid not in self.wc_state.idle_tenants:
                        self.mark_idle(tid)
                    idle.append(tid)
        return idle

    def get_work_conserving_boost(self, tenant_id: str) -> float:
        """
        Work-conserving 부스트 계수 조회

        유휴 워크로드의 자원을 받은 경우 > 1.0
        """
        return self.wc_state.boosted_tenants.get(tenant_id, 1.0)

    def compute_colocation_allocation(
        self,
        tenant_a: str,
        type_a: KernelType,
        tenant_b: str,
        type_b: KernelType,
        priority_a: Priority = Priority.MED,
        priority_b: Priority = Priority.MED
    ) -> Tuple[SMAllocation, SMAllocation]:
        """
        두 워크로드의 Co-location SM 할당 계산

        커널 타입 조합에 따라 최적의 SM 비율 결정
        """
        with self._lock:
            # 타입 조합에 따른 기본 비율
            key = (type_a, type_b)
            if key in SM_RATIO_BY_TYPE:
                ratio_a, ratio_b = SM_RATIO_BY_TYPE[key]
            else:
                ratio_a, ratio_b = 0.5, 0.5

            # 우선순위 보너스 적용
            ratio_a += PRIORITY_SM_BONUS.get(priority_a, 0)
            ratio_b += PRIORITY_SM_BONUS.get(priority_b, 0)

            # 정규화
            total = ratio_a + ratio_b
            ratio_a = ratio_a / total
            ratio_b = ratio_b / total

            # SM 비율 → 개수
            sm_pct_a = ratio_a * 100.0
            sm_pct_b = ratio_b * 100.0

            sm_pct_a = max(self.min_sm_pct, min(self.max_sm_pct, sm_pct_a))
            sm_pct_b = max(self.min_sm_pct, min(self.max_sm_pct, sm_pct_b))

            sm_count_a = int(self.total_sms * ratio_a)
            sm_count_b = int(self.total_sms * ratio_b)

            # 시간 보상
            time_comp_a = self._compute_time_compensation(sm_pct_a)
            time_comp_b = self._compute_time_compensation(sm_pct_b)

            alloc_a = SMAllocation(
                tenant_id=tenant_a,
                kernel_type=type_a,
                priority=priority_a,
                sm_count=sm_count_a,
                sm_pct=sm_pct_a,
                time_compensation=time_comp_a
            )

            alloc_b = SMAllocation(
                tenant_id=tenant_b,
                kernel_type=type_b,
                priority=priority_b,
                sm_count=sm_count_b,
                sm_pct=sm_pct_b,
                time_compensation=time_comp_b
            )

            self.allocations[tenant_a] = alloc_a
            self.allocations[tenant_b] = alloc_b

            return alloc_a, alloc_b

    # ==================== Private Methods ====================

    def _compute_sm_pct(
        self,
        kernel_type: KernelType,
        priority: Priority,
        colocated_with: Optional[str]
    ) -> float:
        """SM 비율 계산"""
        # 기본: 활성 워크로드 수에 따른 균등 분배
        active_count = len([a for a in self.allocations.values() if a.is_active])
        if active_count == 0:
            base_pct = 100.0
        else:
            base_pct = 100.0 / (active_count + 1)

        # 커널 타입에 따른 조정
        if kernel_type == KernelType.COMPUTE_BOUND:
            base_pct *= 1.1  # COMPUTE_BOUND는 SM 10% 더
        elif kernel_type == KernelType.MEMORY_BOUND:
            base_pct *= 0.9  # MEMORY_BOUND는 SM 10% 덜

        # 우선순위 보너스
        base_pct += base_pct * PRIORITY_SM_BONUS.get(priority, 0)

        # Co-location 파트너가 있으면 조정
        if colocated_with and colocated_with in self.allocations:
            partner = self.allocations[colocated_with]
            key = (kernel_type, partner.kernel_type)
            if key in SM_RATIO_BY_TYPE:
                my_ratio, _ = SM_RATIO_BY_TYPE[key]
                base_pct = my_ratio * 100.0

        return max(self.min_sm_pct, min(self.max_sm_pct, base_pct))

    def _compute_time_compensation(self, sm_pct: float) -> float:
        """
        시간 보상 계수 계산

        SM 비율이 50% 미만이면 시간 크레딧 보상
        공식: compensation = 50 / sm_pct (최대 2.0)
        """
        if sm_pct >= 50.0:
            return 1.0

        comp = 50.0 / max(sm_pct, self.min_sm_pct)
        return min(2.0, comp)  # 최대 2배 보상

    def _trigger_work_conserving(self, freed_tenant: str, freed_alloc: SMAllocation):
        """Work-conserving 트리거"""
        # 활성 워크로드에 부스트 적용
        active_tenants = [
            tid for tid, alloc in self.allocations.items()
            if alloc.is_active and tid != freed_tenant
        ]

        if not active_tenants:
            return

        # 해제된 SM 비율을 활성 워크로드에 분배
        freed_pct = freed_alloc.sm_pct
        boost_per_tenant = freed_pct / len(active_tenants)

        for tid in active_tenants:
            current_boost = self.wc_state.boosted_tenants.get(tid, 1.0)
            # 부스트: 추가 SM 비율에 비례
            new_boost = current_boost + (boost_per_tenant / 100.0)
            self.wc_state.boosted_tenants[tid] = min(2.0, new_boost)

        self.wc_state.last_update = time.time()

    def reset_work_conserving(self):
        """Work-conserving 상태 리셋"""
        with self._lock:
            self.wc_state = WorkConservingState()


# ==================== Singleton Instance ====================
_sm_manager: Optional[SMManager] = None

def get_sm_manager() -> SMManager:
    """SM Manager 싱글톤 인스턴스"""
    global _sm_manager
    if _sm_manager is None:
        _sm_manager = SMManager()
    return _sm_manager


# ==================== Utility Functions ====================
def kernel_type_from_string(s: str) -> KernelType:
    """문자열에서 KernelType 변환"""
    s = s.upper().replace("-", "_")
    if "COMPUTE" in s:
        return KernelType.COMPUTE_BOUND
    elif "MEMORY" in s:
        return KernelType.MEMORY_BOUND
    elif "MIXED" in s:
        return KernelType.MIXED
    return KernelType.UNKNOWN

def priority_from_string(s: str) -> Priority:
    """문자열에서 Priority 변환"""
    s = s.upper()
    if s == "HIGH":
        return Priority.HIGH
    elif s == "LOW":
        return Priority.LOW
    return Priority.MED


# ==================== CLI ====================
def main():
    """CLI 테스트"""
    mgr = SMManager(total_sms=84)

    print("=== SM Manager Test ===\n")

    # 단일 할당
    print("1. Single allocation (ResNet50 - COMPUTE_BOUND)")
    alloc = mgr.allocate("resnet", KernelType.COMPUTE_BOUND, Priority.HIGH)
    print(f"   SM: {alloc.sm_count} ({alloc.sm_pct:.1f}%)")
    print(f"   Time compensation: {alloc.time_compensation:.2f}x\n")

    # Co-location 할당
    print("2. Co-location allocation (ResNet50 + BERT)")
    alloc_a, alloc_b = mgr.compute_colocation_allocation(
        "resnet", KernelType.COMPUTE_BOUND,
        "bert", KernelType.MIXED
    )
    print(f"   ResNet: SM={alloc_a.sm_count} ({alloc_a.sm_pct:.1f}%), Time comp={alloc_a.time_compensation:.2f}x")
    print(f"   BERT:   SM={alloc_b.sm_count} ({alloc_b.sm_pct:.1f}%), Time comp={alloc_b.time_compensation:.2f}x\n")

    # Work-conserving 테스트
    print("3. Work-conserving test (ResNet finishes)")
    mgr.mark_idle("resnet")
    boost = mgr.get_work_conserving_boost("bert")
    print(f"   BERT boost factor: {boost:.2f}x\n")

    # 할당 해제
    print("4. Deallocate ResNet")
    mgr.deallocate("resnet")
    boost = mgr.get_work_conserving_boost("bert")
    print(f"   BERT boost factor after dealloc: {boost:.2f}x\n")

    print("=== All allocations ===")
    for tid, alloc in mgr.get_all_allocations().items():
        print(f"   {tid}: SM={alloc.sm_count}, active={alloc.is_active}")


if __name__ == "__main__":
    main()
