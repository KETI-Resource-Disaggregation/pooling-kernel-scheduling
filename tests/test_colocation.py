#!/usr/bin/env python3
"""
Co-location + Profiler 연동 테스트
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'controller'))

from scheduling_coordinator import SchedulingCoordinator

def test_colocation_with_profiler():
    """Profiler 연동 Co-location 테스트"""
    print("=" * 60)
    print("Co-location + Profiler Integration Test")
    print("=" * 60 + "\n")

    coord = SchedulingCoordinator(total_sms=84)

    # 1. Profiler 연결 확인
    print("[1] Profiler Connection")
    if coord.profiler:
        print("    ✓ Connected to profiler")
    else:
        print("    ✗ Profiler not connected (using defaults)")
    print()

    # 2. Co-location 설정 (ResNet50 + BERT)
    print("[2] Setup Co-location: ResNet50 + BERT")
    state_a, state_b = coord.setup_colocation(
        "workload_A", "workload_B",
        model_a="resnet50", model_b="bert"
    )

    print(f"    ResNet50 (A):")
    print(f"      - Kernel Type: {state_a.kernel_type.name}")
    print(f"      - SM Count: {state_a.sm_allocation.sm_count} ({state_a.sm_allocation.sm_pct:.1f}%)")
    print(f"      - Time Compensation: {state_a.time_compensation:.2f}x")

    print(f"    BERT (B):")
    print(f"      - Kernel Type: {state_b.kernel_type.name}")
    print(f"      - SM Count: {state_b.sm_allocation.sm_count} ({state_b.sm_allocation.sm_pct:.1f}%)")
    print(f"      - Time Compensation: {state_b.time_compensation:.2f}x")
    print()

    # 3. 크레딧 계산 (동시 실행 중)
    print("[3] Credit Allocation (Both Active)")
    credits = coord.compute_credits(["workload_A", "workload_B"], 524288)
    for tid, alloc in credits.items():
        state = coord.get_workload(tid)
        model = state.model if state else "unknown"
        print(f"    {tid} ({model}):")
        print(f"      - Base: {alloc.base_credits:,}")
        print(f"      - SM Compensated: {alloc.sm_compensated_credits:,}")
        print(f"      - Final: {alloc.final_credits:,}")
    print()

    # 4. Work-conserving (A 종료)
    print("[4] Work-conserving: ResNet50 (A) Finishes")
    coord.handle_workload_idle("workload_A")

    credits = coord.compute_credits(["workload_A", "workload_B"], 524288)
    for tid, alloc in credits.items():
        state = coord.get_workload(tid)
        print(f"    {tid}:")
        print(f"      - WC Boost: {state.wc_boost:.2f}x")
        print(f"      - Final Credits: {alloc.final_credits:,}")
    print()

    # 5. 상태 요약
    print("[5] Status Summary")
    status = coord.get_status()
    print(f"    Total SMs: {status['sm_manager']['total_sms']}")
    print(f"    Active Allocations: {status['sm_manager']['allocations']}")
    print(f"    Work-conserving Enabled: {status['work_conserving']['enabled']}")
    print(f"    Idle Tenants: {status['work_conserving']['idle_tenants']}")
    print(f"    Boosted Tenants: {status['work_conserving']['boosted_tenants']}")
    print()

    print("=" * 60)
    print("Test Complete!")
    print("=" * 60)


def test_different_combinations():
    """다양한 워크로드 조합 테스트"""
    print("\n" + "=" * 60)
    print("Different Workload Combinations Test")
    print("=" * 60 + "\n")

    coord = SchedulingCoordinator(total_sms=84)

    combinations = [
        ("resnet50", "bert"),      # COMPUTE + MIXED
        ("resnet50", "vgg16"),     # COMPUTE + COMPUTE
        ("bert", "gpt2"),          # MIXED + MIXED
        ("mobilenet", "bert"),     # COMPUTE + MIXED
    ]

    for model_a, model_b in combinations:
        # Reset coordinator
        coord = SchedulingCoordinator(total_sms=84)

        state_a, state_b = coord.setup_colocation(
            "A", "B", model_a=model_a, model_b=model_b
        )

        print(f"{model_a} ({state_a.kernel_type.name}) + {model_b} ({state_b.kernel_type.name}):")
        print(f"  {model_a}: SM={state_a.sm_allocation.sm_count} ({state_a.sm_allocation.sm_pct:.0f}%), "
              f"time_comp={state_a.time_compensation:.2f}x")
        print(f"  {model_b}: SM={state_b.sm_allocation.sm_count} ({state_b.sm_allocation.sm_pct:.0f}%), "
              f"time_comp={state_b.time_compensation:.2f}x")
        print()


if __name__ == "__main__":
    test_colocation_with_profiler()
    test_different_combinations()
