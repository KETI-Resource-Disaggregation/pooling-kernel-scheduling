#!/usr/bin/env python3
"""
통합 테스트 - Soft SM Partitioning + Work-conserving + Profiler

전체 시스템 흐름 검증:
1. Profiler에서 커널 타입 조회
2. SM Manager가 타입 기반 SM 할당
3. 시간 보상 계수 적용
4. Work-conserving 동작 확인
5. Scheduler Coordinator 통합 동작
"""
import os
import sys
import time
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'controller'))

from sm_manager import SMManager, KernelType, Priority, SM_RATIO_BY_TYPE
from scheduling_coordinator import SchedulingCoordinator

# Test Results
results = {"passed": 0, "failed": 0, "tests": []}


def test(name):
    """Test decorator"""
    def decorator(func):
        def wrapper():
            try:
                func()
                results["passed"] += 1
                results["tests"].append({"name": name, "status": "PASS"})
                print(f"  ✓ {name}")
            except AssertionError as e:
                results["failed"] += 1
                results["tests"].append({"name": name, "status": "FAIL", "error": str(e)})
                print(f"  ✗ {name}: {e}")
            except Exception as e:
                results["failed"] += 1
                results["tests"].append({"name": name, "status": "ERROR", "error": str(e)})
                print(f"  ✗ {name}: ERROR - {e}")
        return wrapper
    return decorator


# ==================== SM Manager Tests ====================
print("\n" + "=" * 60)
print("[1] SM Manager Unit Tests")
print("=" * 60)

@test("SM allocation - single workload")
def test_sm_single():
    mgr = SMManager(84)
    alloc = mgr.allocate("A", KernelType.COMPUTE_BOUND, Priority.HIGH)
    assert alloc.sm_count > 0, "SM count should be > 0"
    assert alloc.sm_pct > 0, "SM pct should be > 0"

test_sm_single()


@test("SM allocation - co-location COMPUTE + MIXED")
def test_sm_colocation_compute_mixed():
    mgr = SMManager(84)
    a, b = mgr.compute_colocation_allocation(
        "A", KernelType.COMPUTE_BOUND,
        "B", KernelType.MIXED
    )
    expected_a, expected_b = SM_RATIO_BY_TYPE[(KernelType.COMPUTE_BOUND, KernelType.MIXED)]
    assert abs(a.sm_pct - expected_a * 100) < 1, f"A should get ~{expected_a*100}%, got {a.sm_pct}%"
    assert abs(b.sm_pct - expected_b * 100) < 1, f"B should get ~{expected_b*100}%, got {b.sm_pct}%"

test_sm_colocation_compute_mixed()


@test("SM allocation - co-location COMPUTE + COMPUTE")
def test_sm_colocation_same_type():
    mgr = SMManager(84)
    a, b = mgr.compute_colocation_allocation(
        "A", KernelType.COMPUTE_BOUND,
        "B", KernelType.COMPUTE_BOUND
    )
    assert abs(a.sm_pct - 50) < 5, f"Same type should be ~50/50, got {a.sm_pct}/{b.sm_pct}"

test_sm_colocation_same_type()


@test("Time compensation - low SM gets boost")
def test_time_compensation():
    mgr = SMManager(84)
    a, b = mgr.compute_colocation_allocation(
        "A", KernelType.COMPUTE_BOUND,
        "B", KernelType.MIXED
    )
    # B has less SM (45%), should get time compensation
    assert b.time_compensation > 1.0, f"B should have time_comp > 1.0, got {b.time_compensation}"
    # A has more SM (55%), no compensation needed
    assert a.time_compensation == 1.0, f"A should have time_comp = 1.0, got {a.time_compensation}"

test_time_compensation()


@test("Work-conserving - idle boost")
def test_work_conserving():
    mgr = SMManager(84)
    mgr.compute_colocation_allocation(
        "A", KernelType.COMPUTE_BOUND,
        "B", KernelType.MIXED
    )

    # Initially, no boost
    assert mgr.get_work_conserving_boost("B") == 1.0, "Initial boost should be 1.0"

    # Mark A as idle
    mgr.mark_idle("A")
    boost = mgr.get_work_conserving_boost("B")
    assert boost > 1.0, f"B should get boost when A is idle, got {boost}"

test_work_conserving()


@test("Work-conserving - deallocation boost")
def test_deallocation_boost():
    mgr = SMManager(84)
    mgr.compute_colocation_allocation(
        "A", KernelType.COMPUTE_BOUND,
        "B", KernelType.MIXED
    )

    # Deallocate A
    mgr.deallocate("A")
    boost = mgr.get_work_conserving_boost("B")
    assert boost > 1.0, f"B should get boost when A deallocated, got {boost}"

test_deallocation_boost()


# ==================== Coordinator Tests ====================
print("\n" + "=" * 60)
print("[2] Scheduling Coordinator Tests")
print("=" * 60)


@test("Coordinator - workload registration")
def test_coordinator_register():
    coord = SchedulingCoordinator(84)
    state = coord.register_workload("A", kernel_type="COMPUTE_BOUND", priority="HIGH")
    assert state is not None, "State should not be None"
    assert state.kernel_type == KernelType.COMPUTE_BOUND, "Kernel type should be COMPUTE_BOUND"
    assert state.sm_allocation is not None, "SM allocation should not be None"

test_coordinator_register()


@test("Coordinator - credit allocation")
def test_coordinator_credits():
    coord = SchedulingCoordinator(84)
    coord.register_workload("A", kernel_type="COMPUTE_BOUND")
    coord.register_workload("B", kernel_type="MIXED")

    credits = coord.compute_credits(["A", "B"], 1000000)
    assert "A" in credits and "B" in credits, "Both should have credits"

    total = credits["A"].final_credits + credits["B"].final_credits
    assert abs(total - 1000000) < 100, f"Total credits should be ~1000000, got {total}"

test_coordinator_credits()


@test("Coordinator - work-conserving credit boost")
def test_coordinator_wc_credits():
    coord = SchedulingCoordinator(84)
    coord.register_workload("A", kernel_type="COMPUTE_BOUND")
    coord.register_workload("B", kernel_type="MIXED")

    # Before idle
    before = coord.compute_credits(["A", "B"], 1000000)
    b_before = before["B"].final_credits

    # Mark A as idle
    coord.handle_workload_idle("A")

    # After idle
    after = coord.compute_credits(["A", "B"], 1000000)
    b_after = after["B"].final_credits

    assert b_after > b_before, f"B credits should increase after A idle: {b_before} -> {b_after}"

test_coordinator_wc_credits()


@test("Coordinator - co-location setup")
def test_coordinator_colocation():
    coord = SchedulingCoordinator(84)
    state_a, state_b = coord.setup_colocation(
        "A", "B",
        model_a="resnet50", model_b="bert"
    )

    # Should have different SM counts based on type
    assert state_a.sm_allocation is not None, "A should have SM allocation"
    assert state_b.sm_allocation is not None, "B should have SM allocation"

    # Verify types were detected (if profiler available)
    if coord.profiler:
        assert state_a.kernel_type == KernelType.COMPUTE_BOUND, "ResNet should be COMPUTE_BOUND"
        assert state_b.kernel_type == KernelType.MIXED, "BERT should be MIXED"

test_coordinator_colocation()


@test("Coordinator - status report")
def test_coordinator_status():
    coord = SchedulingCoordinator(84)
    coord.register_workload("A", kernel_type="COMPUTE_BOUND")

    status = coord.get_status()
    assert "workloads" in status, "Status should have workloads"
    assert "A" in status["workloads"], "A should be in workloads"
    assert "sm_manager" in status, "Status should have sm_manager info"
    assert "work_conserving" in status, "Status should have work_conserving info"

test_coordinator_status()


# ==================== Profiler Integration Tests ====================
print("\n" + "=" * 60)
print("[3] Profiler Integration Tests")
print("=" * 60)

coord_with_profiler = SchedulingCoordinator(84)

@test("Profiler connection")
def test_profiler_connection():
    if coord_with_profiler.profiler:
        assert coord_with_profiler.profiler.health_check(), "Profiler should be healthy"
    else:
        print("    (profiler not available, skipping)")

test_profiler_connection()


@test("Profiler - get profile for known model")
def test_profiler_profile():
    if not coord_with_profiler.profiler:
        print("    (profiler not available, skipping)")
        return

    profile = coord_with_profiler.profiler.get_profile("resnet50")
    assert profile is not None, "Should get profile for resnet50"
    assert profile.arithmetic_intensity > 0, "Should have arithmetic intensity"

test_profiler_profile()


@test("Profiler - SM allocation with profiler type")
def test_profiler_sm_allocation():
    if not coord_with_profiler.profiler:
        print("    (profiler not available, skipping)")
        return

    coord = SchedulingCoordinator(84)
    state_a, state_b = coord.setup_colocation("A", "B", "resnet50", "bert")

    # ResNet (COMPUTE_BOUND) should get more SM than BERT (MIXED)
    assert state_a.sm_allocation.sm_count > state_b.sm_allocation.sm_count, \
        f"ResNet should get more SM: {state_a.sm_allocation.sm_count} vs {state_b.sm_allocation.sm_count}"

test_profiler_sm_allocation()


# ==================== End-to-End Scenario Tests ====================
print("\n" + "=" * 60)
print("[4] End-to-End Scenario Tests")
print("=" * 60)


@test("Scenario - Full co-location lifecycle")
def test_full_lifecycle():
    coord = SchedulingCoordinator(84)

    # 1. Setup co-location
    state_a, state_b = coord.setup_colocation("W1", "W2", "resnet50", "bert")

    # 2. Both running - compute credits
    credits1 = coord.compute_credits(["W1", "W2"], 524288)
    w1_c1 = credits1["W1"].final_credits
    w2_c1 = credits1["W2"].final_credits

    # 3. W1 finishes (idle)
    coord.handle_workload_idle("W1")

    # 4. W2 gets boosted credits
    credits2 = coord.compute_credits(["W1", "W2"], 524288)
    w2_c2 = credits2["W2"].final_credits

    assert w2_c2 > w2_c1, f"W2 credits should increase: {w2_c1} -> {w2_c2}"

    # 5. W1 deallocated
    coord.unregister_workload("W1")

    # 6. Verify final state
    status = coord.get_status()
    assert "W1" not in status["workloads"], "W1 should be removed"
    assert "W2" in status["workloads"], "W2 should remain"

test_full_lifecycle()


@test("Scenario - Priority-based allocation")
def test_priority_allocation():
    coord = SchedulingCoordinator(84)

    # Register with different priorities
    coord.register_workload("HIGH", kernel_type="COMPUTE_BOUND", priority="HIGH")
    coord.register_workload("LOW", kernel_type="COMPUTE_BOUND", priority="LOW")

    credits = coord.compute_credits(["HIGH", "LOW"], 1000000)

    assert credits["HIGH"].base_credits > credits["LOW"].base_credits, \
        "HIGH priority should get more base credits"

test_priority_allocation()


# ==================== Summary ====================
print("\n" + "=" * 60)
print("Test Summary")
print("=" * 60)
print(f"  Passed: {results['passed']}")
print(f"  Failed: {results['failed']}")
print(f"  Total:  {results['passed'] + results['failed']}")

if results["failed"] > 0:
    print("\nFailed tests:")
    for t in results["tests"]:
        if t["status"] != "PASS":
            print(f"  - {t['name']}: {t.get('error', 'unknown')}")
    sys.exit(1)
else:
    print("\n✓ All tests passed!")
    sys.exit(0)
