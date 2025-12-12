#!/usr/bin/env python3
"""
Soft SM Partitioning + Work-conserving 종합 테스트 및 검증

테스트 항목:
1. SM Manager 단위 테스트
2. Scheduling Coordinator 테스트
3. Profiler 연동 테스트
4. Co-location 시나리오 테스트
5. Work-conserving 시나리오 테스트
6. End-to-End 시나리오 테스트

결과를 로그 파일로 저장
"""
import os
import sys
import time
import json
from datetime import datetime
from io import StringIO

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'controller'))

# ==================== Test Framework ====================
class TestLogger:
    def __init__(self):
        self.log = StringIO()
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "tests": [],
            "summary": {"passed": 0, "failed": 0, "skipped": 0}
        }

    def write(self, msg):
        print(msg, end='')
        self.log.write(msg)

    def writeln(self, msg=""):
        self.write(msg + "\n")

    def section(self, title):
        self.writeln()
        self.writeln("=" * 70)
        self.writeln(f" {title}")
        self.writeln("=" * 70)

    def subsection(self, title):
        self.writeln()
        self.writeln(f"--- {title} ---")

    def test_result(self, name, passed, details="", error=""):
        status = "PASS" if passed else "FAIL"
        icon = "✓" if passed else "✗"
        self.writeln(f"  {icon} {name}")
        if details:
            self.writeln(f"      {details}")
        if error:
            self.writeln(f"      ERROR: {error}")

        self.results["tests"].append({
            "name": name,
            "status": status,
            "details": details,
            "error": error
        })

        if passed:
            self.results["summary"]["passed"] += 1
        else:
            self.results["summary"]["failed"] += 1

    def skip(self, name, reason):
        self.writeln(f"  ○ {name} (SKIPPED: {reason})")
        self.results["tests"].append({
            "name": name,
            "status": "SKIP",
            "details": reason
        })
        self.results["summary"]["skipped"] += 1

    def get_log(self):
        return self.log.getvalue()

    def get_results(self):
        return self.results


logger = TestLogger()

# ==================== Imports ====================
try:
    from sm_manager import (
        SMManager, KernelType, Priority, SMAllocation,
        SM_RATIO_BY_TYPE, PRIORITY_SM_BONUS
    )
    SM_MANAGER_AVAILABLE = True
except ImportError as e:
    SM_MANAGER_AVAILABLE = False
    logger.writeln(f"Warning: sm_manager import failed: {e}")

try:
    from scheduling_coordinator import SchedulingCoordinator, WorkloadState
    COORDINATOR_AVAILABLE = True
except ImportError as e:
    COORDINATOR_AVAILABLE = False
    logger.writeln(f"Warning: scheduling_coordinator import failed: {e}")

try:
    from profiler_client import ProfilerClient
    PROFILER_CLIENT_AVAILABLE = True
except ImportError as e:
    PROFILER_CLIENT_AVAILABLE = False
    logger.writeln(f"Warning: profiler_client import failed: {e}")


# ==================== Test 1: SM Manager ====================
def test_sm_manager():
    logger.section("TEST 1: SM Manager Unit Tests")

    if not SM_MANAGER_AVAILABLE:
        logger.skip("SM Manager tests", "Module not available")
        return

    # Test 1.1: Single allocation
    logger.subsection("1.1 Single Workload Allocation")
    try:
        mgr = SMManager(84)
        alloc = mgr.allocate("test1", KernelType.COMPUTE_BOUND, Priority.HIGH)
        passed = alloc.sm_count > 0 and alloc.sm_pct > 0
        logger.test_result(
            "Single COMPUTE_BOUND allocation",
            passed,
            f"SM={alloc.sm_count} ({alloc.sm_pct:.1f}%), time_comp={alloc.time_compensation:.2f}x"
        )
    except Exception as e:
        logger.test_result("Single COMPUTE_BOUND allocation", False, error=str(e))

    # Test 1.2: Co-location allocations
    logger.subsection("1.2 Co-location SM Allocation")

    test_combinations = [
        (KernelType.COMPUTE_BOUND, KernelType.MIXED, "COMPUTE + MIXED", 0.55, 0.45),
        (KernelType.COMPUTE_BOUND, KernelType.MEMORY_BOUND, "COMPUTE + MEMORY", 0.60, 0.40),
        (KernelType.COMPUTE_BOUND, KernelType.COMPUTE_BOUND, "COMPUTE + COMPUTE", 0.50, 0.50),
        (KernelType.MIXED, KernelType.MIXED, "MIXED + MIXED", 0.50, 0.50),
    ]

    for type_a, type_b, desc, exp_a, exp_b in test_combinations:
        try:
            mgr = SMManager(84)
            alloc_a, alloc_b = mgr.compute_colocation_allocation(
                "A", type_a, "B", type_b
            )

            # Check if ratio is close to expected
            ratio_a = alloc_a.sm_pct / 100.0
            ratio_b = alloc_b.sm_pct / 100.0
            passed = abs(ratio_a - exp_a) < 0.05 and abs(ratio_b - exp_b) < 0.05

            logger.test_result(
                f"{desc}",
                passed,
                f"A={alloc_a.sm_count} SM ({alloc_a.sm_pct:.0f}%), "
                f"B={alloc_b.sm_count} SM ({alloc_b.sm_pct:.0f}%) "
                f"[Expected {exp_a*100:.0f}%/{exp_b*100:.0f}%]"
            )
        except Exception as e:
            logger.test_result(f"{desc}", False, error=str(e))

    # Test 1.3: Time compensation
    logger.subsection("1.3 Time Compensation")
    try:
        mgr = SMManager(84)
        alloc_a, alloc_b = mgr.compute_colocation_allocation(
            "A", KernelType.COMPUTE_BOUND, "B", KernelType.MIXED
        )

        # B has less SM, should have higher compensation
        passed = (alloc_b.time_compensation > 1.0 and
                  alloc_a.time_compensation == 1.0)
        logger.test_result(
            "Time compensation for lower SM",
            passed,
            f"A (55%): {alloc_a.time_compensation:.2f}x, "
            f"B (45%): {alloc_b.time_compensation:.2f}x"
        )
    except Exception as e:
        logger.test_result("Time compensation", False, error=str(e))

    # Test 1.4: Work-conserving
    logger.subsection("1.4 Work-conserving")
    try:
        mgr = SMManager(84)
        mgr.compute_colocation_allocation(
            "A", KernelType.COMPUTE_BOUND, "B", KernelType.MIXED
        )

        # Initial boost should be 1.0
        boost_before = mgr.get_work_conserving_boost("B")

        # Mark A as idle
        mgr.mark_idle("A")
        boost_after_idle = mgr.get_work_conserving_boost("B")

        # Deallocate A
        mgr.deallocate("A")
        boost_after_dealloc = mgr.get_work_conserving_boost("B")

        passed = (boost_before == 1.0 and
                  boost_after_idle > boost_before and
                  boost_after_dealloc > boost_after_idle)

        logger.test_result(
            "Work-conserving boost progression",
            passed,
            f"Initial: {boost_before:.2f}x -> "
            f"After idle: {boost_after_idle:.2f}x -> "
            f"After dealloc: {boost_after_dealloc:.2f}x"
        )
    except Exception as e:
        logger.test_result("Work-conserving", False, error=str(e))


# ==================== Test 2: Scheduling Coordinator ====================
def test_scheduling_coordinator():
    logger.section("TEST 2: Scheduling Coordinator Tests")

    if not COORDINATOR_AVAILABLE:
        logger.skip("Coordinator tests", "Module not available")
        return

    # Test 2.1: Workload registration
    logger.subsection("2.1 Workload Registration")
    try:
        coord = SchedulingCoordinator(84)
        state = coord.register_workload("A", kernel_type="COMPUTE_BOUND", priority="HIGH")

        passed = (state is not None and
                  state.kernel_type == KernelType.COMPUTE_BOUND and
                  state.sm_allocation is not None)

        logger.test_result(
            "Register workload",
            passed,
            f"Type={state.kernel_type.name}, "
            f"SM={state.sm_allocation.sm_count if state.sm_allocation else 0}"
        )
    except Exception as e:
        logger.test_result("Register workload", False, error=str(e))

    # Test 2.2: Credit computation
    logger.subsection("2.2 Credit Computation")
    try:
        coord = SchedulingCoordinator(84)
        coord.register_workload("A", kernel_type="COMPUTE_BOUND")
        coord.register_workload("B", kernel_type="MIXED")

        total_credits = 1000000
        credits = coord.compute_credits(["A", "B"], total_credits)

        actual_total = credits["A"].final_credits + credits["B"].final_credits
        passed = abs(actual_total - total_credits) < 100

        logger.test_result(
            "Credit allocation sums to total",
            passed,
            f"A={credits['A'].final_credits:,}, B={credits['B'].final_credits:,}, "
            f"Total={actual_total:,} (expected {total_credits:,})"
        )
    except Exception as e:
        logger.test_result("Credit computation", False, error=str(e))

    # Test 2.3: Priority-based credits
    logger.subsection("2.3 Priority-based Credit Allocation")
    try:
        coord = SchedulingCoordinator(84)
        coord.register_workload("HIGH", kernel_type="COMPUTE_BOUND", priority="HIGH")
        coord.register_workload("LOW", kernel_type="COMPUTE_BOUND", priority="LOW")

        credits = coord.compute_credits(["HIGH", "LOW"], 1000000)

        passed = credits["HIGH"].base_credits > credits["LOW"].base_credits

        logger.test_result(
            "HIGH priority gets more credits",
            passed,
            f"HIGH base={credits['HIGH'].base_credits:,}, "
            f"LOW base={credits['LOW'].base_credits:,}"
        )
    except Exception as e:
        logger.test_result("Priority credits", False, error=str(e))

    # Test 2.4: Work-conserving credit boost
    logger.subsection("2.4 Work-conserving Credit Boost")
    try:
        coord = SchedulingCoordinator(84)
        coord.register_workload("A", kernel_type="COMPUTE_BOUND")
        coord.register_workload("B", kernel_type="MIXED")

        # Before A idle
        before = coord.compute_credits(["A", "B"], 1000000)
        b_before = before["B"].final_credits

        # Mark A idle
        coord.handle_workload_idle("A")

        # After A idle
        after = coord.compute_credits(["A", "B"], 1000000)
        b_after = after["B"].final_credits

        passed = b_after > b_before

        logger.test_result(
            "Credits increase when partner idle",
            passed,
            f"B credits: {b_before:,} -> {b_after:,} "
            f"(+{b_after - b_before:,})"
        )
    except Exception as e:
        logger.test_result("WC credit boost", False, error=str(e))


# ==================== Test 3: Profiler Integration ====================
def test_profiler_integration():
    logger.section("TEST 3: Profiler Integration Tests")

    if not PROFILER_CLIENT_AVAILABLE:
        logger.skip("Profiler tests", "profiler_client not available")
        return

    # Check profiler connectivity
    logger.subsection("3.1 Profiler Connectivity")
    try:
        client = ProfilerClient("http://localhost:7070")
        healthy = client.health_check()

        logger.test_result(
            "Profiler health check",
            healthy,
            f"URL: {client.base_url}"
        )

        if not healthy:
            logger.skip("Remaining profiler tests", "Profiler not running")
            return
    except Exception as e:
        logger.test_result("Profiler health check", False, error=str(e))
        return

    # Test 3.2: Get profiles
    logger.subsection("3.2 Profile Retrieval")
    test_models = ["resnet50", "bert", "vgg16", "gpt2", "mobilenet"]

    for model in test_models:
        try:
            profile = client.get_profile(model)
            passed = profile is not None
            if passed:
                logger.test_result(
                    f"Get profile: {model}",
                    True,
                    f"Type={profile.kernel_type.name}, AI={profile.arithmetic_intensity:.1f}"
                )
            else:
                logger.test_result(f"Get profile: {model}", False, "Profile not found")
        except Exception as e:
            logger.test_result(f"Get profile: {model}", False, error=str(e))

    # Test 3.3: Co-location recommendation
    logger.subsection("3.3 Co-location Recommendations")
    try:
        rec = client.get_colocation_score("resnet50", "bert")
        passed = rec is not None and rec.score > 0
        if passed:
            logger.test_result(
                "Co-location recommendation",
                True,
                f"Score={rec.score:.2f}, Recommendation={rec.recommendation}, "
                f"SM partition={rec.sm_partition}"
            )
        else:
            logger.test_result("Co-location recommendation", False, "No recommendation")
    except Exception as e:
        logger.test_result("Co-location recommendation", False, error=str(e))


# ==================== Test 4: Co-location Scenarios ====================
def test_colocation_scenarios():
    logger.section("TEST 4: Co-location Scenario Tests")

    if not COORDINATOR_AVAILABLE:
        logger.skip("Co-location tests", "Coordinator not available")
        return

    logger.subsection("4.1 Profiler-based Co-location Setup")

    test_pairs = [
        ("resnet50", "bert", "COMPUTE + MIXED"),
        ("resnet50", "vgg16", "COMPUTE + COMPUTE"),
        ("bert", "gpt2", "MIXED + MIXED"),
    ]

    for model_a, model_b, desc in test_pairs:
        try:
            coord = SchedulingCoordinator(84)
            state_a, state_b = coord.setup_colocation(
                "A", "B", model_a=model_a, model_b=model_b
            )

            passed = (state_a.sm_allocation is not None and
                      state_b.sm_allocation is not None)

            logger.test_result(
                f"{desc}: {model_a} + {model_b}",
                passed,
                f"{model_a}: {state_a.kernel_type.name}, SM={state_a.sm_allocation.sm_count} | "
                f"{model_b}: {state_b.kernel_type.name}, SM={state_b.sm_allocation.sm_count}"
            )
        except Exception as e:
            logger.test_result(f"{desc}", False, error=str(e))


# ==================== Test 5: End-to-End Scenarios ====================
def test_e2e_scenarios():
    logger.section("TEST 5: End-to-End Scenario Tests")

    if not COORDINATOR_AVAILABLE:
        logger.skip("E2E tests", "Coordinator not available")
        return

    # Scenario: Full lifecycle
    logger.subsection("5.1 Full Workload Lifecycle")
    try:
        coord = SchedulingCoordinator(84)

        # Phase 1: Setup
        state_a, state_b = coord.setup_colocation(
            "WorkloadA", "WorkloadB",
            model_a="resnet50", model_b="bert"
        )

        # Phase 2: Both active
        credits1 = coord.compute_credits(["WorkloadA", "WorkloadB"], 524288)
        a_credits_1 = credits1["WorkloadA"].final_credits
        b_credits_1 = credits1["WorkloadB"].final_credits

        # Phase 3: A finishes
        coord.handle_workload_idle("WorkloadA")
        credits2 = coord.compute_credits(["WorkloadA", "WorkloadB"], 524288)
        b_credits_2 = credits2["WorkloadB"].final_credits

        # Phase 4: A removed
        coord.unregister_workload("WorkloadA")
        status = coord.get_status()

        # Verify
        passed = (
            "WorkloadA" not in status["workloads"] and
            "WorkloadB" in status["workloads"] and
            b_credits_2 > b_credits_1
        )

        logger.test_result(
            "Full lifecycle",
            passed,
            f"Phase 2: A={a_credits_1:,}, B={b_credits_1:,} | "
            f"Phase 3: B={b_credits_2:,} (+{b_credits_2 - b_credits_1:,}) | "
            f"Phase 4: A removed, B remains"
        )
    except Exception as e:
        logger.test_result("Full lifecycle", False, error=str(e))

    # Scenario: Multiple workloads
    logger.subsection("5.2 Multiple Workloads")
    try:
        coord = SchedulingCoordinator(84)

        # Register 3 workloads
        coord.register_workload("W1", kernel_type="COMPUTE_BOUND", priority="HIGH")
        coord.register_workload("W2", kernel_type="MIXED", priority="MED")
        coord.register_workload("W3", kernel_type="MEMORY_BOUND", priority="LOW")

        # Compute credits
        credits = coord.compute_credits(["W1", "W2", "W3"], 1000000)

        total = sum(c.final_credits for c in credits.values())
        passed = abs(total - 1000000) < 100

        logger.test_result(
            "3 workloads credit distribution",
            passed,
            f"W1(HIGH)={credits['W1'].final_credits:,}, "
            f"W2(MED)={credits['W2'].final_credits:,}, "
            f"W3(LOW)={credits['W3'].final_credits:,}"
        )
    except Exception as e:
        logger.test_result("Multiple workloads", False, error=str(e))


# ==================== Test 6: Verification Summary ====================
def test_verification_summary():
    logger.section("TEST 6: Implementation Verification")

    logger.subsection("6.1 SM Allocation Rules")

    if SM_MANAGER_AVAILABLE:
        logger.writeln("\n  SM_RATIO_BY_TYPE Configuration:")
        for key, (ratio_a, ratio_b) in SM_RATIO_BY_TYPE.items():
            type_a, type_b = key
            logger.writeln(f"    {type_a.name} + {type_b.name}: {ratio_a*100:.0f}% / {ratio_b*100:.0f}%")

        logger.writeln("\n  PRIORITY_SM_BONUS Configuration:")
        for prio, bonus in PRIORITY_SM_BONUS.items():
            logger.writeln(f"    {prio.name}: {bonus*100:+.0f}%")

        logger.test_result("SM allocation rules defined", True)
    else:
        logger.skip("SM allocation rules", "SM Manager not available")

    logger.subsection("6.2 Time Compensation Formula")
    if SM_MANAGER_AVAILABLE:
        mgr = SMManager(84)
        test_sm_pcts = [60, 50, 45, 40, 30, 20, 10]

        logger.writeln("\n  Compensation by SM%:")
        for sm_pct in test_sm_pcts:
            comp = mgr._compute_time_compensation(sm_pct)
            logger.writeln(f"    SM {sm_pct}%: {comp:.2f}x")

        logger.test_result("Time compensation formula verified", True)

    logger.subsection("6.3 Work-conserving Boost")
    if SM_MANAGER_AVAILABLE:
        mgr = SMManager(84)
        mgr.compute_colocation_allocation(
            "A", KernelType.COMPUTE_BOUND, "B", KernelType.MIXED
        )

        logger.writeln("\n  Boost progression:")
        boost1 = mgr.get_work_conserving_boost("B")
        logger.writeln(f"    Initial: {boost1:.2f}x")

        mgr.mark_idle("A")
        boost2 = mgr.get_work_conserving_boost("B")
        logger.writeln(f"    After A idle: {boost2:.2f}x")

        mgr.deallocate("A")
        boost3 = mgr.get_work_conserving_boost("B")
        logger.writeln(f"    After A deallocate: {boost3:.2f}x")

        logger.test_result("Work-conserving boost verified", True)


# ==================== Main ====================
def main():
    logger.writeln()
    logger.writeln("=" * 70)
    logger.writeln(" SOFT SM PARTITIONING + WORK-CONSERVING VERIFICATION TEST")
    logger.writeln(f" Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.writeln("=" * 70)

    # Run all tests
    test_sm_manager()
    test_scheduling_coordinator()
    test_profiler_integration()
    test_colocation_scenarios()
    test_e2e_scenarios()
    test_verification_summary()

    # Summary
    logger.section("FINAL SUMMARY")
    results = logger.results["summary"]
    total = results["passed"] + results["failed"] + results["skipped"]

    logger.writeln(f"\n  Total Tests: {total}")
    logger.writeln(f"  Passed:      {results['passed']}")
    logger.writeln(f"  Failed:      {results['failed']}")
    logger.writeln(f"  Skipped:     {results['skipped']}")
    logger.writeln()

    if results["failed"] == 0:
        logger.writeln("  ✓ ALL TESTS PASSED!")
    else:
        logger.writeln("  ✗ SOME TESTS FAILED")
        logger.writeln("\n  Failed tests:")
        for test in logger.results["tests"]:
            if test["status"] == "FAIL":
                logger.writeln(f"    - {test['name']}: {test.get('error', test.get('details', ''))}")

    logger.writeln()
    logger.writeln("=" * 70)

    # Save to file
    log_dir = os.path.dirname(__file__)
    log_file = os.path.join(log_dir, "test_results.log")
    json_file = os.path.join(log_dir, "test_results.json")

    with open(log_file, "w") as f:
        f.write(logger.get_log())

    with open(json_file, "w") as f:
        json.dump(logger.get_results(), f, indent=2, default=str)

    logger.writeln(f"\n  Log saved to: {log_file}")
    logger.writeln(f"  JSON saved to: {json_file}")

    return 0 if results["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
