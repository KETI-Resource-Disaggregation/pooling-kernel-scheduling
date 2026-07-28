// round_manager.cpp
// 라운드 생애주기 관리 (DD-03)
// 모든 active 테넌트가 WAITING → 마지막 테넌트가 새 라운드 트리거
// 중앙 스케줄러의 policy.version 변경도 이 시점에 반영

#include <stdio.h>
#include "../include/prism_runtime.h"

// 외부 선언 (time_slice.cpp)
extern void timeslice_recharge(int tenant_idx, int64_t slice_us);

// 모든 active 테넌트가 WAITING인지 확인
static bool all_waiting(PrismSharedState* shm) {
    for (int i = 0; i < shm->tenant_count; i++) {
        if (!__atomic_load_n(&shm->tenants[i].active, __ATOMIC_RELAXED))
            continue;
        int32_t state = __atomic_load_n(
            (int32_t*)&shm->tenants[i].gate_state, __ATOMIC_ACQUIRE);
        if (state != GATE_WAITING) return false;
    }
    return true;
}

// slice_us 계산: round_duration × weight[i] / total_weight
static int64_t calc_slice_us(PrismSharedState* shm, int tenant_idx) {
    float total_w = 0.0f;
    for (int i = 0; i < shm->tenant_count; i++) {
        if (__atomic_load_n(&shm->tenants[i].active, __ATOMIC_RELAXED))
            total_w += shm->policy.weights[i];
    }
    if (total_w <= 0.0f) total_w = 1.0f;

    float w = shm->policy.weights[tenant_idx];
    int64_t rd = shm->policy.round_duration_us > 0
                 ? shm->policy.round_duration_us
                 : 100000LL;   // 기본 100ms

    return (int64_t)(rd * w / total_w);
}

// 새 라운드 시작 (마지막 WAITING 테넌트가 CAS 획득 후 호출)
static void start_new_round(PrismSharedState* shm) {
    int my = g_prism.tenant_idx;

    // CAS: round_trigger_lock 0→1 획득 (TOCTOU 방지)
    // 여러 테넌트가 동시에 all_waiting()=true를 봐도 한 테넌트만 진입
    int32_t expected = 0;
    if (!__atomic_compare_exchange_n(
            &shm->round_trigger_lock, &expected, 1,
            false, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) {
        return;  // 다른 테넌트가 이미 라운드를 트리거 중
    }

    // killer policy hot-reload 확인 (version 변경 시 JSON 재로드)
    reload_killer_policy_if_needed();

    // scheduler policy weights 등은 shm 직접 읽으므로 자동 반영

    for (int i = 0; i < shm->tenant_count; i++) {
        if (!__atomic_load_n(&shm->tenants[i].active, __ATOMIC_RELAXED))
            continue;
        int64_t slice = calc_slice_us(shm, i);
        timeslice_recharge(i, slice);   // gate_state → GATE_RUNNING
    }

    // round_id 증가 (내 round_id 기준)
    int32_t rid = __atomic_load_n(
        (int32_t*)&shm->tenants[my].round_id, __ATOMIC_RELAXED);
    for (int i = 0; i < shm->tenant_count; i++) {
        __atomic_store_n((int32_t*)&shm->tenants[i].round_id,
                         rid + 1, __ATOMIC_RELEASE);
    }

    // lock 해제
    __atomic_store_n(&shm->round_trigger_lock, 0, __ATOMIC_RELEASE);
}

// gate_killer_enter에서 WAITING 전환 직후 호출
// 내가 마지막으로 WAITING이 된 테넌트라면 새 라운드 트리거
void round_check_and_trigger(void) {
    PrismSharedState* shm = g_prism.shm;
    if (!shm) return;

    if (all_waiting(shm)) {
        start_new_round(shm);
    }
}
