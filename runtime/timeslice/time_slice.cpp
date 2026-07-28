// time_slice.cpp
// 자율 시간 슬라이스 관리 (DD-03)
// 각 테넌트가 shm의 time_remain_us를 killer EXIT 시 자율 차감
// 중앙 스케줄러는 policy(weights, round_duration)만 write

#include "../include/prism_runtime.h"

// time_remain에서 elapsed_us 차감 + exec_us 누적
void timeslice_deduct(int64_t elapsed_us) {
    PrismSharedState* shm = g_prism.shm;
    if (!shm) return;

    TenantState* me = &shm->tenants[g_prism.tenant_idx];
    __atomic_fetch_sub(&me->time_remain_us, elapsed_us, __ATOMIC_RELAXED);
    __atomic_fetch_add(&me->total_exec_us,  elapsed_us, __ATOMIC_RELAXED);
}

// 대기 시간 누적 (gate spin 대기: WAITING spin + peer KILLER_ACTIVE spin)
void timeslice_add_wait(int64_t wait_us) {
    PrismSharedState* shm = g_prism.shm;
    if (!shm) return;

    __atomic_fetch_add(&shm->tenants[g_prism.tenant_idx].total_wait_us,
                       wait_us, __ATOMIC_RELAXED);
}

// 슬라이스 소진 여부
bool timeslice_exhausted(void) {
    PrismSharedState* shm = g_prism.shm;
    if (!shm) return false;

    int64_t remain = __atomic_load_n(
        &shm->tenants[g_prism.tenant_idx].time_remain_us,
        __ATOMIC_RELAXED);
    return remain <= 0;
}

// WAITING 상태 전환 + wait_us 누적 시작 기록
void timeslice_set_waiting(void) {
    PrismSharedState* shm = g_prism.shm;
    if (!shm) return;

    TenantState* me = &shm->tenants[g_prism.tenant_idx];
    __atomic_store_n((int32_t*)&me->gate_state, (int32_t)GATE_WAITING,
                     __ATOMIC_RELEASE);
}

// 새 라운드 시작 시 slice_us 재충전 (round_manager가 호출)
void timeslice_recharge(int tenant_idx, int64_t slice_us) {
    PrismSharedState* shm = g_prism.shm;
    if (!shm) return;

    __atomic_store_n(&shm->tenants[tenant_idx].time_remain_us,
                     slice_us, __ATOMIC_RELEASE);
    __atomic_store_n((int32_t*)&shm->tenants[tenant_idx].gate_state,
                     (int32_t)GATE_RUNNING, __ATOMIC_RELEASE);
}
