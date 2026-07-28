// stream_gate.cpp
// killer op 진입/탈출 시 CPU-side 게이팅 로직
//
// 설계 근거 (DD-12):
//   - CUDA Event 방식(Exp16)은 단일 프로세스 전용
//   - 멀티 프로세스: shm atomic + CPU spin으로 제출 전 대기
//   - gate_killer_enter(): 커널 제출 전 조건 검사 및 대기
//   - gate_killer_exit():  time_remain 차감, gate_state 해제

#include <time.h>
#include <unistd.h>
#include "../include/prism_runtime.h"

void gate_killer_enter(void) {
    PrismSharedState* shm = g_prism.shm;
    if (!shm) return;

    int my = g_prism.tenant_idx;
    TenantState* me = &shm->tenants[my];

    // PROFILING 모드: 내가 프로파일 대상이 아니면 완료될 때까지 대기
    while (__atomic_load_n(&shm->mode, __ATOMIC_ACQUIRE) == MODE_PROFILING) {
        if (__atomic_load_n(&shm->profiling_tenant_idx, __ATOMIC_RELAXED) == my)
            break;
        usleep(DEFAULT_SPIN_INTERVAL_US);
    }

    // FREE 모드: 게이팅 없음
    if (__atomic_load_n(&shm->mode, __ATOMIC_ACQUIRE) == MODE_FREE)
        goto mark_active;

    // OVERCOMMIT 모드
    // 1. 내 time slice 소진 여부
    if (timeslice_exhausted()) {
        timeslice_set_waiting();
        round_check_and_trigger();

        // RUNNING이 될 때까지 대기 → 대기 시간 누적
        // spin-wait 중에도 주기적으로 재확인:
        // peer가 먼저 워크로드를 완료하고 active=0이 된 경우
        // round_check_and_trigger를 재호출해서 all_waiting을 재평가.
        struct timespec wait_s1;
        clock_gettime(CLOCK_MONOTONIC, &wait_s1);
        while (__atomic_load_n((int32_t*)&me->gate_state, __ATOMIC_ACQUIRE)
               == GATE_WAITING) {
            usleep(DEFAULT_SPIN_INTERVAL_US);
            round_check_and_trigger();
        }
        struct timespec wait_e1;
        clock_gettime(CLOCK_MONOTONIC, &wait_e1);
        timeslice_add_wait(timespec_diff_us(wait_s1, wait_e1));
    }

    // 2. peer가 killer 실행 중이면 대기 → 대기 시간 누적
    for (int i = 0; i < shm->tenant_count; i++) {
        if (i == my) continue;
        if (!__atomic_load_n(&shm->tenants[i].active, __ATOMIC_RELAXED)) continue;
        if (__atomic_load_n((int32_t*)&shm->tenants[i].gate_state, __ATOMIC_ACQUIRE)
                != GATE_KILLER_ACTIVE) continue;

        struct timespec wait_s2;
        clock_gettime(CLOCK_MONOTONIC, &wait_s2);
        while (__atomic_load_n((int32_t*)&shm->tenants[i].gate_state, __ATOMIC_ACQUIRE)
               == GATE_KILLER_ACTIVE) {
            usleep(DEFAULT_SPIN_INTERVAL_US);
        }
        struct timespec wait_e2;
        clock_gettime(CLOCK_MONOTONIC, &wait_e2);
        timeslice_add_wait(timespec_diff_us(wait_s2, wait_e2));
    }

mark_active:
    __atomic_store_n((int32_t*)&me->gate_state, (int32_t)GATE_KILLER_ACTIVE,
                     __ATOMIC_RELEASE);
    clock_gettime(CLOCK_MONOTONIC, &g_prism.killer_start);
}

void gate_killer_exit(void) {
    PrismSharedState* shm = g_prism.shm;
    if (!shm) return;

    int my = g_prism.tenant_idx;
    TenantState* me = &shm->tenants[my];

    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    int64_t elapsed_us = timespec_diff_us(g_prism.killer_start, now);

    // time_remain 차감 + stats
    // NOTE: timeslice_deduct() 내부에서 total_exec_us를 누산하므로 여기서는 하지 않음
    timeslice_deduct(elapsed_us);

    __atomic_fetch_add(&me->killer_count, 1, __ATOMIC_RELAXED);

    // gate 해제
    __atomic_store_n((int32_t*)&me->gate_state, (int32_t)GATE_RUNNING,
                     __ATOMIC_RELEASE);
}
