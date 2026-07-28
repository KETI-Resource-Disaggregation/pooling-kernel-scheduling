// virtual_sm.cpp
// Virtual SM 오버커밋 감지 및 모드 전환 (DD-03)
//
// 동작:
//   controller가 각 테넌트의 virtual_sm을 alloc[i].virtual_sm에 기록
//   shm->virtual_sm_total = Σ active 테넌트의 virtual_sm
//   virtual_sm_total > physical_sm_total → MODE_OVERCOMMIT
//   그 이하 → MODE_FREE (게이팅 없음)
//
// resource_is_overcommit() : 런타임이 gate_killer_enter에서 모드 확인용으로도 쓰지만
//   실제 mode 필드는 controller가 관리하므로 shm->mode를 직접 읽는 것이 정석.
//   이 함수는 런타임 내부 진단·테스트 용도.

#include "../include/prism_runtime.h"

// 현재 shm의 virtual_sm_total > physical_sm_total 여부
bool resource_is_overcommit(void) {
    PrismSharedState* shm = g_prism.shm;
    if (!shm) return false;

    // virtual_sm_total은 controller 단독 write (non-atomic) → acquire fence 후 plain read
    __atomic_thread_fence(__ATOMIC_ACQUIRE);
    int32_t virt = shm->virtual_sm_total;
    int32_t phys = shm->physical_sm_total;   // 초기화 후 불변
    return virt > phys;
}
