// virtual_mem.cpp
// 테넌트별 가상 메모리 quota 추적 (DD-03)
//
// 동작:
//   cudaMalloc/cuMemAlloc 후킹 → vmem_track_alloc()으로 사용량 누적
//   cudaFree/cuMemFree 후킹    → vmem_track_free()로 해제
//   vmem_check_alloc()         → quota 초과 시 false 반환 → 할당 거부
//
// quota 출처: shm->alloc[tenant_idx].virtual_mem_mb (controller가 설정)
// 실제 사용량: g_prism.mem_used_bytes (프로세스 로컬 카운터)
//   + shm->tenants[tenant_idx].mem_used_bytes (monitor가 읽는 stats)

#include <stddef.h>
#include <stdint.h>
#include "../include/prism_runtime.h"

// 단순 연결 리스트로 ptr → size 매핑 관리
// 할당 빈도가 낮으므로 lock-free 대신 단순 배열 사용
#define MAX_ALLOC_ENTRIES 4096

static struct {
    void*    ptr;
    int64_t  size;
} alloc_table[MAX_ALLOC_ENTRIES];

static int alloc_table_size = 0;

// quota 한도(bytes) 계산
static int64_t quota_bytes(void) {
    PrismSharedState* shm = g_prism.shm;
    if (!shm) return INT64_MAX;
    int idx = g_prism.tenant_idx;
    int64_t mb = shm->alloc[idx].virtual_mem_mb;
    return (mb > 0) ? mb * 1024LL * 1024LL : INT64_MAX;
}

// 할당 전 quota 확인
bool vmem_check_alloc(size_t bytes) {
    int64_t limit = quota_bytes();
    if (limit == INT64_MAX) return true;   // quota 미설정

    int64_t used = __atomic_load_n(&g_prism.mem_used_bytes, __ATOMIC_RELAXED);
    return (used + (int64_t)bytes) <= limit;
}

// 할당 추적 등록
void vmem_track_alloc(void* ptr, size_t bytes) {
    if (!ptr || bytes == 0) return;

    // 테이블에 기록 (선형 탐색, 슬롯 재사용)
    for (int i = 0; i < alloc_table_size; i++) {
        if (alloc_table[i].ptr == nullptr) {
            alloc_table[i].ptr  = ptr;
            alloc_table[i].size = (int64_t)bytes;
            goto update_count;
        }
    }
    if (alloc_table_size < MAX_ALLOC_ENTRIES) {
        alloc_table[alloc_table_size].ptr  = ptr;
        alloc_table[alloc_table_size].size = (int64_t)bytes;
        alloc_table_size++;
    }

update_count:
    __atomic_fetch_add(&g_prism.mem_used_bytes, (int64_t)bytes, __ATOMIC_RELAXED);

    // shm stats 업데이트 (monitor가 읽음)
    PrismSharedState* shm = g_prism.shm;
    if (shm) {
        __atomic_fetch_add(&shm->tenants[g_prism.tenant_idx].mem_used_bytes,
                           (int64_t)bytes, __ATOMIC_RELAXED);
    }
}

// 할당 해제 추적
void vmem_track_free(void* ptr) {
    if (!ptr) return;

    for (int i = 0; i < alloc_table_size; i++) {
        if (alloc_table[i].ptr == ptr) {
            int64_t sz = alloc_table[i].size;
            alloc_table[i].ptr  = nullptr;
            alloc_table[i].size = 0;

            __atomic_fetch_sub(&g_prism.mem_used_bytes, sz, __ATOMIC_RELAXED);

            PrismSharedState* shm = g_prism.shm;
            if (shm) {
                __atomic_fetch_sub(&shm->tenants[g_prism.tenant_idx].mem_used_bytes,
                                   sz, __ATOMIC_RELAXED);
            }
            return;
        }
    }
}
