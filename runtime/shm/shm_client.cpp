// shm_client.cpp
// POSIX 공유 메모리 연결/해제 + 테넌트 등록
//
// 흐름:
//   prism_shm_open(group_id):
//     1. shm_open("/prism_{group_id}") — controller가 미리 생성해 둔 shm
//     2. ftruncate는 controller 담당 (runtime은 읽기/쓰기만)
//     3. mmap → g_prism.shm
//     4. magic 확인
//     5. 빈 슬롯 탐색 → g_prism.tenant_idx 등록
//     6. tenants[idx].active = 1
//
//   prism_shm_close():
//     1. tenants[idx].active = 0 (cuda_hooks prism_fini에서 이미 처리)
//     2. munmap

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../include/prism_runtime.h"

// Injected by K8s Device Plugin Allocate().
// When present and the controller has not already written the alloc slot
// (virtual_sm == 0), the runtime fills in the values so that the overcommit
// manager can compute mode correctly even before the controller's next poll.
#define ENV_VIRTUAL_SM     "PRISM_VIRTUAL_SM"
#define ENV_VIRTUAL_MEM_MB "PRISM_VIRTUAL_MEM_MB"

void prism_shm_open(const char* group_id) {
    char path[128];
    snprintf(path, sizeof(path), SHM_PATH_FMT, group_id);

    int fd = shm_open(path, O_RDWR, 0666);
    if (fd < 0) {
        fprintf(stderr, "[prism] shm_open 실패: %s\n", path);
        return;
    }

    void* addr = mmap(nullptr, sizeof(PrismSharedState),
                      PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);

    if (addr == MAP_FAILED) {
        fprintf(stderr, "[prism] mmap 실패\n");
        return;
    }

    PrismSharedState* shm = (PrismSharedState*)addr;

    // magic 확인
    if (shm->magic != SHM_MAGIC) {
        fprintf(stderr, "[prism] shm magic 불일치 (0x%08X)\n", shm->magic);
        munmap(addr, sizeof(PrismSharedState));
        return;
    }

    // 내 tenant_id로 슬롯 탐색 (재연결) → 없으면 빈 슬롯 할당
    int idx = -1;
    for (int i = 0; i < MAX_TENANTS; i++) {
        if (strcmp(shm->alloc[i].tenant_id, g_prism.tenant_id) == 0) {
            idx = i;
            break;
        }
    }
    if (idx < 0) {
        // 빈 슬롯: active == 0 && tenant_id 미할당
        for (int i = 0; i < MAX_TENANTS; i++) {
            if (!__atomic_load_n(&shm->tenants[i].active, __ATOMIC_ACQUIRE)
                && shm->alloc[i].tenant_id[0] == '\0') {
                idx = i;
                break;
            }
        }
    }
    if (idx < 0) {
        fprintf(stderr, "[prism] 슬롯 부족 (MAX_TENANTS=%d)\n", MAX_TENANTS);
        munmap(addr, sizeof(PrismSharedState));
        return;
    }

    g_prism.shm        = shm;
    g_prism.tenant_idx = idx;

    // tenant_id 기록 (controller가 미리 기록했을 수도 있음)
    strncpy(shm->alloc[idx].tenant_id, g_prism.tenant_id,
            sizeof(shm->alloc[idx].tenant_id) - 1);

    // Device Plugin이 주입한 env에서 virtual_sm / virtual_mem_mb를 읽어
    // 슬롯을 보완한다.  controller가 /register로 이미 채운 경우(virtual_sm > 0)
    // 에는 덮어쓰지 않는다.
    if (shm->alloc[idx].virtual_sm == 0) {
        const char* env_sm  = getenv(ENV_VIRTUAL_SM);
        const char* env_mem = getenv(ENV_VIRTUAL_MEM_MB);
        if (env_sm && env_mem) {
            int32_t vsm  = (int32_t)atoi(env_sm);
            int32_t vmem = (int32_t)atoi(env_mem);
            if (vsm > 0 && vmem > 0) {
                shm->alloc[idx].virtual_sm     = vsm;
                shm->alloc[idx].virtual_mem_mb = vmem;
                // mps_pct: controller가 없을 때도 올바른 값 유지
                int32_t phys = shm->physical_sm_total;
                shm->alloc[idx].mps_pct = (phys > 0)
                    ? (int32_t)(vsm * 100 / phys)
                    : 100;
                // virtual totals도 갱신
                __atomic_fetch_add(&shm->virtual_sm_total,     vsm,  __ATOMIC_RELAXED);
                __atomic_fetch_add(&shm->virtual_mem_total_mb, vmem, __ATOMIC_RELAXED);
                fprintf(stderr,
                    "[prism] env에서 할당: sm=%d mem=%dMB mps=%d%%\n",
                    vsm, vmem, shm->alloc[idx].mps_pct);
            }
        }
    }

    // active 등록
    __atomic_store_n(&shm->tenants[idx].active, 1, __ATOMIC_RELEASE);

    fprintf(stderr, "[prism] shm 연결: %s slot=%d sm=%d mem=%dMB\n",
        path, idx,
        shm->alloc[idx].virtual_sm,
        shm->alloc[idx].virtual_mem_mb);
}

void prism_shm_close(void) {
    if (!g_prism.shm) return;
    munmap(g_prism.shm, sizeof(PrismSharedState));
    g_prism.shm = nullptr;
}
