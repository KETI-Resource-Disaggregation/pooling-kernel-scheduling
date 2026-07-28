// prism_runtime.h
// runtime 내부 공통 헤더
// gating_lib.so 전체에서 공유하는 타입, 전역 상태, 인터페이스

#pragma once

#include <stdint.h>
#include <stdbool.h>
#include <time.h>
#include "../../shm/prism_shm.h"

// ── 환경변수 키 ────────────────────────────────────────────────────────────────
#define ENV_TENANT_ID    "PRISM_TENANT"
#define ENV_GROUP_ID     "PRISM_GROUP"
#define ENV_POLICY_PATH  "PRISM_KILLER_POLICY"

// ── 기본값 ─────────────────────────────────────────────────────────────────────
#define DEFAULT_GROUP_ID         "default"
#define DEFAULT_SPIN_INTERVAL_US 50         // gate 대기 polling 간격 (μs)
#define MAX_KILLER_KERNELS       512        // killer 인덱스/이름 최대 수
#define MAX_KERNEL_NAME_LEN      256

// ── Killer 정책 (프로파일러가 생성, runtime이 시작 시 로드) ────────────────────
typedef struct {
    int     count;
    int     indices[MAX_KILLER_KERNELS];                  // iteration 내 인덱스
    char    names[MAX_KILLER_KERNELS][MAX_KERNEL_NAME_LEN]; // 커널 이름
    int     kernels_per_iter;     // iteration당 총 커널 수 (인덱스 계산용)
    float   top_pct;
    double  avg_killer_us;
    char    workload_type[32];    // COMPUTE_BOUND / MEMORY_BOUND / MIXED
    bool    loaded;               // policy 로드 완료 여부
} KillerPolicy;

// ── 런타임 전역 상태 (프로세스 하나에 하나) ────────────────────────────────────
typedef struct {
    // shm
    PrismSharedState* shm;
    int               tenant_idx;       // 이 프로세스의 테넌트 인덱스
    char              tenant_id[16];
    char              group_id[64];

    // killer policy
    KillerPolicy      policy;
    char              policy_path[256]; // ENV_POLICY_PATH 저장 (hot-reload용)
    uint64_t          loaded_killer_version; // 마지막 로드 시 shm.killer_policy_version

    // kernel 카운터 (iteration 내 인덱스 추적)
    int               kernel_idx;       // 현재 iteration 내 커널 순번
    int               kernels_per_iter; // policy로부터 로드

    // 타이밍
    struct timespec   killer_start;     // killer ENTER 시점

    // 메모리 추적
    int64_t           mem_used_bytes;

    // 초기화 완료 여부
    bool              initialized;
} PrismRuntime;

// ── 전역 인스턴스 (cuda_hooks.cpp에서 정의) ────────────────────────────────────
extern PrismRuntime g_prism;

// ── 인터페이스 선언 ────────────────────────────────────────────────────────────

// shm_client.h
void     prism_shm_open(const char* group_id);
void     prism_shm_close(void);

// gating/stream_gate.h
void     gate_killer_enter(void);
void     gate_killer_exit(void);

// timeslice/time_slice.h
void     timeslice_deduct(int64_t elapsed_us);
void     timeslice_add_wait(int64_t wait_us);
bool     timeslice_exhausted(void);
void     timeslice_set_waiting(void);

// timeslice/round_manager.h
void     round_check_and_trigger(void);

// cuda_hooks.cpp — killer policy hot-reload
void     reload_killer_policy_if_needed(void);

// resource/virtual_sm.h
bool     resource_is_overcommit(void);

// resource/virtual_mem.h
bool     vmem_check_alloc(size_t bytes);
void     vmem_track_alloc(void* ptr, size_t bytes);
void     vmem_track_free(void* ptr);

// ── 인라인 유틸 ───────────────────────────────────────────────────────────────
static inline int64_t timespec_diff_us(struct timespec start, struct timespec end) {
    return (int64_t)(end.tv_sec  - start.tv_sec)  * 1000000LL
         + (int64_t)(end.tv_nsec - start.tv_nsec) / 1000LL;
}
