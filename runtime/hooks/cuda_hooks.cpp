// cuda_hooks.cpp
// LD_PRELOAD 진입점 — cuLaunchKernel (Driver API) 및 메모리 API 후킹
//
// 설계 근거 (DD-11, DD-12, DD-21):
//   - cuLaunchKernel (Driver API) 후킹 → cuDNN/cuBLAS 포함 캡처율 100%
//   - hook 내 CPU blocking = 커널 제출 전 대기 → 실제 GPU 실행 제어
//   - CUDA 11.3+ cuFuncGetName()으로 커널 이름 획득 (CUPTI 불필요)
//   - 멀티프로세스 조율: shm atomic + CPU-side spin (CUDA Event 불필요)
//   - cuBLAS 우회 해결: dlsym 후킹 (DD-21) → dlopen+dlsym 경로도 인터셉트

#include <dlfcn.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <ctype.h>

// hot-reload: profiler가 killer policy를 교체할 때 사용하는 well-known 경로
// 형식: /tmp/prism_killers_{group_id}_{tenant_idx}.json
#define HOT_POLICY_PATH_FMT "/tmp/prism_killers_%s_%d.json"

#include "../include/prism_runtime.h"

// ── forward 선언 (dlsym/cuGetProcAddress hook이 주소를 참조하므로 필요) ─────────
extern "C" CUresult cuLaunchKernel(
    CUfunction, unsigned, unsigned, unsigned,
    unsigned, unsigned, unsigned,
    unsigned, CUstream, void**, void**);
extern "C" CUresult cuLaunchKernelEx(
    const CUlaunchConfig*, CUfunction, void**, void**);
extern "C" CUresult cuGetProcAddress_v2(
    const char*, void**, int, cuuint64_t, CUdriverProcAddressQueryResult*);

// ── dlsym + cuGetProcAddress 후킹 — cuBLAS/cuDNN 우회 차단 (DD-21) ────────────
//
// cuBLAS/cublasLt의 실제 경로:
//   1) dlopen("libcuda.so.1") + dlsym(handle, "cuGetProcAddress_v2")
//   2) cuGetProcAddress_v2("cuLaunchKernel", &fn, ...) → 실제 함수 포인터 획득
//   3) fn(args) 직접 호출 → LD_PRELOAD PLT 인터포저 완전 우회
//
// 해결 (두 단계):
//   A) dlsym hook: cuGetProcAddress_v2 요청 → 우리 wrapper 반환
//   B) cuGetProcAddress_v2 hook: cuLaunchKernel/Ex 요청 → 우리 hook으로 교체
//
// RTLD_NEXT 요청은 통과 (우리 내부 LAZY_RESOLVE에서 real 함수 획득에 사용)

static void* (*real_dlsym)(void*, const char*) = nullptr;

extern "C"
void* dlsym(void* handle, const char* symbol) {
    if (!real_dlsym) return nullptr;

    // RTLD_NEXT/DEFAULT: 내부 조회 또는 전역 심볼 테이블 → 통과
    if (handle != RTLD_NEXT && handle != RTLD_DEFAULT) {
        if (strcmp(symbol, "cuLaunchKernel") == 0)
            return (void*)cuLaunchKernel;
        if (strcmp(symbol, "cuLaunchKernelEx") == 0)
            return (void*)cuLaunchKernelEx;
        // cuGetProcAddress 경로도 차단 (cublasLt가 이 경로로 cuLaunchKernel 획득)
        if (strcmp(symbol, "cuGetProcAddress_v2") == 0 ||
            strcmp(symbol, "cuGetProcAddress") == 0)
            return (void*)cuGetProcAddress_v2;
    }

    return real_dlsym(handle, symbol);
}

// ── 전역 런타임 인스턴스 ───────────────────────────────────────────────────────
PrismRuntime g_prism = {};

// ── cudaLaunchKernel 함수 포인터 (Runtime API) ────────────────────────────────
// <<<>>> 구문으로 컴파일된 CUDA 커널은 cudaLaunchKernel을 통해 제출됨.
// 이 경로는 PLT 기반이므로 LD_PRELOAD로 정상 후킹 가능.
// (cuBLAS/cuDNN의 cuLaunchKernel(Driver API) 경로와 구분)
static cudaError_t (*real_cudaLaunchKernel)(
    const void*, dim3, dim3, void**, size_t, cudaStream_t) = nullptr;

// ── 원본 함수 포인터 (lazy init: 각 hook 내에서 first-call 시 초기화) ─────────
// constructor에서 초기화하면 libcudart 로드 전에 dlsym이 NULL을 반환할 수 있어
// CUDA 초기화를 망가뜨리는 문제가 발생함 (LD_PRELOAD ordering 이슈)
static CUresult (*real_cuLaunchKernel)(
    CUfunction, unsigned, unsigned, unsigned,
    unsigned, unsigned, unsigned,
    unsigned, CUstream, void**, void**) = nullptr;

static CUresult (*real_cuLaunchKernelEx)(
    const CUlaunchConfig*, CUfunction, void**, void**) = nullptr;

static CUresult  (*real_cuStreamSynchronize)(CUstream)    = nullptr;
static cudaError_t (*real_cudaStreamSynchronize)(cudaStream_t) = nullptr;
static CUresult  (*real_cuMemAlloc)(CUdeviceptr*, size_t) = nullptr;
static CUresult  (*real_cuMemFree)(CUdeviceptr)           = nullptr;
static cudaError_t (*real_cudaMalloc)(void**, size_t)     = nullptr;
static cudaError_t (*real_cudaFree)(void*)                = nullptr;

// lazy 초기화 헬퍼 매크로: 함수 포인터가 NULL이면 real_dlsym(RTLD_NEXT)으로 해결.
// dlsym(RTLD_NEXT)를 쓰면 우리 dlsym hook을 경유하지만 RTLD_NEXT는 통과하므로
// 결과는 동일하다. real_dlsym이 null일 경우 대비해 두 경로 모두 지원.
#define LAZY_RESOLVE(ptr, sym) \
    do { if (!(ptr)) { \
        if (real_dlsym) \
            (ptr) = (decltype(ptr))real_dlsym(RTLD_NEXT, sym); \
        else \
            (ptr) = (decltype(ptr))dlsym(RTLD_NEXT, sym); \
    } } while(0)

// ── killer 판단 ────────────────────────────────────────────────────────────────
static bool is_killer_by_name(const char* name) {
    if (!name) return false;
    KillerPolicy* p = &g_prism.policy;
    for (int i = 0; i < p->count; i++) {
        if (strstr(name, p->names[i])) return true;
    }
    return false;
}

static bool is_killer_by_index(int idx) {
    KillerPolicy* p = &g_prism.policy;
    int iter_idx = (p->kernels_per_iter > 0)
                   ? (idx % p->kernels_per_iter)
                   : idx;
    for (int i = 0; i < p->count; i++) {
        if (p->indices[i] == iter_idx) return true;
    }
    return false;
}

static bool is_killer(CUfunction f) {
    const char* name = nullptr;
#if CUDA_VERSION >= 11030
    cuFuncGetName(&name, f);
#endif
    if (name && is_killer_by_name(name)) return true;
    return is_killer_by_index(g_prism.kernel_idx);
}

// ── 초기화 / 종료 ──────────────────────────────────────────────────────────────
static void load_killer_policy(const char* path);

__attribute__((constructor))
static void prism_init(void) {
    // real_dlsym을 제일 먼저 확보: dlsym hook이 즉시 동작 가능하도록.
    // dlvsym은 versioned symbol lookup으로 우리 dlsym hook을 재귀 호출하지 않음.
    real_dlsym = (void*(*)(void*, const char*))
        dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.2.5");

    // 나머지 함수 포인터는 lazy 초기화 (각 hook 내 first-call 시 해결)
    // constructor에서 dlsym을 수행하면 libcudart 로드 전에 NULL을 얻어
    // hook 통과 시 cudaErrorNoDevice를 반환하는 문제 발생

    const char* tenant_id   = getenv(ENV_TENANT_ID);
    const char* group_id    = getenv(ENV_GROUP_ID);
    const char* policy_path = getenv(ENV_POLICY_PATH);

    if (!tenant_id) {
        fprintf(stderr, "[prism] %s 없음, 게이팅 비활성\n", ENV_TENANT_ID);
        return;
    }

    strncpy(g_prism.tenant_id, tenant_id, sizeof(g_prism.tenant_id) - 1);
    strncpy(g_prism.group_id,
            group_id ? group_id : DEFAULT_GROUP_ID,
            sizeof(g_prism.group_id) - 1);

    prism_shm_open(g_prism.group_id);
    if (!g_prism.shm) {
        fprintf(stderr, "[prism] shm 연결 실패, 게이팅 비활성\n");
        return;
    }

    if (policy_path) {
        strncpy(g_prism.policy_path, policy_path, sizeof(g_prism.policy_path) - 1);
        load_killer_policy(policy_path);
    } else {
        // policy 없음: 기본 패턴으로 동작 (1단계 패턴 매칭)
        fprintf(stderr, "[prism] killer policy 없음, 기본 패턴 매칭 모드\n");
        static const char* defaults[] = {
            "sgemm", "dgemm", "hgemm", "gemm",
            "fprop", "xmma", "cutlass", "fmha",
            "matmul", nullptr
        };
        for (int i = 0; defaults[i] && i < MAX_KILLER_KERNELS; i++) {
            strncpy(g_prism.policy.names[i], defaults[i], MAX_KERNEL_NAME_LEN - 1);
            g_prism.policy.count++;
        }
    }

    // 현재 shm.killer_policy_version을 기록 → 초기화 직후 불필요한 재로드 방지
    if (g_prism.shm) {
        g_prism.loaded_killer_version = __atomic_load_n(
            &g_prism.shm->killer_policy_version, __ATOMIC_ACQUIRE);
    }

    g_prism.initialized = true;
    fprintf(stderr, "[prism] 초기화 완료: tenant=%s group=%s killers=%d\n",
            g_prism.tenant_id, g_prism.group_id, g_prism.policy.count);
}

__attribute__((destructor))
static void prism_fini(void) {
    if (g_prism.shm && g_prism.initialized) {
        g_prism.shm->tenants[g_prism.tenant_idx].active = 0;
        prism_shm_close();
    }
}

// ── cuLaunchKernel 후킹 ────────────────────────────────────────────────────────
extern "C"
CUresult cuLaunchKernel(
    CUfunction f,
    unsigned gridDimX,  unsigned gridDimY,  unsigned gridDimZ,
    unsigned blockDimX, unsigned blockDimY, unsigned blockDimZ,
    unsigned sharedMemBytes, CUstream hStream,
    void** kernelParams, void** extra)
{
    LAZY_RESOLVE(real_cuLaunchKernel, "cuLaunchKernel");
    if (!g_prism.initialized || !real_cuLaunchKernel)
        return real_cuLaunchKernel
               ? real_cuLaunchKernel(f, gridDimX, gridDimY, gridDimZ,
                                     blockDimX, blockDimY, blockDimZ,
                                     sharedMemBytes, hStream, kernelParams, extra)
               : CUDA_ERROR_NOT_INITIALIZED;

    bool killer = is_killer(f);
    if (killer) gate_killer_enter();   // 제출 전 CPU blocking

    CUresult r = real_cuLaunchKernel(
        f, gridDimX, gridDimY, gridDimZ,
        blockDimX, blockDimY, blockDimZ,
        sharedMemBytes, hStream, kernelParams, extra);

    if (killer) {
        // 해당 killer가 실행된 스트림만 동기화.
        // cuCtxSynchronize 대신 cuStreamSynchronize(hStream)을 사용해
        // 비-killer 커널들의 대기 없이 이 커널의 GPU 완료만 기다림.
        // → gate_killer_exit의 clock_gettime이 실제 GPU 실행시간을 측정.
        // → GATE_RUNNING 전환 시 GPU 이미 완료 → peer GPU overlap 방지.
        LAZY_RESOLVE(real_cuStreamSynchronize, "cuStreamSynchronize");
        if (real_cuStreamSynchronize) real_cuStreamSynchronize(hStream);
        gate_killer_exit();
    }

    g_prism.kernel_idx++;
    return r;
}

// cuLaunchKernelEx (CUDA 11.6+)
extern "C"
CUresult cuLaunchKernelEx(
    const CUlaunchConfig* config,
    CUfunction f,
    void** kernelParams, void** extra)
{
    LAZY_RESOLVE(real_cuLaunchKernelEx, "cuLaunchKernelEx");
    if (!g_prism.initialized || !real_cuLaunchKernelEx)
        return real_cuLaunchKernelEx
               ? real_cuLaunchKernelEx(config, f, kernelParams, extra)
               : CUDA_ERROR_NOT_INITIALIZED;

    bool killer = is_killer(f);
    if (killer) gate_killer_enter();

    CUresult r = real_cuLaunchKernelEx(config, f, kernelParams, extra);

    if (killer) {
        LAZY_RESOLVE(real_cuStreamSynchronize, "cuStreamSynchronize");
        if (config && real_cuStreamSynchronize) real_cuStreamSynchronize(config->hStream);
        gate_killer_exit();
    }

    g_prism.kernel_idx++;
    return r;
}

// ── cuGetProcAddress_v2 후킹 — cublasLt의 두 번째 우회 경로 차단 ────────────────
// cublasLt가 dlsym(cuda_handle, "cuGetProcAddress_v2")로 함수 포인터를 얻은 뒤
// cuGetProcAddress_v2("cuLaunchKernel", &fn, ...) 를 호출하는 경로를 차단한다.
// 실제 cuGetProcAddress_v2를 먼저 호출해 기존 동작을 보존하되,
// cuLaunchKernel/Ex 결과만 우리 hook으로 교체한다.
static CUresult (*real_cuGetProcAddress_v2)(
    const char*, void**, int, cuuint64_t, CUdriverProcAddressQueryResult*) = nullptr;

extern "C"
CUresult cuGetProcAddress_v2(
    const char* symbol, void** pfn,
    int cudaVersion, cuuint64_t flags,
    CUdriverProcAddressQueryResult* symbolStatus)
{
    LAZY_RESOLVE(real_cuGetProcAddress_v2, "cuGetProcAddress_v2");
    if (!real_cuGetProcAddress_v2) return CUDA_ERROR_NOT_INITIALIZED;

    CUresult r = real_cuGetProcAddress_v2(symbol, pfn, cudaVersion, flags, symbolStatus);

    if (r == CUDA_SUCCESS && pfn) {
        // cuLaunchKernel/_ptsz 및 cuLaunchKernelEx/_ptsz → 우리 hook으로 교체
        if (strncmp(symbol, "cuLaunchKernel", 14) == 0 &&
            (symbol[14] == '\0' || strcmp(symbol + 14, "_ptsz") == 0))
            *pfn = (void*)cuLaunchKernel;
        else if (strncmp(symbol, "cuLaunchKernelEx", 16) == 0 &&
                 (symbol[16] == '\0' || strcmp(symbol + 16, "_ptsz") == 0))
            *pfn = (void*)cuLaunchKernelEx;
        // cuGetProcAddress 자체도 우리 wrapper로 교체.
        // cublasLt는 cuGetProcAddress_v2("cuGetProcAddress", ...) 로 실제 포인터를
        // 얻은 뒤 이를 사용해 cuLaunchKernel 등을 재조회한다.
        // 우리 wrapper를 돌려줌으로써 모든 후속 API 조회도 인터셉트.
        else if (strcmp(symbol, "cuGetProcAddress") == 0 ||
                 strcmp(symbol, "cuGetProcAddress_v2") == 0)
            *pfn = (void*)cuGetProcAddress_v2;
    }
    return r;
}

// ── cudaLaunchKernel 후킹 (Runtime API: <<<>>> 구문) ──────────────────────────
// 커널 이름: dladdr()로 host-side 함수 포인터에서 심볼 이름 획득
static bool is_cuda_rt_killer(const void* func) {
    Dl_info info;
    if (dladdr(func, &info) && info.dli_sname) {
        if (is_killer_by_name(info.dli_sname)) return true;
    }
    return is_killer_by_index(g_prism.kernel_idx);
}

extern "C"
cudaError_t cudaLaunchKernel(
    const void* func,
    dim3 gridDim, dim3 blockDim,
    void** args,
    size_t sharedMem,
    cudaStream_t stream)
{
    LAZY_RESOLVE(real_cudaLaunchKernel, "cudaLaunchKernel");
    if (!g_prism.initialized || !real_cudaLaunchKernel)
        return real_cudaLaunchKernel
               ? real_cudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream)
               : cudaErrorNoDevice;

    bool killer = is_cuda_rt_killer(func);
    if (killer) gate_killer_enter();

    cudaError_t r = real_cudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream);

    if (killer) {
        LAZY_RESOLVE(real_cudaStreamSynchronize, "cudaStreamSynchronize");
        if (real_cudaStreamSynchronize) real_cudaStreamSynchronize(stream);
        gate_killer_exit();
    }

    g_prism.kernel_idx++;
    return r;
}

// ── 메모리 후킹 ────────────────────────────────────────────────────────────────
extern "C"
CUresult cuMemAlloc_v2(CUdeviceptr* dptr, size_t bytesize) {
    LAZY_RESOLVE(real_cuMemAlloc, "cuMemAlloc_v2");
    if (!g_prism.initialized || !real_cuMemAlloc)
        return real_cuMemAlloc ? real_cuMemAlloc(dptr, bytesize) : CUDA_ERROR_NOT_INITIALIZED;

    if (!vmem_check_alloc(bytesize)) {
        fprintf(stderr, "[prism] cuMemAlloc quota 초과 (%zu bytes)\n", bytesize);
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    CUresult r = real_cuMemAlloc(dptr, bytesize);
    if (r == CUDA_SUCCESS) vmem_track_alloc((void*)(uintptr_t)*dptr, bytesize);
    return r;
}

extern "C"
CUresult cuMemFree_v2(CUdeviceptr dptr) {
    LAZY_RESOLVE(real_cuMemFree, "cuMemFree_v2");
    if (!g_prism.initialized || !real_cuMemFree)
        return real_cuMemFree ? real_cuMemFree(dptr) : CUDA_ERROR_NOT_INITIALIZED;

    vmem_track_free((void*)(uintptr_t)dptr);
    return real_cuMemFree(dptr);
}

extern "C"
cudaError_t cudaMalloc(void** devPtr, size_t size) {
    LAZY_RESOLVE(real_cudaMalloc, "cudaMalloc");
    if (!g_prism.initialized || !real_cudaMalloc)
        return real_cudaMalloc ? real_cudaMalloc(devPtr, size) : cudaErrorNoDevice;

    if (!vmem_check_alloc(size)) {
        fprintf(stderr, "[prism] cudaMalloc quota 초과 (%zu bytes)\n", size);
        return cudaErrorMemoryAllocation;
    }
    cudaError_t r = real_cudaMalloc(devPtr, size);
    if (r == cudaSuccess) vmem_track_alloc(*devPtr, size);
    return r;
}

extern "C"
cudaError_t cudaFree(void* devPtr) {
    LAZY_RESOLVE(real_cudaFree, "cudaFree");
    if (!g_prism.initialized || !real_cudaFree)
        return real_cudaFree ? real_cudaFree(devPtr) : cudaErrorNoDevice;

    vmem_track_free(devPtr);
    return real_cudaFree(devPtr);
}

// ── killer policy JSON 경량 파서 ──────────────────────────────────────────────
static void parse_int_array(const char* json, const char* key,
                             int* out, int* count, int max) {
    const char* p = strstr(json, key);
    if (!p) return;
    p = strchr(p, '[');
    if (!p) return;
    p++;
    *count = 0;
    while (*p && *p != ']' && *count < max) {
        while (*p && !isdigit(*p) && *p != ']') p++;
        if (*p == ']') break;
        out[(*count)++] = atoi(p);
        while (*p && isdigit(*p)) p++;
    }
}

static void parse_string_array(const char* json, const char* key,
                                char out[][MAX_KERNEL_NAME_LEN],
                                int* count, int max) {
    const char* p = strstr(json, key);
    if (!p) return;
    p = strchr(p, '[');
    if (!p) return;
    p++;
    *count = 0;
    while (*p && *p != ']' && *count < max) {
        while (*p && *p != '"' && *p != ']') p++;
        if (*p != '"') break;
        p++;
        int i = 0;
        while (*p && *p != '"' && i < MAX_KERNEL_NAME_LEN - 1)
            out[*count][i++] = *p++;
        out[(*count)++][i] = '\0';
        if (*p == '"') p++;
    }
}

static int parse_int_field(const char* json, const char* key) {
    const char* p = strstr(json, key);
    if (!p) return 0;
    p = strchr(p, ':');
    if (!p) return 0;
    p++;
    while (*p && isspace(*p)) p++;
    return atoi(p);
}

static void load_killer_policy(const char* path) {
    FILE* f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "[prism] policy 파일 없음: %s\n", path);
        return;
    }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    rewind(f);

    char* buf = (char*)malloc(sz + 1);
    fread(buf, 1, sz, f);
    buf[sz] = '\0';
    fclose(f);

    KillerPolicy* p = &g_prism.policy;

    parse_int_array(buf, "\"killer_indices\"",
                    p->indices, &p->count, MAX_KILLER_KERNELS);

    int name_count = 0;
    parse_string_array(buf, "\"killer_names\"",
                       p->names, &name_count, MAX_KILLER_KERNELS);

    g_prism.kernels_per_iter = parse_int_field(buf, "\"kernels_per_iter\"");
    p->kernels_per_iter      = g_prism.kernels_per_iter;
    p->loaded = true;

    free(buf);
    fprintf(stderr, "[prism] policy 로드: %d killers, iter_len=%d\n",
            p->count, g_prism.kernels_per_iter);
}

// ── killer policy hot-reload ───────────────────────────────────────────────────
// round_manager.cpp의 start_new_round()에서 호출.
// shm.killer_policy_version이 마지막 로드 이후 바뀌었으면 JSON을 재로드.
void reload_killer_policy_if_needed(void) {
    PrismSharedState* shm = g_prism.shm;
    if (!shm || !g_prism.initialized) return;

    uint64_t shm_ver = __atomic_load_n(&shm->killer_policy_version, __ATOMIC_ACQUIRE);
    if (shm_ver == g_prism.loaded_killer_version) return;

    // well-known hot-reload 경로 확인
    char hot_path[256];
    snprintf(hot_path, sizeof(hot_path), HOT_POLICY_PATH_FMT,
             g_prism.group_id, g_prism.tenant_idx);

    // 기존 policy 초기화
    memset(&g_prism.policy, 0, sizeof(g_prism.policy));
    g_prism.kernels_per_iter = 0;

    if (access(hot_path, R_OK) == 0) {
        load_killer_policy(hot_path);
    } else if (g_prism.policy_path[0] && access(g_prism.policy_path, R_OK) == 0) {
        // hot-path 없으면 원본 ENV_POLICY_PATH 재로드
        load_killer_policy(g_prism.policy_path);
    } else {
        // 둘 다 없으면 기본 패턴 복구
        static const char* defaults[] = {
            "sgemm", "dgemm", "hgemm", "gemm",
            "fprop", "xmma", "cutlass", "fmha",
            "matmul", nullptr
        };
        for (int i = 0; defaults[i] && i < MAX_KILLER_KERNELS; i++) {
            strncpy(g_prism.policy.names[i], defaults[i], MAX_KERNEL_NAME_LEN - 1);
            g_prism.policy.count++;
        }
        fprintf(stderr, "[prism] hot-reload: policy 없음, 기본 패턴 복구\n");
    }

    g_prism.loaded_killer_version = shm_ver;
}
