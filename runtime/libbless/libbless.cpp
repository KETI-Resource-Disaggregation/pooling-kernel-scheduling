#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <pthread.h>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <algorithm>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <sched.h>
#include <time.h>  // nanosleep
#include <chrono>
#include <sys/ioctl.h>  // ioctl interception
#include <sys/mman.h>   // mmap interception
#include <stdarg.h>     // va_list for ioctl
#include <errno.h>      // errno for syscall errors
#include <unordered_map> // for memory allocation tracking

// fwd
static void ensure_init();
static void attach_runtime_if_needed();
static void start_control_server(const std::string& path);

namespace bless { static std::atomic<int> gate{0}; }

// ── ctx 생성 진입점 분기 (Exp_54 O-4) ────────────────────────────────────────
// CUDA 13 헤더는 cuCtxCreate_v3 선언을 legacy 블록으로 숨겨 빌드가 깨진다.
// 드라이버(libcuda)는 v3·v4 심볼을 모두 export 하므로(232 nm 실측) 컴파일 타임
// 선언 대신 ★런타임 dlsym 으로 v4 우선/v3 폴백 — 12.9·13.1 양쪽 헤더에서 빌드,
// 실행은 드라이버가 주는 진입점 사용. affinity 는 v4 의 CUctxCreateParams.
// execAffinityParams 로 계승됨(13.1 cuda.h:2561 공식 확인 — 공간 노브 유지).
typedef CUresult (*bless_ctx_v3_t)(CUcontext*, CUexecAffinityParam*, int,
                                   unsigned int, CUdevice);
typedef CUresult (*bless_ctx_v4_t)(CUcontext*, CUctxCreateParams*,
                                   unsigned int, CUdevice);

static CUresult bless_ctx_create(CUcontext* pctx, CUexecAffinityParam* params,
                                 int nparams, unsigned int flags, CUdevice dev,
                                 const char** entry_used) {
  static bless_ctx_v4_t v4 =
      (bless_ctx_v4_t)dlsym(RTLD_DEFAULT, "cuCtxCreate_v4");
  static bless_ctx_v3_t v3 =
      (bless_ctx_v3_t)dlsym(RTLD_DEFAULT, "cuCtxCreate_v3");
  if (v4) {
    if (entry_used) *entry_used = "v4";
    CUctxCreateParams cp{};
    cp.execAffinityParams = params;
    cp.numExecAffinityParams = nparams;
    return v4(pctx, params ? &cp : nullptr, flags, dev);
  }
  if (v3) {
    if (entry_used) *entry_used = "v3";
    return v3(pctx, params, nparams, flags, dev);
  }
  if (entry_used) *entry_used = "none";
  return CUDA_ERROR_NOT_SUPPORTED;
}



// ---------------- real runtime symbols ----------------
static decltype(&cudaLaunchKernel)  real_cudaLaunchKernel  = nullptr;
static decltype(&cuLaunchKernel)    real_cuLaunchKernel    = nullptr;
static decltype(&cudaMalloc)        real_cudaMalloc        = nullptr;
static decltype(&cudaFree)          real_cudaFree          = nullptr;
static decltype(&cudaMallocManaged) real_cudaMallocManaged = nullptr;
static decltype(&cudaMemcpy)        real_cudaMemcpy        = nullptr;
static decltype(&cudaMemcpyAsync)   real_cudaMemcpyAsync   = nullptr;
static decltype(&cudaStreamCreate)  real_cudaStreamCreate  = nullptr;
static decltype(&cudaStreamDestroy) real_cudaStreamDestroy = nullptr;
static decltype(&cudaGraphLaunch)   real_cudaGraphLaunch   = nullptr;
static decltype(&cudaDeviceSynchronize) real_cudaDeviceSynchronize = nullptr;
static decltype(&cudaStreamSynchronize) real_cudaStreamSynchronize = nullptr;

// System call function pointers
typedef int (*ioctl_func_t)(int fd, unsigned long request, ...);
typedef void* (*mmap_func_t)(void *addr, size_t length, int prot, int flags, int fd, off_t offset);
static ioctl_func_t real_ioctl = nullptr;
static mmap_func_t  real_mmap  = nullptr;

static void resolve_real() {
  if (!real_cudaLaunchKernel)  real_cudaLaunchKernel  = (decltype(&cudaLaunchKernel))  dlsym(RTLD_NEXT, "cudaLaunchKernel");
  if (!real_cuLaunchKernel)    real_cuLaunchKernel    = (decltype(&cuLaunchKernel))    dlsym(RTLD_NEXT, "cuLaunchKernel");
  if (!real_cudaMalloc)        real_cudaMalloc        = (decltype(&cudaMalloc))        dlsym(RTLD_NEXT, "cudaMalloc");
  if (!real_cudaFree)          real_cudaFree          = (decltype(&cudaFree))          dlsym(RTLD_NEXT, "cudaFree");
  if (!real_cudaMallocManaged) real_cudaMallocManaged = (decltype(&cudaMallocManaged)) dlsym(RTLD_NEXT, "cudaMallocManaged");
  if (!real_cudaMemcpy)        real_cudaMemcpy        = (decltype(&cudaMemcpy))        dlsym(RTLD_NEXT, "cudaMemcpy");
  if (!real_cudaMemcpyAsync)   real_cudaMemcpyAsync   = (decltype(&cudaMemcpyAsync))   dlsym(RTLD_NEXT, "cudaMemcpyAsync");
  if (!real_cudaStreamCreate)  real_cudaStreamCreate  = (decltype(&cudaStreamCreate))  dlsym(RTLD_NEXT, "cudaStreamCreate");
  if (!real_cudaStreamDestroy) real_cudaStreamDestroy = (decltype(&cudaStreamDestroy)) dlsym(RTLD_NEXT, "cudaStreamDestroy");
  if (!real_cudaGraphLaunch)   real_cudaGraphLaunch   = (decltype(&cudaGraphLaunch))   dlsym(RTLD_NEXT, "cudaGraphLaunch");
  if (!real_cudaDeviceSynchronize)
    real_cudaDeviceSynchronize = (decltype(&cudaDeviceSynchronize)) dlsym(RTLD_NEXT, "cudaDeviceSynchronize");
  if (!real_cudaStreamSynchronize)
    real_cudaStreamSynchronize = (decltype(&cudaStreamSynchronize)) dlsym(RTLD_NEXT, "cudaStreamSynchronize");
  
  if (!real_ioctl) real_ioctl = (ioctl_func_t) dlsym(RTLD_NEXT, "ioctl");
  if (!real_mmap)  real_mmap  = (mmap_func_t)  dlsym(RTLD_NEXT, "mmap");
}

// ---------------- bless state ----------------
namespace bless {
  enum Route { LIMITED=0, UNLIMITED=1 };

  static std::atomic<int>  route{LIMITED};       // 런타임 중 전환 금지(할당 이후엔 무시)
  static std::atomic<bool> inited{false};
  static pthread_mutex_t   init_mu = PTHREAD_MUTEX_INITIALIZER;

  static CUdevice   dev = 0;
  static CUcontext  ctx_limited   = nullptr;
  static CUcontext  ctx_unlimited = nullptr;

  static std::thread ctrl_thread;
  static std::atomic<bool> ctrl_running{false};
  static std::string sock_path;
  static int g_ctrl_fd = -1;
  static std::atomic<bool> ctrl_ready{false};
  // [Exp_73] time_stats 를 파일로도 발화 — feeder /occupancy 배선용
  // (stderr 는 컨테이너 로그로만 가 feeder 가 못 읽음). BLESS_STATS_LOG 지정 시 open.
  static FILE* stats_log = nullptr;

  static int total_sms = 0;
  static std::atomic<int> limited_sms{0};
  // [Exp_41 O-3] 공간 제한 실효 상태 — 무음 no-op 금지
  static std::atomic<int> limited_applied{0};      // 1=exec affinity 실제 적용
  static std::atomic<int> requested_pct{50};       // BLESS_LIMIT_PCT 요청값
  // [Exp_61] 공간 제한 위임 방식 (BLESS_SPACE_MODE=ctx|mps_env)
  //   SPACE_CTX     : exec-affinity 제한 컨텍스트 (현행 기본 — 하위호환)
  //   SPACE_MPS_ENV : MPS 가 CUDA_MPS_ACTIVE_THREAD_PERCENTAGE 를 직접 해석.
  //                   libbless 는 제한 ctx 를 만들지 않고 ctx 전환도 하지 않는다
  //                   → O-6b(별도 스레드 launch)·O-7(훈련 hang) 이 원천 해소.
  //                   그 env 값은 libbless 가 소비하지 않는다(관측용 echo 만).
  enum { SPACE_CTX = 0, SPACE_MPS_ENV = 1 };
  static std::atomic<int> space_mode{SPACE_CTX};
  static std::atomic<long long> fallback_launches{0}; // limited 경로가 unlimited 로 폴백한 런치 수

  // squad accounting
  static std::atomic<int>        squad_size{100};   // set_squad
  static std::atomic<int>        squad_prog{0};
  static std::atomic<long long>  kernel_seq{0};

  // boost는 "크레딧 게이트 우회" 의미로만 사용 (컨텍스트 전환 X)
  static std::atomic<bool> boost_mode{false};
  static std::atomic<bool> pause_mode{false};

  // share & SD
  static std::atomic<int>  share_quota{0};
  static std::atomic<int>  sd_sent{0};

  // 크레딧 게이트: -1이면 무제한(게이트 off), 0..N은 남은 커널 크레딧
  static std::atomic<int>  credit_remain{-1};

  // ======== 시간 기반 크레딧 시스템 (v2) ========
  // 모드: 0=커널 개수 기반(기존), 1=시간 기반
  static std::atomic<int>  credit_mode{0};

  // 시간 크레딧 (마이크로초 단위)
  static std::atomic<int64_t> time_credit_us{-1};      // -1 = 무제한
  // [Exp_16 보강 ①] 무제한 여부를 credit 부호와 분리. 원본은 "credit<0 = 무제한"
  // 이라 차감으로 적자가 되는 순간 게이트·차감이 모두 우회됨(Exp_16 실측:
  // 가중 2:1/1:2/3:1 전부 무효과). 적자는 "빚"으로 게이트를 계속 막아야 하고,
  // 무제한은 이 플래그로만 표현한다.
  static std::atomic<bool>    time_unlimited{true};
  static std::atomic<int64_t> time_slice_us{1000};     // 타임슬라이스 (기본 1ms)

  // 커널 시간 통계 (EMA)
  static std::atomic<int64_t> avg_kernel_time_us{10};  // 평균 커널 시간 추정
  static std::atomic<int64_t> total_time_us{0};        // 누적 사용 시간

  // 샘플링 측정용
  static std::atomic<int>     sample_counter{0};
  static std::atomic<int64_t> sample_start_us{0};
  static std::atomic<int>     sample_kernel_count{0};
  static constexpr int SAMPLE_INTERVAL = 32;           // N개 커널마다 측정

  // 로컬 버스트 (스레드별, 오버헤드 감소용)
  static constexpr int64_t LOCAL_BURST_US = 100;       // 100us씩 로컬에서 사용

  // 실시간 시간 추적
  static std::atomic<int64_t> last_sync_time_us{0};
  static std::atomic<int64_t> kernels_since_sync{0};
  static std::atomic<bool>    measuring{false};
  // =============================================

  // any GPU allocation happened? (guard reconf/route)
  static std::atomic<bool> any_alloc{false};

  // optional master
  static int master_fd = -1;
  static std::string master_path;
  static std::string tenant_id;

  // ======== 컨텍스트 스위칭 오버헤드 측정 ========
  static std::atomic<int64_t> ctx_switch_count{0};        // 총 컨텍스트 스위치 횟수
  static std::atomic<int64_t> ctx_switch_total_us{0};     // 총 컨텍스트 스위치 시간(us)
  static std::atomic<int64_t> ctx_switch_avg_us{0};       // 평균 컨텍스트 스위치 시간(us)
  static std::atomic<int64_t> ctx_switch_max_us{0};       // 최대 컨텍스트 스위치 시간(us)
  // =============================================

  // ======== 우선순위 기반 커널 큐 시스템 ========
  enum Priority { URGENT=0, NORMAL=1, BACKGROUND=2, NUM_PRIORITIES=3 };
  static std::atomic<int> current_priority{NORMAL};       // 현재 우선순위

  // 우선순위별 대기 커널 수
  static std::atomic<int64_t> queue_len[NUM_PRIORITIES] = {{0}, {0}, {0}};

  // Urgent 선점 플래그
  static std::atomic<bool> urgent_pending{false};         // Urgent 커널 대기 중
  static std::atomic<bool> preempt_requested{false};      // 선점 요청됨
  // =============================================

  // ======== System Call 인터셉션 통계 ========
  static std::atomic<int64_t> ioctl_count{0};             // ioctl 호출 횟수
  static std::atomic<int64_t> ioctl_nvidia_count{0};      // NVIDIA ioctl 횟수
  static std::atomic<int64_t> mmap_count{0};              // mmap 호출 횟수
  static std::atomic<int64_t> mmap_gpu_count{0};          // GPU 관련 mmap 횟수
  static std::atomic<int64_t> mmap_total_bytes{0};        // 총 mmap 바이트
  static std::atomic<bool>    syscall_intercept_enabled{false}; // 인터셉션 활성화
  // =============================================

  // ======== 실시간 메모리 추적 (S-06) ========
  static std::atomic<int64_t> mem_alloc_count{0};         // cudaMalloc 호출 횟수
  static std::atomic<int64_t> mem_free_count{0};          // cudaFree 호출 횟수
  static std::atomic<int64_t> mem_current_bytes{0};       // 현재 할당된 바이트
  static std::atomic<int64_t> mem_peak_bytes{0};          // 최대 할당 바이트 (peak)
  static std::atomic<int64_t> mem_total_alloc_bytes{0};   // 누적 할당 바이트
  static std::atomic<int64_t> mem_quota_bytes{-1};        // 메모리 quota (-1=무제한)
  static std::atomic<bool>    mem_tracking_enabled{true}; // 메모리 추적 활성화

  // 할당 크기 추적용 맵 (cudaFree 시 크기 조회)
  static std::unordered_map<void*, size_t> alloc_size_map;
  static pthread_mutex_t alloc_map_mu = PTHREAD_MUTEX_INITIALIZER;

  // 메모리 추적 헬퍼 함수
  static inline void track_alloc(void* ptr, size_t size) {
    if (!mem_tracking_enabled.load(std::memory_order_relaxed)) return;

    pthread_mutex_lock(&alloc_map_mu);
    alloc_size_map[ptr] = size;
    pthread_mutex_unlock(&alloc_map_mu);

    mem_alloc_count.fetch_add(1, std::memory_order_relaxed);
    mem_total_alloc_bytes.fetch_add(size, std::memory_order_relaxed);
    int64_t current = mem_current_bytes.fetch_add(size, std::memory_order_acq_rel) + size;

    // peak 업데이트 (lock-free)
    int64_t peak = mem_peak_bytes.load(std::memory_order_relaxed);
    while (current > peak) {
      if (mem_peak_bytes.compare_exchange_weak(peak, current, std::memory_order_acq_rel)) break;
    }
  }

  static inline void track_free(void* ptr) {
    if (!mem_tracking_enabled.load(std::memory_order_relaxed)) return;
    if (!ptr) return;  // cudaFree(nullptr) 무시

    size_t size = 0;
    pthread_mutex_lock(&alloc_map_mu);
    auto it = alloc_size_map.find(ptr);
    if (it != alloc_size_map.end()) {
      size = it->second;
      alloc_size_map.erase(it);
    }
    pthread_mutex_unlock(&alloc_map_mu);

    if (size > 0) {
      mem_free_count.fetch_add(1, std::memory_order_relaxed);
      mem_current_bytes.fetch_sub(size, std::memory_order_acq_rel);
    }
  }

  // Quota 체크 (S-03/S-04 연동용)
  static inline bool check_quota(size_t size) {
    int64_t quota = mem_quota_bytes.load(std::memory_order_relaxed);
    if (quota < 0) return true;  // 무제한
    int64_t current = mem_current_bytes.load(std::memory_order_relaxed);
    return (current + (int64_t)size) <= quota;
  }
  // =============================================
}

// ---- master send ----
static inline void master_send(const char* msg) {
  if (bless::master_fd < 0 || bless::master_path.empty()) return;
  sockaddr_un r{}; r.sun_family = AF_UNIX;
  snprintf(r.sun_path, sizeof(r.sun_path), "%s", bless::master_path.c_str());
  sendto(bless::master_fd, msg, (int)strlen(msg), 0, (sockaddr*)&r, sizeof(r));
}

// ---- 시간 유틸리티 ----
static inline int64_t now_us() {
  using namespace std::chrono;
  return duration_cast<microseconds>(steady_clock::now().time_since_epoch()).count();
}

// 스레드 로컬 시간 버짓
static thread_local int64_t tl_time_budget_us = 0;

// ---- 실제 시간 측정 및 크레딧 차감 ----
// 커널 launch 전에 시간 기록
static thread_local int64_t tl_batch_start_us = 0;
static thread_local int tl_batch_kernel_count = 0;

static inline void time_batch_start() {
  if (tl_batch_kernel_count == 0) {
    tl_batch_start_us = now_us();
  }
  tl_batch_kernel_count++;
}

// 동기화 시점에 실제 시간 측정 및 크레딧 차감
static inline void time_batch_end_and_charge() {
  if (tl_batch_kernel_count == 0) return;

  int64_t elapsed_us = now_us() - tl_batch_start_us;
  int kernel_count = tl_batch_kernel_count;

  // 누적 시간 기록
  bless::total_time_us.fetch_add(elapsed_us, std::memory_order_relaxed);

  // 평균 커널 시간 업데이트 (EMA)
  if (kernel_count > 0) {
    int64_t avg = elapsed_us / kernel_count;
    int64_t old_avg = bless::avg_kernel_time_us.load(std::memory_order_relaxed);
    // EMA: new = old * 0.75 + measured * 0.25
    int64_t new_avg = (old_avg * 3 + avg) / 4;
    if (new_avg < 1) new_avg = 1;
    bless::avg_kernel_time_us.store(new_avg, std::memory_order_relaxed);
  }

  // [Exp_16 보강 ②] 사용 시간은 무제한이 아닌 한 항상 차감 — 적자(빚)도 이월.
  // 원본의 "credit>=0 일 때만 차감"은 적자 진입 후 차감이 멈춰 예산이 무의미해짐.
  if (!bless::time_unlimited.load(std::memory_order_relaxed)) {
    bless::time_credit_us.fetch_sub(elapsed_us, std::memory_order_acq_rel);
  }

  // 배치 리셋
  tl_batch_kernel_count = 0;
  tl_batch_start_us = 0;
}

static inline void time_credit_gate() {
  if (__builtin_expect(bless::boost_mode.load(std::memory_order_relaxed), 0)) return;
  if (__builtin_expect(bless::time_unlimited.load(std::memory_order_relaxed), 1)) return;
  int64_t credit = bless::time_credit_us.load(std::memory_order_relaxed);
  if (credit > 0) return;
  // [Exp_16 보강 ③] 대기 전에 열린 배치를 먼저 청구하고 창을 닫는다 —
  // 원본은 배치 창이 열린 채 게이트에서 대기해 "대기 시간까지 사용으로 청구"
  // (Exp_16 실측: 청구량 ≈ 벽시계 전체). 대기 시간은 사용이 아니다.
  time_batch_end_and_charge();
  int waited = 0;
  while (credit <= 0) {
    if (waited < 50) {
      ++waited;
      sched_yield();
    } else {
      struct timespec ts{0, 20000}; /* 20us */
      nanosleep(&ts, nullptr);
      waited = 0;
    }
    if (bless::time_unlimited.load(std::memory_order_relaxed)) return;
    credit = bless::time_credit_us.load(std::memory_order_acquire);
  }
}

// ---- 샘플링 기반 측정 (백업용) ----
static inline void sample_kernel_start() {
  int count = bless::sample_counter.fetch_add(1, std::memory_order_relaxed);
  if (count == 0) {
    bless::sample_start_us.store(now_us(), std::memory_order_relaxed);
  }
  bless::sample_kernel_count.fetch_add(1, std::memory_order_relaxed);
}

static inline void do_sample_measurement() {
  int count = bless::sample_kernel_count.load(std::memory_order_relaxed);
  if (count == 0) return;

  int64_t start = bless::sample_start_us.load(std::memory_order_relaxed);
  int64_t elapsed = now_us() - start;

  if (count > 0 && elapsed > 0) {
    int64_t avg = elapsed / count;
    int64_t old_avg = bless::avg_kernel_time_us.load(std::memory_order_relaxed);
    int64_t new_avg = (old_avg * 3 + avg) / 4;
    if (new_avg < 1) new_avg = 1;
    bless::avg_kernel_time_us.store(new_avg, std::memory_order_relaxed);

    // 시간 크레딧 차감
    int64_t credit = bless::time_credit_us.load(std::memory_order_relaxed);
    if (credit >= 0) {
      bless::time_credit_us.fetch_sub(elapsed, std::memory_order_acq_rel);
    }

    bless::total_time_us.fetch_add(elapsed, std::memory_order_relaxed);
  }

  // 리셋
  bless::sample_counter.store(0, std::memory_order_relaxed);
  bless::sample_kernel_count.store(0, std::memory_order_relaxed);
  bless::sample_start_us.store(now_us(), std::memory_order_relaxed);
}

// push/pop current context according to **route only** (boost는 경로에 영향 X)
// 현재 컨텍스트와 목표 컨텍스트가 같으면 push/pop 자체를 생략 → 핫패스 오버헤드 절감
// 컨텍스트 스위칭 시간 측정 기능 추가
struct ScopedCtxGuard {
  CUcontext target{nullptr};
  bool pushed{false};
  int64_t push_start_us{0};

  explicit ScopedCtxGuard(int override_route = -1) {
    ensure_init();
    int base = bless::route.load(std::memory_order_relaxed);
    int r = (override_route >= 0) ? override_route : base;
    target = (r==bless::UNLIMITED) ? bless::ctx_unlimited : bless::ctx_limited;
    if (target == nullptr) {   // [Exp_41 O-3] limited 생성 실패 → 명시적 폴백
      target = bless::ctx_unlimited;
      // [Exp_61] mps_env 모드의 ctx 부재는 폴백(실패)이 아니라 의도된 위임 경로 —
      // O-3 폴백 카운터를 오염시키지 않는다.
      if (bless::space_mode.load(std::memory_order_relaxed) == bless::SPACE_CTX)
        bless::fallback_launches.fetch_add(1, std::memory_order_relaxed);
    }
    // [Exp_54 O-6] 공간 제한이 실제로 적용되지 않은 상태(limited ctx 생성 실패 =
    // 무-MPS 환경)에서는 컨텍스트 전환이 아무 이득이 없다. 그런데 전환을 하면
    // ★앱이 자기 컨텍스트에서 만든 비-default 스트림이 무효 핸들이 되어
    // (CUDA_ERROR_INVALID_HANDLE) 멀티스트림 워크로드가 즉시 크래시한다
    // (default stream 은 컨텍스트별 특수 핸들이라 무증상 — 그래서 단일 스트림
    // 워크로드에서만 검증돼 온 것). 실효 없는 전환을 생략해 멀티스트림 호환을
    // 회복한다. 시간 게이트(time_credit)는 컨텍스트와 무관하므로 영향 없음.
    if (bless::limited_applied.load(std::memory_order_relaxed) == 0) {
      return;   // pushed=false — 앱의 현재 컨텍스트 유지
    }
    CUcontext cur=nullptr;
    cuCtxGetCurrent(&cur);
    if (cur != target) {
      push_start_us = now_us();
      cuCtxPushCurrent(target);
      pushed = true;

      // 컨텍스트 스위치 시간 측정
      int64_t elapsed = now_us() - push_start_us;
      bless::ctx_switch_count.fetch_add(1, std::memory_order_relaxed);
      bless::ctx_switch_total_us.fetch_add(elapsed, std::memory_order_relaxed);

      // 평균 업데이트 (EMA)
      int64_t old_avg = bless::ctx_switch_avg_us.load(std::memory_order_relaxed);
      int64_t new_avg = (old_avg * 7 + elapsed) / 8;
      bless::ctx_switch_avg_us.store(new_avg, std::memory_order_relaxed);

      // 최대값 업데이트
      int64_t cur_max = bless::ctx_switch_max_us.load(std::memory_order_relaxed);
      while (elapsed > cur_max) {
        if (bless::ctx_switch_max_us.compare_exchange_weak(cur_max, elapsed)) break;
      }
    }
  }

  ~ScopedCtxGuard(){
    if (pushed) {
      int64_t pop_start = now_us();
      CUcontext p=nullptr;
      cuCtxPopCurrent(&p);

      // Pop 시간도 측정에 포함
      int64_t elapsed = now_us() - pop_start;
      bless::ctx_switch_total_us.fetch_add(elapsed, std::memory_order_relaxed);
    }
  }
};

static std::atomic<bool> rt_attached{false};
static pthread_mutex_t   rt_mu = PTHREAD_MUTEX_INITIALIZER;

static void attach_runtime_if_needed() {
  if (rt_attached.load()) return;
  pthread_mutex_lock(&rt_mu);
  if (!rt_attached.load()) {
    resolve_real();
    if (!real_cudaLaunchKernel || !real_cudaFree) {
      void* h = dlopen("libcudart.so", RTLD_LAZY | RTLD_GLOBAL); (void)h; resolve_real();
    }
    CUcontext prev=nullptr;
    // attach to LIMITED
    cuCtxPushCurrent(bless::ctx_limited);
    if (real_cudaFree) real_cudaFree(0);
    cuCtxPopCurrent(&prev);
    // attach to UNLIMITED
    cuCtxPushCurrent(bless::ctx_unlimited);
    if (real_cudaFree) real_cudaFree(0);
    cuCtxPopCurrent(&prev);
    rt_attached.store(true);
  }
  pthread_mutex_unlock(&rt_mu);
}

// safe (re)create LIMITED only if no allocations yet
static void reconf_limited_ctx(int new_sms) {
  // [Exp_61] mps_env 모드: 제한 ctx 를 만들지 않는다. limited_applied=0 이므로
  // ScopedCtxGuard 가 전환을 생략(Exp_54 경로) — 공간 제한은 스폰 시점에 주입된
  // CUDA_MPS_ACTIVE_THREAD_PERCENTAGE 를 MPS 서버가 해석한다. Exp_57 PoC 4/4 ·
  // Exp_60 훈련 12/12 정상이 이 경로다.
  if (bless::space_mode.load(std::memory_order_relaxed) == bless::SPACE_MPS_ENV) {
    bless::ctx_limited = nullptr;
    bless::limited_applied.store(0);
    bless::limited_sms.store(new_sms);
    const char* mp = getenv("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE");   // 관측용 echo
    fprintf(stderr, "[libbless] space_mode=mps_env: ctx affinity 생략 — 공간 제한은 "
                    "MPS env 위임 (CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=%s)\n",
            mp ? mp : "unset");
    bless::gate.store(0);
    return;
  }
  if (bless::any_alloc.load(std::memory_order_acquire)) {
    fprintf(stderr, "[libbless] reconf_sm ignored: allocations already exist\n");
    return;
  }
  if (new_sms < 1) new_sms = 1;
  if (new_sms > bless::total_sms) new_sms = bless::total_sms;

  if (bless::ctx_limited) {
    int expect = 0; bless::gate.compare_exchange_strong(expect, 1);
    CUcontext prev=nullptr; cuCtxPushCurrent(bless::ctx_limited); cuCtxSynchronize(); cuCtxPopCurrent(&prev);
    cuCtxDestroy(bless::ctx_limited);
  }
  CUexecAffinityParam p{}; p.type = CU_EXEC_AFFINITY_TYPE_SM_COUNT; p.param.smCount.val = new_sms;
  const char* ctx_entry = nullptr;
  CUresult rc = bless_ctx_create(&bless::ctx_limited, &p, 1, 0, bless::dev,
                                 &ctx_entry);
  if (rc != CUDA_SUCCESS || bless::ctx_limited == nullptr) {
    // [Exp_41 O-3] 조용한 실패 금지: SM exec affinity 는 MPS 하에서만 지원 —
    // 생성 실패를 감지해 폴백 상태를 기록하고 명확히 경고한다.
    bless::ctx_limited = nullptr;
    bless::limited_applied.store(0);
    const char* en = nullptr; cuGetErrorName(rc, &en);
    fprintf(stderr, "[libbless] limited ctx FALLBACK-unlimited: requested sm=%d "
                    "생성 실패 (%s) — SM affinity 는 MPS 필요 (O-3). 공간 제한 미적용\n",
            new_sms, en ? en : "unknown");
  } else {
    int got = -1; CUcontext prev = nullptr;
    cuCtxPushCurrent(bless::ctx_limited);
    CUexecAffinityParam q{}; q.type = CU_EXEC_AFFINITY_TYPE_SM_COUNT;
    if (cuCtxGetExecAffinity(&q, CU_EXEC_AFFINITY_TYPE_SM_COUNT) == CUDA_SUCCESS)
      got = q.param.smCount.val;
    cuCtxPopCurrent(&prev);
    bless::limited_applied.store(1);
    // 요청값 echo 금지 — 적용값을 조회해 함께 출력 (Exp_41)
    fprintf(stderr, "[libbless] limited ctx applied sm_affinity=%d (verified=%d, entry=%s)\n",
            new_sms, got, ctx_entry ? ctx_entry : "?");
  }
  bless::limited_sms.store(new_sms);
  bless::gate.store(0);
}

static void ensure_init() {
  if (bless::inited.load()) return;
  pthread_mutex_lock(&bless::init_mu);
  if (bless::inited.load()) { pthread_mutex_unlock(&bless::init_mu); return; }

  cuInit(0);
  cuDeviceGet(&bless::dev, 0);
  cuDeviceGetAttribute(&bless::total_sms, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, bless::dev);

  // [Exp_61] 공간 제한 위임 방식 선택. 기본 ctx(하위호환 — 기존 동작 무변경).
  // 미지의 값은 무음 수용 금지(PRS2 정신) — 경고 후 기본값.
  if (const char* m = getenv("BLESS_SPACE_MODE")) {
    if (!strcmp(m, "mps_env")) {
      bless::space_mode.store(bless::SPACE_MPS_ENV);
    } else if (strcmp(m, "ctx") != 0) {
      fprintf(stderr, "[libbless] BLESS_SPACE_MODE=%s 인식 불가 — 기본 ctx 사용 "
                      "(유효값: ctx|mps_env)\n", m);
    }
  }

  int pct = 50;
  if (const char* e = getenv("BLESS_LIMIT_PCT")) { int v=atoi(e); if (v>0 && v<100) pct = v; }
  bless::requested_pct.store(pct);
  if (bless::space_mode.load() == bless::SPACE_MPS_ENV && getenv("BLESS_LIMIT_PCT"))
    fprintf(stderr, "[libbless] 경고: space_mode=mps_env 에서 BLESS_LIMIT_PCT 는 "
                    "공간 제한에 사용되지 않음 — CUDA_MPS_ACTIVE_THREAD_PERCENTAGE 로 지정할 것\n");
  int init_sms = (int)(bless::total_sms * (pct/100.0f));
  if (init_sms < 1) init_sms = 1;
  bless::limited_sms.store(init_sms);

  // UNLIMITED ctx
  (void)bless_ctx_create(&bless::ctx_unlimited, nullptr, 0, 0, bless::dev, nullptr);
  { CUcontext prev=nullptr; cuCtxPushCurrent(bless::ctx_unlimited);
    CUexecAffinityParam q{}; q.type = CU_EXEC_AFFINITY_TYPE_SM_COUNT;
    if (cuCtxGetExecAffinity(&q, CU_EXEC_AFFINITY_TYPE_SM_COUNT)==CUDA_SUCCESS)
      fprintf(stderr, "[libbless] unlimited ctx sm_affinity=%d\n", q.param.smCount.val);
    cuCtxPopCurrent(&prev);
  }

  // LIMITED ctx
  reconf_limited_ctx(init_sms);

  // control socket
  char sp[128]; snprintf(sp, sizeof(sp), "/tmp/bless-%d.sock", (int)getpid());
  bless::sock_path = sp;
  start_control_server(bless::sock_path);

  // wait ready
  for (int i=0;i<60 && !bless::ctrl_ready.load();++i) { usleep(50*1000); }

  // attach cudart upfront
  attach_runtime_if_needed();

  // optional master
  if (const char* mp = getenv("BLESS_MASTER")) {
    bless::master_path = mp;
    bless::master_fd = socket(AF_UNIX, SOCK_DGRAM, 0);
    if (bless::master_fd >= 0) {
      bless::tenant_id = getenv("BLESS_TENANT") ? getenv("BLESS_TENANT") : "";
      char hello[256];
      snprintf(hello, sizeof(hello), "HELLO pid=%d sock=%s tenant=%s",
               (int)getpid(), bless::sock_path.c_str(), bless::tenant_id.c_str());
      master_send(hello);
    }
  }

  // mem quota — env 소비 (Exp_51 O-5): BLESS_MEM_QUOTA_MB 는 device-plugin 이
  // Allocate 때 주입하는 채널(Exp_33)인데 기존에는 소켓 명령(mem_quota)만 읽어
  // K8s 경로 전체가 무집행이었다. 소켓 명령은 이후에도 우선(런타임 재설정).
  if (const char* q = getenv("BLESS_MEM_QUOTA_MB")) {
    long long mb = atoll(q);
    if (mb > 0) {
      bless::mem_quota_bytes.store(mb * 1024LL * 1024LL, std::memory_order_release);
      fprintf(stderr, "[libbless] mem_quota=%lld MB (env)\n", mb);
    }
  }

  // [Exp_61] sm_limit 3상태 — 판정은 이 로그로 한다(Exp_60 측정 함정 교훈):
  //   applied / FALLBACK-unlimited (O-3) / delegated-mps-env
  fprintf(stderr, "[libbless] init: total_sms=%d, limited=%d, sm_limit=%s, sock=%s\n",
          bless::total_sms, bless::limited_sms.load(),
          bless::space_mode.load() == bless::SPACE_MPS_ENV ? "delegated-mps-env"
            : bless::limited_applied.load() ? "applied" : "FALLBACK-unlimited (O-3)",
          bless::sock_path.c_str());
  // [Exp_61] O-7 예방 breadcrumb: 훈련 워크로드 감지는 launch 파라미터에 커널
  // 이름/형상이 없어 불가(Exp_13/48) — 대신 위험 조건(ctx 실적용)에서 1줄 고지.
  if (bless::limited_applied.load())
    fprintf(stderr, "[libbless] notice: ctx affinity 적용 상태 — 훈련 워크로드는 "
                    "hang 위험(O-7). 훈련은 BLESS_SPACE_MODE=mps_env 권장\n");

  // [Exp_73] time_stats 파일 발화 경로 — feeder occupancy 가 이 파일을 읽는다.
  if (const char* sl = getenv("BLESS_STATS_LOG")) {
    if (sl[0]) {
      bless::stats_log = fopen(sl, "a");
      if (bless::stats_log)
        fprintf(stderr, "[libbless] stats_log=%s (feeder occupancy 배선)\n", sl);
    }
  }

  // [Exp_75] 워크로드 클래스 선언 파일 발화 — relaxed 자동 분류용. 파드가 env
  // PRISM_WORKLOAD_CLASS=compute|memory 로 선언하면 여기서 class 파일에 1회 기록,
  // bless_feeder(wirer)가 쌍 판정(aggressor+victim 공존 시 victim 해제)에 쓴다.
  // ★선언 기반이라 남용 가능(memory 선언=gate 해제 유리) — 관측 검증은 MPS 하
  //   DCGM 프로세스 귀속 불가로 미구현(Exp_75 A-1). 안전 기본값은 미선언=strict.
  if (const char* cl = getenv("BLESS_CLASS_LOG")) {
    const char* wc = getenv("PRISM_WORKLOAD_CLASS");
    if (cl[0] && wc && wc[0]) {
      if (FILE* cf = fopen(cl, "w")) {
        fprintf(cf, "%s\n", wc);
        fclose(cf);
        fprintf(stderr, "[libbless] workload_class=%s → %s (relaxed 분류 선언)\n", wc, cl);
      }
    }
  }

  bless::inited.store(true);
  pthread_mutex_unlock(&bless::init_mu);
}

static void start_control_server(const std::string& path) {
  bless::ctrl_running.store(true);
  bless::ctrl_thread = std::thread([path]{
    int fd = socket(AF_UNIX, SOCK_DGRAM, 0);
    if (fd < 0) return;
    bless::g_ctrl_fd = fd;
    struct timeval tv{0,200000}; setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

    sockaddr_un addr{}; addr.sun_family = AF_UNIX;
    snprintf(addr.sun_path, sizeof(addr.sun_path), "%s", path.c_str());
    unlink(addr.sun_path);
    if (bind(fd, (sockaddr*)&addr, sizeof(addr)) < 0) { close(fd); return; }

    bless::ctrl_ready.store(true);

    char buf[128];
    while (bless::ctrl_running.load()) {
      ssize_t n = recv(fd, buf, sizeof(buf)-1, 0);
      if (n < 0) continue;
      buf[n]=0;

      // -------- route/boost/pause ----------
      if (!strncmp(buf,"limited",7)) {
        if (!bless::any_alloc.load()) bless::route.store(bless::LIMITED);
      }
      else if (!strncmp(buf,"unlimited",9)) {
        if (!bless::any_alloc.load()) bless::route.store(bless::UNLIMITED);
      }
      else if (!strncmp(buf,"set_route ",10)) {
        int r = atoi(buf+10);
        if (!bless::any_alloc.load()) {
          bless::route.store((r==1)?bless::UNLIMITED:bless::LIMITED);
        } else {
          fprintf(stderr, "[libbless] set_route ignored after allocations\n");
        }
      }
      else if (!strncmp(buf,"pause",5))   { bless::pause_mode.store(true); }
      else if (!strncmp(buf,"resume",6))  { bless::pause_mode.store(false); }
      else if (!strncmp(buf,"boost_on",8)) {
        bless::boost_mode.store(true);
        if (bless::master_fd >= 0) {
          char msg[160];
          snprintf(msg, sizeof(msg), "BE pid=%d t=%s what=BOOST_ON",
                   (int)getpid(), bless::tenant_id.c_str());
          master_send(msg);
        }
      }
      else if (!strncmp(buf,"boost_off",9)) {
        bless::boost_mode.store(false);
        if (bless::master_fd >= 0) {
          char msg[160];
          snprintf(msg, sizeof(msg), "BE pid=%d t=%s what=BOOST_OFF",
                   (int)getpid(), bless::tenant_id.c_str());
          master_send(msg);
        }
      }

      // -------- squad/share ----------
      else if (!strncmp(buf,"set_squad ",10)){
        int k = atoi(buf+10); if (k>0){ bless::squad_size.store(k); }
      }
      else if (!strncmp(buf,"set_share ",10)){
        int q = atoi(buf+10);
        if (q >= 0){
          bless::share_quota.store(q);
          if (q == 0) bless::sd_sent.store(0);
        }
      }
      else if (!strncmp(buf,"squad_reset",11)){
        bless::squad_prog.store(0);
        bless::sd_sent.store(0);
      }

      // -------- credit gate (커널 개수 기반) ----------
      else if (!strncmp(buf,"credit_set ",11)) {
        int q = atoi(buf+11);
        bless::credit_remain.store(q, std::memory_order_release);
      }
      else if (!strncmp(buf,"credit_off",10)) {
        bless::credit_remain.store(-1, std::memory_order_release);
      }

      // -------- 시간 기반 크레딧 (v2) ----------
      // 모드 전환: time_mode 0=커널개수, 1=시간
      else if (!strncmp(buf,"time_mode ",10)) {
        int m = atoi(buf+10);
        bless::credit_mode.store(m, std::memory_order_release);
        fprintf(stderr, "[libbless] credit_mode=%d (%s)\n", m, m?"TIME":"KERNEL");
      }
      // 시간 크레딧 설정 (마이크로초). [Exp_16 보강 ①] 음수 = 무제한 플래그로 분리
      else if (!strncmp(buf,"time_credit ",12)) {
        int64_t t = atoll(buf+12);
        if (t < 0) {
          bless::time_unlimited.store(true, std::memory_order_release);
          bless::time_credit_us.store(-1, std::memory_order_release);
          fprintf(stderr, "[libbless] time_credit=unlimited\n");
        } else {
          bless::time_unlimited.store(false, std::memory_order_release);
          bless::time_credit_us.store(t, std::memory_order_release);
          fprintf(stderr, "[libbless] time_credit=%lld us\n", (long long)t);
        }
      }
      // 시간 크레딧 무제한
      else if (!strncmp(buf,"time_unlimited",14)) {
        bless::time_unlimited.store(true, std::memory_order_release);
        bless::time_credit_us.store(-1, std::memory_order_release);
        fprintf(stderr, "[libbless] time_credit=unlimited\n");
      }
      // 타임슬라이스 설정 (마이크로초)
      else if (!strncmp(buf,"time_slice ",11)) {
        int64_t s = atoll(buf+11);
        if (s > 0) {
          bless::time_slice_us.store(s, std::memory_order_release);
          fprintf(stderr, "[libbless] time_slice=%lld us\n", (long long)s);
        }
      }
      // 시간 크레딧 추가 (가변 조정용)
      else if (!strncmp(buf,"time_add ",9)) {
        int64_t add = atoll(buf+9);
        bless::time_credit_us.fetch_add(add, std::memory_order_acq_rel);
      }
      // 통계 조회
      else if (!strncmp(buf,"time_stats",10)) {
        fprintf(stderr, "[libbless] mode=%d time_credit=%lld slice=%lld avg_kernel=%lld total=%lld kernels=%lld\n",
                bless::credit_mode.load(),
                (long long)bless::time_credit_us.load(),
                (long long)bless::time_slice_us.load(),
                (long long)bless::avg_kernel_time_us.load(),
                (long long)bless::total_time_us.load(),
                (long long)bless::kernel_seq.load());
        // [Exp_73] feeder occupancy 파싱용 파일 발화 (read_time_stats 정규식 호환:
        // "total=<μs> kernels=<n>"). 매번 flush — 피더가 즉시 읽는다.
        if (bless::stats_log) {
          fprintf(bless::stats_log, "total=%lld kernels=%lld\n",
                  (long long)bless::total_time_us.load(),
                  (long long)bless::kernel_seq.load());
          fflush(bless::stats_log);
        }
      }

      // -------- 컨텍스트 스위칭 통계 ----------
      else if (!strncmp(buf,"ctx_stats",9)) {
        fprintf(stderr, "[libbless] ctx_switch: count=%lld total_us=%lld avg_us=%lld max_us=%lld\n",
                (long long)bless::ctx_switch_count.load(),
                (long long)bless::ctx_switch_total_us.load(),
                (long long)bless::ctx_switch_avg_us.load(),
                (long long)bless::ctx_switch_max_us.load());
      }

      // -------- 메모리 추적 제어 (S-06) ----------
      // 메모리 통계 조회
      else if (!strncmp(buf,"mem_stats",9)) {
        fprintf(stderr, "[libbless] mem: alloc_count=%lld free_count=%lld current_MB=%.2f peak_MB=%.2f total_MB=%.2f quota_MB=%.2f\n",
                (long long)bless::mem_alloc_count.load(),
                (long long)bless::mem_free_count.load(),
                bless::mem_current_bytes.load() / (1024.0*1024.0),
                bless::mem_peak_bytes.load() / (1024.0*1024.0),
                bless::mem_total_alloc_bytes.load() / (1024.0*1024.0),
                bless::mem_quota_bytes.load() < 0 ? -1.0 : bless::mem_quota_bytes.load() / (1024.0*1024.0));
      }
      // 메모리 quota 설정 (MB 단위, -1=무제한)
      else if (!strncmp(buf,"mem_quota ",10)) {
        int64_t mb = atoll(buf+10);
        if (mb < 0) {
          bless::mem_quota_bytes.store(-1, std::memory_order_release);
          fprintf(stderr, "[libbless] mem_quota=UNLIMITED\n");
        } else {
          bless::mem_quota_bytes.store(mb * 1024 * 1024, std::memory_order_release);
          fprintf(stderr, "[libbless] mem_quota=%lld MB\n", (long long)mb);
        }
      }
      // 메모리 추적 on/off
      else if (!strncmp(buf,"mem_track ",10)) {
        int enabled = atoi(buf+10);
        bless::mem_tracking_enabled.store(enabled != 0, std::memory_order_release);
        fprintf(stderr, "[libbless] mem_tracking=%s\n", enabled ? "ON" : "OFF");
      }
      // 메모리 통계 리셋
      else if (!strncmp(buf,"mem_reset",9)) {
        bless::mem_alloc_count.store(0);
        bless::mem_free_count.store(0);
        bless::mem_current_bytes.store(0);
        bless::mem_peak_bytes.store(0);
        bless::mem_total_alloc_bytes.store(0);
        pthread_mutex_lock(&bless::alloc_map_mu);
        bless::alloc_size_map.clear();
        pthread_mutex_unlock(&bless::alloc_map_mu);
        fprintf(stderr, "[libbless] mem_stats RESET\n");
      }

      // -------- 우선순위 큐 제어 ----------
      // 우선순위 설정: priority_set 0=URGENT, 1=NORMAL, 2=BACKGROUND
      else if (!strncmp(buf,"priority_set ",13)) {
        int p = atoi(buf+13);
        if (p >= 0 && p < bless::NUM_PRIORITIES) {
          bless::current_priority.store(p, std::memory_order_release);
          fprintf(stderr, "[libbless] priority=%d (%s)\n", p,
                  p==0 ? "URGENT" : (p==1 ? "NORMAL" : "BACKGROUND"));
        }
      }
      // Urgent 선점 요청
      else if (!strncmp(buf,"urgent_request",14)) {
        bless::urgent_pending.store(true, std::memory_order_release);
        bless::preempt_requested.store(true, std::memory_order_release);
        fprintf(stderr, "[libbless] urgent preemption requested\n");
      }
      // Urgent 선점 해제
      else if (!strncmp(buf,"urgent_clear",12)) {
        bless::urgent_pending.store(false, std::memory_order_release);
        bless::preempt_requested.store(false, std::memory_order_release);
        fprintf(stderr, "[libbless] urgent preemption cleared\n");
      }
      // 큐 통계 조회
      else if (!strncmp(buf,"queue_stats",11)) {
        fprintf(stderr, "[libbless] queues: urgent=%lld normal=%lld background=%lld current_priority=%d\n",
                (long long)bless::queue_len[bless::URGENT].load(),
                (long long)bless::queue_len[bless::NORMAL].load(),
                (long long)bless::queue_len[bless::BACKGROUND].load(),
                bless::current_priority.load());
      }

      // -------- System Call 인터셉션 ----------
      // 인터셉션 활성화/비활성화
      else if (!strncmp(buf,"syscall_on",10)) {
        bless::syscall_intercept_enabled.store(true, std::memory_order_release);
        fprintf(stderr, "[libbless] syscall interception enabled\n");
      }
      else if (!strncmp(buf,"syscall_off",11)) {
        bless::syscall_intercept_enabled.store(false, std::memory_order_release);
        fprintf(stderr, "[libbless] syscall interception disabled\n");
      }
      // System call 통계 조회
      else if (!strncmp(buf,"syscall_stats",13)) {
        fprintf(stderr, "[libbless] syscall: ioctl=%lld nvidia_ioctl=%lld mmap=%lld gpu_mmap=%lld total_mmap_bytes=%lld\n",
                (long long)bless::ioctl_count.load(),
                (long long)bless::ioctl_nvidia_count.load(),
                (long long)bless::mmap_count.load(),
                (long long)bless::mmap_gpu_count.load(),
                (long long)bless::mmap_total_bytes.load());
      }

      // [Exp_41 O-3] 공간 노브 실효 상태 조회 (syscall_stats 패턴)
      else if (!strncmp(buf,"sm_stats",8)) {
        fprintf(stderr, "[libbless] sm: space_mode=%s requested_pct=%d limited_sms=%d "
                        "applied=%d fallback_launches=%lld total_sms=%d\n",
                bless::space_mode.load()==bless::SPACE_MPS_ENV ? "mps_env" : "ctx",
                bless::requested_pct.load(), bless::limited_sms.load(),
                bless::limited_applied.load(),
                (long long)bless::fallback_launches.load(), bless::total_sms);
      }

      // -------- reconf ----------
      // [Exp_61] mps_env 모드에서는 공간 재설정이 프로세스 밖(MPS env, 스폰 시점
      // 고정)에 있으므로 런타임 재설정 요청을 무음 no-op 이 아니라 명시 거부한다.
      else if (!strncmp(buf,"reconf_sm ",10) || !strncmp(buf,"set_limit_pct ",14)) {
        if (bless::space_mode.load() == bless::SPACE_MPS_ENV) {
          fprintf(stderr, "[libbless] REJECT %s: space_mode=mps_env 는 런타임 공간 "
                          "재설정 불가(프로세스 단위 env) — 재스폰으로 처리할 것\n", buf);
        } else if (!strncmp(buf,"reconf_sm ",10)) {
          int k = atoi(buf+10); if (k>0) reconf_limited_ctx(k);
        } else {
          int pct = atoi(buf+14);
          if (pct < 1) pct = 1; if (pct > 99) pct = 99;
          bless::requested_pct.store(pct);
          int sms = (int)(bless::total_sms * (pct/100.0f));
          if (sms < 1) sms = 1;
          reconf_limited_ctx(sms);
        }
      }

      else if (!strncmp(buf,"quit",4)) break;
    }
    close(fd); unlink(path.c_str());
  });
}

// --------- credit gate (경량/버스트 캐시) ----------
static inline void kernel_credit_gate() {
  // boost 시에는 게이트 우회
  if (__builtin_expect(bless::boost_mode.load(std::memory_order_relaxed), 0)) return;

  int cr = bless::credit_remain.load(std::memory_order_relaxed);
  if (__builtin_expect(cr < 0, 1)) return; // 무제한

  // [Exp_55 결정 — Fix2/Fix3 영구 기각] Exp_7 이 제안한 Fix2(청크 cur>>2)·
  // Fix3(credit_refill 이월+cap)는 이 커널-수 경로의 정밀도 보정안이었다.
  // 기각 근거(3): ① 제어 평면(controller feeder·device-plugin·bench)이
  // credit_set/credit_refill 을 호출하지 않는다 — 실사용 경로는 전부 TIME 모드
  // (time_mode 1 + time_add). ② Fix3 가 겨냥한 "덮어쓰기로 잔량 손실"은 시간
  // 경로에 구조적으로 없다(time_add=누적, Exp_16 보강 ② 로 적자도 이월).
  // ③ 시간 경로 정확도가 이미 목표 달성: 2:1/1:2/3:1/★10:1 오차 0.0002~0.0008
  // (Exp_55 §Phase 1). → 보류 종결. 커널 경로를 되살릴 일이 생기면 그때 재평가.
  static thread_local int tl_burst = 0;
  if (__builtin_expect(tl_burst > 0, 1)) { --tl_burst; return; }

  constexpr int BURST = 16;
  int waited = 0;

  // [Exp_7 Fix 1] 언더플로 클램프: fetch_sub(BURST)+undo는 잔량<BURST일 때
  // credit_remain을 음수로 만들고, 그 음수가 다음 게이트에서 cr<0 "무제한"으로
  // 오인되어 비례가 깨졌다. CAS로 정확히 min(cur,BURST)만 차감하여 0 미만으로
  // 내려가지 않게 한다. 이제 cr<0는 오직 credit_off(-1) 센티넬에서만 발생.
  while (true) {
    int cur = bless::credit_remain.load(std::memory_order_acquire);
    if (cur < 0) return;            // -1 센티넬 = 무제한(credit_off)
    if (cur > 0) {
      int chunk = (cur >= BURST) ? BURST : cur;   // 절대 초과 차감 안 함
      if (bless::credit_remain.compare_exchange_weak(
              cur, cur - chunk, std::memory_order_acq_rel)) {
        tl_burst = chunk - 1;
        return;
      }
      continue;                     // CAS 경쟁 → 재시도
    }
    // cur == 0 : 소진 → 다음 refill 대기
    if (waited < 50) { ++waited; sched_yield(); }
    else {
      struct timespec ts{0, 20000}; /* 20µs */
      nanosleep(&ts, nullptr);
      waited = 0;
    }
  }
}

// --------- 우선순위 큐 게이트 (선점 포함) ----------
// 현재 워크로드의 우선순위에 따라 실행 여부 결정
// Urgent 선점 요청이 있으면 낮은 우선순위 커널은 대기
static inline void priority_queue_gate() {
  int my_priority = bless::current_priority.load(std::memory_order_relaxed);

  // 큐 길이 증가 (커널 진입)
  bless::queue_len[my_priority].fetch_add(1, std::memory_order_relaxed);

  // Urgent 선점 체크: 내가 Urgent가 아니고, Urgent가 대기 중이면 양보
  if (my_priority != bless::URGENT) {
    int waited = 0;
    while (bless::urgent_pending.load(std::memory_order_acquire)) {
      // Urgent 대기 중 - 내 큐 길이 감소 후 대기
      bless::queue_len[my_priority].fetch_sub(1, std::memory_order_relaxed);

      if (waited < 100) {
        ++waited;
        sched_yield();
      } else {
        struct timespec ts{0, 50000}; // 50us
        nanosleep(&ts, nullptr);
        waited = 0;
      }

      // 큐 길이 다시 증가
      bless::queue_len[my_priority].fetch_add(1, std::memory_order_relaxed);
    }
  }

  // Background는 Normal/Urgent가 없을 때만 실행
  if (my_priority == bless::BACKGROUND) {
    int waited = 0;
    while (bless::queue_len[bless::URGENT].load(std::memory_order_relaxed) > 0 ||
           bless::queue_len[bless::NORMAL].load(std::memory_order_relaxed) > bless::queue_len[bless::BACKGROUND].load(std::memory_order_relaxed)) {
      if (waited < 50) {
        ++waited;
        sched_yield();
      } else {
        struct timespec ts{0, 100000}; // 100us
        nanosleep(&ts, nullptr);
        waited = 0;
      }

      // 타임아웃: 최대 10ms 대기
      if (waited > 100) break;
    }
  }
}

// 큐 길이 감소 (커널 완료)
static inline void priority_queue_done() {
  int my_priority = bless::current_priority.load(std::memory_order_relaxed);
  int64_t prev = bless::queue_len[my_priority].fetch_sub(1, std::memory_order_relaxed);
  if (prev <= 0) {
    // 언더플로우 방지
    bless::queue_len[my_priority].store(0, std::memory_order_relaxed);
  }
}

// ------------- interceptors -------------
extern "C" cudaError_t cudaLaunchKernel(const void *hostFunc,
                                        dim3 gridDim, dim3 blockDim,
                                        void **args, size_t sharedMem,
                                        cudaStream_t stream)
{
  ensure_init(); attach_runtime_if_needed();

  priority_queue_gate();

  // reconf gate / pause
  if (bless::route.load()==bless::LIMITED) {
    int g = bless::gate.load(std::memory_order_acquire);
    if (g!=0) { while ((g = bless::gate.load(std::memory_order_acquire))!=0) sched_yield(); }
    while (bless::pause_mode.load(std::memory_order_acquire)) sched_yield();
  }
  if (bless::preempt_requested.load(std::memory_order_acquire) &&
      bless::current_priority.load(std::memory_order_relaxed) != bless::URGENT) {
    priority_queue_done();
    sched_yield();
    priority_queue_gate();
  }
  int mode = bless::credit_mode.load(std::memory_order_relaxed);
  if (mode == 1) {
    time_credit_gate();
    time_batch_start();
  } else {
    kernel_credit_gate();
  }

  long long kseq = bless::kernel_seq.fetch_add(1) + 1;
  int sp = bless::squad_prog.fetch_add(1) + 1;
  if (sp >= bless::squad_size.load()) {
    bless::squad_prog.store(0);
  }
  int quota = bless::share_quota.load(std::memory_order_relaxed);
  if (quota > 0 && sp >= quota) {
    int was = bless::sd_sent.exchange(1);
    if (was == 0 && bless::master_fd >= 0) {
      char msg[200];
      snprintf(msg, sizeof(msg), "SD pid=%d t=%s sp=%d kseq=%lld",
               (int)getpid(), bless::tenant_id.c_str(), sp, (long long)kseq);
      master_send(msg);
    }
  }

  ScopedCtxGuard s; // route만 고려 (boost는 경로에 영향 X)
  cudaError_t result = real_cudaLaunchKernel(hostFunc, gridDim, blockDim, args, sharedMem, stream);

  // 커널 완료 후 큐 길이 감소
  priority_queue_done();

  return result;
}

extern "C" CUresult cuLaunchKernel(CUfunction f,
                                   unsigned int gridX, unsigned int gridY, unsigned int gridZ,
                                   unsigned int blockX, unsigned int blockY, unsigned int blockZ,
                                   unsigned int sharedMemBytes,
                                   CUstream hStream,
                                   void **kernelParams, void **extra)
{
  ensure_init(); attach_runtime_if_needed();

  // 우선순위 큐 게이트 (선점 체크 포함)
  priority_queue_gate();

  if (bless::route.load()==bless::LIMITED) {
    int g = bless::gate.load(std::memory_order_acquire);
    if (g!=0) { while ((g = bless::gate.load(std::memory_order_acquire))!=0) sched_yield(); }
    while (bless::pause_mode.load(std::memory_order_acquire)) sched_yield();
  }

  // 선점 요청 확인
  if (bless::preempt_requested.load(std::memory_order_acquire) &&
      bless::current_priority.load(std::memory_order_relaxed) != bless::URGENT) {
    priority_queue_done();
    sched_yield();
    priority_queue_gate();
  }

  // 크레딧 게이트 (모드에 따라 선택)
  int mode = bless::credit_mode.load(std::memory_order_relaxed);
  if (mode == 1) {
    time_credit_gate();
    time_batch_start();
  } else {
    kernel_credit_gate();
  }

  long long kseq = bless::kernel_seq.fetch_add(1) + 1;
  int sp = bless::squad_prog.fetch_add(1) + 1;
  if (sp >= bless::squad_size.load()) {
    bless::squad_prog.store(0);
  }

  int quota = bless::share_quota.load(std::memory_order_relaxed);
  if (quota > 0 && sp >= quota) {
    int was = bless::sd_sent.exchange(1);
    if (was == 0 && bless::master_fd >= 0) {
      char msg[200];
      snprintf(msg, sizeof(msg), "SD pid=%d t=%s sp=%d kseq=%lld",
               (int)getpid(), bless::tenant_id.c_str(), sp, (long long)kseq);
      master_send(msg);
    }
  }

  ScopedCtxGuard s;
  CUresult result = real_cuLaunchKernel(f, gridX, gridY, gridZ,
                             blockX, blockY, blockZ,
                             sharedMemBytes, hStream, kernelParams, extra);

  // 커널 완료 후 큐 길이 감소
  priority_queue_done();

  return result;
}

// alloc/mem/stream/graph/sync pass-throughs
extern "C" cudaError_t cudaMalloc(void **p, size_t n){
  ensure_init(); attach_runtime_if_needed();
  bless::any_alloc.store(true, std::memory_order_release);

  // Quota 체크 (S-03/S-04 연동)
  if (!bless::check_quota(n)) {
    fprintf(stderr, "[libbless] cudaMalloc REJECTED: size=%zu exceeds quota (current=%lld, quota=%lld)\n",
            n, (long long)bless::mem_current_bytes.load(), (long long)bless::mem_quota_bytes.load());
    return cudaErrorMemoryAllocation;
  }

  ScopedCtxGuard s;
  cudaError_t ret = real_cudaMalloc(p, n);

  // 성공 시 메모리 추적
  if (ret == cudaSuccess && p && *p) {
    bless::track_alloc(*p, n);
  }
  return ret;
}
extern "C" cudaError_t cudaFree(void *devPtr){
  ensure_init(); attach_runtime_if_needed();

  // 메모리 추적 (해제 전에 기록)
  bless::track_free(devPtr);

  CUcontext owner=nullptr;
  CUresult r=cuPointerGetAttribute(&owner, CU_POINTER_ATTRIBUTE_CONTEXT, (CUdeviceptr)devPtr);
  if (r==CUDA_SUCCESS && owner){
    CUcontext prev=nullptr; cuCtxPushCurrent(owner);
    cudaError_t e=real_cudaFree(devPtr);
    cuCtxPopCurrent(&prev); return e;
  }
  ScopedCtxGuard s; return real_cudaFree(devPtr);
}
extern "C" cudaError_t cudaMallocManaged(void **p, size_t n, unsigned int f){
  ensure_init(); attach_runtime_if_needed();
  bless::any_alloc.store(true, std::memory_order_release);

  // Quota 체크
  if (!bless::check_quota(n)) {
    fprintf(stderr, "[libbless] cudaMallocManaged REJECTED: size=%zu exceeds quota\n", n);
    return cudaErrorMemoryAllocation;
  }

  ScopedCtxGuard s;
  cudaError_t ret = real_cudaMallocManaged(p, n, f);

  // 성공 시 메모리 추적
  if (ret == cudaSuccess && p && *p) {
    bless::track_alloc(*p, n);
  }
  return ret;
}
extern "C" cudaError_t cudaMemcpy(void *d,const void *s,size_t c,cudaMemcpyKind k){
  ensure_init(); return real_cudaMemcpy(d,s,c,k);
}
extern "C" cudaError_t cudaMemcpyAsync(void *d,const void *s,size_t c,cudaMemcpyKind k,cudaStream_t st){
  ensure_init(); return real_cudaMemcpyAsync(d,s,c,k,st);
}
extern "C" cudaError_t cudaStreamCreate(cudaStream_t *st){
  ensure_init(); attach_runtime_if_needed();
  return real_cudaStreamCreate(st);
}
extern "C" cudaError_t cudaStreamDestroy(cudaStream_t st){
  ensure_init(); attach_runtime_if_needed();
  return real_cudaStreamDestroy(st);
}
extern "C" cudaError_t cudaGraphLaunch(cudaGraphExec_t g, cudaStream_t st){
  ensure_init(); return real_cudaGraphLaunch(g, st);
}
extern "C" cudaError_t cudaDeviceSynchronize(){
  ensure_init(); attach_runtime_if_needed();
  cudaError_t ret = real_cudaDeviceSynchronize();

  // 시간 모드에서 동기화 시점에 실제 시간 측정 및 크레딧 차감
  if (bless::credit_mode.load(std::memory_order_relaxed) == 1) {
    time_batch_end_and_charge();
  }

  return ret;
}
extern "C" cudaError_t cudaStreamSynchronize(cudaStream_t st){
  ensure_init(); attach_runtime_if_needed();
  cudaError_t ret = real_cudaStreamSynchronize(st);

  // 시간 모드에서 동기화 시점에 실제 시간 측정 및 크레딧 차감
  if (bless::credit_mode.load(std::memory_order_relaxed) == 1) {
    time_batch_end_and_charge();
  }

  return ret;
}

// queries / helpers
extern "C" __attribute__((visibility("default"))) int bless_query_sm_affinity(){
  ensure_init();
  ScopedCtxGuard s;
  CUexecAffinityParam q{}; q.type=CU_EXEC_AFFINITY_TYPE_SM_COUNT;
  if (cuCtxGetExecAffinity(&q, CU_EXEC_AFFINITY_TYPE_SM_COUNT)==CUDA_SUCCESS) return (int)q.param.smCount.val;
  return -1;
}
extern "C" __attribute__((visibility("default")))
void bless_bind_thread(int route) {
  ensure_init(); attach_runtime_if_needed();
  int r = (route==1) ? bless::UNLIMITED : bless::LIMITED;
  // any_alloc 이후 route 전환은 무시되므로, 여기서는 "설정된 route"로만 바인딩
  CUcontext target = (r==bless::UNLIMITED) ? bless::ctx_unlimited : bless::ctx_limited;
  CUcontext cur = nullptr;
  cuCtxGetCurrent(&cur);
  if (cur == target) return;
  CUresult st = cuCtxSetCurrent(target);
  if (st != CUDA_SUCCESS) {
    CUcontext prev = nullptr;
    cuCtxPushCurrent(target);
    cuCtxPopCurrent(&prev);
    cuCtxSetCurrent(target);
  }
}
extern "C" __attribute__((visibility("default"))) int bless_current_route(){
  ensure_init(); return bless::route.load();
}
extern "C" __attribute__((visibility("default"))) int bless_squad_progress(){
  return bless::squad_prog.load();
}
extern "C" __attribute__((visibility("default"))) long long bless_kernel_seq(){
  return bless::kernel_seq.load();
}
extern "C" __attribute__((visibility("default"))) int bless_is_boosting(){
  return bless::boost_mode.load() ? 1 : 0;
}

// ---- 시간 기반 크레딧 쿼리 API ----
extern "C" __attribute__((visibility("default"))) int bless_get_credit_mode(){
  return bless::credit_mode.load();
}
extern "C" __attribute__((visibility("default"))) long long bless_get_time_credit_us(){
  return bless::time_credit_us.load();
}
extern "C" __attribute__((visibility("default"))) long long bless_get_time_slice_us(){
  return bless::time_slice_us.load();
}
extern "C" __attribute__((visibility("default"))) long long bless_get_avg_kernel_time_us(){
  return bless::avg_kernel_time_us.load();
}
extern "C" __attribute__((visibility("default"))) long long bless_get_total_time_us(){
  return bless::total_time_us.load();
}

// ---- System Call 인터셉션 API ----
extern "C" __attribute__((visibility("default"))) long long bless_get_ioctl_count(){
  return bless::ioctl_count.load();
}
extern "C" __attribute__((visibility("default"))) long long bless_get_mmap_count(){
  return bless::mmap_count.load();
}
extern "C" __attribute__((visibility("default"))) long long bless_get_mmap_total_bytes(){
  return bless::mmap_total_bytes.load();
}

// NVIDIA ioctl magic numbers (from nvidia-uvm)
#define NV_IOCTL_MAGIC      0x46  // 'F' for nvidia frontend
#define NV_ESC_CARD_INFO    0x00
#define NV_ESC_REGISTER_FD  0x02

// ---- ioctl 인터셉터 ----
// NVIDIA GPU 드라이버 ioctl 호출을 추적
extern "C" int ioctl(int fd, unsigned long request, ...) {
  // 실제 함수 해결
  if (!real_ioctl) {
    real_ioctl = (ioctl_func_t) dlsym(RTLD_NEXT, "ioctl");
    if (!real_ioctl) {
      errno = ENOSYS;
      return -1;
    }
  }

  // 가변 인자 처리
  va_list args;
  va_start(args, request);
  void* arg = va_arg(args, void*);
  va_end(args);

  // 인터셉션이 활성화된 경우에만 통계 수집
  if (bless::syscall_intercept_enabled.load(std::memory_order_relaxed)) {
    bless::ioctl_count.fetch_add(1, std::memory_order_relaxed);

    // NVIDIA ioctl 감지 (magic number 확인)
    unsigned int magic = (request >> 8) & 0xFF;
    if (magic == NV_IOCTL_MAGIC || magic == 0x2A) { // NVIDIA frontend or UVM
      bless::ioctl_nvidia_count.fetch_add(1, std::memory_order_relaxed);
    }
  }

  return real_ioctl(fd, request, arg);
}

// ---- mmap 인터셉터 ----
// GPU 메모리 매핑 추적
extern "C" void* mmap(void *addr, size_t length, int prot, int flags, int fd, off_t offset) {
  // 실제 함수 해결
  if (!real_mmap) {
    real_mmap = (mmap_func_t) dlsym(RTLD_NEXT, "mmap");
    if (!real_mmap) {
      errno = ENOSYS;
      return MAP_FAILED;
    }
  }

  void* result = real_mmap(addr, length, prot, flags, fd, offset);

  // 인터셉션이 활성화된 경우에만 통계 수집
  if (bless::syscall_intercept_enabled.load(std::memory_order_relaxed)) {
    bless::mmap_count.fetch_add(1, std::memory_order_relaxed);

    if (result != MAP_FAILED) {
      bless::mmap_total_bytes.fetch_add(length, std::memory_order_relaxed);

      // GPU 메모리 매핑 감지 (특정 플래그 조합 또는 /dev/nvidia* fd)
      // 실제로는 fd를 /proc/self/fd에서 확인해야 하지만, 간단히 크기로 추정
      if (length >= 4096 && (flags & MAP_SHARED)) {
        bless::mmap_gpu_count.fetch_add(1, std::memory_order_relaxed);
      }
    }
  }

  return result;
}

// dtor
__attribute__((destructor)) static void bless_dtor() {
  if (bless::master_fd >= 0) {
    char bye[128];
    snprintf(bye, sizeof(bye), "BYE pid=%d t=%s",
             (int)getpid(), bless::tenant_id.c_str());
    master_send(bye);
  }
  bless::ctrl_running.store(false);
  if (!bless::sock_path.empty()) {
    int t = socket(AF_UNIX, SOCK_DGRAM, 0);
    if (t >= 0) {
      sockaddr_un r{}; r.sun_family = AF_UNIX;
      snprintf(r.sun_path, sizeof(r.sun_path), "%s", bless::sock_path.c_str());
      sendto(t, "quit", 4, 0, (sockaddr*)&r, sizeof(r));
      close(t);
    }
  }
  if (bless::ctrl_thread.joinable()) bless::ctrl_thread.join();
  if (bless::g_ctrl_fd >= 0) { close(bless::g_ctrl_fd); bless::g_ctrl_fd = -1; }
  if (bless::ctx_limited)   cuCtxDestroy(bless::ctx_limited);
  if (bless::ctx_unlimited) cuCtxDestroy(bless::ctx_unlimited);
  if (bless::master_fd >= 0){ close(bless::master_fd); bless::master_fd = -1; }
}
