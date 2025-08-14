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
#include <sched.h>   // <-- sched_yield 위해 필요

// fwd
static void ensure_init();
static void attach_runtime_if_needed();
static void start_control_server(const std::string& path);

namespace bless { static std::atomic<int> gate{0}; }

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
}

// ---------------- bless state ----------------
namespace bless {
  enum Route { LIMITED=0, UNLIMITED=1 };

  static std::atomic<int>  route{LIMITED};
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

  static int total_sms = 0;
  static std::atomic<int> limited_sms{0};

  // squad accounting
  static std::atomic<int>        squad_size{100};   // set_squad 로 설정
  static std::atomic<int>        squad_prog{0};     // 이번 스쿼드 내 진행 커널 수
  static std::atomic<long long>  kernel_seq{0};

  // boost mode
  static std::atomic<bool> boost_mode{false};

  // share & SD(share done) 플래그
  static std::atomic<int>  share_quota{0};          // set_share 로 설정
  static std::atomic<int>  sd_sent{0};              // 이번 스쿼드에서 SD 보냈는지

  // any GPU allocation happened? (guard reconf)
  static std::atomic<bool> any_alloc{false};

  // optional master
  static int master_fd = -1;
  static std::string master_path;
  static std::string tenant_id;
}

// ---- master send ----
static inline void master_send(const char* msg) {
  if (bless::master_fd < 0 || bless::master_path.empty()) return;
  sockaddr_un r{}; r.sun_family = AF_UNIX;
  snprintf(r.sun_path, sizeof(r.sun_path), "%s", bless::master_path.c_str());
  sendto(bless::master_fd, msg, (int)strlen(msg), 0, (sockaddr*)&r, sizeof(r));
}

// push/pop current context according to route/boost
struct ScopedRoute {
  explicit ScopedRoute(int override_route = -1) {
    ensure_init();
    int base = bless::route.load();
    int r = (override_route >= 0) ? override_route
            : (bless::boost_mode.load() ? bless::UNLIMITED : base);
    CUcontext target = (r==bless::UNLIMITED) ? bless::ctx_unlimited : bless::ctx_limited;
    cuCtxPushCurrent(target);
  }
  ~ScopedRoute(){ CUcontext p=nullptr; cuCtxPopCurrent(&p); }
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
    // attach cudart to LIMITED
    cuCtxPushCurrent(bless::ctx_limited);
    if (real_cudaFree) real_cudaFree(0);
    cuCtxPopCurrent(&prev);
    // attach cudart to UNLIMITED
    cuCtxPushCurrent(bless::ctx_unlimited);
    if (real_cudaFree) real_cudaFree(0);
    cuCtxPopCurrent(&prev);
    rt_attached.store(true);
  }
  pthread_mutex_unlock(&rt_mu);
}

// safe (re)create LIMITED only if no allocations yet
static void reconf_limited_ctx(int new_sms) {
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
  (void)cuCtxCreate_v3(&bless::ctx_limited, &p, 1, 0, bless::dev);
  bless::limited_sms.store(new_sms);
  fprintf(stderr, "[libbless] limited   ctx sm_affinity=%d\n", new_sms);
  bless::gate.store(0);
}

static void ensure_init() {
  if (bless::inited.load()) return;
  pthread_mutex_lock(&bless::init_mu);
  if (bless::inited.load()) { pthread_mutex_unlock(&bless::init_mu); return; }

  cuInit(0);
  cuDeviceGet(&bless::dev, 0);
  cuDeviceGetAttribute(&bless::total_sms, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, bless::dev);

  int pct = 50;
  if (const char* e = getenv("BLESS_LIMIT_PCT")) { int v=atoi(e); if (v>0 && v<100) pct = v; }
  int init_sms = (int)(bless::total_sms * (pct/100.0f));
  if (init_sms < 1) init_sms = 1;
  bless::limited_sms.store(init_sms);

  // UNLIMITED
  (void)cuCtxCreate_v3(&bless::ctx_unlimited, nullptr, 0, 0, bless::dev);
  { CUcontext prev=nullptr; cuCtxPushCurrent(bless::ctx_unlimited);
    CUexecAffinityParam q{}; q.type = CU_EXEC_AFFINITY_TYPE_SM_COUNT;
    if (cuCtxGetExecAffinity(&q, CU_EXEC_AFFINITY_TYPE_SM_COUNT)==CUDA_SUCCESS)
      fprintf(stderr, "[libbless] unlimited ctx sm_affinity=%d\n", q.param.smCount.val);
    cuCtxPopCurrent(&prev);
  }

  // LIMITED
  reconf_limited_ctx(init_sms);

  // control socket (thread)
  char sp[128]; snprintf(sp, sizeof(sp), "/tmp/bless-%d.sock", (int)getpid());
  bless::sock_path = sp;
  start_control_server(bless::sock_path);

  // wait ready before HELLO
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

  fprintf(stderr, "[libbless] init: total_sms=%d, limited=%d, sock=%s\n",
          bless::total_sms, bless::limited_sms.load(), bless::sock_path.c_str());

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

      if      (!strncmp(buf,"limited",7))    bless::route.store(bless::LIMITED);
      else if (!strncmp(buf,"unlimited",9))  bless::route.store(bless::UNLIMITED);
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
      else if (!strncmp(buf,"set_squad ",10)){
        int k = atoi(buf+10); if (k>0){ bless::squad_size.store(k); }
      }
      else if (!strncmp(buf,"set_share ",10)){
        int q = atoi(buf+10); if (q>0){ bless::share_quota.store(q); }
      }
      else if (!strncmp(buf,"squad_reset",11)){
        bless::squad_prog.store(0);
        bless::boost_mode.store(false);
        bless::sd_sent.store(0);          // 새 스쿼드 시작 → SD 플래그 리셋
      }
      else if (!strncmp(buf,"reconf_sm ",10)) {
        int k = atoi(buf+10); if (k>0) reconf_limited_ctx(k); // guarded
      }
      else if (!strncmp(buf,"quit",4)) break;
    }
    close(fd); unlink(path.c_str());
  });
}

// ------------- interceptors -------------
extern "C" cudaError_t cudaLaunchKernel(const void *hostFunc,
                                        dim3 gridDim, dim3 blockDim,
                                        void **args, size_t sharedMem,
                                        cudaStream_t stream)
{
  resolve_real(); ensure_init(); attach_runtime_if_needed();

  // gate 중이면 양보(컨텍스트 재구성 중 보호)
  if (!bless::boost_mode.load() && bless::route.load()==bless::LIMITED) {
    int g = bless::gate.load(std::memory_order_acquire);
    if (g!=0) { while ((g = bless::gate.load(std::memory_order_acquire))!=0) sched_yield(); }
  }

  long long kseq = bless::kernel_seq.fetch_add(1) + 1;
  int sp = bless::squad_prog.fetch_add(1) + 1;
  if (sp >= bless::squad_size.load()) {
    bless::squad_prog.store(0);
  }

  // quota 도달 시 1회 SD 전송
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

  // 실제 실행 컨텍스트 선택
  ScopedRoute s;
  return real_cudaLaunchKernel(hostFunc, gridDim, blockDim, args, sharedMem, stream);
}

extern "C" CUresult cuLaunchKernel(CUfunction f,
                                   unsigned int gridX, unsigned int gridY, unsigned int gridZ,
                                   unsigned int blockX, unsigned int blockY, unsigned int blockZ,
                                   unsigned int sharedMemBytes,
                                   CUstream hStream,
                                   void **kernelParams, void **extra)
{
  resolve_real(); ensure_init(); attach_runtime_if_needed();

  if (!bless::boost_mode.load() && bless::route.load()==bless::LIMITED) {
    int g = bless::gate.load(std::memory_order_acquire);
    if (g!=0) { while ((g = bless::gate.load(std::memory_order_acquire))!=0) sched_yield(); }
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

  ScopedRoute s;
  return real_cuLaunchKernel(f, gridX, gridY, gridZ,
                             blockX, blockY, blockZ,
                             sharedMemBytes, hStream, kernelParams, extra);
}

// alloc/mem/stream/graph/sync pass-throughs
extern "C" cudaError_t cudaMalloc(void **p, size_t n){
  resolve_real(); ensure_init(); attach_runtime_if_needed();
  bless::any_alloc.store(true, std::memory_order_release);
  ScopedRoute s; return real_cudaMalloc(p,n);
}
extern "C" cudaError_t cudaFree(void *devPtr){
  resolve_real(); ensure_init(); attach_runtime_if_needed();
  CUcontext owner=nullptr;
  CUresult r=cuPointerGetAttribute(&owner, CU_POINTER_ATTRIBUTE_CONTEXT, (CUdeviceptr)devPtr);
  if (r==CUDA_SUCCESS && owner){
    CUcontext prev=nullptr; cuCtxPushCurrent(owner);
    cudaError_t e=real_cudaFree(devPtr);
    cuCtxPopCurrent(&prev); return e;
  }
  ScopedRoute s; return real_cudaFree(devPtr);
}
extern "C" cudaError_t cudaMallocManaged(void **p, size_t n, unsigned int f){
  resolve_real(); ensure_init(); attach_runtime_if_needed();
  bless::any_alloc.store(true, std::memory_order_release);
  ScopedRoute s; return real_cudaMallocManaged(p,n,f);
}
extern "C" cudaError_t cudaMemcpy(void *d,const void *s,size_t c,cudaMemcpyKind k){
  resolve_real(); ensure_init(); return real_cudaMemcpy(d,s,c,k);
}
extern "C" cudaError_t cudaMemcpyAsync(void *d,const void *s,size_t c,cudaMemcpyKind k,cudaStream_t st){
  resolve_real(); ensure_init(); return real_cudaMemcpyAsync(d,s,c,k,st);
}
extern "C" cudaError_t cudaStreamCreate(cudaStream_t *st){
  resolve_real(); ensure_init(); attach_runtime_if_needed();
  return real_cudaStreamCreate(st);
}
extern "C" cudaError_t cudaStreamDestroy(cudaStream_t st){
  resolve_real(); ensure_init(); attach_runtime_if_needed();
  return real_cudaStreamDestroy(st);
}
extern "C" cudaError_t cudaGraphLaunch(cudaGraphExec_t g, cudaStream_t st){
  resolve_real(); ensure_init(); return real_cudaGraphLaunch(g, st);
}
extern "C" cudaError_t cudaDeviceSynchronize(){
  resolve_real(); ensure_init(); attach_runtime_if_needed();
  return real_cudaDeviceSynchronize();
}
extern "C" cudaError_t cudaStreamSynchronize(cudaStream_t st){
  resolve_real(); ensure_init(); attach_runtime_if_needed();
  return real_cudaStreamSynchronize(st);
}

// queries / helpers
extern "C" __attribute__((visibility("default"))) int bless_query_sm_affinity(){
  resolve_real(); ensure_init();
  ScopedRoute s;
  CUexecAffinityParam q{}; q.type=CU_EXEC_AFFINITY_TYPE_SM_COUNT;
  if (cuCtxGetExecAffinity(&q, CU_EXEC_AFFINITY_TYPE_SM_COUNT)==CUDA_SUCCESS) return (int)q.param.smCount.val;
  return -1;
}
extern "C" __attribute__((visibility("default")))
void bless_bind_thread(int route) {
  resolve_real(); ensure_init(); attach_runtime_if_needed();
  int r = (route==1) ? bless::UNLIMITED : bless::LIMITED;
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
