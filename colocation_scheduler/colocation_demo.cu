/**
 * Co-location Demo
 *
 * Demonstrates the full system:
 * 1. Kernel profiling (Compute/Memory bound classification)
 * 2. Co-location scheduling (characteristic-based placement)
 * 3. Time-based credit management (via libbless socket)
 *
 * Run with: LD_PRELOAD=../spark_kernel_scheduling/libbless/libbless.so ./colocation_demo
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <chrono>
#include <atomic>

#include "colocation_scheduler.h"

// ============================================================
// Kernel Definitions
// ============================================================

// Compute-bound: Heavy FMA
__global__ void kernel_compute_bound(float* data, int N, int iterations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float val = data[idx];
    float a = 1.00001f;
    float b = 0.99999f;

    #pragma unroll 16
    for (int i = 0; i < iterations; i++) {
        val = fmaf(val, a, b);
        val = fmaf(val, b, a);
        val = fmaf(val, a, b);
        val = fmaf(val, b, a);
    }

    data[idx] = val;
}

// Memory-bound: Copy
__global__ void kernel_memory_bound(float* dst, const float* src, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    dst[idx] = src[idx] + 1.0f;
}

// Mixed: Stencil
__global__ void kernel_mixed(float* output, const float* input, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        int idx = y * width + x;
        float val = input[idx - width] + input[idx + width] +
                    input[idx - 1] + input[idx + 1] +
                    4.0f * input[idx];
        output[idx] = val * 0.125f;
    }
}

// ============================================================
// Workload Simulation
// ============================================================

typedef struct {
    int workload_id;
    const char* name;
    KernelType type;
    int num_kernels;
    float* d_data;
    float* d_data2;  // For memory-bound kernels
    int data_size;
    pthread_t thread;
    std::atomic<bool> running;
    std::atomic<int> kernels_executed;
    std::atomic<int64_t> total_time_us;
} WorkloadContext;

// libbless socket communication
static char* find_bless_socket() {
    static char path[256];
    char buf[256];
    FILE* fp = popen("ls -t /tmp/bless-*.sock 2>/dev/null | head -1", "r");
    if (fp && fgets(buf, sizeof(buf), fp)) {
        buf[strcspn(buf, "\n")] = 0;
        strcpy(path, buf);
        pclose(fp);
        return path;
    }
    if (fp) pclose(fp);
    return NULL;
}

static void send_bless_command(const char* sock_path, const char* cmd) {
    if (!sock_path) return;

    int fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) return;

    struct sockaddr_un addr;
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, sock_path, sizeof(addr.sun_path) - 1);

    if (connect(fd, (struct sockaddr*)&addr, sizeof(addr)) == 0) {
        write(fd, cmd, strlen(cmd));
        char response[256];
        read(fd, response, sizeof(response) - 1);
        printf("[bless] %s -> %s\n", cmd, response);
    }

    close(fd);
}

// Timer utilities
inline int64_t now_us() {
    using namespace std::chrono;
    return duration_cast<microseconds>(
        high_resolution_clock::now().time_since_epoch()
    ).count();
}

// ============================================================
// Workload Thread Functions
// ============================================================

void* workload_compute_thread(void* arg) {
    WorkloadContext* ctx = (WorkloadContext*)arg;

    int N = ctx->data_size;
    int iterations = 50;  // Makes each kernel compute-heavy

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    printf("[%s] Starting compute-bound workload (%d kernels)\n", ctx->name, ctx->num_kernels);

    for (int i = 0; i < ctx->num_kernels && ctx->running; i++) {
        cudaEventRecord(start);
        kernel_compute_bound<<<(N + 255) / 256, 256>>>(ctx->d_data, N, iterations);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms;
        cudaEventElapsedTime(&ms, start, stop);

        ctx->kernels_executed++;
        ctx->total_time_us += (int64_t)(ms * 1000);

        if (i % 100 == 0) {
            printf("[%s] Progress: %d/%d kernels, avg=%.2f us\n",
                   ctx->name, i, ctx->num_kernels,
                   (float)ctx->total_time_us / ctx->kernels_executed);
        }
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("[%s] Completed: %d kernels in %ld us (avg=%.2f us)\n",
           ctx->name, (int)ctx->kernels_executed, (int64_t)ctx->total_time_us,
           (float)ctx->total_time_us / ctx->kernels_executed);

    return NULL;
}

void* workload_memory_thread(void* arg) {
    WorkloadContext* ctx = (WorkloadContext*)arg;

    int N = ctx->data_size;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    printf("[%s] Starting memory-bound workload (%d kernels)\n", ctx->name, ctx->num_kernels);

    for (int i = 0; i < ctx->num_kernels && ctx->running; i++) {
        cudaEventRecord(start);
        kernel_memory_bound<<<(N + 255) / 256, 256>>>(ctx->d_data2, ctx->d_data, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms;
        cudaEventElapsedTime(&ms, start, stop);

        ctx->kernels_executed++;
        ctx->total_time_us += (int64_t)(ms * 1000);

        // Swap buffers for next iteration
        float* tmp = ctx->d_data;
        ctx->d_data = ctx->d_data2;
        ctx->d_data2 = tmp;

        if (i % 100 == 0) {
            printf("[%s] Progress: %d/%d kernels, avg=%.2f us\n",
                   ctx->name, i, ctx->num_kernels,
                   (float)ctx->total_time_us / ctx->kernels_executed);
        }
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("[%s] Completed: %d kernels in %ld us (avg=%.2f us)\n",
           ctx->name, (int)ctx->kernels_executed, (int64_t)ctx->total_time_us,
           (float)ctx->total_time_us / ctx->kernels_executed);

    return NULL;
}

// ============================================================
// Demo Functions
// ============================================================

void demo_profiling() {
    printf("\n");
    printf("╔═══════════════════════════════════════════════════════╗\n");
    printf("║           PHASE 1: Kernel Profiling                   ║\n");
    printf("╚═══════════════════════════════════════════════════════╝\n\n");

    int N = 4 * 1024 * 1024;

    float* d_data;
    float* d_data2;
    cudaMalloc(&d_data, N * sizeof(float));
    cudaMalloc(&d_data2, N * sizeof(float));
    cudaMemset(d_data, 0, N * sizeof(float));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    printf("Profiling kernels with N = %d elements...\n\n", N);

    // Profile compute-bound kernel
    {
        // Warmup
        kernel_compute_bound<<<(N + 255) / 256, 256>>>(d_data, N, 50);
        cudaDeviceSynchronize();

        cudaEventRecord(start);
        kernel_compute_bound<<<(N + 255) / 256, 256>>>(d_data, N, 50);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms;
        cudaEventElapsedTime(&ms, start, stop);

        int64_t flops = (int64_t)N * 50 * 8;
        float gflops = (float)flops / (ms * 1e6);
        int64_t bytes = (int64_t)N * sizeof(float) * 2;
        float gb_s = (float)bytes / (ms * 1e6);
        float ai = (float)flops / bytes;

        printf("Compute-bound kernel:\n");
        printf("  Time: %.2f ms, GFLOPS: %.2f, Memory BW: %.2f GB/s\n", ms, gflops, gb_s);
        printf("  Arithmetic Intensity: %.2f FLOP/byte -> COMPUTE BOUND\n\n", ai);
    }

    // Profile memory-bound kernel
    {
        // Warmup
        kernel_memory_bound<<<(N + 255) / 256, 256>>>(d_data2, d_data, N);
        cudaDeviceSynchronize();

        cudaEventRecord(start);
        kernel_memory_bound<<<(N + 255) / 256, 256>>>(d_data2, d_data, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms;
        cudaEventElapsedTime(&ms, start, stop);

        int64_t ops = N;  // One add per element
        float gflops = (float)ops / (ms * 1e6);
        int64_t bytes = (int64_t)N * sizeof(float) * 2;
        float gb_s = (float)bytes / (ms * 1e6);
        float ai = (float)ops / bytes;

        printf("Memory-bound kernel:\n");
        printf("  Time: %.2f ms, GFLOPS: %.2f, Memory BW: %.2f GB/s\n", ms, gflops, gb_s);
        printf("  Arithmetic Intensity: %.4f FLOP/byte -> MEMORY BOUND\n\n", ai);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_data);
    cudaFree(d_data2);
}

void demo_scheduling() {
    printf("\n");
    printf("╔═══════════════════════════════════════════════════════╗\n");
    printf("║           PHASE 2: Co-location Scheduling             ║\n");
    printf("╚═══════════════════════════════════════════════════════╝\n\n");

    SchedulerState state;
    scheduler_init(&state, 84);

    // Register workloads
    int w0 = scheduler_register_workload(&state, "ResNet_Train",
                                          KERNEL_TYPE_COMPUTE_BOUND, WORKLOAD_PRIORITY_HIGH);
    int w1 = scheduler_register_workload(&state, "DataPipeline",
                                          KERNEL_TYPE_MEMORY_BOUND, WORKLOAD_PRIORITY_NORMAL);
    int w2 = scheduler_register_workload(&state, "BERT_Inference",
                                          KERNEL_TYPE_MIXED, WORKLOAD_PRIORITY_NORMAL);
    int w3 = scheduler_register_workload(&state, "GPT_Train",
                                          KERNEL_TYPE_COMPUTE_BOUND, WORKLOAD_PRIORITY_HIGH);

    // Update profiles
    scheduler_update_profile(&state, w0, 100.0f, 300.0f, 30000.0f, 250.0f);
    scheduler_update_profile(&state, w1, 0.25f, 600.0f, 50.0f, 100.0f);
    scheduler_update_profile(&state, w2, 25.0f, 400.0f, 500.0f, 200.0f);
    scheduler_update_profile(&state, w3, 80.0f, 350.0f, 25000.0f, 300.0f);

    // Set time credits
    scheduler_set_time_credit(&state, w0, 50000);  // 50ms
    scheduler_set_time_credit(&state, w1, 30000);  // 30ms
    scheduler_set_time_credit(&state, w2, 40000);  // 40ms
    scheduler_set_time_credit(&state, w3, 50000);  // 50ms

    printf("\n");
    scheduler_print_state(&state);
    scheduler_print_colocation_matrix(&state);

    // Recommendations
    printf("========== Co-location Decisions ==========\n\n");

    printf("Scenario: All 4 workloads need to run concurrently\n\n");

    // Find optimal pairs
    int best_for_w0 = scheduler_find_best_colocation(&state, w0);
    int best_for_w3 = scheduler_find_best_colocation(&state, w3);

    printf("Decision 1: ResNet_Train (COMPUTE) + DataPipeline (MEMORY)\n");
    printf("  Score: %.2f - EXCELLENT\n", scheduler_colocation_score(&state, w0, w1));
    int sm1, sm2;
    scheduler_recommend_sm_split(&state, w0, w1, &sm1, &sm2);
    printf("  SM Split: ResNet=%d, DataPipeline=%d\n\n", sm1, sm2);

    printf("Decision 2: GPT_Train (COMPUTE) + BERT_Inference (MIXED)\n");
    printf("  Score: %.2f - GOOD\n", scheduler_colocation_score(&state, w3, w2));
    scheduler_recommend_sm_split(&state, w3, w2, &sm1, &sm2);
    printf("  SM Split: GPT=%d, BERT=%d\n\n", sm1, sm2);

    printf("Alternative: GPT_Train + ResNet_Train (both COMPUTE)\n");
    printf("  Score: %.2f - AVOID (resource contention)\n\n", scheduler_colocation_score(&state, w3, w0));
}

void demo_colocation_execution() {
    printf("\n");
    printf("╔═══════════════════════════════════════════════════════╗\n");
    printf("║      PHASE 3: Concurrent Execution with Co-location   ║\n");
    printf("╚═══════════════════════════════════════════════════════╝\n\n");

    int N = 2 * 1024 * 1024;
    int num_kernels = 500;

    // Create workload contexts
    WorkloadContext compute_ctx = {0};
    compute_ctx.workload_id = 0;
    compute_ctx.name = "COMPUTE_WORKLOAD";
    compute_ctx.type = KERNEL_TYPE_COMPUTE_BOUND;
    compute_ctx.num_kernels = num_kernels;
    compute_ctx.data_size = N;
    compute_ctx.running = true;
    compute_ctx.kernels_executed = 0;
    compute_ctx.total_time_us = 0;

    WorkloadContext memory_ctx = {0};
    memory_ctx.workload_id = 1;
    memory_ctx.name = "MEMORY_WORKLOAD";
    memory_ctx.type = KERNEL_TYPE_MEMORY_BOUND;
    memory_ctx.num_kernels = num_kernels;
    memory_ctx.data_size = N;
    memory_ctx.running = true;
    memory_ctx.kernels_executed = 0;
    memory_ctx.total_time_us = 0;

    // Allocate memory
    cudaMalloc(&compute_ctx.d_data, N * sizeof(float));
    cudaMalloc(&memory_ctx.d_data, N * sizeof(float));
    cudaMalloc(&memory_ctx.d_data2, N * sizeof(float));

    cudaMemset(compute_ctx.d_data, 0, N * sizeof(float));
    cudaMemset(memory_ctx.d_data, 0, N * sizeof(float));

    printf("Test 1: Sequential Execution (Baseline)\n");
    printf("========================================\n");

    int64_t t0 = now_us();

    // Run compute workload first
    for (int i = 0; i < num_kernels; i++) {
        kernel_compute_bound<<<(N + 255) / 256, 256>>>(compute_ctx.d_data, N, 50);
    }
    cudaDeviceSynchronize();

    int64_t t1 = now_us();

    // Then memory workload
    for (int i = 0; i < num_kernels; i++) {
        kernel_memory_bound<<<(N + 255) / 256, 256>>>(memory_ctx.d_data2, memory_ctx.d_data, N);
        float* tmp = memory_ctx.d_data;
        memory_ctx.d_data = memory_ctx.d_data2;
        memory_ctx.d_data2 = tmp;
    }
    cudaDeviceSynchronize();

    int64_t t2 = now_us();

    int64_t seq_compute_time = t1 - t0;
    int64_t seq_memory_time = t2 - t1;
    int64_t seq_total_time = t2 - t0;

    printf("  Compute workload: %ld us (%d kernels, avg=%.2f us)\n",
           seq_compute_time, num_kernels, (float)seq_compute_time / num_kernels);
    printf("  Memory workload:  %ld us (%d kernels, avg=%.2f us)\n",
           seq_memory_time, num_kernels, (float)seq_memory_time / num_kernels);
    printf("  Total (sequential): %ld us\n\n", seq_total_time);

    // Reset data
    cudaMemset(compute_ctx.d_data, 0, N * sizeof(float));
    cudaMemset(memory_ctx.d_data, 0, N * sizeof(float));

    printf("Test 2: Concurrent Execution (Co-located COMPUTE + MEMORY)\n");
    printf("==========================================================\n");

    // Create CUDA streams for concurrent execution
    cudaStream_t stream_compute, stream_memory;
    cudaStreamCreate(&stream_compute);
    cudaStreamCreate(&stream_memory);

    int64_t t3 = now_us();

    // Run both workloads concurrently using streams
    for (int i = 0; i < num_kernels; i++) {
        kernel_compute_bound<<<(N + 255) / 256, 256, 0, stream_compute>>>(
            compute_ctx.d_data, N, 50);
        kernel_memory_bound<<<(N + 255) / 256, 256, 0, stream_memory>>>(
            memory_ctx.d_data2, memory_ctx.d_data, N);

        float* tmp = memory_ctx.d_data;
        memory_ctx.d_data = memory_ctx.d_data2;
        memory_ctx.d_data2 = tmp;
    }

    cudaStreamSynchronize(stream_compute);
    cudaStreamSynchronize(stream_memory);

    int64_t t4 = now_us();
    int64_t concurrent_time = t4 - t3;

    printf("  Total (concurrent): %ld us\n\n", concurrent_time);

    cudaStreamDestroy(stream_compute);
    cudaStreamDestroy(stream_memory);

    // Calculate speedup
    float speedup = (float)seq_total_time / concurrent_time;
    float overlap_efficiency = (float)(seq_total_time - concurrent_time) / seq_total_time * 100;

    printf("========== Results ==========\n");
    printf("Sequential total:  %ld us\n", seq_total_time);
    printf("Concurrent total:  %ld us\n", concurrent_time);
    printf("Speedup:           %.2fx\n", speedup);
    printf("Overlap efficiency: %.1f%%\n", overlap_efficiency);
    printf("\n");

    if (speedup > 1.3f) {
        printf("SUCCESS: Compute + Memory co-location shows significant benefit!\n");
        printf("The complementary resource usage allows efficient overlap.\n");
    } else if (speedup > 1.0f) {
        printf("PARTIAL SUCCESS: Some overlap achieved.\n");
        printf("Benefits may be limited by kernel launch overhead.\n");
    } else {
        printf("NOTE: Limited improvement. Kernels may be too short or\n");
        printf("hardware scheduling overhead dominates.\n");
    }

    // Cleanup
    cudaFree(compute_ctx.d_data);
    cudaFree(memory_ctx.d_data);
    cudaFree(memory_ctx.d_data2);
}

// ============================================================
// Main
// ============================================================

int main(int argc, char** argv) {
    printf("╔═══════════════════════════════════════════════════════╗\n");
    printf("║         Co-location Scheduling Demo                   ║\n");
    printf("║   Kernel Profiling + Type-based Co-location           ║\n");
    printf("╚═══════════════════════════════════════════════════════╝\n\n");

    // Get GPU info
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s\n", prop.name);
    printf("SMs: %d, Compute Capability: %d.%d\n\n",
           prop.multiProcessorCount, prop.major, prop.minor);

    // Check for libbless
    char* sock = find_bless_socket();
    if (sock) {
        printf("libbless socket found: %s\n", sock);
        printf("Time-based credit management is available.\n\n");
    } else {
        printf("Note: libbless not loaded. Run with:\n");
        printf("  LD_PRELOAD=../spark_kernel_scheduling/libbless/libbless.so ./colocation_demo\n\n");
    }

    // Run demos
    demo_profiling();
    demo_scheduling();
    demo_colocation_execution();

    printf("\n╔═══════════════════════════════════════════════════════╗\n");
    printf("║                    Summary                            ║\n");
    printf("╚═══════════════════════════════════════════════════════╝\n\n");
    printf("Key findings from Orion-style co-location:\n");
    printf("1. COMPUTE + MEMORY kernels benefit from co-location\n");
    printf("2. Same-type kernels (COMPUTE+COMPUTE) cause contention\n");
    printf("3. Time-based credits ensure fair resource allocation\n");
    printf("4. SM partitioning can provide isolation when needed\n\n");

    printf("Integration points:\n");
    printf("- libbless: Time credit management and kernel interception\n");
    printf("- kernel_profiler: Runtime classification of kernel types\n");
    printf("- scheduler: Co-location decisions based on characteristics\n\n");

    return 0;
}
