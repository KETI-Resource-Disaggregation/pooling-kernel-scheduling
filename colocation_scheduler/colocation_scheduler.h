/**
 * Co-location Scheduler Header
 *
 * Manages multi-tenant GPU workload scheduling based on:
 * 1. Time-based credits (from libbless)
 * 2. Kernel type classification (Compute/Memory bound)
 *
 * Key principles (from Orion research):
 * - Co-locate Compute-bound + Memory-bound kernels for better utilization
 * - Avoid co-locating same-type kernels (resource contention)
 * - Use SM partitioning for isolation when needed
 */

#ifndef COLOCATION_SCHEDULER_H
#define COLOCATION_SCHEDULER_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================
// Kernel Type Classification
// ============================================================

typedef enum {
    KERNEL_TYPE_UNKNOWN = 0,
    KERNEL_TYPE_COMPUTE_BOUND = 1,
    KERNEL_TYPE_MEMORY_BOUND = 2,
    KERNEL_TYPE_MIXED = 3
} KernelType;

typedef enum {
    WORKLOAD_PRIORITY_LOW = 0,
    WORKLOAD_PRIORITY_NORMAL = 1,
    WORKLOAD_PRIORITY_HIGH = 2,
    WORKLOAD_PRIORITY_REALTIME = 3
} WorkloadPriority;

// ============================================================
// Workload Profile
// ============================================================

typedef struct {
    int workload_id;
    char name[64];

    // Kernel characteristics
    KernelType primary_type;
    float avg_arithmetic_intensity;  // FLOP/byte
    float avg_memory_bandwidth;      // GB/s
    float avg_compute_throughput;    // GFLOPS
    float avg_kernel_time_us;

    // Scheduling parameters
    WorkloadPriority priority;
    int64_t time_credit_us;          // Remaining time credit
    int64_t time_slice_us;           // Time slice for this workload
    int assigned_sms;                // Number of SMs assigned (0 = unlimited)

    // Runtime stats
    int64_t total_kernels;
    int64_t total_time_us;
    int64_t last_update_time;
} WorkloadProfile;

// ============================================================
// Scheduler State
// ============================================================

#define MAX_WORKLOADS 16

typedef struct {
    int total_sms;
    int available_sms;

    WorkloadProfile workloads[MAX_WORKLOADS];
    int workload_count;

    // Co-location pairs (indices into workloads array)
    int colocated_pairs[MAX_WORKLOADS / 2][2];
    int colocated_count;

    // Scheduler settings
    bool auto_colocation;
    float colocation_threshold;  // Minimum score to auto-colocate
    bool time_sharing_enabled;
    int default_time_slice_us;
} SchedulerState;

// ============================================================
// API Functions
// ============================================================

// Initialize scheduler
void scheduler_init(SchedulerState* state, int total_sms);

// Register a new workload
int scheduler_register_workload(SchedulerState* state, const char* name,
                                 KernelType type, WorkloadPriority priority);

// Update workload profile with measured characteristics
void scheduler_update_profile(SchedulerState* state, int workload_id,
                               float arithmetic_intensity,
                               float memory_bandwidth,
                               float compute_throughput,
                               float avg_kernel_time);

// Set time credit for a workload
void scheduler_set_time_credit(SchedulerState* state, int workload_id,
                                int64_t credit_us);

// Consume time credit
bool scheduler_consume_credit(SchedulerState* state, int workload_id,
                               int64_t time_us);

// Check if workload has remaining credit
bool scheduler_has_credit(SchedulerState* state, int workload_id);

// Get co-location score between two workloads
float scheduler_colocation_score(SchedulerState* state, int w1_id, int w2_id);

// Find best workload to co-locate with given workload
int scheduler_find_best_colocation(SchedulerState* state, int workload_id);

// Assign SM partition to workload
void scheduler_assign_sms(SchedulerState* state, int workload_id, int num_sms);

// Get recommended SM assignment for co-location pair
void scheduler_recommend_sm_split(SchedulerState* state, int w1_id, int w2_id,
                                   int* w1_sms, int* w2_sms);

// Print scheduler state
void scheduler_print_state(SchedulerState* state);

// Print co-location matrix
void scheduler_print_colocation_matrix(SchedulerState* state);

#ifdef __cplusplus
}
#endif

#endif // COLOCATION_SCHEDULER_H
