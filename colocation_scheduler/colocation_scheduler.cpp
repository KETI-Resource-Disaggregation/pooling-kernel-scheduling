/**
 * Co-location Scheduler Implementation
 *
 * Implements multi-tenant GPU scheduling with:
 * 1. Time-based credit system
 * 2. Kernel type-aware co-location
 * 3. SM partitioning recommendations
 */

#include "colocation_scheduler.h"
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <chrono>

// ============================================================
// Helper Functions
// ============================================================

static inline int64_t now_us() {
    using namespace std::chrono;
    return duration_cast<microseconds>(
        high_resolution_clock::now().time_since_epoch()
    ).count();
}

static const char* kernel_type_str(KernelType type) {
    switch (type) {
        case KERNEL_TYPE_COMPUTE_BOUND: return "COMPUTE";
        case KERNEL_TYPE_MEMORY_BOUND:  return "MEMORY";
        case KERNEL_TYPE_MIXED:         return "MIXED";
        default:                        return "UNKNOWN";
    }
}

static const char* priority_str(WorkloadPriority p) {
    switch (p) {
        case WORKLOAD_PRIORITY_LOW:      return "LOW";
        case WORKLOAD_PRIORITY_NORMAL:   return "NORMAL";
        case WORKLOAD_PRIORITY_HIGH:     return "HIGH";
        case WORKLOAD_PRIORITY_REALTIME: return "REALTIME";
        default:                         return "UNKNOWN";
    }
}

// ============================================================
// Scheduler Initialization
// ============================================================

void scheduler_init(SchedulerState* state, int total_sms) {
    memset(state, 0, sizeof(SchedulerState));
    state->total_sms = total_sms;
    state->available_sms = total_sms;
    state->workload_count = 0;
    state->colocated_count = 0;

    // Default settings
    state->auto_colocation = true;
    state->colocation_threshold = 0.6f;  // Minimum score for auto-colocation
    state->time_sharing_enabled = true;
    state->default_time_slice_us = 1000;  // 1ms default slice
}

// ============================================================
// Workload Management
// ============================================================

int scheduler_register_workload(SchedulerState* state, const char* name,
                                 KernelType type, WorkloadPriority priority) {
    if (state->workload_count >= MAX_WORKLOADS) {
        fprintf(stderr, "[scheduler] Error: Maximum workloads reached\n");
        return -1;
    }

    int id = state->workload_count;
    WorkloadProfile* w = &state->workloads[id];

    w->workload_id = id;
    strncpy(w->name, name, sizeof(w->name) - 1);
    w->name[sizeof(w->name) - 1] = '\0';

    w->primary_type = type;
    w->avg_arithmetic_intensity = 0.0f;
    w->avg_memory_bandwidth = 0.0f;
    w->avg_compute_throughput = 0.0f;
    w->avg_kernel_time_us = 10.0f;  // Default estimate

    w->priority = priority;
    w->time_credit_us = -1;  // Unlimited by default
    w->time_slice_us = state->default_time_slice_us;
    w->assigned_sms = 0;  // Unlimited

    w->total_kernels = 0;
    w->total_time_us = 0;
    w->last_update_time = now_us();

    state->workload_count++;

    printf("[scheduler] Registered workload %d: '%s' type=%s priority=%s\n",
           id, name, kernel_type_str(type), priority_str(priority));

    return id;
}

void scheduler_update_profile(SchedulerState* state, int workload_id,
                               float arithmetic_intensity,
                               float memory_bandwidth,
                               float compute_throughput,
                               float avg_kernel_time) {
    if (workload_id < 0 || workload_id >= state->workload_count) {
        return;
    }

    WorkloadProfile* w = &state->workloads[workload_id];

    // Use EMA for smooth updates
    const float alpha = 0.3f;

    if (w->avg_arithmetic_intensity == 0.0f) {
        w->avg_arithmetic_intensity = arithmetic_intensity;
    } else {
        w->avg_arithmetic_intensity = alpha * arithmetic_intensity +
                                      (1.0f - alpha) * w->avg_arithmetic_intensity;
    }

    if (w->avg_memory_bandwidth == 0.0f) {
        w->avg_memory_bandwidth = memory_bandwidth;
    } else {
        w->avg_memory_bandwidth = alpha * memory_bandwidth +
                                  (1.0f - alpha) * w->avg_memory_bandwidth;
    }

    if (w->avg_compute_throughput == 0.0f) {
        w->avg_compute_throughput = compute_throughput;
    } else {
        w->avg_compute_throughput = alpha * compute_throughput +
                                    (1.0f - alpha) * w->avg_compute_throughput;
    }

    if (avg_kernel_time > 0.0f) {
        w->avg_kernel_time_us = alpha * avg_kernel_time +
                                (1.0f - alpha) * w->avg_kernel_time_us;
    }

    // Re-classify kernel type based on updated metrics
    if (w->avg_arithmetic_intensity < 10.0f) {
        w->primary_type = KERNEL_TYPE_MEMORY_BOUND;
    } else if (w->avg_arithmetic_intensity > 50.0f) {
        w->primary_type = KERNEL_TYPE_COMPUTE_BOUND;
    } else {
        w->primary_type = KERNEL_TYPE_MIXED;
    }

    w->last_update_time = now_us();
}

// ============================================================
// Time Credit Management
// ============================================================

void scheduler_set_time_credit(SchedulerState* state, int workload_id,
                                int64_t credit_us) {
    if (workload_id < 0 || workload_id >= state->workload_count) {
        return;
    }

    state->workloads[workload_id].time_credit_us = credit_us;
    printf("[scheduler] Set time credit for workload %d: %ld us\n",
           workload_id, credit_us);
}

bool scheduler_consume_credit(SchedulerState* state, int workload_id,
                               int64_t time_us) {
    if (workload_id < 0 || workload_id >= state->workload_count) {
        return false;
    }

    WorkloadProfile* w = &state->workloads[workload_id];

    // Update stats
    w->total_kernels++;
    w->total_time_us += time_us;

    // Consume credit if limited
    if (w->time_credit_us >= 0) {
        w->time_credit_us -= time_us;
        if (w->time_credit_us < 0) {
            w->time_credit_us = 0;
            return false;  // Credit exhausted
        }
    }

    return true;
}

bool scheduler_has_credit(SchedulerState* state, int workload_id) {
    if (workload_id < 0 || workload_id >= state->workload_count) {
        return false;
    }

    int64_t credit = state->workloads[workload_id].time_credit_us;
    return credit < 0 || credit > 0;  // -1 = unlimited, >0 = has credit
}

// ============================================================
// Co-location Scoring
// ============================================================

float scheduler_colocation_score(SchedulerState* state, int w1_id, int w2_id) {
    if (w1_id < 0 || w1_id >= state->workload_count ||
        w2_id < 0 || w2_id >= state->workload_count ||
        w1_id == w2_id) {
        return 0.0f;
    }

    WorkloadProfile* w1 = &state->workloads[w1_id];
    WorkloadProfile* w2 = &state->workloads[w2_id];

    float score = 0.5f;  // Base score

    // Type-based scoring (Orion approach)
    KernelType t1 = w1->primary_type;
    KernelType t2 = w2->primary_type;

    if ((t1 == KERNEL_TYPE_COMPUTE_BOUND && t2 == KERNEL_TYPE_MEMORY_BOUND) ||
        (t1 == KERNEL_TYPE_MEMORY_BOUND && t2 == KERNEL_TYPE_COMPUTE_BOUND)) {
        // Ideal: Compute + Memory
        score = 0.9f;
    } else if (t1 == KERNEL_TYPE_MIXED || t2 == KERNEL_TYPE_MIXED) {
        // Mixed can co-locate with anything
        score = 0.7f;
    } else if (t1 == t2) {
        // Same type = resource contention
        score = 0.2f;
    }

    // Adjust by arithmetic intensity difference
    float ai_diff = fabsf(w1->avg_arithmetic_intensity - w2->avg_arithmetic_intensity);
    float ai_bonus = fminf(ai_diff / 100.0f, 0.1f);  // Up to 0.1 bonus
    score += ai_bonus;

    // Adjust by memory bandwidth balance
    float total_bw = w1->avg_memory_bandwidth + w2->avg_memory_bandwidth;
    if (total_bw > 0) {
        // Penalize if both are memory-heavy
        if (total_bw > 1000.0f) {  // > 1000 GB/s combined
            score -= 0.1f;
        }
    }

    // Priority consideration
    if (w1->priority == WORKLOAD_PRIORITY_REALTIME ||
        w2->priority == WORKLOAD_PRIORITY_REALTIME) {
        // Real-time workloads prefer isolation
        score -= 0.2f;
    }

    return fmaxf(0.0f, fminf(1.0f, score));
}

int scheduler_find_best_colocation(SchedulerState* state, int workload_id) {
    if (workload_id < 0 || workload_id >= state->workload_count) {
        return -1;
    }

    int best_id = -1;
    float best_score = 0.0f;

    for (int i = 0; i < state->workload_count; i++) {
        if (i == workload_id) continue;

        float score = scheduler_colocation_score(state, workload_id, i);
        if (score > best_score && score >= state->colocation_threshold) {
            best_score = score;
            best_id = i;
        }
    }

    return best_id;
}

// ============================================================
// SM Partitioning
// ============================================================

void scheduler_assign_sms(SchedulerState* state, int workload_id, int num_sms) {
    if (workload_id < 0 || workload_id >= state->workload_count) {
        return;
    }

    // Validate SM count
    if (num_sms > state->total_sms) {
        num_sms = state->total_sms;
    }
    if (num_sms < 0) {
        num_sms = 0;  // 0 = unlimited
    }

    state->workloads[workload_id].assigned_sms = num_sms;
    printf("[scheduler] Assigned %d SMs to workload %d (%s)\n",
           num_sms, workload_id, state->workloads[workload_id].name);
}

void scheduler_recommend_sm_split(SchedulerState* state, int w1_id, int w2_id,
                                   int* w1_sms, int* w2_sms) {
    if (w1_id < 0 || w1_id >= state->workload_count ||
        w2_id < 0 || w2_id >= state->workload_count) {
        *w1_sms = state->total_sms / 2;
        *w2_sms = state->total_sms / 2;
        return;
    }

    WorkloadProfile* w1 = &state->workloads[w1_id];
    WorkloadProfile* w2 = &state->workloads[w2_id];

    // Base: split by type
    // Compute-bound kernels benefit more from SMs
    // Memory-bound kernels are limited by bandwidth anyway

    float w1_weight = 0.5f;
    float w2_weight = 0.5f;

    // Compute-bound gets more SMs
    if (w1->primary_type == KERNEL_TYPE_COMPUTE_BOUND) {
        w1_weight += 0.2f;
    }
    if (w2->primary_type == KERNEL_TYPE_COMPUTE_BOUND) {
        w2_weight += 0.2f;
    }

    // Memory-bound gets fewer SMs (doesn't need them)
    if (w1->primary_type == KERNEL_TYPE_MEMORY_BOUND) {
        w1_weight -= 0.1f;
    }
    if (w2->primary_type == KERNEL_TYPE_MEMORY_BOUND) {
        w2_weight -= 0.1f;
    }

    // Adjust by priority
    if (w1->priority > w2->priority) {
        w1_weight += 0.1f * (w1->priority - w2->priority);
    } else if (w2->priority > w1->priority) {
        w2_weight += 0.1f * (w2->priority - w1->priority);
    }

    // Normalize
    float total = w1_weight + w2_weight;
    w1_weight /= total;
    w2_weight /= total;

    // Calculate SM allocation
    *w1_sms = (int)(state->total_sms * w1_weight);
    *w2_sms = state->total_sms - *w1_sms;

    // Ensure minimum SMs
    if (*w1_sms < 1) *w1_sms = 1;
    if (*w2_sms < 1) *w2_sms = 1;
}

// ============================================================
// Debug / Printing
// ============================================================

void scheduler_print_state(SchedulerState* state) {
    printf("\n========== Scheduler State ==========\n");
    printf("Total SMs: %d, Available: %d\n", state->total_sms, state->available_sms);
    printf("Workloads: %d, Co-located pairs: %d\n",
           state->workload_count, state->colocated_count);
    printf("Auto co-location: %s (threshold: %.2f)\n",
           state->auto_colocation ? "ON" : "OFF", state->colocation_threshold);
    printf("\n");

    printf("%-4s | %-15s | %-8s | %-8s | %-10s | %-10s | %-8s | %-8s\n",
           "ID", "Name", "Type", "Priority", "Credit(us)", "AvgKern", "SMs", "Total");
    printf("-------------------------------------------------------------------------------------------\n");

    for (int i = 0; i < state->workload_count; i++) {
        WorkloadProfile* w = &state->workloads[i];
        printf("%-4d | %-15s | %-8s | %-8s | %-10ld | %-10.2f | %-8d | %-8ld\n",
               w->workload_id, w->name,
               kernel_type_str(w->primary_type),
               priority_str(w->priority),
               w->time_credit_us,
               w->avg_kernel_time_us,
               w->assigned_sms,
               w->total_kernels);
    }
    printf("=====================================\n\n");
}

void scheduler_print_colocation_matrix(SchedulerState* state) {
    if (state->workload_count < 2) {
        printf("Need at least 2 workloads for co-location matrix\n");
        return;
    }

    printf("\n========== Co-location Score Matrix ==========\n");
    printf("     ");
    for (int i = 0; i < state->workload_count; i++) {
        printf("| W%-3d ", i);
    }
    printf("\n");
    printf("-----");
    for (int i = 0; i < state->workload_count; i++) {
        printf("+------");
    }
    printf("\n");

    for (int i = 0; i < state->workload_count; i++) {
        printf("W%-3d ", i);
        for (int j = 0; j < state->workload_count; j++) {
            if (i == j) {
                printf("|  --  ");
            } else {
                float score = scheduler_colocation_score(state, i, j);
                printf("| %.2f ", score);
            }
        }
        printf("\n");
    }
    printf("==============================================\n");
    printf("Legend: 0.8+ = Excellent, 0.6-0.8 = Good, 0.4-0.6 = Fair, <0.4 = Avoid\n\n");
}

// ============================================================
// Test Main
// ============================================================

#ifdef SCHEDULER_TEST_MAIN

int main() {
    printf("==============================================\n");
    printf("     Co-location Scheduler Test              \n");
    printf("==============================================\n\n");

    SchedulerState state;
    scheduler_init(&state, 84);  // RTX A6000 has 84 SMs

    // Register test workloads
    int w_compute = scheduler_register_workload(&state, "ResNet_Train",
                                                  KERNEL_TYPE_COMPUTE_BOUND,
                                                  WORKLOAD_PRIORITY_HIGH);

    int w_memory = scheduler_register_workload(&state, "DataLoader",
                                                 KERNEL_TYPE_MEMORY_BOUND,
                                                 WORKLOAD_PRIORITY_NORMAL);

    int w_mixed = scheduler_register_workload(&state, "BERT_Infer",
                                                KERNEL_TYPE_MIXED,
                                                WORKLOAD_PRIORITY_NORMAL);

    int w_compute2 = scheduler_register_workload(&state, "GPT_Train",
                                                   KERNEL_TYPE_COMPUTE_BOUND,
                                                   WORKLOAD_PRIORITY_HIGH);

    // Update profiles with measured characteristics
    scheduler_update_profile(&state, w_compute, 100.0f, 300.0f, 30000.0f, 250.0f);
    scheduler_update_profile(&state, w_memory, 0.5f, 600.0f, 50.0f, 130.0f);
    scheduler_update_profile(&state, w_mixed, 25.0f, 400.0f, 500.0f, 200.0f);
    scheduler_update_profile(&state, w_compute2, 80.0f, 350.0f, 25000.0f, 300.0f);

    // Set time credits
    scheduler_set_time_credit(&state, w_compute, 20000);  // 20ms
    scheduler_set_time_credit(&state, w_memory, 10000);   // 10ms
    scheduler_set_time_credit(&state, w_mixed, 15000);    // 15ms
    scheduler_set_time_credit(&state, w_compute2, 20000); // 20ms

    // Print state
    scheduler_print_state(&state);

    // Print co-location matrix
    scheduler_print_colocation_matrix(&state);

    // Find best co-location partners
    printf("========== Co-location Recommendations ==========\n\n");

    for (int i = 0; i < state.workload_count; i++) {
        int best = scheduler_find_best_colocation(&state, i);
        if (best >= 0) {
            float score = scheduler_colocation_score(&state, i, best);
            printf("Workload '%s' -> Best partner: '%s' (score: %.2f)\n",
                   state.workloads[i].name, state.workloads[best].name, score);

            int w1_sms, w2_sms;
            scheduler_recommend_sm_split(&state, i, best, &w1_sms, &w2_sms);
            printf("  Recommended SM split: %d vs %d\n\n", w1_sms, w2_sms);
        } else {
            printf("Workload '%s' -> No suitable partner found\n\n",
                   state.workloads[i].name);
        }
    }

    printf("=================================================\n");

    // Test credit consumption
    printf("\n========== Credit Consumption Test ==========\n");
    printf("Consuming 5000us from ResNet_Train...\n");
    scheduler_consume_credit(&state, w_compute, 5000);
    printf("Remaining credit: %ld us\n", state.workloads[w_compute].time_credit_us);

    printf("Consuming 8000us from DataLoader...\n");
    scheduler_consume_credit(&state, w_memory, 8000);
    printf("Remaining credit: %ld us\n", state.workloads[w_memory].time_credit_us);

    printf("Has credit? ResNet_Train: %s, DataLoader: %s\n",
           scheduler_has_credit(&state, w_compute) ? "YES" : "NO",
           scheduler_has_credit(&state, w_memory) ? "YES" : "NO");

    return 0;
}

#endif // SCHEDULER_TEST_MAIN
