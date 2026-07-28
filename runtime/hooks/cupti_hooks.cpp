// cupti_hooks.cpp
// CUPTI callback-based killer op detection
// Detects operator boundaries and triggers gating at killer kernels
//
// Based on: Master_Experiment/Exp12_cupti_gate, Exp13_proactive_gating

#include <cupti.h>
#include <cuda.h>
#include <string>
#include <unordered_set>

// TODO: implement
// - CUpti_CallbackFunc for CUPTI_CB_DOMAIN_RUNTIME_API
// - kernel name matching against loaded killer policy
// - on ENTER: notify stream_gate (killer start)
// - on EXIT:  notify stream_gate (killer end), log elapsed time
