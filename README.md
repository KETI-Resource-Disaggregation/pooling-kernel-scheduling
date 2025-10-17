# 🚀 GPU Kernel Scheduling System

> **Multi-Tenant GPU Virtualization with MIG + MPS Integration**  
> Fine-grained kernel scheduling, SM partitioning, and credit-based rate limiting for efficient GPU sharing

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Components](#-components)
- [Getting Started](#-getting-started)
- [Configuration](#-configuration)
- [API Reference](#-api-reference)
- [Performance](#-performance)
- [Troubleshooting](#-troubleshooting)

---

## 🎯 Overview

This system provides **fine-grained GPU resource management** by combining:

1. **MIG (Multi-Instance GPU)**: Hardware-level GPU partitioning
2. **MPS (Multi-Process Service)**: SM time-slicing within MIG instances
3. **libbless**: User-space kernel scheduling library with credit-based gating
4. **blessctl**: Centralized control plane for runtime management

### Key Features

✅ **Dual-Context Routing**: Switch between LIMITED (SM-constrained) and UNLIMITED contexts  
✅ **Credit-Based Gating**: Token bucket algorithm for kernel launch rate limiting  
✅ **Boost Mode**: Bypass gating for urgent/priority workloads  
✅ **Squad & Share Quota**: Batch-based scheduling with preemption signals  
✅ **Dynamic SM Reconfiguration**: Adjust SM allocation without restart  
✅ **Multi-Tenant Support**: Isolation via contexts, quotas, and namespaces  
✅ **Zero Application Changes**: LD_PRELOAD-based interception

---

## 🏗️ Architecture

### System Block Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                      Application Layer                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │ Tenant A │  │ Tenant B │  │ Tenant C │  │ Tenant D │       │
│  │(Container│  │(Container│  │(Bare-    │  │(Container│       │
│  │ PyTorch) │  │TensorFlow│  │ metal)   │  │   JAX)   │       │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘       │
└───────┼─────────────┼─────────────┼─────────────┼──────────────┘
        │             │             │             │
        └─────────────┴─────────────┴─────────────┘
                      │
              CUDA API / Framework Calls
                      │
┌─────────────────────▼─────────────────────────────────────────┐
│                    User Space                                 │
│ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌────────┐│
│ │ CUDA Runtime │ │  MPS Client  │ │   Resource   │ │Monitor││
│ │              │ │              │ │   Manager    │ │ Agent  ││
│ │ cudaMalloc() │ │Context Queue │ │  Scheduler   │ │Metrics ││
│ │ cudaLaunch() │ │Memory Alloc  │ │Tenant Quota  │ │Collect ││
│ └──────────────┘ └──────────────┘ └──────────────┘ └────────┘│
└───────────────────────┬───────────────────────────────────────┘
                        │
                ioctl() / System Calls
                        │
┌───────────────────────▼───────────────────────────────────────┐
│                   Kernel Space                                │
│ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌────────┐│
│ │  MPS Server  │ │MIG Controller│ │  NVIDIA KMD  │ │ Memory││
│ │              │ │              │ │              │ │Manager ││
│ │Context Switch│ │MIG Create/   │ │  nvidia.ko   │ │Physical││
│ │Time Slice    │ │Delete        │ │GPU State Mgmt│ │ Memory ││
│ │SM Scheduling │ │Placement     │ │Interrupt     │ │  DMA   ││
│ └──────────────┘ └──────────────┘ └──────────────┘ └────────┘│
└───────────────────────┬───────────────────────────────────────┘
                        │
                MMIO / PCIe / DMA
                        │
┌───────────────────────▼───────────────────────────────────────┐
│                   GPU Hardware                                │
│ ┌─────────────────────────────────────────────────────────┐  │
│ │ GPU 0: RTX Pro 6000 (97GB)                              │  │
│ │ ┌─────────┐ ┌─────────┐ ┌─────────┐                    │  │
│ │ │  MIG-0  │ │  MIG-1  │ │  MIG-2  │                    │  │
│ │ │1g.24gb  │ │1g.24gb  │ │2g.48gb  │                    │  │
│ │ │ 46 SM   │ │ 46 SM   │ │ 94 SM   │                    │  │
│ │ │MPS: A,B │ │ MPS: C  │ │ MPS: D  │                    │  │
│ │ └─────────┘ └─────────┘ └─────────┘                    │  │
│ │ Physical: 186 SM | HBM3: 97GB | L2: 96MB               │  │
│ └─────────────────────────────────────────────────────────┘  │
│ ┌─────────────────────────────────────────────────────────┐  │
│ │ GPU 1: RTX Pro 6000 (97GB)                              │  │
│ │ ┌─────────┐ ┌─────────┐ ┌─────────┐                    │  │
│ │ │  MIG-3  │ │  MIG-4  │ │  MIG-5  │                    │  │
│ │ │1g.24gb  │ │1g.24gb  │ │2g.48gb  │                    │  │
│ │ │ 46 SM   │ │ 46 SM   │ │ 94 SM   │                    │  │
│ │ │MPS: E,F │ │ MPS: G  │ │ MPS: H  │                    │  │
│ │ └─────────┘ └─────────┘ └─────────┘                    │  │
│ │ Physical: 186 SM | HBM3: 97GB | L2: 96MB               │  │
│ └─────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────┘
```

### MIG Instance with MPS Time-Slicing

```
┌────────────────────────────────────────────────────────┐
│           MIG-0 (1g.24gb, 46 SM)                      │
├────────────────────────────────────────────────────────┤
│  SM Utilization Over Time (MPS Time-Slicing)         │
│  ┌──────────────────────────────────────────────────┐ │
│  │ [Tenant A] [Tenant B] [A] [Tenant C] [B] [A]     │ │
│  │    25%       35%      15%    20%     15%  10%    │ │
│  └──────────────────────────────────────────────────┘ │
│  ↑          ↑          ↑          ↑                   │
│  t=0       t=50ms     t=100ms    t=150ms              │
│                                                        │
│  • Context switching preserves state                  │
│  • No preemption overhead                             │
│  • Cooperative time-slicing                           │
└────────────────────────────────────────────────────────┘
```

---

## 🧩 Components

### 1️⃣ **libbless.so** - Kernel Scheduling Library

**Purpose**: LD_PRELOAD interception layer for CUDA API hooking and context management

**Key Functions**:
- **CUDA API Interception**: Hooks `cudaLaunchKernel`, `cuLaunchKernel`, memory ops
- **Dual Context Management**: LIMITED (SM-constrained) + UNLIMITED (full GPU)
- **Credit-Based Gating**: Token bucket with burst optimization (16-kernel cache)
- **Boost Mode**: Bypass credit gate for priority execution
- **Squad & Share Quota**: Batch scheduling with scheduler decision signals
- **Dynamic SM Reconfiguration**: Runtime SM allocation adjustment
- **Control Socket**: `/tmp/bless-{pid}.sock` for runtime commands

**Environment Variables**:
```bash
BLESS_LIMIT_PCT=50        # Initial SM percentage for LIMITED context (default: 50%)
BLESS_MASTER=/path/sock   # Master coordinator socket path
BLESS_TENANT=tenant_name  # Tenant identifier for multi-tenant setups
```

**Control Commands** (via socket):
```bash
limited              # Switch to LIMITED context
unlimited            # Switch to UNLIMITED context
set_route 0|1        # Set route (0=LIMITED, 1=UNLIMITED)
pause                # Pause kernel launches
resume               # Resume kernel launches
boost_on             # Enable boost mode (bypass credit gate)
boost_off            # Disable boost mode
set_squad N          # Set squad size (batch size)
set_share N          # Set share quota (preemption trigger)
credit_set N         # Set credit quota (N kernels allowed)
credit_off           # Disable credit gating
reconf_sm N          # Reconfigure LIMITED context to N SMs
set_limit_pct N      # Set LIMITED context to N% of total SMs
quit                 # Shutdown control server
```

---

### 2️⃣ **blessctl** - Control Plane CLI

**Purpose**: External orchestrator for sending commands to tenant processes

**Usage**:
```bash
# List all active bless tenants
blessctl --list

# Target by PID
blessctl -p 12345 boost_on
blessctl -p 12345 13456 set_limit_pct 33

# Target by tenant name
blessctl -t TenantA pause
blessctl -t TenantA TenantB resume

# Target by socket path
blessctl -S /tmp/bless-12345.sock credit_set 1000

# Broadcast to all tenants
blessctl -A set_squad 100

# Backward-compatible syntax
blessctl 12345 set_share 250
```

**Key Features**:
- Socket discovery via `/tmp/bless-*.sock` scanning
- Tenant discovery via `/proc/{pid}/environ` parsing
- Multi-target batch commands
- Retry logic with configurable delays
- Pretty-printed tenant status table

---

### 3️⃣ **PyTorch Safe GEMM Wrapper**

**Purpose**: Ensure PyTorch operations respect bless context routing

**Usage**:
```python
import bless_torch  # Custom module

# Use safe linear layer (CPU-side param init)
model = nn.Sequential(
    bless_torch.MyLinear(512, 1024),  # Instead of nn.Linear
    nn.ReLU(),
    bless_torch.MyLinear(1024, 10)
)

# Or use safe matmul directly
result = bless_torch.matmul2d(A, B)
```

**Key Features**:
- Dynamic symbol loading via `ctypes.CDLL(None)`
- Pre-operation context binding
- CPU-side parameter initialization (avoids context conflicts)
- Lazy `.to(device)` in forward pass
- Type coercion & contiguous memory layout

**Environment Variables**:
```bash
BLESS_SAFE_GEMM=1  # Enable safe wrapper (default: ON)
```

---

## 🚀 Getting Started

### Prerequisites

- **GPU**: NVIDIA RTX Pro 6000 Blackwell or MIG-capable GPU
- **Driver**: NVIDIA 580.x+ with MIG support
- **CUDA**: 12.0+
- **OS**: Linux with `/proc` filesystem

### Setup MIG + MPS

```bash
# 1. Enable MIG mode
sudo nvidia-smi -i 0,1 -mig 1

# 2. Create MIG instances (1g + 1g + 2g per GPU)
sudo nvidia-smi mig -cgi 14,14,5 -C -i 0
sudo nvidia-smi mig -cgi 14,14,5 -C -i 1

# 3. Start MPS daemon
sudo nvidia-cuda-mps-control -d

# 4. Verify setup
nvidia-smi
nvidia-smi mig -lgi
```

Or use the automated script:
```bash
sudo ./setup_mig_mps.sh
```

### Build libbless.so

```bash
# Compile the interception library
g++ -O3 -shared -fPIC -o libbless.so libbless.cpp \
    -I/usr/local/cuda/include \
    -L/usr/local/cuda/lib64 \
    -lcuda -lcudart -lpthread -ldl

# Verify symbols
nm -D libbless.so | grep bless_
```

### Run a Tenant

```bash
# Launch with LD_PRELOAD
LD_PRELOAD=./libbless.so \
BLESS_LIMIT_PCT=30 \
BLESS_TENANT=TenantA \
CUDA_VISIBLE_DEVICES=MIG-xxx-xxx \
python train.py

# In another terminal, control the tenant
blessctl -t TenantA set_squad 200
blessctl -t TenantA credit_set 5000
blessctl -t TenantA boost_on
```

---

## ⚙️ Configuration

### Scenario 1: Fixed SM Allocation

```bash
# Tenant A: 30% SM, strict credit limit
LD_PRELOAD=./libbless.so \
BLESS_LIMIT_PCT=30 \
BLESS_TENANT=TenantA \
python train.py &

# Control via blessctl
blessctl -t TenantA credit_set 1000  # 1000 kernels per time window
blessctl -t TenantA set_squad 100    # Batch size = 100
```

### Scenario 2: Dynamic SM Scaling

```bash
# Start with 50% SM
LD_PRELOAD=./libbless.so \
BLESS_LIMIT_PCT=50 \
BLESS_TENANT=TenantB \
python inference.py &

# Scale down at runtime
blessctl -t TenantB set_limit_pct 25
blessctl -t TenantB reconf_sm 46  # Explicit SM count
```

### Scenario 3: Boost for Priority Workloads

```bash
# Normal execution with credit gating
blessctl -t TenantC credit_set 2000

# Urgent task arrives → boost
blessctl -t TenantC boost_on
# ... run priority job ...
blessctl -t TenantC boost_off
```

### Scenario 4: Pause/Resume for Maintenance

```bash
# Pause all tenants
blessctl -A pause

# Perform GPU maintenance (e.g., profiling, checkpoint)
# ...

# Resume execution
blessctl -A resume
```

---

## 📚 API Reference

### C API (libbless.so)

```c
// Query current SM affinity
int bless_query_sm_affinity();

// Bind calling thread to specific route
void bless_bind_thread(int route);  // 0=LIMITED, 1=UNLIMITED

// Query current route
int bless_current_route();  // Returns 0 or 1

// Query squad progress
int bless_squad_progress();

// Query kernel launch sequence number
long long bless_kernel_seq();

// Check if boost mode is active
int bless_is_boosting();  // Returns 1 if boost, 0 otherwise
```

**Example**:
```c
#include <dlfcn.h>

typedef int (*query_fn)();
void* lib = dlopen(NULL, RTLD_NOW);
query_fn query = (query_fn)dlsym(lib, "bless_query_sm_affinity");
int sms = query();
printf("Current SM affinity: %d\n", sms);
```

### Python API (ctypes)

```python
import ctypes

lib = ctypes.CDLL(None)
bind_fn = lib.bless_bind_thread
bind_fn.argtypes = [ctypes.c_int]
bind_fn(1)  # Bind to UNLIMITED route

route_fn = lib.bless_current_route
route_fn.restype = ctypes.c_int
current = route_fn()
print(f"Current route: {current}")
```
