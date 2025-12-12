# Pooling Kernel Scheduling - Makefile

NVCC = nvcc
CXX = g++
NVCC_FLAGS = -O3 -arch=sm_86 -cudart shared
CXX_FLAGS = -O2 -std=c++17

.PHONY: all clean libbless colocation_scheduler test

all: libbless colocation_scheduler

# ==================== libbless ====================
libbless:
	@echo "Building libbless..."
	$(MAKE) -C libbless

# ==================== colocation_scheduler ====================
colocation_scheduler:
	@echo "Building colocation_scheduler..."
	$(NVCC) $(NVCC_FLAGS) -o colocation_scheduler/colocation_demo \
		colocation_scheduler/colocation_demo.cu \
		colocation_scheduler/colocation_scheduler.cpp \
		-I colocation_scheduler
	$(CXX) $(CXX_FLAGS) -DSCHEDULER_TEST_MAIN -o colocation_scheduler/scheduler_test \
		colocation_scheduler/colocation_scheduler.cpp

# ==================== Clean ====================
clean:
	@echo "Cleaning..."
	$(MAKE) -C libbless clean
	rm -f colocation_scheduler/colocation_demo
	rm -f colocation_scheduler/scheduler_test

# ==================== Test ====================
test: all
	@echo "Running tests..."
	@echo "\n=== Scheduler Test ===" && cd colocation_scheduler && ./scheduler_test
	@echo "\n=== Colocation Demo ===" && cd colocation_scheduler && ./colocation_demo

# ==================== Run Services ====================
.PHONY: run-scheduler run-api

run-scheduler:
	@echo "Starting SPARK Scheduler..."
	python3 controller/scheduler.py

run-api:
	@echo "Starting Scheduler API..."
	python3 controller/sched_api.py
