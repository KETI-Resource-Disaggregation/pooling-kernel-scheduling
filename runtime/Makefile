CXX      = g++
CXXFLAGS = -O2 -fPIC -std=c++17 \
           -I../shm \
           -Iinclude

# CUDA_HOME 기본값 (심볼릭 링크 없으면 버전 직접 지정)
CUDA_HOME ?= $(shell ls -d /usr/local/cuda-* 2>/dev/null | sort -V | tail -1)
CUDA_HOME := $(if $(CUDA_HOME),$(CUDA_HOME),/usr/local/cuda)
CUDA_LIB   = $(CUDA_HOME)/targets/x86_64-linux/lib

CXXFLAGS  += -I$(CUDA_HOME)/include
LIBS       = -lcuda -lcudart -ldl -lpthread -lrt -L$(CUDA_LIB) -Wl,-rpath,$(CUDA_LIB)

TARGET = gating_lib.so

SRCS = hooks/cuda_hooks.cpp         \
       gating/stream_gate.cpp       \
       timeslice/time_slice.cpp     \
       timeslice/round_manager.cpp  \
       resource/virtual_sm.cpp      \
       resource/virtual_mem.cpp     \
       shm/shm_client.cpp

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) -shared -o $@ $^ $(LIBS)

clean:
	rm -f $(TARGET) *.o
