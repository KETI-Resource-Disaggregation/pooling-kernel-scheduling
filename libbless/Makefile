CUDA_HOME ?= /usr/local/cuda
CXXFLAGS  := -fPIC -O2 -std=c++17 -I$(CUDA_HOME)/include
LDFLAGS   := -shared -L$(CUDA_HOME)/lib64 -lcuda -lcudart -ldl -lpthread

all: libbless.so bench

libbless.so: libbless.cpp context_manager.hpp routing.hpp
	$(CXX) $(CXXFLAGS) -o $@ libbless.cpp $(LDFLAGS)

bench: ../tests/bench_kernels.cu
	nvcc -O3 -o ../tests/bench ../tests/bench_kernels.cu

clean:
	rm -f libbless.so ../tests/bench