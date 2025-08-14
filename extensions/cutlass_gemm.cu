// extensions/cutlass_gemm.cu
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/library.h>

#ifndef USE_CUTLASS
#define USE_CUTLASS 0
#endif

// (빠른 안정화를 위해 fp32만)
// A:[M,K], B:[K,N] -> C:[M,N]
__global__ void gemm_naive_f32(const float* __restrict__ A,
                               const float* __restrict__ B,
                               float* __restrict__ C,
                               int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float acc = 0.f;
        #pragma unroll 4
        for (int k = 0; k < K; ++k) {
            acc += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = acc;
    }
}

static at::Tensor bless_gemm_impl(const at::Tensor& A, const at::Tensor& B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "A/B must be CUDA")
    TORCH_CHECK(A.scalar_type() == at::kFloat && B.scalar_type() == at::kFloat, "fp32 only in naive build")
    TORCH_CHECK(A.dim()==2 && B.dim()==2, "expect 2D")
    TORCH_CHECK(A.size(1) == B.size(0), "K mismatch")

    auto M = (int)A.size(0);
    auto K = (int)A.size(1);
    auto N = (int)B.size(1);

    auto C = at::empty({M, N}, A.options());
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1)/block.x, (M + block.y - 1)/block.y);
    auto stream = at::cuda::getCurrentCUDAStream();

    gemm_naive_f32<<<grid, block, 0, stream>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);
    return C;
}

TORCH_LIBRARY(bless, m) { m.def("gemm(Tensor A, Tensor B) -> Tensor"); }
TORCH_LIBRARY_IMPL(bless, CUDA, m) {
    m.impl("gemm", [](const at::Tensor& A, const at::Tensor& B){
        return bless_gemm_impl(A, B);
    });
}

#include <pybind11/pybind11.h>
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}  // 파이썬 모듈 초기화 스텁