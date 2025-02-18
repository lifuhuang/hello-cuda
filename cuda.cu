#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <iostream>

#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define uint unsigned int

inline unsigned int cdiv(unsigned int a, unsigned int b) { return (a + b - 1) / b;}

__device__ void corank(uint k, float *a, uint lenA, float *b, uint lenB) {

}

__global__ void kernel_merge(float *a, uint lenA, float *b, uint lenB, float *c, uint lenC) {
    // uint c_start = blockIdx.x * blockDim.x;
    // uint c_end = min((blockIdx.x + 1) * blockDim.x, lenC);
}

torch::Tensor merge(torch::Tensor a, torch::Tensor b) {
    CHECK_INPUT(a);
    CHECK_INPUT(b);

    uint lenA = a.size(0);
    uint lenB = b.size(0);
    uint lenC = lenA + lenB;
    auto c = torch::empty(lenC, a.options());

    uint blockSize = 256;
    uint numBlocks = cdiv(lenC, blockSize);
    kernel_merge<<<numBlocks, blockSize>>>(a.data_ptr<float>(), lenA, b.data_ptr<float>(), lenB, c.data_ptr<float>(), lenC);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return c;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("merge", &merge, "Tensor Merge");
}

int main() {
    torch::Tensor a = torch::arange(0, 1000000, torch::kFloat).cuda();
    torch::Tensor b = torch::arange(0, 1000000, torch::kFloat).cuda();
    auto c = merge(a, b);

    auto res = std::get<0>(torch::concat({a, b}, 0).sort(0)).equal(c);
    printf("correct = %s\n", res ? "true" : "false");
}