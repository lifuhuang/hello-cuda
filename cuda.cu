#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <assert.h>

#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__host__ __device__ inline int cdiv(int a, int b) { return (a + b - 1) / b;}

template <int M>
__device__ int corank(int i, float *a, int curA, int m, float *b, int curB, int n) {
    // max(0, i - n) <= j <= min(i, m)
    // max(0, i - m) <= k <= min(i, n)
    int lo = max(0, i - n), hi = min(i, m);
    while (lo <= hi) {
        int j = (lo + hi) / 2;
        int k = i - j;
        if (j > 0 && k < n && a[(curA + j - 1) % M] > b[(curB + k) % M]) {
            hi = j - 1;
        } else if (k > 0 && j < m && b[(curB + k - 1) % M] >= a[(curA + j) % M]) {
            lo = j + 1;
        } else {
            return j;
        }
    }
    assert(false);
}

template <int M>
__device__ void sequential_merge(float *c, int cStart, float *a, int curA, int lenA, float *b, int curB, int lenB) {
    int i = 0, j = 0;
    while (i < lenA && j < lenB) {
        if (a[(curA + i) % M] < b[(curB + j) % M]) {
            c[cStart++] = a[(curA + i++) % M];
        } else {
            c[cStart++] = b[(curB + j++) % M];
        }
    }
    while (i < lenA) {
        c[cStart++] = a[(curA + i++) % M];
    }
    while (j < lenB) {
        c[cStart++] = b[(curB + j++) % M];
    }
}

constexpr int INF = 0x7fffffff;
template <int TILE_SIZE>
__global__ void kernel_merge(float *a, int lenA, float *b, int lenB, float *c) {
    extern __shared__ float tile_buffer[];
    int tx = threadIdx.x;
    int lenC = lenA + lenB;
    int windowLength = cdiv(lenC, gridDim.x);
    int cBlockStart = blockIdx.x * windowLength;
    int cBlockEnd = min((blockIdx.x + 1) * windowLength, lenC);
    float *aTile_s = tile_buffer;
    float *bTile_s = tile_buffer + TILE_SIZE;

    if (tx == 0) {
        aTile_s[0] = corank<INF>(cBlockStart, a, 0, lenA, b, 0, lenB);
    }
    __syncthreads();
    int aBlockStart = aTile_s[0];
    int bBlockStart = cBlockStart - aBlockStart;
    int aConsumed = 0;
    int bConsumed = 0;
    int cCompleted = 0;
    int aProduced = TILE_SIZE;
    int bProduced = TILE_SIZE;

    for (int cStart = cBlockStart; cStart < cBlockEnd; cStart += TILE_SIZE) {
        __syncthreads();
        const int cEnd = min(cStart + TILE_SIZE, cBlockEnd);
 
        for (int i = tx; i < TILE_SIZE; i += blockDim.x) {
            if (i + aConsumed < aProduced && aBlockStart + aConsumed + i < lenA) {
                aTile_s[(i + aConsumed) % TILE_SIZE] = a[aBlockStart + aConsumed + i];
            }
            if (i + aConsumed < aProduced && bBlockStart + bConsumed + i < lenB) {
                bTile_s[(i + bConsumed) % TILE_SIZE] = b[bBlockStart + bConsumed + i];
            }
        }
        __syncthreads();
        
        int threadLength = cdiv(cEnd - cStart, blockDim.x);
        int cThreadStart = min(tx * threadLength, cEnd - cStart);
        int cThreadEnd = min((tx + 1) * threadLength, cEnd - cStart);
        int asLength = aBlockStart + aConsumed + TILE_SIZE <= lenA ? TILE_SIZE : lenA - (aBlockStart + aConsumed);
        int bsLength = bBlockStart + bConsumed + TILE_SIZE <= lenB ? TILE_SIZE : lenB - (bBlockStart + bConsumed);
        int aThreadStart = corank<TILE_SIZE>(cThreadStart, aTile_s, aConsumed, asLength, bTile_s, bConsumed, bsLength);
        int aThreadEnd = corank<TILE_SIZE>(cThreadEnd, aTile_s, aConsumed, asLength, bTile_s, bConsumed, bsLength);
        int bThreadStart = cThreadStart - aThreadStart;
        int bThreadEnd = cThreadEnd - aThreadEnd;
        sequential_merge<TILE_SIZE>(c, cStart + cThreadStart, aTile_s, aConsumed + aThreadStart, aThreadEnd - aThreadStart, bTile_s, bConsumed + bThreadStart, bThreadEnd - bThreadStart);
        if (cStart + TILE_SIZE < cBlockEnd) {
            cCompleted += TILE_SIZE;
            aConsumed += corank<TILE_SIZE>(TILE_SIZE, aTile_s, aConsumed, TILE_SIZE, bTile_s, bConsumed, TILE_SIZE);
            aProduced = aConsumed + TILE_SIZE;
            bConsumed = cCompleted - aConsumed;
            bProduced = bConsumed + TILE_SIZE;
        }
    }
}

torch::Tensor merge(torch::Tensor a, torch::Tensor b) {
    CHECK_INPUT(a);
    CHECK_INPUT(b);

    int lenA = a.size(0);
    int lenB = b.size(0);
    int lenC = lenA + lenB;
    auto c = torch::empty(lenC, a.options());

    constexpr int TILE_SIZE = 8192;
    constexpr int TILES_PER_BLOCK = 4;
    int blockSize = 1024;
    int numBlocks = cdiv(lenC, TILE_SIZE * TILES_PER_BLOCK);
    
    cudaFuncSetAttribute(kernel_merge<TILE_SIZE>, cudaFuncAttributeMaxDynamicSharedMemorySize , 0x10000);
    kernel_merge<TILE_SIZE><<<numBlocks, blockSize, 0x10000>>>(a.data_ptr<float>(), lenA, b.data_ptr<float>(), lenB, c.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return c;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("merge", &merge, "Tensor Merge");
}

using namespace std;
int main() {
    int sizes[] = {100, 1000000, 10000000};

    for (int i = 0; i < 3; ++i ) {
        int size = sizes[i];
        torch::Tensor a = std::get<0>(torch::randn({size}, torch::kFloat).cuda().sort(0));
        torch::Tensor b = std::get<0>(torch::randn({size}, torch::kFloat).cuda().sort(0));
        auto c = merge(a, b);
        auto c_p = std::get<0>(torch::concat({a, b}, 0).sort(0));
        auto res = c.equal(c_p);
        printf("correct = %s\n", res ? "true" : "false");
    }

    return 0;
}
