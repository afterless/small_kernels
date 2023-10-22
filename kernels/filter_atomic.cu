#include <cstdint>

__global__ void filter_atomic(const float *inp, float *dst, int *cnt, int size, const float thresh)
{
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (i >= size)
    {
        return;
    }

    if (std::abs(inp[i]) < thresh)
    {
        dst[atomicAdd(cnt, 1)] = inp[i];
    }
}