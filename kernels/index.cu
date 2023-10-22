#include <cstdint>

__global__ void index(const float *a, const int64_t *b, float *dst, int64_t aSize, int64_t size)
{
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (i >= size)
    {
        return;
    }

    int64_t idx = b[i];
    if (idx < 0 || idx >= aSize)
    {
        printf("\nidx out of range");
        return;
    }

    dst[i] = a[idx];
}