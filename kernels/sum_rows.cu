#include <cstdint>

__global__ void sum_rows(float *dst, const float *inp, int64_t ncols, int64_t nrows)
{
    const int64_t i = static_cast<int64_t>(blockDim.x) * blockIdx.x + threadIdx.x;

    if (i >= ncols)
    {
        return;
    }
    float total = 0.f;
    for (int64_t row = 0; row < nrows; ++row)
    {
        total += inp[i * nrows + row];
    }
    dst[i] = total;
}