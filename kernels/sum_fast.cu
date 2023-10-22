#include <cstdint>

__global__ void sum_1024(float *dst, const float *inp, int64_t size)
{
    static __shared__ float data[1024];
    int64_t tidx = threadIdx.x;
    int64_t idx = threadIdx.x + blockIdx.x * 1024;
    data[tidx] = (idx < size) ? inp[idx] : 0;

    for (int chunk_size = 1024 / 2; chunk_size > 0; chunk_size /= 2)
    {
        __syncthreads();
        if (tidx < chunk_size)
        {
            data[tidx] += data[tidx + chunk_size];
        }
    }

    if (tidx == 0)
    {
        dst[blockIdx.x] = data[tidx];
    }
}