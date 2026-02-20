// Elementwise operations: add_inplace and scale_inplace.

// dst[i] += src[i]
extern "C" __global__ void add_inplace(
    float* __restrict__ dst,
    const float* __restrict__ src,
    int n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < n; i += stride) {
        dst[i] += src[i];
    }
}

// dst[i] *= factor
extern "C" __global__ void scale_inplace(
    float* __restrict__ dst,
    float factor,
    int n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < n; i += stride) {
        dst[i] *= factor;
    }
}
