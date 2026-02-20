// In-place softmax: values[i] = exp(values[i] - max) / sum(exp(values - max))
//
// Three-pass single-block kernel:
//   1. Find max (shared-memory reduction)
//   2. Compute exp(x - max) and sum (shared-memory reduction)
//   3. Normalize by dividing each element by sum

extern "C" __global__ void softmax(
    float* __restrict__ values,
    int n
) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int stride = blockDim.x;

    // Pass 1: find max.
    float local_max = -1e30f;
    for (int i = tid; i < n; i += stride) {
        local_max = fmaxf(local_max, values[i]);
    }
    sdata[tid] = local_max;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    float max_val = sdata[0];
    __syncthreads();

    // Pass 2: exp and sum.
    float local_sum = 0.0f;
    for (int i = tid; i < n; i += stride) {
        float e = expf(values[i] - max_val);
        values[i] = e;
        local_sum += e;
    }
    sdata[tid] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    float total = sdata[0];
    __syncthreads();

    // Pass 3: normalize.
    if (total != 0.0f) {
        float inv_total = 1.0f / total;
        for (int i = tid; i < n; i += stride) {
            values[i] *= inv_total;
        }
    }
}
