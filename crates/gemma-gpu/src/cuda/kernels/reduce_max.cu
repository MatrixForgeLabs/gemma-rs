extern "C" __global__ void reduce_block_max(const float* __restrict__ x, int n, float* out_vals, int* out_idx) {
    // Each block reduces a contiguous chunk. blockIdx.x selects chunk.
    int start = blockIdx.x * blockDim.x;
    int idx = start + threadIdx.x;
    float val = -1e30f;
    int best = -1;
    if (idx < n) {
        val = x[idx];
        best = idx;
    }
    // reduce in block
    __shared__ float s_val[256];
    __shared__ int s_idx[256];
    s_val[threadIdx.x] = val;
    s_idx[threadIdx.x] = best;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            float other = s_val[threadIdx.x + offset];
            int other_idx = s_idx[threadIdx.x + offset];
            if (other > s_val[threadIdx.x]) {
                s_val[threadIdx.x] = other;
                s_idx[threadIdx.x] = other_idx;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        out_vals[blockIdx.x] = s_val[0];
        out_idx[blockIdx.x] = s_idx[0];
    }
}
