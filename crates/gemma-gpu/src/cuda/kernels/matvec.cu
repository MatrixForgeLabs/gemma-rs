// Matrix-vector multiply: y = A * x
// A is row-major (rows × cols), x has cols elements, y has rows elements.
//
// Each thread block handles one row. Threads within the block cooperatively
// compute the dot product using shared memory reduction.

extern "C" __global__
void matvec_f32(const float* __restrict__ A,
                const float* __restrict__ x,
                float* __restrict__ y,
                int rows,
                int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;

    extern __shared__ float sdata[];

    const float* row_ptr = A + row * cols;
    float sum = 0.0f;

    // Grid-stride accumulation within the block
    for (int j = threadIdx.x; j < cols; j += blockDim.x) {
        sum += row_ptr[j] * x[j];
    }

    sdata[threadIdx.x] = sum;
    __syncthreads();

    // Tree reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        y[row] = sdata[0];
    }
}
