#include "snake_1d.cuh"

// Snake 1D: y = x + sin^2(alpha * x) / (alpha + 1e-9)
// Input:  x     [T, C] F32  (time-first, ne[0]=T, ne[1]=C)
//         alpha [C]    F32  per-channel parameter
// Output: dst   [T, C] F32  same shape as x

static __global__ void snake_1d_kernel(
        const float * __restrict__ x,
        const float * __restrict__ alpha,
        float * __restrict__ dst,
        int T, int C) {

    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    const int total = C * T;
    if (idx >= total) return;

    const int ic = idx / T;  // ne[0]=T, flat: t + c*T → c = idx/T
    const float a = alpha[ic] + 1e-9f;
    const float ax = a * x[idx];
    const float s = sinf(ax);
    dst[idx] = x[idx] + s * s / a;
}

void ggml_cuda_op_snake_1d(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x     = dst->src[0];
    const ggml_tensor * alpha = dst->src[1];

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(alpha->type == GGML_TYPE_F32);

    const int T = (int)x->ne[0];
    const int C = (int)x->ne[1];
    const int total = C * T;

    const float * x_d = (const float *)x->data;
    const float * a_d = (const float *)alpha->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    const int block_size = 256;
    const int num_blocks = (total + block_size - 1) / block_size;
    snake_1d_kernel<<<num_blocks, block_size, 0, stream>>>(x_d, a_d, dst_d, T, C);
}
