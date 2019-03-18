//#include <ATen/ATen.h>
//#include <torch/torch.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <assert.h>
# include "cublas_v2.h"
#define DIM_GRID 128
#define DIM_BLOCK 1024

#define DEF_KERNEL __global__
#define start_dev_kernel(kernel, args) (kernel<<<DIM_GRID,DIM_BLOCK,0>>>  args);
#define elem_atomic_add(x, v) atomicAdd(x, v)
#define CudaMemcpy(y, x, size) (cudaMemcpyAsync(y, x, size, cudaMemcpyDeviceToDevice))
#define CudaMemset(s, c, size) (cudaMemsetAsync(s, c, size, 0))

DEF_KERNEL void lstm_fwd_kernel_for_low_memory(
    int n_batch,
    int n_cells,
    const float* mask,
    float* intern,
    float* prev_c,
    float* y,
    float* c)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    while (idx < n_cells * n_batch) {
        int batch_idx = idx / n_cells;
        int cell_idx = idx % n_cells;
        int intern_offset = batch_idx * 4 * n_cells + cell_idx;
        float prev_c_b = prev_c[idx];
        float mask_b = mask[batch_idx];

        // cell-in + input, forget and output gates
        float cellIn = tanhf(intern[intern_offset]);
        float inpGate = 1.f / (1.f + expf(-intern[intern_offset + n_cells]));
        float fgtGate = 1.f / (1.f + expf(-intern[intern_offset + 2 * n_cells]));
        float outGate = 1.f / (1.f + expf(-intern[intern_offset + 3 * n_cells]));

        float c_b = (prev_c_b * fgtGate + cellIn * inpGate) * mask_b + prev_c_b * (1.f - mask_b);
        c[idx] = c_b;
        y[idx] = tanhf(c_b) * outGate * mask_b;

        idx += gridDim.x * blockDim.x;
    }
}

DEF_KERNEL void lstm_bwd_kernel_for_low_memory(
    int n_batch, int n_in, int n_cells, const float* mask,
    float* x_h,
    float* intern,
    float* prev_c,
    float* y,
    float* c,
    float* d_y,
    float* d_h,
    float* d_c,
    float* d_intern,
    float* d_b)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    while (idx < n_cells * n_batch) {
        int batch_idx = idx / n_cells;
        int cell_idx = idx % n_cells;
        int intern_offset = batch_idx * 4 * n_cells + cell_idx;
        float mask_b = mask[batch_idx];
        float d_y_b = d_y[idx] * mask_b + d_h[idx];
        float d_c_b = d_c[idx] * mask_b;
        float prev_c_b = prev_c[idx];

        // cell-in + input, forget and output gates
        float cellIn = tanhf(intern[intern_offset]);
        float inpGate = 1.f / (1.f + expf(-intern[intern_offset + n_cells]));
        float fgtGate = 1.f / (1.f + expf(-intern[intern_offset + 2 * n_cells]));
        float outGate = 1.f / (1.f + expf(-intern[intern_offset + 3 * n_cells]));

        float c_b = prev_c_b * fgtGate + cellIn * inpGate;
        float gc = tanhf(c_b);
        float d_outGate_in = (1.f - outGate) * outGate * gc * d_y_b;
        float d_c2 = d_c_b + outGate * d_y_b * (1.f - gc * gc);
        float d_cellIn_in = (1.f - cellIn * cellIn) * inpGate * d_c2;
        float d_inpGate_in = (1.f - inpGate) * inpGate * cellIn * d_c2;
        float d_fgtGate_in = (1.f - fgtGate) * fgtGate * prev_c_b * d_c2;
        d_c[idx] = fgtGate * d_c2 + d_c[idx] * (1.f - mask_b);

        d_intern[intern_offset] = d_cellIn_in;
        d_intern[intern_offset + n_cells] = d_inpGate_in;
        d_intern[intern_offset + 2 * n_cells] = d_fgtGate_in;
        d_intern[intern_offset + 3 * n_cells] = d_outGate_in;

        elem_atomic_add(&d_b[cell_idx], d_cellIn_in);
        elem_atomic_add(&d_b[cell_idx + n_cells], d_inpGate_in);
        elem_atomic_add(&d_b[cell_idx + 2 * n_cells], d_fgtGate_in);
        elem_atomic_add(&d_b[cell_idx + 3 * n_cells], d_outGate_in);

        idx += gridDim.x * blockDim.x;
    }
}

DEF_KERNEL void lstm_fwd_kernel(
    int n_batch, int n_cells, const float* mask,
    float* h,
    float* prev_y,
    float* prev_c,
    float* y,
    float* c,
    float* y_prev_out
)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    while (idx < n_cells * n_batch) {
        int batch_idx = idx / n_cells;
        int cell_idx = idx % n_cells;
        int intern_offset = batch_idx * 4 * n_cells + cell_idx;
        float prev_c_b = prev_c[idx];
        float mask_b = mask[batch_idx];

        // cell-in + input, forget and output gates
        float cellIn = tanhf(h[intern_offset]);
        float inpGate = 1.f / (1.f + expf(-h[intern_offset + n_cells]));
        float fgtGate = 1.f / (1.f + expf(-h[intern_offset + 2 * n_cells]));
        float outGate = 1.f / (1.f + expf(-h[intern_offset + 3 * n_cells]));

        h[intern_offset] = cellIn;
        h[intern_offset + n_cells] = inpGate;
        h[intern_offset + 2 * n_cells] = fgtGate;
        h[intern_offset + 3 * n_cells] = outGate;

        float c_b = (prev_c_b * fgtGate + cellIn * inpGate) * mask_b + prev_c_b * (1.f - mask_b);
        c[idx] = c_b;
        float y_b = tanhf(c_b) * outGate * mask_b;
        y[idx] = y_b;
        y_prev_out[idx] = y_b + prev_y[idx] * (1.f - mask_b);

        idx += gridDim.x * blockDim.x;
    }
}

DEF_KERNEL void lstm_bwd_kernel(
    int n_batch, int n_cells, const float* mask,
    float* h,
    float* prev_c,
    float* y,
    float* c,
    float* d_y,
    float* d_h,
    float* d_c,
    float* d_x,
    float* d_x0
)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    while (idx < n_cells * n_batch) {
        int batch_idx = idx / n_cells;
        int cell_idx = idx % n_cells;
        int intern_offset = batch_idx * 4 * n_cells + cell_idx;
        float mask_b = mask[batch_idx];
        float d_y_b = (d_y[idx] + d_h[idx]) * mask_b;
        float d_c_b = d_c[idx] * mask_b;
        float prev_c_b = prev_c[idx];

        // cell-in + input, forget and output gates
        float cellIn = h[intern_offset];
        float inpGate = h[intern_offset + n_cells];
        float fgtGate = h[intern_offset + 2 * n_cells];
        float outGate = h[intern_offset + 3 * n_cells];

        float c_b = prev_c_b * fgtGate + cellIn * inpGate;
        float gc = tanhf(c_b);
        float d_outGate_in = (1.f - outGate) * outGate * gc * d_y_b;
        float d_c2 = d_c_b + outGate * d_y_b * (1.f - gc * gc);
        float d_cellIn_in = (1.f - cellIn * cellIn) * inpGate * d_c2;
        float d_inpGate_in = (1.f - inpGate) * inpGate * cellIn * d_c2;
        float d_fgtGate_in = (1.f - fgtGate) * fgtGate * prev_c_b * d_c2;
        d_c[idx] = fgtGate * d_c2 + d_c[idx] * (1.f - mask_b);

        d_x[intern_offset] = d_cellIn_in;
        d_x[intern_offset + n_cells] = d_inpGate_in;
        d_x[intern_offset + 2 * n_cells] = d_fgtGate_in;
        d_x[intern_offset + 3 * n_cells] = d_outGate_in;

        #define set_x0(off) { d_x0[off] = d_x[off] + d_x0[off] * (1.f - mask_b); }
        set_x0(intern_offset);
        set_x0(intern_offset + n_cells);
        set_x0(intern_offset + 2 * n_cells);
        set_x0(intern_offset + 3 * n_cells);
        #undef set_x0

    // Reset if used frame, otherwise leave as-is.
        d_h[idx] *= (1.f - mask_b);

        idx += gridDim.x * blockDim.x;
    }
}

DEF_KERNEL
void lstm_fwd_kernel_base(
    float* data, const float* old_state, bool old_state_strided,
    float* output, float* state_out, int n_cells, int n_batch, const float* i) {
    //layout:
    //data[0*n_cells..1*n_cells-1] : cell state
    //data[1*n_cells..2*n_cells-1] : input gate
    //data[2*n_cells..3*n_cells-1] : forget gate
    //data[3*n_cells..4*n_cells-1] : output gate
    //output[0*n_cells..1*n_cells-1]: cell output
    //repeated for every mini-batch

    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    while (idx < n_cells * n_batch) {
        int batch_idx = idx / n_cells;
        int start = batch_idx * 4 * n_cells + idx % n_cells;
        float i_batch = i[batch_idx];

        //input, forget and output gates
        float inpGate = 1.f / (1.f + expf(-data[start + n_cells]));
        float fgtGate = 1.f / (1.f + expf(-data[start + 2 * n_cells]));
        float outGate = 1.f / (1.f + expf(-data[start + 3 * n_cells]));
        float state = inpGate * tanhf(data[start]);
        float old_state_batch = old_state_strided ? old_state[start] : old_state[idx];

        state += fgtGate * old_state_batch;
        state = state * i_batch + old_state_batch * (1.f - i_batch);

        //cell output
        output[idx] = outGate * tanhf(state) * i_batch;

        data[start] = state;
        data[start + n_cells] = inpGate;
        data[start + 2 * n_cells] = fgtGate;
        data[start + 3 * n_cells] = outGate;
        if(state_out)
        state_out[idx] = state;

        idx += gridDim.x * blockDim.x;
    }
}

DEF_KERNEL
void lstm_bwd_kernel_base(
    float* delta, float* epsilon, const float* next_epsilon, const float* old_state,
    bool old_state_strided, const float* Y, int n_cells, int n_batch, const float* i
    )
{
    //layout:
    //delta[0*n_cells..1*n_cells-1] : input gate
    //delta[1*n_cells..2*n_cells-1] : forget gate
    //delta[2*n_cells..3*n_cells-1] : output gate
    //delta[3*n_cells..4*n_cells-1] : cell state
    //epsilon[0*n_cells..1*n_cells-1]: cell output derivative (later overwritten, see below)
    //next_epsilon[0*n_cells..1*n_cells-1]: cell state derivative * forget_gate (of next timestep)
    //repeated for every mini-batch

    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    while (idx < n_cells * n_batch) {
        int batch_idx = idx / n_cells;
        int batch_offset = batch_idx * 4 * n_cells;
        int cell_offset = idx % n_cells;
        int start = batch_offset + cell_offset;
        float i_batch = i[batch_idx];

        float inpGate = delta[start + n_cells];
        float fgtGate = delta[start + 2 * n_cells];
        float outGate = delta[start + 3 * n_cells];
        float oldState = old_state_strided ? old_state[start] : old_state[idx];
        float state = delta[start];
        float eps = epsilon[idx];

        //avoid division by 0
        float gc = tanhf(state); //g(c(t))
        float gzc = (state - fgtGate * oldState) / fmaxf(inpGate, float(1e-16)); //g(z_c(t))

        //delta_output
        delta[start + 3 * n_cells] = outGate * (1.f - outGate) * gc * eps * i_batch;

        //epsilon_c
        float epsilon_c = (1.f - (gc * gc)) * outGate * eps;
        epsilon_c += next_epsilon[idx];
        epsilon[idx] = epsilon_c * fgtGate * i_batch + next_epsilon[idx] * (1.f - i_batch);

        //delta_cell
        delta[start] = inpGate * (1.f - (gzc * gzc)) * epsilon_c * i_batch;

        //delta_forget
        delta[start + 2 * n_cells] = fgtGate * (1.f - fgtGate) * oldState * epsilon_c * i_batch;

        //delta_input
        delta[start + n_cells] = inpGate * (1.f - inpGate) * gzc * epsilon_c * i_batch;

        idx += gridDim.x * blockDim.x;
    }
}


DEF_KERNEL void add_bias_kernel(int n_batch, int n_dim, float* x, float* b)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    while (idx < n_batch * n_dim) {
        int dim_idx = idx % n_dim;
        x[idx] += b[dim_idx];
        idx += gridDim.x * blockDim.x;
    }
}

DEF_KERNEL void copy_x_h_kernel(int n_batch, int n_in, int n_cells, float* x_h, float* x, float* h)
{
    int n_total_in = n_in + n_cells;
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    while (idx < n_batch * n_total_in) {
        int batch_idx = idx / n_total_in;
        int in_dim_idx = idx % n_total_in;

        if(in_dim_idx < n_in)
        x_h[idx] = x[batch_idx * n_in + in_dim_idx];
        else
        x_h[idx] = h[batch_idx * n_cells + in_dim_idx - n_in];

        idx += gridDim.x * blockDim.x;
    }
}

DEF_KERNEL void inv_copy_x_h_kernel(int n_batch, int n_in, int n_cells, float* x_h, float* x, float* h)
{
    int n_total_in = n_in + n_cells;
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    while (idx < n_batch * n_total_in) {
        int batch_idx = idx / n_total_in;
        int in_dim_idx = idx % n_total_in;

        if(in_dim_idx < n_in)
          x[batch_idx * n_in + in_dim_idx] = x_h[idx];
        else
          h[batch_idx * n_cells + in_dim_idx - n_in] = x_h[idx];

        idx += gridDim.x * blockDim.x;
    }
}

void addmmatb(at::Tensor target,
    at::Tensor a, int r_start, int r_end,
    at::Tensor b, int c_start, int c_end,
    float al, float bet)
{
    assert((r_end - r_start) == (c_end - c_start));
    assert(a.size(0) * a.size(1) == b.size(0) * b.size(1));
    long m = a.size(2);
    long k = (r_end - r_start) * a.size(1);
    long n = b.size(2);
    cublasHandle_t handle;
    cublasCreate(&handle);
    float* da = a.data<float>() + r_start * a.size(1) * a.size(2);
    float* db = b.data<float>() + c_start * b.size(1) * b.size(2);
    float* dc = target.data<float>();
    auto stat = cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,n,m,k,&al,db,n,da,m,&bet,dc,n);
    //cout << a << endl;
    //cout << b << endl;
    //cout << target << endl;
}

void lstm_fwd_func_low_mem(
    at::Tensor X, at::Tensor W, at::Tensor b, at::Tensor y0, at::Tensor c0, at::Tensor i,
    int start, int step,
    at::Tensor Y, at::Tensor C, at::Tensor d
)
{
    assert(X.dim() == 3);   assert(i.dim() == 2);
    assert(W.dim() == 2);   assert(b.dim() == 1);
    assert(y0.dim() == 2);  assert(c0.dim() == 2);
    assert(Y.dim() == 3);   assert(C.dim() == 3);    assert(d.dim() == 2);

    long T = i.size(0);
    int n_batch = i.size(1);
    int n_cells = y0.size(1);
    int n_in = X.size(2);

    assert(X.size(0) == T);                    assert(X.size(1) == n_batch);
    assert(W.size(0) == n_in + n_cells);       assert(W.size(1) == n_cells * 4);
    assert(b.size(0) == n_cells * 4);
    assert(y0.size(0) == n_batch);             assert(y0.size(1) == n_cells);
    assert(c0.size(0) == n_batch);             assert(c0.size(1) == n_cells);
    assert(Y.size(0) == T);                    assert(Y.size(1) == n_batch);          assert(Y.size(2) == n_cells);
    assert(C.size(0) == T);                    assert(C.size(1) == n_batch);          assert(C.size(2) == n_cells);
    assert(d.size(0) == n_batch);              assert(d.size(1) == n_cells);

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, 0);
    // at::Tensor x_h = at::empty({n_batch, n_in + n_cells}, options);
    auto x_h = torch::zeros({n_batch, n_in + n_cells}, options);
    // at::Tensor intern = at::empty({n_batch, n_cells * 4}, options);
    auto intern = torch::zeros({n_batch, n_cells * 4}, options);

    assert(T > 0);
    assert(start >= 0);
    assert(start < T);
    assert(step != 0);

    int end = T - 1;
    if(step < 0) {
        end = start;
        start = T - start - 1;
    }

    int t = start;
    for(; (step > 0) ? (t <= end) : (t >= end); t += step) {
        start_dev_kernel(
            copy_x_h_kernel,
            (
                n_batch, n_in, n_cells,
                x_h.data<float>(),
                X[t].data<float>(),
                (t != start) ? Y[t-step].data<float>() : y0.data<float>()
            )
        );

        // intern = x_h * W
        // mm(x_h, W, intern);
        torch::mm_out(intern, x_h, W);

        // intern += b
        start_dev_kernel(
            add_bias_kernel,
            (
                n_batch, n_cells * 4, intern.data<float>(), b.data<float>()
            )
        );

        start_dev_kernel(
            lstm_fwd_kernel_for_low_memory,
            (
                n_batch,
                n_cells,
                i.data<float>() + t * n_batch,
                intern.data<float>(),
                (t != start) ? C[t-step].data<float>() : c0.data<float>(),
                Y[t].data<float>(),
                C[t].data<float>()
            )
        );

    }

    // device_free(x_h);
    // device_free(intern);
    CudaMemcpy(d.data<float>(), C[t-step].data<float>(), n_batch * n_cells * sizeof(float));
    // Ndarray_memcpy(Ndarray_DEV_DATA(d), data_ptr(C, t - step), n_batch * n_cells * sizeof(float));
}

void lstm_bwd_func_low_mem(
    at::Tensor X, at::Tensor W, at::Tensor b, at::Tensor y0, at::Tensor c0, at::Tensor i, int start, int step,
    at::Tensor Y, at::Tensor C,
    at::Tensor DY, at::Tensor Dd,
    at::Tensor DX, at::Tensor DW, at::Tensor Db, at::Tensor Dh, at::Tensor Dc
)
{
    // check dimension of forward inputs (bwd input part 1)
    assert(X.dim() == 3);        assert(i.dim() == 2);
    assert(W.dim() == 2);        assert(b.dim() == 1);
    assert(y0.dim() == 2);       assert(c0.dim() == 2);
    // check dimension of forward outputs (bwd input part 2)
    assert(Y.dim() == 3);        assert(C.dim() == 3);
    // check dimension of gradient of forward outputs (bwd input part 3)
    assert(DY.dim() == 3);       assert(Dd.dim() == 2);
    // check dimension of gradient of forwaard inputs (bwd output storage)
    assert(DX.dim() == 3);       assert(DW.dim() == 2);        assert(Db.dim() == 1);
    assert(Dh.dim() == 2);       assert(Dc.dim() == 2);

    long T = i.size(0);
    int n_batch = i.size(1);
    int n_cells = y0.size(1);
    int n_in = X.size(2);

    // check shape of forward inputs (bwd input part 1)
    assert(X.size(0) == T);                assert(X.size(1) == n_batch);
    assert(W.size(0) == n_in + n_cells);   assert(W.size(1) == n_cells * 4);
    assert(b.size(0) == n_cells * 4);
    assert(y0.size(0) == n_batch);         assert(y0.size(1) == n_cells);
    assert(c0.size(0) == n_batch);         assert(c0.size(1) == n_cells);
    // check shape of forward outputs (bwd input part 2)
    assert(Y.size(0) == T);                assert(Y.size(1) == n_batch);        assert(Y.size(2) == n_cells);
    assert(C.size(0) == T);                assert(C.size(1) == n_batch);        assert(C.size(2) == n_cells);
    // check shape of gradient of forward outputs (bwd input part 3)
    assert(DY.size(0) == T);               assert(DY.size(1) == n_batch);       assert(DY.size(2) == n_cells);
    assert(Dd.size(0) == n_batch);         assert(Dd.size(1) == n_cells);
    // check shape of gradient of forwaard inputs (bwd output storage)
    assert(DX.size(0) == T);               assert(DX.size(1) == n_batch);       assert(DX.size(2) == n_in);
    assert(DW.size(0) == n_in + n_cells);  assert(DW.size(1) == n_cells * 4);
    assert(Db.size(0) == n_cells * 4);
    assert(Dh.size(0) == n_batch);         assert(Dh.size(1) == n_cells);
    assert(Dc.size(0) == n_batch);         assert(Dc.size(1) == n_cells);

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, 0);
    auto x_h = torch::zeros({n_batch, n_in + n_cells}, options);
    auto intern = torch::zeros({n_batch, n_cells * 4}, options);
    auto Dx_h = torch::zeros({n_batch, n_in + n_cells}, options);
    auto Dintern = torch::zeros({n_batch, n_cells * 4}, options);

    // We will work inplace on DX/DW/Db.
    // Ndarray_memset(Ndarray_DEV_DATA(DX), 0, T * n_batch * n_in * sizeof(float));
    // Ndarray_memset(Ndarray_DEV_DATA(DW), 0, (n_in + n_cells) * n_cells * 4 * sizeof(float));
    // Ndarray_memset(Ndarray_DEV_DATA(Db), 0, n_cells * 4 * sizeof(float));
    // We will work inplace on Dh.
    // Ndarray_memset(Ndarray_DEV_DATA(Dh), 0, n_batch * n_cells * sizeof(float));
    // We will work inplace on Dc, and init it with Dd.
    // Ndarray_memcpy(Ndarray_DEV_DATA(Dc), Ndarray_DEV_DATA(Dd), n_batch * n_cells * sizeof(float));
    CudaMemcpy(Dc.data<float>(), Dd.data<float>(), n_batch * n_cells * sizeof(float));

    assert(T > 0);
    assert(start >= 0);
    assert(start < T);
    assert(step != 0);

    int end = T - 1;
    if(step < 0) {
        end = start;
        start = T - start - 1;
    }
    int t = end;  // go backwards
    for(; (step > 0) ? (t >= start) : (t <= start); t -= step) {
        bool right = (step > 0) ? (t - step >= start) : (t - step <= start);

        // TODO: correct handling of mask in grad, fwd, initial cell,hidden, etc
        // x_h = X[t], Y[t-1]
        start_dev_kernel(
            copy_x_h_kernel,
            (
                n_batch, n_in, n_cells,
                x_h.data<float>(), X[t].data<float>(),
                right ? Y[t-step].data<float>() : y0.data<float>()
            )
        );

        // intern = x_h * W
        // mm(x_h, W, intern);
        torch::mm_out(intern, x_h, W);
//        affine_raw(
//        x_h, n_batch, n_in + n_cells,
//        Ndarray_DEV_DATA(W), n_in + n_cells, n_cells * 4,
//        intern, n_batch, n_cells * 4,
//        false, false, 0.0);

        // intern += b
        start_dev_kernel(
            add_bias_kernel, (
                n_batch, n_cells * 4, intern.data<float>(), b.data<float>()
            )
        );

        start_dev_kernel(
            lstm_bwd_kernel_for_low_memory, (
                n_batch,
                n_in,
                n_cells,
                i.data<float>() + t * n_batch,
                x_h.data<float>(),
                intern.data<float>(),
                right ? C[t-step].data<float>() : c0.data<float>(),
                Y[t].data<float>(),
                C[t].data<float>(),
                DY[t].data<float>(),
                Dh.data<float>(),  // error from prev frame, excluding DY. updated below
                Dc.data<float>(),  // in+out, working inplace. also error from prev frame, initially Dd
                Dintern.data<float>(),  // out
                Db.data<float>()  // out
            )
        );

        // Dx_h = Dintern * W^T
        // mmabt(Dintern, W, Dx_h);
        torch::mm_out(Dx_h, Dintern, W.t());
        // DW += x_h^T * Dintern

        //addmmatb(x_h, Dintern, DW);
        torch::addmm(DW, x_h.t(), Dintern);
        // DX[t], Dh = Dx_h
        start_dev_kernel(
            inv_copy_x_h_kernel,
            (
                n_batch, n_in, n_cells,
                Dx_h.data<float>(), DX[t].data<float>(), Dh.data<float>()
            )
        );
    }
    // TODO: should we free intermediate tensors in pytorch c++ extensions?
    //device_free(x_h);
    //device_free(intern);
    //device_free(Dx_h);
    //device_free(Dintern);
}

void lstm_fwd_func_low_mem_var(
    at::Tensor X, at::Tensor W, at::Tensor b, at::Tensor y0, at::Tensor c0, at::Tensor i,
    int start, int step,
    at::Tensor Y, at::Tensor C, at::Tensor d,
    at::Tensor x_h, at::Tensor intern
)
{
    assert(X.dim() == 3);   assert(i.dim() == 2);
    assert(W.dim() == 2);   assert(b.dim() == 1);
    assert(y0.dim() == 2);  assert(c0.dim() == 2);
    assert(Y.dim() == 3);   assert(C.dim() == 3);    assert(d.dim() == 2);

    long T = i.size(0);
    int n_batch = i.size(1);
    int n_cells = y0.size(1);
    int n_in = X.size(2);

    assert(X.size(0) == T);                    assert(X.size(1) == n_batch);
    assert(W.size(0) == n_in + n_cells);       assert(W.size(1) == n_cells * 4);
    assert(b.size(0) == n_cells * 4);
    assert(y0.size(0) == n_batch);             assert(y0.size(1) == n_cells);
    assert(c0.size(0) == n_batch);             assert(c0.size(1) == n_cells);
    assert(Y.size(0) == T);                    assert(Y.size(1) == n_batch);          assert(Y.size(2) == n_cells);
    assert(C.size(0) == T);                    assert(C.size(1) == n_batch);          assert(C.size(2) == n_cells);
    assert(d.size(0) == n_batch);              assert(d.size(1) == n_cells);
    assert(x_h.size(0) == T);                  assert(x_h.size(1) == n_batch);        assert(x_h.size(2) == n_in + n_cells);
    assert(intern.size(0) == T);               assert(intern.size(1) == n_batch);     assert(intern.size(2) == 4 * n_cells);

    // auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, 0);
    // at::Tensor x_h = at::empty({n_batch, n_in + n_cells}, options);
    // auto x_h = torch::zeros({n_batch, n_in + n_cells}, options);
    // at::Tensor intern = at::empty({n_batch, n_cells * 4}, options);
    // auto intn = torch::zeros({n_batch, n_cells * 4}, options);

    assert(T > 0);
    assert(start >= 0);
    assert(start < T);
    assert(step != 0);

    int end = T - 1;
    if(step < 0) {
        end = start;
        start = T - start - 1;
    }

    int t = start;
    for(; (step > 0) ? (t <= end) : (t >= end); t += step) {
        start_dev_kernel(
            copy_x_h_kernel,
            (
                n_batch, n_in, n_cells,
                x_h[t].data<float>(),
                X[t].data<float>(),
                (t != start) ? Y[t-step].data<float>() : y0.data<float>()
            )
        );

        // intern = x_h * W
        // mm(x_h, W, intern);
        auto intn = torch::mm(x_h[t], W);
        CudaMemcpy(intern[t].data<float>(), intn.data<float>(), 4 * n_batch * n_cells * sizeof(float));

        // intern += b
        start_dev_kernel(
            add_bias_kernel,
            (
                n_batch, n_cells * 4, intern[t].data<float>(), b.data<float>()
            )
        );

        start_dev_kernel(
            lstm_fwd_kernel_for_low_memory,
            (
                n_batch,
                n_cells,
                i.data<float>() + t * n_batch,
                intern[t].data<float>(),
                (t != start) ? C[t-step].data<float>() : c0.data<float>(),
                Y[t].data<float>(),
                C[t].data<float>()
            )
        );

    }

    // device_free(x_h);
    // device_free(intern);
    CudaMemcpy(d.data<float>(), C[t-step].data<float>(), n_batch * n_cells * sizeof(float));
    // Ndarray_memcpy(Ndarray_DEV_DATA(d), data_ptr(C, t - step), n_batch * n_cells * sizeof(float));
}

void lstm_bwd_func_low_mem_var(
    at::Tensor X, at::Tensor W, at::Tensor b, at::Tensor y0, at::Tensor c0, at::Tensor i, int start, int step,
    at::Tensor Y, at::Tensor C,
    at::Tensor DY, at::Tensor Dd,
    at::Tensor DX, at::Tensor DW, at::Tensor Db, at::Tensor Dh, at::Tensor Dc,
    at::Tensor x_h, at::Tensor intern
)
{
    // check dimension of forward inputs (bwd input part 1)
    assert(X.dim() == 3);        assert(i.dim() == 2);
    assert(W.dim() == 2);        assert(b.dim() == 1);
    assert(y0.dim() == 2);       assert(c0.dim() == 2);
    // check dimension of forward outputs (bwd input part 2)
    assert(Y.dim() == 3);        assert(C.dim() == 3);
    // check dimension of gradient of forward outputs (bwd input part 3)
    assert(DY.dim() == 3);       assert(Dd.dim() == 2);
    // check dimension of gradient of forwaard inputs (bwd output storage)
    assert(DX.dim() == 3);       assert(DW.dim() == 2);        assert(Db.dim() == 1);
    assert(Dh.dim() == 2);       assert(Dc.dim() == 2);

    long T = i.size(0);
    int n_batch = i.size(1);
    int n_cells = y0.size(1);
    int n_in = X.size(2);

    // check shape of forward inputs (bwd input part 1)
    assert(X.size(0) == T);                assert(X.size(1) == n_batch);
    assert(W.size(0) == n_in + n_cells);   assert(W.size(1) == n_cells * 4);
    assert(b.size(0) == n_cells * 4);
    assert(y0.size(0) == n_batch);         assert(y0.size(1) == n_cells);
    assert(c0.size(0) == n_batch);         assert(c0.size(1) == n_cells);
    // check shape of forward outputs (bwd input part 2)
    assert(Y.size(0) == T);                assert(Y.size(1) == n_batch);        assert(Y.size(2) == n_cells);
    assert(C.size(0) == T);                assert(C.size(1) == n_batch);        assert(C.size(2) == n_cells);
    // check shape of gradient of forward outputs (bwd input part 3)
    assert(DY.size(0) == T);               assert(DY.size(1) == n_batch);       assert(DY.size(2) == n_cells);
    assert(Dd.size(0) == n_batch);         assert(Dd.size(1) == n_cells);
    // check shape of gradient of forwaard inputs (bwd output storage)
    assert(DX.size(0) == T);               assert(DX.size(1) == n_batch);       assert(DX.size(2) == n_in);
    assert(DW.size(0) == n_in + n_cells);  assert(DW.size(1) == n_cells * 4);
    assert(Db.size(0) == n_cells * 4);
    assert(Dh.size(0) == n_batch);         assert(Dh.size(1) == n_cells);
    assert(Dc.size(0) == n_batch);         assert(Dc.size(1) == n_cells);
    assert(x_h.size(0) == T);                  assert(x_h.size(1) == n_batch);        assert(x_h.size(2) == n_in + n_cells);
    assert(intern.size(0) == T);               assert(intern.size(1) == n_batch);     assert(intern.size(2) == 4 * n_cells);

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, 0);
    // auto x_h = torch::zeros({n_batch, n_in + n_cells}, options);
    // auto intern = torch::zeros({n_batch, n_cells * 4}, options);
    auto Dx_h = torch::zeros({n_batch, n_in + n_cells}, options);
    auto Dintern = torch::zeros({n_batch, n_cells * 4}, options);

    // We will work inplace on DX/DW/Db.
    // Ndarray_memset(Ndarray_DEV_DATA(DX), 0, T * n_batch * n_in * sizeof(float));
    // Ndarray_memset(Ndarray_DEV_DATA(DW), 0, (n_in + n_cells) * n_cells * 4 * sizeof(float));
    // Ndarray_memset(Ndarray_DEV_DATA(Db), 0, n_cells * 4 * sizeof(float));
    // We will work inplace on Dh.
    // Ndarray_memset(Ndarray_DEV_DATA(Dh), 0, n_batch * n_cells * sizeof(float));
    // We will work inplace on Dc, and init it with Dd.
    // Ndarray_memcpy(Ndarray_DEV_DATA(Dc), Ndarray_DEV_DATA(Dd), n_batch * n_cells * sizeof(float));
    CudaMemcpy(Dc.data<float>(), Dd.data<float>(), n_batch * n_cells * sizeof(float));

    assert(T > 0);
    assert(start >= 0);
    assert(start < T);
    assert(step != 0);

    int end = T - 1;
    if(step < 0) {
        end = start;
        start = T - start - 1;
    }
    int t = end;  // go backwards
    for(; (step > 0) ? (t >= start) : (t <= start); t -= step) {
        bool right = (step > 0) ? (t - step >= start) : (t - step <= start);

        start_dev_kernel(
            lstm_bwd_kernel_for_low_memory, (
                n_batch,
                n_in,
                n_cells,
                i.data<float>() + t * n_batch,
                x_h[t].data<float>(),
                intern[t].data<float>(),
                right ? C[t-step].data<float>() : c0.data<float>(),
                Y[t].data<float>(),
                C[t].data<float>(),
                DY[t].data<float>(),
                Dh.data<float>(),  // error from prev frame, excluding DY. updated below
                Dc.data<float>(),  // in+out, working inplace. also error from prev frame, initially Dd
                Dintern.data<float>(),  // out
                Db.data<float>()  // out
            )
        );

        // Dx_h = Dintern * W^T
        torch::mm_out(Dx_h, Dintern, W.t());
        // DW += x_h^T * Dintern

        //addmmatb(x_h, Dintern, DW);
        torch::addmm(DW, x_h[t].t(), Dintern);
        // DX[t], Dh = Dx_h
        start_dev_kernel(
            inv_copy_x_h_kernel,
            (
                n_batch, n_in, n_cells,
                Dx_h.data<float>(), DX[t].data<float>(), Dh.data<float>()
            )
        );
    }
    // TODO: should we free intermediate tensors in pytorch c++ extensions?
    //device_free(x_h);
    //device_free(intern);
    //device_free(Dx_h);
    //device_free(Dintern);
}

void lstm_fwd_func(
    at::Tensor X, at::Tensor W, at::Tensor y0, at::Tensor c0, at::Tensor i,
    int start, int step,
    at::Tensor Y, at::Tensor C, at::Tensor H, at::Tensor d, at::Tensor y_prev
)
{
    assert(X.dim() == 3);   assert(W.dim() == 2);
    assert(y0.dim() == 2);  assert(c0.dim() == 2);   assert(i.dim() == 2);
    assert(Y.dim() == 3);   assert(C.dim() == 3);    assert(H.dim() == 3);   assert(d.dim() == 2);

    long T = i.size(0);
    int n_batch = i.size(1);
    int n_cells = y0.size(1);
    int n_in = X.size(2);

    assert(X.size(0) == T);                    assert(X.size(1) == n_batch);          assert(X.size(2) == 4 * n_cells);
    assert(W.size(0) == n_cells);              assert(W.size(1) == n_cells * 4);
    assert(y0.size(0) == n_batch);             assert(y0.size(1) == n_cells);
    assert(c0.size(0) == n_batch);             assert(c0.size(1) == n_cells);
    assert(Y.size(0) == T);                    assert(Y.size(1) == n_batch);          assert(Y.size(2) == n_cells);
    assert(C.size(0) == T);                    assert(C.size(1) == n_batch);          assert(C.size(2) == n_cells);
    assert(d.size(0) == n_batch);              assert(d.size(1) == n_cells);
    assert(H.size(0) == T);                    assert(H.size(1) == n_batch);          assert(H.size(2) == n_cells * 4);

    if(T == 0) {
        CudaMemcpy(d.data_ptr(), c0.data_ptr(), n_batch * n_cells * sizeof(float));
    } else {  // T > 0
        // It makes the backprop with step<0 easier to implement,
        // esp. the DW = Y[0..T-2]^T * DX[1..T-1] calculation,
        // if we can have Y[t] = 0 where mask[t] = 0.
        // That is why we need to keep track of Y[t-1] explicitly.
        // float* y_prev = (float*) device_malloc(n_batch * n_cells * sizeof(float));
        // auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, 0).requires_grad(false);
        // auto y_prev = torch::zeros_like(y0);
        //std::cout << y_prev.data_ptr() << std::endl;
        // H = X
        CudaMemcpy(H.data_ptr(), X.data_ptr(), T * n_batch * n_cells * 4 * sizeof(float));
        // Ndarray_memcpy(Ndarray_DEV_DATA(H), Ndarray_DEV_DATA(X), T * n_batch * n_cells * 4 * sizeof(float));

        assert(T > 0);             assert(step != 0);
        assert(start >= 0);        assert(start < T);

        int end = T - 1;
        if(step < 0) {
            end = 0;
            start = T - start - 1;
        }
        int t = start;

        for(; (step > 0) ? (t <= end) : (t >= end); t += step) {
        // H[t] += Y[t-1] * W
            H[t] += torch::mm((t != start) ? y_prev : y0, W);
//            affine_raw(
//              (t != start) ? y_prev : Ndarray_DEV_DATA(y0), n_batch, n_cells,
//              Ndarray_DEV_DATA(W), n_cells, n_cells * 4,
//              data_ptr(H, t), n_batch, n_cells * 4,
//              false, false);
            start_dev_kernel(
                  lstm_fwd_kernel, (
                      n_batch,
                      n_cells,
                      i[t].data<float>(),
                      H[t].data<float>(), // inplace
                      (t != start) ? y_prev.data<float>() : y0.data<float>(),
                      (t != start) ? C[t-step].data<float>() : c0.data<float>(),
                      Y[t].data<float>(),
                      C[t].data<float>(),
                      y_prev.data<float>()
                  )
            );
        }
        CudaMemcpy(d.data_ptr(), C[t-step].data_ptr(), n_batch * n_cells * sizeof(float));
        // Ndarray_memcpy(Ndarray_DEV_DATA(d), data_ptr(C, t - step), n_batch * n_cells * sizeof(float));
    }
}

void lstm_bwd_func(
    at::Tensor X, at::Tensor W, at::Tensor y0, at::Tensor c0, at::Tensor i,
    int start, int step,
    at::Tensor Y, at::Tensor C, at::Tensor H,
    at::Tensor DY, at::Tensor Dd, at::Tensor DX, at::Tensor DW, at::Tensor Dy0, at::Tensor Dc0,
    at::Tensor dx0
)
{
    assert(X.dim() == 3);        assert(W.dim() == 2);
    assert(y0.dim() == 2);       assert(c0.dim() == 2);            assert(i.dim() == 2);
    assert(Y.dim() == 3);        assert(C.dim() == 3);             assert(H.dim() == 3);
    assert(DY.dim() == 3);       assert(Dd.dim() == 2);            assert(DX.dim() == 3);
    assert(DW.dim() == 2);       assert(Dy0.dim() == 2);           assert(Dc0.dim() == 2);

    long T = i.size(0);
    int n_batch = i.size(1), n_cells = y0.size(1);

    assert(X.size(0) == T);          assert(X.size(1) == n_batch);         assert(X.size(2) == n_cells * 4);
    assert(W.size(0) == n_cells);    assert(W.size(1) == n_cells * 4);
    assert(y0.size(0) == n_batch);   assert(y0.size(1) == n_cells);
    assert(c0.size(0) == n_batch);   assert(c0.size(1) == n_cells);
    assert(Y.size(0) == T);          assert(Y.size(1) == n_batch);         assert(Y.size(2) == n_cells);
    assert(C.size(0) == T);          assert(C.size(1) == n_batch);         assert(C.size(2) == n_cells);
    assert(H.size(0) == T);          assert(H.size(1) == n_batch);         assert(H.size(2) == n_cells * 4);
    assert(DY.size(0) == T);         assert(DY.size(1) == n_batch);        assert(DY.size(2) == n_cells);
    assert(Dd.size(0) == n_batch);   assert(Dd.size(1) == n_cells);
    assert(DX.size(0) == T);         assert(DX.size(1) == n_batch);        assert(DX.size(2) == n_cells * 4);
    assert(DW.size(0) == n_cells);   assert(DW.size(1) == n_cells * 4);
    assert(Dy0.size(0) == n_batch);  assert(Dy0.size(1) == n_cells);
    assert(Dc0.size(0) == n_batch);  assert(Dc0.size(1) == n_cells);

    // We will work inplace on DW.
    // Ndarray_memset(Ndarray_DEV_DATA(DW), 0, n_cells * n_cells * 4 * sizeof(float));
    CudaMemset(DW.data_ptr(), 0, n_cells * n_cells * 4 * sizeof(float));
    // We will work inplace on (Dy0) DY[t], initially 0.
    // Ndarray_memset(Ndarray_DEV_DATA(Dy0), 0, n_batch * n_cells * sizeof(float));
    CudaMemset(Dy0.data_ptr(), 0, n_batch * n_cells * sizeof(float));
    // We will work inplace on (Dc0) DC[t], and init it with Dd.
    // Ndarray_memcpy(Ndarray_DEV_DATA(Dc0), Ndarray_DEV_DATA(Dd), n_batch * n_cells * sizeof(float));
    CudaMemcpy(Dc0.data_ptr(), Dd.data_ptr(), n_batch * n_cells * sizeof(float));

    if(T == 0) {
        // just do nothing. at least do not crash
    } else {
        assert(T > 0);           assert(step != 0);
        assert(start >= 0);      assert(start < T);

        // Need to keep track of (logical) DX[0], which in practice (masking, step<0)
        // can be different from data_ptr(DX, start).
        // float* dx0 = (float*) device_malloc(n_batch * n_cells * 4 * sizeof(float));
        // Ndarray_memset(dx0, 0, n_batch * n_cells * 4 * sizeof(float));
        // auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, 0).requires_grad(false);
        // auto dx0 = torch::zeros_like(X[0]);
        //std::cout << dx0.device() << " " << dx0.dim() << " " << dx0.size(1) << std::endl;
        CudaMemset(dx0.data_ptr(), 0, n_batch * n_cells * 4 * sizeof(float));

        int abs_step = std::abs(step);
        int num_steps = (T - start + abs_step - 1) / abs_step;
        assert(num_steps > 0);
        if(step < 0) start = T - start - 1;
        int end = start + (num_steps - 1) * step;  // inclusive
        assert(end >= 0);
        assert(end < T);
        int t = end;  // go backwards

        for(; (step > 0) ? (t >= start) : (t <= start); t -= step) {

            bool right = (step > 0) ? (t - step >= start) : (t - step <= start);

            start_dev_kernel(
                lstm_bwd_kernel, (
                    n_batch,
                    n_cells,
                    i[t].data<float>(),
                    H[t].data<float>(),
                    right ? C[t-step].data<float>() : c0.data<float>(),
                    Y[t].data<float>(),
                    C[t].data<float>(),
                    DY[t].data<float>(),
                    Dy0.data<float>(), // in+out, error from prev frame, excluding DY. reset here, updated below
                    Dc0.data<float>(), // in+out, working inplace. also error from prev frame, initially Dd
                    DX[t].data<float>(), // out
                    dx0.data<float>()  // out
                )
            );

            // (Dy0) DY[t-1] += DX[t] * W^T
            //affine_raw(
            //  data_ptr(DX, t), n_batch, n_cells * 4,
            //  Ndarray_DEV_DATA(W), n_cells, n_cells * 4,
            //  Ndarray_DEV_DATA(Dy0), n_batch, n_cells,
            //  false, true);
            torch::addmm(Dy0, DX[t], W.t());
        }

        if(num_steps > 1){
            //DW = Y[0..T-2]^T * DX[1..T-1]  (if step==1)
            // affine_raw(
            //    data_ptr(Y, std::min(start, end) + std::max(0, -step)), (num_steps - 1) * n_batch, n_cells,
            //    data_ptr(DX, std::min(start, end) + std::max(0, step)), (num_steps - 1) * n_batch, n_cells * 4,
            //    Ndarray_DEV_DATA(DW), n_cells, n_cells * 4,
            //    true, false, 0.0f, 1.0f,
            //    abs_step, abs_step);
            int Y_start = std::min(start, end) + std::max(0, -step);
            int DX_start = std::min(start, end) + std::max(0, step);
            auto Y_rows = torch::arange(Y_start, Y_start + num_steps - 1, torch::kLong);
            auto DX_rows = torch::arange(DX_start, DX_start + num_steps - 1, torch::kLong);
            torch::mm_out(DW, Y.index(Y_rows).view({-1, Y.size(2)}).t(), DX.index(DX_rows).view({-1, DX.size(2)}));
        }

        //DW += y0^T * DX[0]
        //affine_raw(
        //Ndarray_DEV_DATA(y0), n_batch, n_cells,
        //dx0, n_batch, n_cells * 4,
        //Ndarray_DEV_DATA(DW), n_cells, n_cells * 4,
        //true, false);
        torch::addmm(DW, y0.t(), dx0);
    }
}

void lstm_fwd_func_base(
    at::Tensor Z, at::Tensor Wr, at::Tensor c, at::Tensor i,
    at::Tensor Y, at::Tensor d
)
{
    long T = i.size(0);
    int B = i.size(1);
    assert(Z.size(2) % 4 == 0);
    int D = Z.size(2) / 4;
    assert(Z.size(0) == T);           assert(Z.size(1) == B);       assert(Z.size(2) == 4 * D);
    assert(Wr.size(0) == D);          assert(Wr.size(1) == 4 * D);
    assert(c.size(0) == B);           assert(c.size(1) == D);
    assert(i.size(0) == T);           assert(i.size(1) == B);
    assert(Y.size(0) == T);           assert(Y.size(1) == B);       assert(Y.size(2) == D);
    assert(d.size(0) == B);           assert(d.size(1) == D);

    assert(T > 0);

    for(int x = 0; x < T; ++x){
        if(x > 0){
            Z[x] += torch::mm(Y[x-1], Wr);
        }
        start_dev_kernel(
            lstm_fwd_kernel_base,
            (
                Z[x].data<float>(),
                x > 0 ? Z[x-1].data<float>() : c.data<float>(),
                x > 0,
                Y[x].data<float>(),
                (x == T - 1) ? d.data<float>() : 0,
                D,
                B,
                i[x].data<float>()
            )
        );
    }
}

void lstm_bwd_func_base(
    at::Tensor Wr, at::Tensor c, at::Tensor i, at::Tensor Y,
    at::Tensor Dd, at::Tensor DZ, at::Tensor DWr, at::Tensor Dc, at::Tensor tmpDc
)
{
    long T = i.size(0);
    int B = i.size(1);
    assert(DZ.size(2) % 4 == 0);
    int D = DZ.size(2) / 4;

    assert(T > 0);
    assert(Wr.size(0) == D);         assert(Wr.size(1) == 4 * D);
    assert(c.size(0) == B);          assert(c.size(1) == D);
    assert(Y.size(0) == T);          assert(Y.size(1) == B);             assert(Y.size(2) == D);
    assert(Dd.size(0) == B);         assert(Dd.size(1) == D);
    assert(DZ.size(0) == T);         assert(DZ.size(1) == B);            assert(DZ.size(2) == 4 * D);
    assert(DWr.size(0) == D);        assert(DWr.size(1) == 4 * D);
    assert(Dc.size(0) == B);         assert(Dc.size(1) == D);
    //assert(tmpDc.size(0) == B);      assert(tmpDc.size(1) == D);

    for(int x = T- 1; x >= 0; --x){
        bool rightBorder = (x == (T - 1));
        if(!rightBorder){
            // tmpDc[x] += DZ[x+1] * Wr[x]^T;
            tmpDc[x] += torch::mm(DZ[x+1], Wr.t());
        }
        start_dev_kernel(
            lstm_bwd_kernel_base,
            (
                DZ[x].data<float>(),
                tmpDc[x].data<float>(),
                rightBorder ? Dd.data<float>() : tmpDc[x+1].data<float>(),
                x > 0 ? DZ[x-1].data<float>() : c.data<float>(),
                x > 0,
                Y[x].data<float>(),
                D,
                B,
                i[x].data<float>()
            )
        );

    }
    auto Y_R = torch::arange(0, T-1, torch::kLong);
    auto DZ_R = torch::arange(1, T, torch::kLong);
    torch::mm_out(DWr, Y.index(Y_R).view({-1, Y.size(2)}).t(), DZ.index(DZ_R).view({-1, DZ.size(2)}));
    CudaMemcpy(Dc.data_ptr(), tmpDc.data_ptr(), B * D * sizeof(float));
}


