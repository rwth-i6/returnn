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
#define CudaMemcpy(y, x, size) (cudaMemcpyAsync(y, x, size, cudaMemcpyDeviceToDevice))
#define CudaMemset(s, c, size) (cudaMemsetAsync(s, c, size, 0))

DEF_KERNEL void lstm_fwd_kernel_gpu(
    float* data, const float* old_state, bool old_state_strided,
    float* output, float* state_out, int n_cells, int n_batch, const float* i
)
{
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

DEF_KERNEL void lstm_bwd_kernel_gpu(
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

void lstm_fwd_kernel_cpu(
    float* data, const float* old_state, bool old_state_strided,
    float* output, float* state_out, int n_cells, int n_batch, const float* i
)
{
    //layout:
    //data[0*n_cells..1*n_cells-1] : cell state
    //data[1*n_cells..2*n_cells-1] : input gate
    //data[2*n_cells..3*n_cells-1] : forget gate
    //data[3*n_cells..4*n_cells-1] : output gate
    //output[0*n_cells..1*n_cells-1]: cell output
    //repeated for every mini-batch
    for (int idx = 0; idx < n_cells * n_batch; ++idx) {
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
    }
}

void lstm_bwd_kernel_cpu(
    float* delta, float* epsilon, const float* next_epsilon, const float* old_state,
    bool old_state_strided, const float* Y, int n_cells, int n_batch, const float* i
)
{
    for (int idx = 0; idx < n_cells * n_batch; ++idx) {
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
    }
}

void lstm_fwd_func_gpu(
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
            lstm_fwd_kernel_gpu,
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

void lstm_bwd_func_gpu(
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
            lstm_bwd_kernel_gpu,
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

void lstm_fwd_func_cpu(
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
        lstm_fwd_kernel_cpu(
            Z[x].data<float>(),
            x > 0 ? Z[x-1].data<float>() : c.data<float>(),
            x > 0,
            Y[x].data<float>(),
            (x == T - 1) ? d.data<float>() : 0,
            D,
            B,
            i[x].data<float>()
        );
    }
}

void lstm_bwd_func_cpu(
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
        lstm_bwd_kernel_cpu(
            DZ[x].data<float>(),
            tmpDc[x].data<float>(),
            rightBorder ? Dd.data<float>() : tmpDc[x+1].data<float>(),
            x > 0 ? DZ[x-1].data<float>() : c.data<float>(),
            x > 0,
            Y[x].data<float>(),
            D,
            B,
            i[x].data<float>()
        );
    }
    auto Y_R = torch::arange(0, T-1, torch::kLong);
    auto DZ_R = torch::arange(1, T, torch::kLong);
    torch::mm_out(DWr, Y.index(Y_R).view({-1, Y.size(2)}).t(), DZ.index(DZ_R).view({-1, DZ.size(2)}));
    Dc = tmpDc[0];
}

