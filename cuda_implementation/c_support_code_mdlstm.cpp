#define STABLE_CELL

CudaNdarray * sumOverAllButLastDimensions(const CudaNdarray * A)
{
  int lastDim = CudaNdarray_HOST_DIMS(A)[A->nd - 1];
  int N = CudaNdarray_SIZE(A) / lastDim;
  thrust::device_vector<float> v(N, 1.0f);
  float alpha = 1.0f;
  float beta = 0.0f;
  const float * dataA = CudaNdarray_DEV_DATA(A);
  const float * dataX = thrust::raw_pointer_cast(&v[0]);
  int dims[] = { lastDim };
  CudaNdarray * dst = (CudaNdarray*)CudaNdarray_NewDims(1, dims);
  float * dataDst = CudaNdarray_DEV_DATA(dst);
  int lda = lastDim;
  HANDLE_ERROR(cublasSgemv(handle, CUBLAS_OP_N, lastDim, N, &alpha, dataA, lda, dataX, 1, &beta, dataDst, 1));
  return dst;
}

//if nd is 2 then assume a weight matrix and just return beginning of data
//else nd should be 4 and we pick the (y,x) part
const float * data_ptr(const CudaNdarray * a, int y, int x, int outer_dim=0)
{
  assert(a->nd == 2 || a->nd == 4 || a->nd == 5);
  if (a->nd == 2)
  {
    return CudaNdarray_DEV_DATA(a);
  }
  else if(a->nd == 4)
  {
    const int * dims = CudaNdarray_HOST_DIMS(a);
    return CudaNdarray_DEV_DATA(a) + y * dims[1] * dims[2] * dims[3] + x * dims[2] * dims[3];
  }
  else
  {
        const int * dims = CudaNdarray_HOST_DIMS(a);
    return CudaNdarray_DEV_DATA(a) + outer_dim * dims[1] * dims[2] * dims[3] * dims[4] +
     y * dims[2] * dims[3] * dims[4] + x * dims[3] * dims[4];
  }
}

float * data_ptr(CudaNdarray * a, int y, int x, int outer_dim=0)
{
  const CudaNdarray * ca = a;
  return const_cast<float *>(data_ptr(ca, y, x, outer_dim));
}

void lastTwoDims(const CudaNdarray * a, int out[2])
{
  const int * dims = CudaNdarray_HOST_DIMS(a);
  assert(a->nd >= 2);
  out[0] = dims[a->nd - 2];
  out[1] = dims[a->nd - 1];
}

int lastTwoDimsStride(const CudaNdarray * a)
{
  int dims[2];
  lastTwoDims(a, dims);
  return dims[0] * dims[1];
}

#ifdef STABLE_CELL
__global__ void lstm_stable_cell_kernel_batched(float ** datas, const float ** old_state_ys, const float ** old_state_xs,
 float ** outputs, const float ** valids, int n_outer_batch, int n_cells, int n_minibatch)
{
  //layout (for every outer batch):
  //data[0*n_cells..1*n_cells-1] : input gate
  //data[1*n_cells..2*n_cells-1] : forget gate
  //data[2*n_cells..3*n_cells-1] : lambda gate
  //data[3*n_cells..4*n_cells-1] : output gate
  //data[5*n_cells..6*n_cells-1] : cell state
  //output[0*n_cells..1*n_cells-1]: cell output
  //valids: either 1.0 or 0.0, indicating if the current (y,x) position is still inside the image in this minibatch
  //repeated for every mini-batch

  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  while (idx < n_outer_batch * n_cells * n_minibatch)
  {
    int size_per_outer_batch = n_cells * n_minibatch;
    int outer_batch_idx = idx / size_per_outer_batch;
    float * data = datas[outer_batch_idx];
    const float * old_state_y = old_state_ys[outer_batch_idx];
    const float * old_state_x = old_state_xs[outer_batch_idx];
    float * output = outputs[outer_batch_idx];
    const float * valid = valids[outer_batch_idx];

    int inner_idx = idx % size_per_outer_batch;
    int minibatch_idx = inner_idx / n_cells;
    int batch_offset = minibatch_idx * 5 * n_cells;
    int cell_offset = inner_idx % n_cells;
    int start = batch_offset + cell_offset;

    float valid_batch = valid[minibatch_idx];

    //input, forget and output gates
    float inpGate = 1.f / (1.f + expf(-data[start]));
    float fgtGate = 1.f / (1.f + expf(-data[start + n_cells]));
    float lambdaGate = 1.f / (1.f + expf(-data[start + 2 * n_cells]));
    float outGate = 1.f / (1.f + expf(-data[start + 3 * n_cells]));
    float state = inpGate * tanhf(data[start + 4 * n_cells]);
    if (old_state_y)
    {
      state += fgtGate * lambdaGate * old_state_y[start];
    }
    if (old_state_x)
    {
      state += fgtGate * (1.0f - lambdaGate) * old_state_x[start];
    }
    state *= valid_batch;

    //cell output
    output[inner_idx] = outGate * tanhf(state) * valid_batch;

    data[start] = inpGate;
    data[start + n_cells] = fgtGate;
    data[start + 2 * n_cells] = lambdaGate;
    data[start + 3 * n_cells] = outGate;
    data[start + 4 * n_cells] = state;

    idx += gridDim.x * blockDim.x;
  }
}

__global__ void lstm_bwd_stable_cell_kernel_batched(float ** deltas, const float ** epsilons,
  const float ** next_epsilon_ys, const float ** next_epsilon_xs, float ** epsilon_ys, float ** epsilon_xs,
  const float ** last_state_ys, const float ** last_state_xs, const float ** Ys, const float ** valids,
  int n_outer_batch, int n_cells, int n_minibatch)
{
  //layout (for every outer batch):
  //delta[0*n_cells..1*n_cells-1] : input gate
  //delta[1*n_cells..2*n_cells-1] : forget gate
  //delta[2*n_cells..3*n_cells-1] : lambda gate
  //delta[3*n_cells..4*n_cells-1] : output gate
  //delta[4*n_cells..5*n_cells-1] : cell state
  //epsilon[0*n_cells..1*n_cells-1]: cell output derivative
  //next_epsilon_y[0*n_cells..1*n_cells-1]: cell state derivative * forget_gate * lambda_gate (of next timestep)
  //next_epsilon_x[0*n_cells..1*n_cells-1]: cell state derivative * forget_gate * (-1*)lambda_gate (of next timestep)
  //epsilon_y[0*n_cells..1*n_cells-1]: cell state derivative * forget_gate * lambda_gate (of current timestep, as output)
  //epsilon_x[0*n_cells..1*n_cells-1]: cell state derivative * forget_gate * (1-lambda_gate) (of current timestep, as output)
  //valids: either 1.0 or 0.0, indicating if the current (y,x) position is still inside the image in this minibatch
  //repeated for every mini-batch

  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  while (idx < n_outer_batch * n_cells * n_minibatch)
  {
    int size_per_outer_batch = n_cells * n_minibatch;
    int outer_batch_idx = idx / size_per_outer_batch;
    const float * valid = valids[outer_batch_idx];

    float * delta = deltas[outer_batch_idx];
    const float * epsilon = epsilons[outer_batch_idx];
    const float * next_epsilon_y = next_epsilon_ys[outer_batch_idx];
    const float * next_epsilon_x = next_epsilon_xs[outer_batch_idx];
    float * epsilon_y = epsilon_ys[outer_batch_idx];
    float * epsilon_x = epsilon_xs[outer_batch_idx];
    const float * last_state_y = last_state_ys[outer_batch_idx];
    const float * last_state_x = last_state_xs[outer_batch_idx];
    const float * Y = Ys[outer_batch_idx];

    int inner_idx = idx % size_per_outer_batch;
    int minibatch_idx = inner_idx / n_cells;
    int batch_offset = minibatch_idx * 5 * n_cells;
    int cell_offset = inner_idx % n_cells;
    int start = batch_offset + cell_offset;
    float valid_batch = valid[minibatch_idx];

    float inpGate = delta[start];
    float fgtGate = delta[start + n_cells];
    float lambdaGate = delta[start + 2 * n_cells];
    float outGate = delta[start + 3 * n_cells];
    float state = delta[start + 4 * n_cells];
    float lastState_y = last_state_y ? last_state_y[start] : 0.f;
    float lastState_x = last_state_x ? last_state_x[start] : 0.f;
    float eps = epsilon[inner_idx];

    //avoid division by 0
    float gc = 0.f; //g(c(t))
    float gzc = 0.f; //g(z_c(t))
    if (outGate != 0)
    {
      gc = Y[inner_idx] / outGate;
    }

    if (inpGate != 0)
    {
      gzc = (state - fgtGate * lambdaGate * lastState_y - fgtGate * (1.0f - lambdaGate) * lastState_x) / inpGate;
    }

    //delta_output
    delta[start + 3 * n_cells] = outGate * (1.f - outGate) * gc * eps * valid_batch;

    //epsilon_c
    float epsilon_c = (1.f - (gc * gc)) * outGate * eps;
    if (next_epsilon_y)
    {
      epsilon_c += next_epsilon_y[inner_idx];
    }
    if (next_epsilon_x)
    {
      epsilon_c += next_epsilon_x[inner_idx];
    }

    //TODO: clip epsilon_c?
    //epsilon_c = max(epsilon_c, -10.f);
    //epsilon_c = min(epsilon_c, 10.f);

    epsilon_y[inner_idx] = epsilon_c * fgtGate * lambdaGate * valid_batch;
    epsilon_x[inner_idx] = epsilon_c * fgtGate * (1.0f - lambdaGate) * valid_batch;

    //delta_cell
    delta[start + 4 * n_cells] = inpGate * (1.f - (gzc * gzc)) * epsilon_c * valid_batch;

    //delta_forget
    delta[start + n_cells] = fgtGate * (1.f - fgtGate) * epsilon_c *
                             (lastState_y * lambdaGate + lastState_x * (1.0f - lambdaGate)) * valid_batch;

    //delta_lambda
    delta[start + 2 * n_cells] = fgtGate * lambdaGate * (1.f - lambdaGate) * epsilon_c
                                 * (lastState_y - lastState_x) * valid_batch;

    //delta_input
    delta[start] = inpGate * (1.f - inpGate) * gzc * epsilon_c * valid_batch;

    idx += gridDim.x * blockDim.x;
  }
}

#else

__global__ void lstm_stable_cell_kernel_batched(float ** datas, const float ** old_state_ys, const float ** old_state_xs,
 float ** outputs, const float ** valids, int n_outer_batch, int n_cells, int n_minibatch)
{
  //layout (for every outer batch):
  //data[0*n_cells..1*n_cells-1] : input gate
  //data[1*n_cells..2*n_cells-1] : forget gate
  //data[2*n_cells..3*n_cells-1] : lambda gate
  //data[3*n_cells..4*n_cells-1] : output gate
  //data[5*n_cells..6*n_cells-1] : cell state
  //output[0*n_cells..1*n_cells-1]: cell output
  //valids: either 1.0 or 0.0, indicating if the current (y,x) position is still inside the image in this minibatch
  //repeated for every mini-batch

  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  while (idx < n_outer_batch * n_cells * n_minibatch)
  {
    int size_per_outer_batch = n_cells * n_minibatch;
    int outer_batch_idx = idx / size_per_outer_batch;
    float * data = datas[outer_batch_idx];
    const float * old_state_y = old_state_ys[outer_batch_idx];
    const float * old_state_x = old_state_xs[outer_batch_idx];
    float * output = outputs[outer_batch_idx];
    const float * valid = valids[outer_batch_idx];

    int inner_idx = idx % size_per_outer_batch;
    int minibatch_idx = inner_idx / n_cells;
    int batch_offset = minibatch_idx * 5 * n_cells;
    int cell_offset = inner_idx % n_cells;
    int start = batch_offset + cell_offset;

    float valid_batch = valid[minibatch_idx];

    //input, forget and output gates
    float inpGate = 1.f / (1.f + expf(-data[start]));
    float fgtGate_y = 1.f / (1.f + expf(-data[start + n_cells]));
    float fgtGate_x = 1.f / (1.f + expf(-data[start + 2 * n_cells]));
    float outGate = 1.f / (1.f + expf(-data[start + 3 * n_cells]));
    float state = inpGate * tanhf(data[start + 4 * n_cells]);
    if (old_state_y)
    {
      //!!
      state += 0.5 * fgtGate_y * old_state_y[start];
    }
    if (old_state_x)
    {
      //!!
      state += 0.5 * fgtGate_x * old_state_x[start];
    }
    state *= valid_batch;

    //cell output
    output[inner_idx] = outGate * tanhf(state) * valid_batch;

    data[start] = inpGate;
    data[start + n_cells] = fgtGate_y;
    data[start + 2 * n_cells] = fgtGate_x;
    data[start + 3 * n_cells] = outGate;
    data[start + 4 * n_cells] = state;

    idx += gridDim.x * blockDim.x;
  }
}

__global__ void lstm_bwd_stable_cell_kernel_batched(float ** deltas, const float ** epsilons,
  const float ** next_epsilon_ys, const float ** next_epsilon_xs, float ** epsilon_ys, float ** epsilon_xs,
  const float ** last_state_ys, const float ** last_state_xs, const float ** Ys, const float ** valids,
  int n_outer_batch, int n_cells, int n_minibatch)
{
  //layout (for every outer batch):
  //delta[0*n_cells..1*n_cells-1] : input gate
  //delta[1*n_cells..2*n_cells-1] : forget gate
  //delta[2*n_cells..3*n_cells-1] : lambda gate
  //delta[3*n_cells..4*n_cells-1] : output gate
  //delta[4*n_cells..5*n_cells-1] : cell state
  //epsilon[0*n_cells..1*n_cells-1]: cell output derivative
  //next_epsilon_y[0*n_cells..1*n_cells-1]: cell state derivative * forget_gate * lambda_gate (of next timestep)
  //next_epsilon_x[0*n_cells..1*n_cells-1]: cell state derivative * forget_gate * (-1*)lambda_gate (of next timestep)
  //epsilon_y[0*n_cells..1*n_cells-1]: cell state derivative * forget_gate * lambda_gate (of current timestep, as output)
  //epsilon_x[0*n_cells..1*n_cells-1]: cell state derivative * forget_gate * (1-lambda_gate) (of current timestep, as output)
  //valids: either 1.0 or 0.0, indicating if the current (y,x) position is still inside the image in this minibatch
  //repeated for every mini-batch

  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  while (idx < n_outer_batch * n_cells * n_minibatch)
  {
    int size_per_outer_batch = n_cells * n_minibatch;
    int outer_batch_idx = idx / size_per_outer_batch;
    const float * valid = valids[outer_batch_idx];

    float * delta = deltas[outer_batch_idx];
    const float * epsilon = epsilons[outer_batch_idx];
    const float * next_epsilon_y = next_epsilon_ys[outer_batch_idx];
    const float * next_epsilon_x = next_epsilon_xs[outer_batch_idx];
    float * epsilon_y = epsilon_ys[outer_batch_idx];
    float * epsilon_x = epsilon_xs[outer_batch_idx];
    const float * last_state_y = last_state_ys[outer_batch_idx];
    const float * last_state_x = last_state_xs[outer_batch_idx];
    const float * Y = Ys[outer_batch_idx];

    int inner_idx = idx % size_per_outer_batch;
    int minibatch_idx = inner_idx / n_cells;
    int batch_offset = minibatch_idx * 5 * n_cells;
    int cell_offset = inner_idx % n_cells;
    int start = batch_offset + cell_offset;
    float valid_batch = valid[minibatch_idx];

    float inpGate = delta[start];
    float fgtGate_y = delta[start + n_cells];
    float fgtGate_x = delta[start + 2 * n_cells];
    float outGate = delta[start + 3 * n_cells];
    float state = delta[start + 4 * n_cells];
    float lastState_y = last_state_y ? last_state_y[start] : 0.f;
    float lastState_x = last_state_x ? last_state_x[start] : 0.f;
    float eps = epsilon[inner_idx];

    //avoid division by 0
    float gc = 0.f; //g(c(t))
    float gzc = 0.f; //g(z_c(t))
    if (outGate != 0)
    {
      gc = Y[inner_idx] / outGate;
    }

    if (inpGate != 0)
    {
      gzc = (state - fgtGate_y * lastState_y - fgtGate_x * lastState_x) / inpGate;
    }

    //delta_output
    delta[start + 3 * n_cells] = outGate * (1.f - outGate) * gc * eps * valid_batch;

    //epsilon_c
    float epsilon_c = (1.f - (gc * gc)) * outGate * eps;
    if (next_epsilon_y)
    {
      epsilon_c += next_epsilon_y[inner_idx];
    }
    if (next_epsilon_x)
    {
      epsilon_c += next_epsilon_x[inner_idx];
    }

    //TODO: clip epsilon_c?
    //epsilon_c = max(epsilon_c, -10.f);
    //epsilon_c = min(epsilon_c, 10.f);

    //!!
    epsilon_y[inner_idx] = 0.5 * epsilon_c * fgtGate_y;
		epsilon_x[inner_idx] = 0.5 * epsilon_c * fgtGate_x;

    //delta_cell
    delta[start + 4 * n_cells] = inpGate * (1.f - (gzc * gzc)) * epsilon_c * valid_batch;

    //delta_forget_y
    delta[start + n_cells] = fgtGate_y * (1.f - fgtGate_y) * lastState_y * epsilon_c;

    //delta_forget_x
		delta[start + 2 * n_cells] = fgtGate_x * (1.f - fgtGate_x) * lastState_x * epsilon_c;

    //delta_input
    delta[start] = inpGate * (1.f - inpGate) * gzc * epsilon_c * valid_batch;

    idx += gridDim.x * blockDim.x;
  }
}

#endif

void do_lstm_batched_multidir(CudaNdarray * H1, CudaNdarray * H2, CudaNdarray * H3, CudaNdarray * H4,
 CudaNdarray * out1, CudaNdarray * out2, CudaNdarray * out3, CudaNdarray * out4,
 const vector<int>& ys, const vector<int>& xs,
 CudaNdarray * ptr_storage, CudaNdarray * valid_storage, PyArrayObject * sizes, cudaStream_t stream=0)
{
  //CudaNdarray_print_part(H1);

  int n_outer_batch = ys.size();
  const int * H1_dims = CudaNdarray_HOST_DIMS(H1);
  int height = H1_dims[0];
  int width = H1_dims[1];
  int n_minibatch = H1_dims[2];
  int n_cells = H1_dims[3] / 5;
  assert(H1_dims[3] % 5 == 0); //4 gates + cell

  vector<float*> ptrs(4 * 5 * n_outer_batch); //4 dirs * 5 arrays
  vector<float> valid(4 * n_minibatch * n_outer_batch, 1.0f);
  for(int i = 0; i < n_outer_batch; ++i)
  {
    int y = ys[i];
    int x = xs[i];

    //fill valid
    for(int n = 0; n < n_minibatch; ++n)
    {
      //these are the sizes of a single image in the batch, while height/width are the maximum sizes in the batch
      int img_height = int(*(reinterpret_cast<const float*>(PyArray_DATA(sizes)) + 2 * n));
      int img_width = int(*(reinterpret_cast<const float*>(PyArray_DATA(sizes)) + 2 * n + 1));
      valid[i * 4 * n_minibatch + 0 * n_minibatch + n] = float(y < img_height && x < img_width);
      valid[i * 4 * n_minibatch + 1 * n_minibatch + n] = float((height - y) <= img_height && x < img_width);
      valid[i * 4 * n_minibatch + 2 * n_minibatch + n] = float(y < img_height && (width - x) <= img_width);
      valid[i * 4 * n_minibatch + 3 * n_minibatch + n] = float((height - y) <= img_height && (width - x) <= img_width);
    }

    //y not flipped, x not flipped
    float * data_H1 = data_ptr(H1, y, x);
    //y flipped, x not flipped
    float * data_H2 = data_ptr(H2, (height - 1) - y, x);
    //y not flipped, x flipped
    float * data_H3 = data_ptr(H3, y, (width - 1) - x);
    //y flipped, x flipped
    float * data_H4 = data_ptr(H4, (height - 1) - y, (width - 1) - x);

    //y not flipped, x not flipped
    float * data_old_state_y1 = y > 0 ? data_ptr(H1, y - 1, x) + 4 * n_cells : 0;
    //y flipped, x not flipped
    float * data_old_state_y2 = y > 0 ? data_ptr(H2, (height - 1) - (y - 1), x) + 4 * n_cells : 0;
    //y not flipped, x flipped
    float * data_old_state_y3 = y > 0 ? data_ptr(H3, y - 1, (width - 1) - x) + 4 * n_cells : 0;
    //y flipped, x flipped
    float * data_old_state_y4 = y > 0 ? data_ptr(H4, (height - 1) - (y - 1), (width - 1) - x) + 4 * n_cells : 0;

    //y not flipped, x not flipped
    float * data_old_state_x1 = x > 0 ? data_ptr(H1, y, x - 1) + 4 * n_cells : 0;
    //y flipped, x not flipped
    float * data_old_state_x2 = x > 0 ? data_ptr(H2, (height - 1) - y, x - 1) + 4 * n_cells : 0;
    //y not flipped, x flipped
    float * data_old_state_x3 = x > 0 ? data_ptr(H3, y, (width - 1) - (x - 1)) + 4 * n_cells : 0;
    //y flipped, x flipped
    float * data_old_state_x4 = x > 0 ? data_ptr(H4, (height - 1) - y, (width - 1) - (x - 1)) + 4 * n_cells : 0;

    //y not flipped, x not flipped
    float * data_out1 = data_ptr(out1, y, x);
    //y flipped, x not flipped
    float * data_out2 = data_ptr(out2, (height - 1) - y, x);
    //y not flipped, x flipped
    float * data_out3 = data_ptr(out3, y, (width - 1) - x);
    //y flipped, x flipped
    float * data_out4 = data_ptr(out4, (height - 1) - y, (width - 1) - x);

    float * valid1 = CudaNdarray_DEV_DATA(valid_storage) + i * 4 * n_minibatch + 0 * n_minibatch;
    float * valid2 = CudaNdarray_DEV_DATA(valid_storage) + i * 4 * n_minibatch + 1 * n_minibatch;
    float * valid3 = CudaNdarray_DEV_DATA(valid_storage) + i * 4 * n_minibatch + 2 * n_minibatch;
    float * valid4 = CudaNdarray_DEV_DATA(valid_storage) + i * 4 * n_minibatch + 3 * n_minibatch;

    ptrs[0 * 4 * n_outer_batch + 0 * n_outer_batch + i] = data_H1;
    ptrs[0 * 4 * n_outer_batch + 1 * n_outer_batch + i] = data_H2;
    ptrs[0 * 4 * n_outer_batch + 2 * n_outer_batch + i] = data_H3;
    ptrs[0 * 4 * n_outer_batch + 3 * n_outer_batch + i] = data_H4;

    ptrs[1 * 4 * n_outer_batch + 0 * n_outer_batch + i] = data_old_state_y1;
    ptrs[1 * 4 * n_outer_batch + 1 * n_outer_batch + i] = data_old_state_y2;
    ptrs[1 * 4 * n_outer_batch + 2 * n_outer_batch + i] = data_old_state_y3;
    ptrs[1 * 4 * n_outer_batch + 3 * n_outer_batch + i] = data_old_state_y4;

    ptrs[2 * 4 * n_outer_batch + 0 * n_outer_batch + i] = data_old_state_x1;
    ptrs[2 * 4 * n_outer_batch + 1 * n_outer_batch + i] = data_old_state_x2;
    ptrs[2 * 4 * n_outer_batch + 2 * n_outer_batch + i] = data_old_state_x3;
    ptrs[2 * 4 * n_outer_batch + 3 * n_outer_batch + i] = data_old_state_x4;

    ptrs[3 * 4 * n_outer_batch + 0 * n_outer_batch + i] = data_out1;
    ptrs[3 * 4 * n_outer_batch + 1 * n_outer_batch + i] = data_out2;
    ptrs[3 * 4 * n_outer_batch + 2 * n_outer_batch + i] = data_out3;
    ptrs[3 * 4 * n_outer_batch + 3 * n_outer_batch + i] = data_out4;

    ptrs[4 * 4 * n_outer_batch + 0 * n_outer_batch + i] = valid1;
    ptrs[4 * 4 * n_outer_batch + 1 * n_outer_batch + i] = valid2;
    ptrs[4 * 4 * n_outer_batch + 2 * n_outer_batch + i] = valid3;
    ptrs[4 * 4 * n_outer_batch + 3 * n_outer_batch + i] = valid4;
  }

  HANDLE_ERROR(cudaMemcpy(CudaNdarray_DEV_DATA(valid_storage), valid.data(),
    valid.size() * sizeof(float), cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(CudaNdarray_DEV_DATA(ptr_storage), ptrs.data(),
    ptrs.size() * sizeof(float*), cudaMemcpyHostToDevice));
  //HANDLE_ERROR(cudaMemPrefetchAsync(*ptr_storage, ptrs.size() * sizeof(float*), 0, stream));
  float ** ptr_storage_data = reinterpret_cast<float**>(CudaNdarray_DEV_DATA(ptr_storage));
  float ** data_Hs = ptr_storage_data + 0 * 4 * n_outer_batch;
  const float ** data_old_state_ys = (const float**) ptr_storage_data + 1 * 4 * n_outer_batch;
  const float ** data_old_state_xs = (const float**) ptr_storage_data + 2 * 4 * n_outer_batch;
  float ** data_outs = ptr_storage_data + 3 * 4 * n_outer_batch;
  const float ** data_valids = (const float**) (ptr_storage_data + 4 * 4 * n_outer_batch);

  lstm_stable_cell_kernel_batched<<<DIM_GRID, DIM_BLOCK, 0, stream>>>(data_Hs, data_old_state_ys, data_old_state_xs,
    data_outs, data_valids, 4 * n_outer_batch, n_cells, n_minibatch);
  CHECK_KERNEL_ERROR();
}

//epsilon are the derivates w.r.t. Z, delta stores the gate and cell activations and will store the derivatives later
void do_lstm_bwd_batched_multidir(CudaNdarray * delta1, CudaNdarray * delta2, CudaNdarray * delta3, CudaNdarray * delta4,
 CudaNdarray * epsilon1, CudaNdarray * epsilon2, CudaNdarray * epsilon3, CudaNdarray * epsilon4,
 const CudaNdarray * Y1, const CudaNdarray * Y2, const CudaNdarray * Y3, const CudaNdarray * Y4,
 CudaNdarray * workmem1, CudaNdarray * workmem2, CudaNdarray * workmem3, CudaNdarray * workmem4,
 int height, int width, const vector<int>& ys, const vector<int>& xs,
 CudaNdarray * ptr_storage, CudaNdarray * valid_storage, PyArrayObject * sizes, cudaStream_t stream=0)
{
  int n_outer_batch = ys.size();
  int dims[2];
  lastTwoDims(delta1, dims);
  assert(dims[1] % 5 == 0); //4 gates + cell
  int n_cells = dims[1] / 5;
  int n_minibatch = dims[0];

  vector<const float*> ptrs(4 * 10 * n_outer_batch); //4 dirs * 10 arrays
  vector<float> valid(4 * n_minibatch * n_outer_batch, 1.0f);
  for(int i = 0; i < n_outer_batch; ++i)
  {
    int y = ys[i];
    int x = xs[i];

    //fill valid
    for(int n = 0; n < n_minibatch; ++n)
    {
      //these are the sizes of a single image in the batch, while height/width are the maximum sizes in the batch
      int img_height = int(*(reinterpret_cast<const float*>(PyArray_DATA(sizes)) + 2 * n));
      int img_width = int(*(reinterpret_cast<const float*>(PyArray_DATA(sizes)) + 2 * n + 1));
      valid[i * 4 * n_minibatch + 0 * n_minibatch + n] = float(y < img_height && x < img_width);
      valid[i * 4 * n_minibatch + 1 * n_minibatch + n] = float((height - y) <= img_height && x < img_width);
      valid[i * 4 * n_minibatch + 2 * n_minibatch + n] = float(y < img_height && (width - x) <= img_width);
      valid[i * 4 * n_minibatch + 3 * n_minibatch + n] = float((height - y) <= img_height && (width - x) <= img_width);
    }

    bool botBorder = (y == height-1);
    bool rightBorder = (x == width-1);
    int y_flipped = (height - 1) - y;
    int x_flipped = (width - 1) - x;
    int yp1 = y + 1;
    int yp1_flipped = (height - 1) - (y + 1);
    int xp1 = x + 1;
    int xp1_flipped = (width - 1) - (x + 1);
    int ym1 = y - 1;
    int ym1_flipped = (height - 1) - (y - 1);
    int xm1 = x - 1;
    int xm1_flipped = (width - 1) - (x - 1);

    float * data_delta1 = data_ptr(delta1, y, x);
    float * data_delta2 = data_ptr(delta2, y_flipped, x);
    float * data_delta3 = data_ptr(delta3, y, x_flipped);
    float * data_delta4 = data_ptr(delta4, y_flipped, x_flipped);
    const float * data_epsilon1 = data_ptr(epsilon1, y, x);
    const float * data_epsilon2 = data_ptr(epsilon2, y_flipped, x);
    const float * data_epsilon3 = data_ptr(epsilon3, y, x_flipped);
    const float * data_epsilon4 = data_ptr(epsilon4, y_flipped, x_flipped);
    const float * data_next_epsilon_y1 = botBorder ? 0 : data_ptr(workmem1, yp1, x, 0);
    const float * data_next_epsilon_y2 = botBorder ? 0 : data_ptr(workmem2, yp1_flipped, x, 0);
    const float * data_next_epsilon_y3 = botBorder ? 0 : data_ptr(workmem3, yp1, x_flipped, 0);
    const float * data_next_epsilon_y4 = botBorder ? 0 : data_ptr(workmem4, yp1_flipped, x_flipped, 0);
    const float * data_next_epsilon_x1 = rightBorder ? 0 : data_ptr(workmem1, y, xp1, 1);
    const float * data_next_epsilon_x2 = rightBorder ? 0 : data_ptr(workmem2, y_flipped, xp1, 1);
    const float * data_next_epsilon_x3 = rightBorder ? 0 : data_ptr(workmem3, y, xp1_flipped, 1);
    const float * data_next_epsilon_x4 = rightBorder ? 0 : data_ptr(workmem4, y_flipped, xp1_flipped, 1);
    float * data_epsilon_y1 = data_ptr(workmem1, y, x, 0);
    float * data_epsilon_y2 = data_ptr(workmem2, y_flipped, x, 0);
    float * data_epsilon_y3 = data_ptr(workmem3, y, x_flipped, 0);
    float * data_epsilon_y4 = data_ptr(workmem4, y_flipped, x_flipped, 0);
    float * data_epsilon_x1 = data_ptr(workmem1, y, x, 1);
    float * data_epsilon_x2 = data_ptr(workmem2, y_flipped, x, 1);
    float * data_epsilon_x3 = data_ptr(workmem3, y, x_flipped, 1);
    float * data_epsilon_x4 = data_ptr(workmem4, y_flipped, x_flipped, 1);
    const float * data_last_state_y1 = y > 0 ? data_ptr(delta1, ym1, x) + 4 * n_cells : 0;
    const float * data_last_state_y2 = y > 0 ? data_ptr(delta2, ym1_flipped, x) + 4 * n_cells : 0;
    const float * data_last_state_y3 = y > 0 ? data_ptr(delta3, ym1, x_flipped) + 4 * n_cells : 0;
    const float * data_last_state_y4 = y > 0 ? data_ptr(delta4, ym1_flipped, x_flipped) + 4 * n_cells : 0;
    const float * data_last_state_x1 = x > 0 ? data_ptr(delta1, y, xm1) + 4 * n_cells : 0;
    const float * data_last_state_x2 = x > 0 ? data_ptr(delta2, y_flipped, xm1) + 4 * n_cells : 0;
    const float * data_last_state_x3 = x > 0 ? data_ptr(delta3, y, xm1_flipped) + 4 * n_cells : 0;
    const float * data_last_state_x4 = x > 0 ? data_ptr(delta4, y_flipped, xm1_flipped) + 4 * n_cells : 0;
    const float * data_Y1 = data_ptr(Y1, y, x);
    const float * data_Y2 = data_ptr(Y2, y_flipped, x);
    const float * data_Y3 = data_ptr(Y3, y, x_flipped);
    const float * data_Y4 = data_ptr(Y4, y_flipped, x_flipped);
    float * valid1 = CudaNdarray_DEV_DATA(valid_storage) + i * 4 * n_minibatch + 0 * n_minibatch;
    float * valid2 = CudaNdarray_DEV_DATA(valid_storage) + i * 4 * n_minibatch + 1 * n_minibatch;
    float * valid3 = CudaNdarray_DEV_DATA(valid_storage) + i * 4 * n_minibatch + 2 * n_minibatch;
    float * valid4 = CudaNdarray_DEV_DATA(valid_storage) + i * 4 * n_minibatch + 3 * n_minibatch;

    ptrs[0 * 4 * n_outer_batch + 0 * n_outer_batch + i] = data_delta1;
    ptrs[0 * 4 * n_outer_batch + 1 * n_outer_batch + i] = data_delta2;
    ptrs[0 * 4 * n_outer_batch + 2 * n_outer_batch + i] = data_delta3;
    ptrs[0 * 4 * n_outer_batch + 3 * n_outer_batch + i] = data_delta4;
    ptrs[1 * 4 * n_outer_batch + 0 * n_outer_batch + i] = data_epsilon1;
    ptrs[1 * 4 * n_outer_batch + 1 * n_outer_batch + i] = data_epsilon2;
    ptrs[1 * 4 * n_outer_batch + 2 * n_outer_batch + i] = data_epsilon3;
    ptrs[1 * 4 * n_outer_batch + 3 * n_outer_batch + i] = data_epsilon4;
    ptrs[2 * 4 * n_outer_batch + 0 * n_outer_batch + i] = data_next_epsilon_y1;
    ptrs[2 * 4 * n_outer_batch + 1 * n_outer_batch + i] = data_next_epsilon_y2;
    ptrs[2 * 4 * n_outer_batch + 2 * n_outer_batch + i] = data_next_epsilon_y3;
    ptrs[2 * 4 * n_outer_batch + 3 * n_outer_batch + i] = data_next_epsilon_y4;
    ptrs[3 * 4 * n_outer_batch + 0 * n_outer_batch + i] = data_next_epsilon_x1;
    ptrs[3 * 4 * n_outer_batch + 1 * n_outer_batch + i] = data_next_epsilon_x2;
    ptrs[3 * 4 * n_outer_batch + 2 * n_outer_batch + i] = data_next_epsilon_x3;
    ptrs[3 * 4 * n_outer_batch + 3 * n_outer_batch + i] = data_next_epsilon_x4;
    ptrs[4 * 4 * n_outer_batch + 0 * n_outer_batch + i] = data_epsilon_y1;
    ptrs[4 * 4 * n_outer_batch + 1 * n_outer_batch + i] = data_epsilon_y2;
    ptrs[4 * 4 * n_outer_batch + 2 * n_outer_batch + i] = data_epsilon_y3;
    ptrs[4 * 4 * n_outer_batch + 3 * n_outer_batch + i] = data_epsilon_y4;
    ptrs[5 * 4 * n_outer_batch + 0 * n_outer_batch + i] = data_epsilon_x1;
    ptrs[5 * 4 * n_outer_batch + 1 * n_outer_batch + i] = data_epsilon_x2;
    ptrs[5 * 4 * n_outer_batch + 2 * n_outer_batch + i] = data_epsilon_x3;
    ptrs[5 * 4 * n_outer_batch + 3 * n_outer_batch + i] = data_epsilon_x4;
    ptrs[6 * 4 * n_outer_batch + 0 * n_outer_batch + i] = data_last_state_y1;
    ptrs[6 * 4 * n_outer_batch + 1 * n_outer_batch + i] = data_last_state_y2;
    ptrs[6 * 4 * n_outer_batch + 2 * n_outer_batch + i] = data_last_state_y3;
    ptrs[6 * 4 * n_outer_batch + 3 * n_outer_batch + i] = data_last_state_y4;
    ptrs[7 * 4 * n_outer_batch + 0 * n_outer_batch + i] = data_last_state_x1;
    ptrs[7 * 4 * n_outer_batch + 1 * n_outer_batch + i] = data_last_state_x2;
    ptrs[7 * 4 * n_outer_batch + 2 * n_outer_batch + i] = data_last_state_x3;
    ptrs[7 * 4 * n_outer_batch + 3 * n_outer_batch + i] = data_last_state_x4;
    ptrs[8 * 4 * n_outer_batch + 0 * n_outer_batch + i] = data_Y1;
    ptrs[8 * 4 * n_outer_batch + 1 * n_outer_batch + i] = data_Y2;
    ptrs[8 * 4 * n_outer_batch + 2 * n_outer_batch + i] = data_Y3;
    ptrs[8 * 4 * n_outer_batch + 3 * n_outer_batch + i] = data_Y4;
    ptrs[9 * 4 * n_outer_batch + 0 * n_outer_batch + i] = valid1;
    ptrs[9 * 4 * n_outer_batch + 1 * n_outer_batch + i] = valid2;
    ptrs[9 * 4 * n_outer_batch + 2 * n_outer_batch + i] = valid3;
    ptrs[9 * 4 * n_outer_batch + 3 * n_outer_batch + i] = valid4;
  }
  HANDLE_ERROR(cudaMemcpy(CudaNdarray_DEV_DATA(valid_storage), valid.data(),
    valid.size() * sizeof(float), cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(CudaNdarray_DEV_DATA(ptr_storage), ptrs.data(),
    ptrs.size() * sizeof(float*), cudaMemcpyHostToDevice));
  float ** ptr_storage_data = reinterpret_cast<float**>(CudaNdarray_DEV_DATA(ptr_storage));
  float ** data_deltas = ptr_storage_data + 0 * 4 * n_outer_batch;
  const float ** data_epsilons = (const float**) ptr_storage_data + 1 * 4 * n_outer_batch;
  const float ** data_next_epsilon_ys = (const float**) ptr_storage_data + 2 * 4 * n_outer_batch;
  const float ** data_next_epsilon_xs = (const float**) ptr_storage_data + 3 * 4 * n_outer_batch;
  float ** data_epsilon_ys = ptr_storage_data + 4 * 4 * n_outer_batch;
  float ** data_epsilon_xs = ptr_storage_data + 5 * 4 * n_outer_batch;
  const float ** data_last_state_ys = (const float**) ptr_storage_data + 6 * 4 * n_outer_batch;
  const float ** data_last_state_xs = (const float**) ptr_storage_data + 7 * 4 * n_outer_batch;
  const float ** data_Ys = (const float**) ptr_storage_data + 8 * 4 * n_outer_batch;
  const float ** data_valids = (const float**) (ptr_storage_data + 9 * 4 * n_outer_batch);

  lstm_bwd_stable_cell_kernel_batched<<<DIM_GRID, DIM_BLOCK, 0, stream>>>(data_deltas, data_epsilons, data_next_epsilon_ys,
    data_next_epsilon_xs, data_epsilon_ys, data_epsilon_xs, data_last_state_ys, data_last_state_xs,
    data_Ys, data_valids, 4 * n_outer_batch, n_cells, n_minibatch);
  CHECK_KERNEL_ERROR();
}

void do_lstm_batched_bidir(CudaNdarray * H1, CudaNdarray * H2, CudaNdarray * out1, CudaNdarray * out2,
 const vector<int>& ys, const vector<int>& xs,
 CudaNdarray * ptr_storage, CudaNdarray * valid_storage, PyArrayObject * sizes, cudaStream_t stream=0)
{
  //CudaNdarray_print_part(H1);

  int n_outer_batch = ys.size();
  const int * H1_dims = CudaNdarray_HOST_DIMS(H1);
  int height = H1_dims[0];
  int width = H1_dims[1];
  int n_minibatch = H1_dims[2];
  int n_cells = H1_dims[3] / 5;
  assert(H1_dims[3] % 5 == 0); //4 gates + cell

  vector<float*> ptrs(2 * 5 * n_outer_batch); //4 dirs * 5 arrays
  vector<float> valid(2 * n_minibatch * n_outer_batch, 1.0f);
  for(int i = 0; i < n_outer_batch; ++i)
  {
    int y = ys[i];
    int x = xs[i];

    //fill valid
    for(int n = 0; n < n_minibatch; ++n)
    {
      //these are the sizes of a single image in the batch, while height/width are the maximum sizes in the batch
      int img_height = int(*(reinterpret_cast<const float*>(PyArray_DATA(sizes)) + 2 * n));
      int img_width = int(*(reinterpret_cast<const float*>(PyArray_DATA(sizes)) + 2 * n + 1));
      valid[i * 2 * n_minibatch + 0 * n_minibatch + n] = float(y < img_height && x < img_width);
      valid[i * 2 * n_minibatch + 1 * n_minibatch + n] = float(y < img_height && (width - x) <= img_width);
    }

    //y not flipped, x not flipped
    float * data_H1 = data_ptr(H1, y, x);
    //y not flipped, x flipped
    float * data_H2 = data_ptr(H2, y, (width - 1) - x);

    //y not flipped, x not flipped
    float * data_old_state_y1 = y > 0 ? data_ptr(H1, y - 1, x) + 2 * n_cells : 0;
    //y flipped, x not flipped
    float * data_old_state_y2 = y > 0 ? data_ptr(H2, y - 1, (width - 1) - x) + 2 * n_cells : 0;

    //y not flipped, x not flipped
    float * data_old_state_x1 = x > 0 ? data_ptr(H1, y, x - 1) + 2 * n_cells : 0;
    //y not flipped, x flipped
    float * data_old_state_x2 = x > 0 ? data_ptr(H2, y, (width - 1) - (x - 1)) + 2 * n_cells : 0;

    //y not flipped, x not flipped
    float * data_out1 = data_ptr(out1, y, x);
    //y not flipped, x flipped
    float * data_out2 = data_ptr(out2, y, (width - 1) - x);

    float * valid1 = CudaNdarray_DEV_DATA(valid_storage) + i * 2 * n_minibatch + 0 * n_minibatch;
    float * valid2 = CudaNdarray_DEV_DATA(valid_storage) + i * 2 * n_minibatch + 1 * n_minibatch;

    ptrs[0 * 2 * n_outer_batch + 0 * n_outer_batch + i] = data_H1;
    ptrs[0 * 2 * n_outer_batch + 1 * n_outer_batch + i] = data_H2;

    ptrs[1 * 2 * n_outer_batch + 0 * n_outer_batch + i] = data_old_state_y1;
    ptrs[1 * 2 * n_outer_batch + 1 * n_outer_batch + i] = data_old_state_y2;

    ptrs[2 * 2 * n_outer_batch + 0 * n_outer_batch + i] = data_old_state_x1;
    ptrs[2 * 2 * n_outer_batch + 1 * n_outer_batch + i] = data_old_state_x2;

    ptrs[3 * 2 * n_outer_batch + 0 * n_outer_batch + i] = data_out1;
    ptrs[3 * 2 * n_outer_batch + 1 * n_outer_batch + i] = data_out2;

    ptrs[4 * 2 * n_outer_batch + 0 * n_outer_batch + i] = valid1;
    ptrs[4 * 2 * n_outer_batch + 1 * n_outer_batch + i] = valid2;
  }

  HANDLE_ERROR(cudaMemcpy(CudaNdarray_DEV_DATA(valid_storage), valid.data(),
    valid.size() * sizeof(float), cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(CudaNdarray_DEV_DATA(ptr_storage), ptrs.data(),
    ptrs.size() * sizeof(float*), cudaMemcpyHostToDevice));
  float ** ptr_storage_data = reinterpret_cast<float**>(CudaNdarray_DEV_DATA(ptr_storage));
  float ** data_Hs = ptr_storage_data + 0 * 2 * n_outer_batch;
  const float ** data_old_state_ys = (const float**) ptr_storage_data + 1 * 2 * n_outer_batch;
  const float ** data_old_state_xs = (const float**) ptr_storage_data + 2 * 2 * n_outer_batch;
  float ** data_outs = ptr_storage_data + 3 * 2 * n_outer_batch;
  const float ** data_valids = (const float**) (ptr_storage_data + 4 * 2 * n_outer_batch);

  lstm_stable_cell_kernel_batched<<<DIM_GRID, DIM_BLOCK, 0, stream>>>(data_Hs, data_old_state_ys, data_old_state_xs,
    data_outs, data_valids, 2 * n_outer_batch, n_cells, n_minibatch);
  CHECK_KERNEL_ERROR();
}

//epsilon are the derivates w.r.t. Z, delta stores the gate and cell activations and will store the derivatives later
void do_lstm_bwd_batched_bidir(CudaNdarray * delta1, CudaNdarray * delta2, CudaNdarray * epsilon1, CudaNdarray * epsilon2,
 const CudaNdarray * Y1, const CudaNdarray * Y2, CudaNdarray * workmem1, CudaNdarray * workmem2,
 int height, int width, const vector<int>& ys, const vector<int>& xs,
 CudaNdarray * ptr_storage, CudaNdarray * valid_storage, PyArrayObject * sizes, cudaStream_t stream=0)
{
  int n_outer_batch = ys.size();
  int dims[2];
  lastTwoDims(delta1, dims);
  assert(dims[1] % 5 == 0); //4 gates + cell
  int n_cells = dims[1] / 5;
  int n_minibatch = dims[0];

  vector<const float*> ptrs(2 * 10 * n_outer_batch); //2 dirs * 10 arrays
  vector<float> valid(2 * n_minibatch * n_outer_batch, 1.0f);
  for(int i = 0; i < n_outer_batch; ++i)
  {
    int y = ys[i];
    int x = xs[i];

    //fill valid
    for(int n = 0; n < n_minibatch; ++n)
    {
      //these are the sizes of a single image in the batch, while height/width are the maximum sizes in the batch
      int img_height = int(*(reinterpret_cast<const float*>(PyArray_DATA(sizes)) + 2 * n));
      int img_width = int(*(reinterpret_cast<const float*>(PyArray_DATA(sizes)) + 2 * n + 1));
      valid[i * 2 * n_minibatch + 0 * n_minibatch + n] = float(y < img_height && x < img_width);
      valid[i * 2 * n_minibatch + 1 * n_minibatch + n] = float(y < img_height && (width - x) <= img_width);
    }

    bool botBorder = (y == height-1);
    bool rightBorder = (x == width-1);
    int y_flipped = (height - 1) - y;
    int x_flipped = (width - 1) - x;
    int yp1 = y + 1;
    int yp1_flipped = (height - 1) - (y + 1);
    int xp1 = x + 1;
    int xp1_flipped = (width - 1) - (x + 1);
    int ym1 = y - 1;
    int ym1_flipped = (height - 1) - (y - 1);
    int xm1 = x - 1;
    int xm1_flipped = (width - 1) - (x - 1);

    float * data_delta1 = data_ptr(delta1, y, x);
    float * data_delta2 = data_ptr(delta2, y, x_flipped);
    const float * data_epsilon1 = data_ptr(epsilon1, y, x);
    const float * data_epsilon2 = data_ptr(epsilon2, y, x_flipped);
    const float * data_next_epsilon_y1 = botBorder ? 0 : data_ptr(workmem1, yp1, x, 0);
    const float * data_next_epsilon_y2 = botBorder ? 0 : data_ptr(workmem2, yp1, x_flipped, 0);
    const float * data_next_epsilon_x1 = rightBorder ? 0 : data_ptr(workmem1, y, xp1, 1);
    const float * data_next_epsilon_x2 = rightBorder ? 0 : data_ptr(workmem2, y, xp1_flipped, 1);
    float * data_epsilon_y1 = data_ptr(workmem1, y, x, 0);
    float * data_epsilon_y2 = data_ptr(workmem2, y, x_flipped, 0);
    float * data_epsilon_x1 = data_ptr(workmem1, y, x, 1);
    float * data_epsilon_x2 = data_ptr(workmem2, y, x_flipped, 1);
    const float * data_last_state_y1 = y > 0 ? data_ptr(delta1, ym1, x) + 2 * n_cells : 0;
    const float * data_last_state_y2 = y > 0 ? data_ptr(delta2, ym1, x_flipped) + 2 * n_cells : 0;
    const float * data_last_state_x1 = x > 0 ? data_ptr(delta1, y, xm1) + 2 * n_cells : 0;
    const float * data_last_state_x2 = x > 0 ? data_ptr(delta2, y, xm1_flipped) + 2 * n_cells : 0;
    const float * data_Y1 = data_ptr(Y1, y, x);
    const float * data_Y2 = data_ptr(Y2, y, x_flipped);
    float * valid1 = CudaNdarray_DEV_DATA(valid_storage) + i * 2 * n_minibatch + 0 * n_minibatch;
    float * valid2 = CudaNdarray_DEV_DATA(valid_storage) + i * 2 * n_minibatch + 1 * n_minibatch;

    ptrs[0 * 2 * n_outer_batch + 0 * n_outer_batch + i] = data_delta1;
    ptrs[0 * 2 * n_outer_batch + 1 * n_outer_batch + i] = data_delta2;
    ptrs[1 * 2 * n_outer_batch + 0 * n_outer_batch + i] = data_epsilon1;
    ptrs[1 * 2 * n_outer_batch + 1 * n_outer_batch + i] = data_epsilon2;
    ptrs[2 * 2 * n_outer_batch + 0 * n_outer_batch + i] = data_next_epsilon_y1;
    ptrs[2 * 2 * n_outer_batch + 1 * n_outer_batch + i] = data_next_epsilon_y2;
    ptrs[3 * 2 * n_outer_batch + 0 * n_outer_batch + i] = data_next_epsilon_x1;
    ptrs[3 * 2 * n_outer_batch + 1 * n_outer_batch + i] = data_next_epsilon_x2;
    ptrs[4 * 2 * n_outer_batch + 0 * n_outer_batch + i] = data_epsilon_y1;
    ptrs[4 * 2 * n_outer_batch + 1 * n_outer_batch + i] = data_epsilon_y2;
    ptrs[5 * 2 * n_outer_batch + 0 * n_outer_batch + i] = data_epsilon_x1;
    ptrs[5 * 2 * n_outer_batch + 1 * n_outer_batch + i] = data_epsilon_x2;
    ptrs[6 * 2 * n_outer_batch + 0 * n_outer_batch + i] = data_last_state_y1;
    ptrs[6 * 2 * n_outer_batch + 1 * n_outer_batch + i] = data_last_state_y2;
    ptrs[7 * 2 * n_outer_batch + 0 * n_outer_batch + i] = data_last_state_x1;
    ptrs[7 * 2 * n_outer_batch + 1 * n_outer_batch + i] = data_last_state_x2;
    ptrs[8 * 2 * n_outer_batch + 0 * n_outer_batch + i] = data_Y1;
    ptrs[8 * 2 * n_outer_batch + 1 * n_outer_batch + i] = data_Y2;
    ptrs[9 * 2 * n_outer_batch + 0 * n_outer_batch + i] = valid1;
    ptrs[9 * 2 * n_outer_batch + 1 * n_outer_batch + i] = valid2;
  }
  HANDLE_ERROR(cudaMemcpy(CudaNdarray_DEV_DATA(valid_storage), valid.data(),
    valid.size() * sizeof(float), cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(CudaNdarray_DEV_DATA(ptr_storage), ptrs.data(),
    ptrs.size() * sizeof(float*), cudaMemcpyHostToDevice));
  float ** ptr_storage_data = reinterpret_cast<float**>(CudaNdarray_DEV_DATA(ptr_storage));
  float ** data_deltas = ptr_storage_data + 0 * 2 * n_outer_batch;
  const float ** data_epsilons = (const float**) ptr_storage_data + 1 * 2 * n_outer_batch;
  const float ** data_next_epsilon_ys = (const float**) ptr_storage_data + 2 * 2 * n_outer_batch;
  const float ** data_next_epsilon_xs = (const float**) ptr_storage_data + 3 * 2 * n_outer_batch;
  float ** data_epsilon_ys = ptr_storage_data + 4 * 2 * n_outer_batch;
  float ** data_epsilon_xs = ptr_storage_data + 5 * 2 * n_outer_batch;
  const float ** data_last_state_ys = (const float**) ptr_storage_data + 6 * 2 * n_outer_batch;
  const float ** data_last_state_xs = (const float**) ptr_storage_data + 7 * 2 * n_outer_batch;
  const float ** data_Ys = (const float**) ptr_storage_data + 8 * 2 * n_outer_batch;
  const float ** data_valids = (const float**) (ptr_storage_data + 9 * 2 * n_outer_batch);

  lstm_bwd_stable_cell_kernel_batched<<<DIM_GRID, DIM_BLOCK, 0, stream>>>(data_deltas, data_epsilons, data_next_epsilon_ys,
    data_next_epsilon_xs, data_epsilon_ys, data_epsilon_xs, data_last_state_ys, data_last_state_xs,
    data_Ys, data_valids, 2 * n_outer_batch, n_cells, n_minibatch);
  CHECK_KERNEL_ERROR();
}


__global__ void repvec(const float * v, int vlen, int nCopies, float * dest)
{
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  while (idx < vlen * nCopies)
  {
    dest[idx] = v[idx % vlen];
    idx += gridDim.x * blockDim.x;
  }
}

void fillmat(const CudaNdarray * b, CudaNdarray * dst)
{
  const float * data_b = CudaNdarray_DEV_DATA(b);
  float * data_dst = CudaNdarray_DEV_DATA(dst);
  const int * dims_b = CudaNdarray_HOST_DIMS(b);
  int dims_dst[2];
  lastTwoDims(dst, dims_dst);
  assert(dims_b[0] == dims_dst[1]);
  repvec<<<DIM_GRID, DIM_BLOCK>>>(data_b, dims_dst[1], CudaNdarray_SIZE(dst)/dims_dst[1], data_dst);
  CHECK_KERNEL_ERROR();
}

//ys and xs: base indices, offset by y_A, x_A (-1,0,1)
void affine_y_x_batched_multidir(int y_A, int x_A,
  const CudaNdarray * A1, const CudaNdarray * A2, const CudaNdarray * A3, const CudaNdarray * A4,
  const CudaNdarray * B1, const CudaNdarray * B2, const CudaNdarray * B3, const CudaNdarray * B4,
  CudaNdarray * C1, CudaNdarray * C2, CudaNdarray * C3, CudaNdarray * C4,
  const vector<int>& ys, const vector<int>& xs, CudaNdarray * ptr_storage, int height, int width,
  cudaStream_t stream = 0, bool transpose_A=false, bool transpose_B=false)
{
  const int batch_size = ys.size();
  if(batch_size == 0)
  {
    return;
  }
  vector<const float*> ABC_ptrs(3 * 4 * batch_size); //content layout: 3x4xbatch_size (3: A,B,C, 4: dirs)

  for(int i = 0; i < batch_size; ++i)
  {
    //A
    //y not flipped, x not flipped
    ABC_ptrs[0 * 4 * batch_size + 0 * batch_size + i] = data_ptr(A1, y_A + ys[i], x_A + xs[i]);
    //y flipped, x not flipped
    ABC_ptrs[0 * 4 * batch_size + 1 * batch_size + i] = data_ptr(A2, (height - 1) - (y_A + ys[i]), x_A + xs[i]);
    //y not flipped, x flipped
    ABC_ptrs[0 * 4 * batch_size + 2 * batch_size + i] = data_ptr(A3, y_A + ys[i], (width - 1) - (x_A + xs[i]));
    //y flipped, x flipped
    ABC_ptrs[0 * 4 * batch_size + 3 * batch_size + i] = data_ptr(A4, (height - 1) - (y_A + ys[i]),
                                                                  (width - 1) - (x_A + xs[i]));

    //B
    //index doesent matter here, as B is only 2dimensional
    ABC_ptrs[1 * 4 * batch_size + 0 * batch_size + i] = data_ptr(B1, 0, 0);
    ABC_ptrs[1 * 4 * batch_size + 1 * batch_size + i] = data_ptr(B2, 0, 0);
    ABC_ptrs[1 * 4 * batch_size + 2 * batch_size + i] = data_ptr(B3, 0, 0);
    ABC_ptrs[1 * 4 * batch_size + 3 * batch_size + i] = data_ptr(B4, 0, 0);

    //we write the result (C) in the same destination (y,x) as the source (A), so we don't need to flip later
    //C
    //y not flipped, x not flipped
    ABC_ptrs[2 * 4 * batch_size + 0 * batch_size + i] = data_ptr(C1, ys[i], xs[i]);
    //y flipped, x not flipped
    ABC_ptrs[2 * 4 * batch_size + 1 * batch_size + i] = data_ptr(C2, (height - 1) - ys[i], xs[i]);
    //y not flipped, x flipped
    ABC_ptrs[2 * 4 * batch_size + 2 * batch_size + i] = data_ptr(C3, ys[i], (width - 1) - xs[i]);
    //y flipped, x flipped
    ABC_ptrs[2 * 4 * batch_size + 3 * batch_size + i] = data_ptr(C4, (height - 1) - ys[i], (width - 1) - xs[i]);
  }

  HANDLE_ERROR(cudaMemcpy(CudaNdarray_DEV_DATA(ptr_storage), ABC_ptrs.data(),
    ABC_ptrs.size() * sizeof(float*), cudaMemcpyHostToDevice));
  float ** ptr_storage_data = reinterpret_cast<float**>(CudaNdarray_DEV_DATA(ptr_storage));
  const float ** A_ptrs_data = (const float**) ptr_storage_data + 0 * 4 * batch_size;
  const float ** B_ptrs_data = (const float**) ptr_storage_data + 1 * 4 * batch_size;
  float ** C_ptrs_data = ptr_storage_data + 2 * 4 * batch_size;

  int A_dim[2], B_dim[2];
  lastTwoDims(A1, A_dim);
  lastTwoDims(B1, B_dim);
  int ldB = B_dim[1];
  int ldA = A_dim[1];
  cublasOperation_t transA = transpose_A ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t transB = transpose_B ? CUBLAS_OP_T : CUBLAS_OP_N;
  if (transpose_A)
  {
    std::swap(A_dim[0], A_dim[1]);
  }
  if (transpose_B)
  {
    std::swap(B_dim[0], B_dim[1]);
  }

  const float alpha = 1;
  const float beta = 1;
  if(stream)
  {
    HANDLE_ERROR(cublasSetStream(handle, stream));
  }
  HANDLE_ERROR(cublasSgemmBatched(handle, transB, transA, B_dim[1], A_dim[0], A_dim[1], &alpha, B_ptrs_data, ldB,
    A_ptrs_data, ldA, &beta, C_ptrs_data, B_dim[1], 4 * batch_size));
}

//ys and xs: base indices, offset by y_A, x_A (-1,0,1)
void affine_y_x_batched_bidir(int y_A, int x_A,
  const CudaNdarray * A1, const CudaNdarray * A2,
  const CudaNdarray * B1, const CudaNdarray * B2,
  CudaNdarray * C1, CudaNdarray * C2,
  const vector<int>& ys, const vector<int>& xs, CudaNdarray * ptr_storage, int height, int width,
  cudaStream_t stream = 0, bool transpose_A=false, bool transpose_B=false)
{
  const int batch_size = ys.size();
  if(batch_size == 0)
  {
    return;
  }
  vector<const float*> ABC_ptrs(3 * 2 * batch_size); //content layout: 3x2xbatch_size (3: A,B,C, 2: dirs)

  for(int i = 0; i < batch_size; ++i)
  {
    //A
    //y not flipped, x not flipped
    ABC_ptrs[0 * 2 * batch_size + 0 * batch_size + i] = data_ptr(A1, y_A + ys[i], x_A + xs[i]);
    //y not flipped, x flipped
    ABC_ptrs[0 * 2 * batch_size + 1 * batch_size + i] = data_ptr(A2, y_A + ys[i], (width - 1) - (x_A + xs[i]));

    //B
    //index doesent matter here, as B is only 2dimensional
    ABC_ptrs[1 * 2 * batch_size + 0 * batch_size + i] = data_ptr(B1, 0, 0);
    ABC_ptrs[1 * 2 * batch_size + 1 * batch_size + i] = data_ptr(B2, 0, 0);

    //we write the result (C) in the same destination (y,x) as the source (A), so we don't need to flip later
    //C
    //y not flipped, x not flipped
    ABC_ptrs[2 * 2 * batch_size + 0 * batch_size + i] = data_ptr(C1, ys[i], xs[i]);
    //y not flipped, x flipped
    ABC_ptrs[2 * 2 * batch_size + 1 * batch_size + i] = data_ptr(C2, ys[i], (width - 1) - xs[i]);
  }
  HANDLE_ERROR(cudaMemcpy(CudaNdarray_DEV_DATA(ptr_storage), ABC_ptrs.data(),
    ABC_ptrs.size() * sizeof(float*), cudaMemcpyHostToDevice));
  float ** ptr_storage_data = reinterpret_cast<float**>(CudaNdarray_DEV_DATA(ptr_storage));
  const float ** A_ptrs_data = (const float**) ptr_storage_data + 0 * 2 * batch_size;
  const float ** B_ptrs_data = (const float**) ptr_storage_data + 1 * 2 * batch_size;
  float ** C_ptrs_data = ptr_storage_data + 2 * 2 * batch_size;

  int A_dim[2], B_dim[2];
  lastTwoDims(A1, A_dim);
  lastTwoDims(B1, B_dim);
  int ldB = B_dim[1];
  int ldA = A_dim[1];
  cublasOperation_t transA = transpose_A ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t transB = transpose_B ? CUBLAS_OP_T : CUBLAS_OP_N;
  if (transpose_A)
  {
    std::swap(A_dim[0], A_dim[1]);
  }
  if (transpose_B)
  {
    std::swap(B_dim[0], B_dim[1]);
  }

  const float alpha = 1;
  const float beta = 1;
  if(stream)
  {
    HANDLE_ERROR(cublasSetStream(handle, stream));
  }
  HANDLE_ERROR(cublasSgemmBatched(handle, transB, transA, B_dim[1], A_dim[0], A_dim[1], &alpha, B_ptrs_data, ldB,
    A_ptrs_data, ldA, &beta, C_ptrs_data, B_dim[1], 2 * batch_size));
}

//offsets is used for x time-shifts of A and B
void affine_global(const CudaNdarray * A, const CudaNdarray * B, CudaNdarray * C,
  bool transpose_A=false, bool transpose_B=false, int offsetA = 0, int offsetB = 0, float beta = 1.0)
{
  assert(offsetA == 0 || offsetB == 0);
  int totalOffset = offsetA + offsetB;

  float * data_C = CudaNdarray_DEV_DATA(C);
  int A_dim[2], B_dim[2];
  lastTwoDims(A, A_dim);
  lastTwoDims(B, B_dim);
  int shiftA = A_dim[1] * A_dim[0];
  int shiftB = B_dim[1] * B_dim[0];
  A_dim[0] = CudaNdarray_SIZE(A) / A_dim[1] - totalOffset * A_dim[0];
  //TODO: maybe this is wrong, as it uses the updated value of A_dim[0]
  B_dim[0] = CudaNdarray_SIZE(B) / B_dim[1] - totalOffset * A_dim[0];
  const float * data_A = CudaNdarray_DEV_DATA(A) + offsetA * shiftA;
  const float * data_B = CudaNdarray_DEV_DATA(B) + offsetB * shiftB;

  int ldB = B_dim[1];
  int ldA = A_dim[1];
  cublasOperation_t transA = transpose_A ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t transB = transpose_B ? CUBLAS_OP_T : CUBLAS_OP_N;
  if (transpose_A)
  {
    std::swap(A_dim[0], A_dim[1]);
  }
  if (transpose_B)
  {
    std::swap(B_dim[0], B_dim[1]);
  }

  const float alpha = 1;
  HANDLE_ERROR(cublasSgemm(handle, transB, transA, B_dim[1], A_dim[0], A_dim[1], &alpha, data_B, ldB,
    data_A, ldA, &beta, data_C, B_dim[1]));
}
