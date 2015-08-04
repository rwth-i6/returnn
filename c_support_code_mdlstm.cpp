#include <thrust/device_vector.h>

#define DIM_GRID 64
#define DIM_BLOCK 512

static const char *_cudaGetErrorEnum(cublasStatus_t error)
{
	switch (error)
	{
	case CUBLAS_STATUS_SUCCESS:
		return "CUBLAS_STATUS_SUCCESS";

	case CUBLAS_STATUS_NOT_INITIALIZED:
		return "CUBLAS_STATUS_NOT_INITIALIZED";

	case CUBLAS_STATUS_ALLOC_FAILED:
		return "CUBLAS_STATUS_ALLOC_FAILED";

	case CUBLAS_STATUS_INVALID_VALUE:
		return "CUBLAS_STATUS_INVALID_VALUE";

	case CUBLAS_STATUS_ARCH_MISMATCH:
		return "CUBLAS_STATUS_ARCH_MISMATCH";

	case CUBLAS_STATUS_MAPPING_ERROR:
		return "CUBLAS_STATUS_MAPPING_ERROR";

	case CUBLAS_STATUS_EXECUTION_FAILED:
		return "CUBLAS_STATUS_EXECUTION_FAILED";

	case CUBLAS_STATUS_INTERNAL_ERROR:
		return "CUBLAS_STATUS_INTERNAL_ERROR";
	}

	return "<unknown>";
}

static void HandleError(cudaError_t err, const char *file, int line) 
{
	if (err != cudaSuccess) 
	{
		printf("%s in %s at line %d\n", cudaGetErrorString(err),
			file, line);
		exit(EXIT_FAILURE);
	}
}

static void HandleError(cublasStatus_t status, const char *file, int line)
{
	if (status != CUBLAS_STATUS_SUCCESS)
	{
		printf("%s in %s at line %d\n", _cudaGetErrorEnum(status),
			file, line);
		exit(EXIT_FAILURE);
	}
}

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

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

CudaNdarray * CudaNdarray_zeros_like(CudaNdarray* a)
{
	const int * dim = CudaNdarray_HOST_DIMS(a);
	CudaNdarray * res = (CudaNdarray*) CudaNdarray_NewDims(a->nd, dim);
	int n = CudaNdarray_SIZE(a);	
	HANDLE_ERROR(cudaMemset(CudaNdarray_DEV_DATA(res), 0, sizeof(float) * n));
	return res;
}

CudaNdarray * CudaNdarray_uninitialized_like(CudaNdarray* a)
{
	const int * dim = CudaNdarray_HOST_DIMS(a);
	CudaNdarray * res = (CudaNdarray*)CudaNdarray_NewDims(a->nd, dim);
	return res;
}

//if nd is 2 then assume a weight matrix and just return beginning of data
//else nd should be 3 and we pick the x part
const float * data_ptr(const CudaNdarray * a, int y, int x)
{
	assert(a->nd == 2 || a->nd == 3);
	if (a->nd == 2)
	{
		return CudaNdarray_DEV_DATA(a);
	} 
	else
	{
		const int * dims = CudaNdarray_HOST_DIMS(a);
		return CudaNdarray_DEV_DATA(a) + x * dims[1] * dims[2];
	}
}

float * data_ptr(CudaNdarray * a, int y, int x)
{
	const CudaNdarray * ca = a;
	return const_cast<float *>(data_ptr(ca, y, x));
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

__global__ void tanh_kernel(float * dst, int len)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	while (idx < len)
	{
		dst[idx] = tanhf(dst[idx]);
		idx += gridDim.x * blockDim.x;
	}
}

__global__ void lstm_kernel(float * data, const float * old_state, bool old_state_strided,
 float * output, float * state_out, int n_cells, int n_batch)
{
	//layout: 
	//data[0*n_cells..1*n_cells-1] : input gate
	//data[1*n_cells..2*n_cells-1] : forget gate
	//data[2*n_cells..3*n_cells-1] : output gate
	//data[3*n_cells..4*n_cells-1] : cell state
	//output[0*n_cells..1*n_cells-1]: cell output
	//repeated for every mini-batch

	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	while (idx < n_cells * n_batch)
	{
		int start = (idx/n_cells) * 4 * n_cells + idx % n_cells;

		//input, forget and output gates
		float inpGate = 1.f / (1.f + expf(-data[start]));
		float fgtGate = 1.f / (1.f + expf(-data[start + n_cells]));
		float outGate = 1.f / (1.f + expf(-data[start + 2 * n_cells]));
		float state = inpGate * tanhf(data[start + 3 * n_cells]);
		if(old_state_strided)
		{
		    state += fgtGate * old_state[start];
		}
		else
		{
		    state += fgtGate * old_state[idx];
		}

		//cell output
		output[idx] = outGate * tanhf(state);

		data[start] = inpGate;
		data[start + n_cells] = fgtGate;
		data[start + 2 * n_cells] = outGate;
		data[start + 3 * n_cells] = state;
		if(state_out)
		{
		    state_out[idx] = state;
		}

		idx += gridDim.x * blockDim.x;
	}
}

__global__ void lstm_bwd_kernel(float * delta, float * epsilon, const float * next_epsilon, const float * old_state,
  bool old_state_strided, const float * Y, int n_cells, int n_batch)
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
	while (idx < n_cells * n_batch)
	{
		int batch_idx = idx / n_cells;
		int batch_offset = batch_idx * 4 * n_cells;
		int cell_offset = idx % n_cells;
		int start = batch_offset + cell_offset;

		float inpGate = delta[start];
		float fgtGate = delta[start + n_cells];
		float outGate = delta[start + 2 * n_cells];
		float oldState = old_state_strided ? old_state[start] : old_state[idx];
		float state = delta[start + 3 * n_cells];
		float eps = epsilon[idx];

		//avoid division by 0 (TODO: check if this is needed)
		float gc = 0.f; //g(c(t))		
		float gzc = 0.f; //g(z_c(t))
		if (outGate != 0)
		{
			gc = Y[idx] / outGate;
		}
		if (inpGate != 0)
		{
			gzc = (state - fgtGate * oldState) / inpGate;
		}
		
		//delta_output
		delta[start + 2 * n_cells] = outGate * (1.f - outGate) * gc * eps;
		
		//epsilon_c
		float epsilon_c = (1.f - (gc * gc)) * outGate * eps;
		if (next_epsilon)
		{
			epsilon_c += next_epsilon[idx];
		}
		epsilon[idx] = epsilon_c * fgtGate;

		//delta_cell
		delta[start + 3 * n_cells] = inpGate * (1.f - (gzc * gzc)) * epsilon_c;

		//delta_forget
		delta[start + n_cells] = fgtGate * (1.f - fgtGate) * oldState * epsilon_c;

		//delta_input
		delta[start] = inpGate * (1.f - inpGate) * gzc * epsilon_c;

		idx += gridDim.x * blockDim.x;
	}
}

//input is already tanh
__global__ void mul_with_tanh_deriv_kernel(float * dst, const float * tanhVals, int len)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	while (idx < len)
	{
		//tanh' = 1 - tanh^2
		dst[idx] *= 1.0f - tanhVals[idx] * tanhVals[idx];
		idx += gridDim.x * blockDim.x;
	}
}

void do_tanh(CudaNdarray * a, int y, int x)
{
	float * data_a = data_ptr(a, y, x);
	int size = lastTwoDimsStride(a);
	//TODO tune launch configuration
	tanh_kernel<<<DIM_GRID, DIM_BLOCK>>>(data_a, size);
}

void do_lstm(CudaNdarray * H, CudaNdarray * out, const CudaNdarray * prev, float * state_out, int y, int x)
{		
	assert(y == 0 && "2d LSTM not supported yet");
	int dims[2];
	lastTwoDims(H, dims);	
	assert(dims[1] % 4 == 0); //3 gates + cell
	int n_cells = dims[1] / 4;
	int n_batch = dims[0];
	
	float * data_H = data_ptr(H, y, x);
	const float * data_prev = CudaNdarray_DEV_DATA(prev);
	const float * data_old_state = x > 0 ? data_ptr(H, y, x - 1) + 3 * n_cells : data_prev;
	float * data_out = data_ptr(out, y, x);	
	//TODO tune launch configuration
	lstm_kernel<<<DIM_GRID, DIM_BLOCK>>>(data_H, data_old_state, x > 0, data_out, state_out, n_cells, n_batch);
}

//epsilon are the derivates w.r.t. Z, delta stores the gate and cell activations and will store the derivatives later
//Dd stores the derivative w.r.t. end state
void do_lstm_bwd(CudaNdarray * delta, CudaNdarray * epsilon, const CudaNdarray * Y, const CudaNdarray * Dd,
 const CudaNdarray * c, int y, int x, bool rightBorder)
{
	assert(y == 0 && "2d LSTM not supported yet");
	int dims[2];
	lastTwoDims(delta, dims);	
	assert(dims[1] % 4 == 0); //3 gates + cell
	int n_cells = dims[1] / 4;
	int n_batch = dims[0];

	float * data_delta = data_ptr(delta, y, x);	
	float * data_epsilon = data_ptr(epsilon, y, x);
	const float * data_next_epsilon = rightBorder ? CudaNdarray_DEV_DATA(Dd) : data_ptr(epsilon, y, x + 1);
	const float * data_old_state = x > 0 ? data_ptr(delta, y, x - 1) + 3 * n_cells : CudaNdarray_DEV_DATA(c);
	const float * data_Y = data_ptr(Y, y, x);
	//TODO tune launch configuration
	lstm_bwd_kernel<<<DIM_GRID, DIM_BLOCK>>>(data_delta, data_epsilon, data_next_epsilon,
	 data_old_state, x > 0, data_Y, n_cells, n_batch);
}

void mul_with_tanh_deriv(CudaNdarray * dst, const CudaNdarray * tanhVals, int y, int x)
{
	float * data_dst = data_ptr(dst, y, x);
	const float * data_tanhVals = data_ptr(tanhVals, y, x);
	int size = lastTwoDimsStride(dst);
	//TODO tune launch configuration
	mul_with_tanh_deriv_kernel<<<DIM_GRID, DIM_BLOCK>>>(data_dst, data_tanhVals, size);
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
	//TODO tune launch configuration
	repvec<<<DIM_GRID, DIM_BLOCK>>>(data_b, dims_dst[1], CudaNdarray_SIZE(dst)/dims_dst[1], data_dst);
}

//C[y,x] += A[y,x]*B[y,x]
//(if not 4-dimensional, then indexing [y,x] is ignored (e.g. for weight matrices))
void affine_y_x(int y_A, int x_A, const CudaNdarray * A, int y_B, int x_B, const CudaNdarray * B,
	int y_C, int x_C, CudaNdarray * C, bool transpose_A=false, bool transpose_B=false)
{
	const float * data_A = data_ptr(A, y_A, x_A);	
	const float * data_B = data_ptr(B, y_B, x_B);
	float * data_C = data_ptr(C, y_C, x_C);
	int A_dim[2], B_dim[2];
	lastTwoDims(A, A_dim);
	lastTwoDims(B, B_dim);	
	
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
	HANDLE_ERROR(cublasSgemm(handle, transB, transA, B_dim[1], A_dim[0], A_dim[1], &alpha, data_B, ldB,
		data_A, ldA, &beta, data_C, B_dim[1]));
}

//offset is used for x time-shift between A and B 
//if offset == 1, then we will calculate A[0..end-1] * B[1..end]
void affine_global(const CudaNdarray * A, const CudaNdarray * B, CudaNdarray * C, 
	bool transpose_A=false, bool transpose_B=false, int offset = 0, float beta = 1.0)
{	
	float * data_C = CudaNdarray_DEV_DATA(C);
	int A_dim[2], B_dim[2];
	lastTwoDims(A, A_dim);
	lastTwoDims(B, B_dim);
	int shiftA = A_dim[1] * A_dim[0];
	int shiftB = B_dim[1] * B_dim[0];
	A_dim[0] = CudaNdarray_SIZE(A) / A_dim[1] - offset * A_dim[0];
	B_dim[0] = CudaNdarray_SIZE(B) / B_dim[1] - offset * A_dim[0];
	const float * data_A = CudaNdarray_DEV_DATA(A);
	const float * data_B = CudaNdarray_DEV_DATA(B) + offset * shiftB;

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
