

#ifdef STANDALONE
#define CUDA 0
#include "../NativeOp.cpp"
#endif

DEF_KERNEL
void find_best_n_kernel(float* in, long in_idx_stride, long out_idx_stride, long max_idx, long n, float* out) {
	long idx = threadIdx.x + blockDim.x * blockIdx.x;
	while(idx < max_idx) {
		long in_start = idx * in_idx_stride;
		long out_start = idx * out_idx_stride;
		float out_min_v = in[in_start];
		long out_min_i = 0;
		// the first n elements will just be our initial n-best list
		for(long i = 0; i < n; ++i) {
			out[out_start + i] = (float) i;
			long in_idx = in_start + i;
			float in_v = in[in_idx];
			if(in_v < out_min_v) {
				out_min_v = in_v;
				out_min_i = i;
			}
		}
		for(long i = n; i < in_idx_stride; ++i) {
			long in_idx = in_start + i;
			float in_v = in[in_idx];
			if(in_v > out_min_v) {
				// we found a new value to fill into the n-best list
				out[out_start + out_min_i] = (float) i;
				// search for new lowest value
				for(long j = 0; j < n; ++j) {
					long out_idx = out_start + j;
					long out_v = out[out_idx];
					if(out_v < out_min_v) {
						out_min_v = out_v;
						out_min_i = j;
					}
				}
			}
		}
		idx += gridDim.x * blockDim.x;
	}
}

void make_best_path(Ndarray* posteriors, int local_n_best_num) {
	assert(Ndarray_NDIM(posteriors) == 3);
	long T = Ndarray_DIMS(posteriors)[0];
	long n_batch = Ndarray_DIMS(posteriors)[1];
	long n_dim = Ndarray_DIMS(posteriors)[2];
	
	Ndarray_DIM_Type local_n_best_dims[] = {T, n_batch, local_n_best_num};
	Ndarray* local_n_best = (Ndarray*) Ndarray_NewDims(/*ndim*/3, local_n_best_dims);
	start_dev_kernel(find_best_n_kernel, (
		/*in*/ Ndarray_DEV_DATA(posteriors),
		/*in_idx_stride*/ n_dim,
		/*out_idx_stride*/ local_n_best_num,
		/*max_idx*/ T * n_batch,
		/*n*/ local_n_best_num,
		/*out*/ Ndarray_DEV_DATA(local_n_best)
	));

	// now: which are the possible allophone states regarding these n-best labels?
	// what are possible sequences?
	// what is the best sequence?

}


#ifdef STANDALONE
int main() {

}
#endif
