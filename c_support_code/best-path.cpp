

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

DEF_KERNEL
void subtensor_idxs_kernel(float* in, long in_idx_stride, long out_idx_stride, long max_idx, long n, float* in_idxs, float* out) {
	long idx = threadIdx.x + blockDim.x * blockIdx.x;
	while(idx < max_idx) {
		long in_start = idx * in_idx_stride;
		long out_start = idx * out_idx_stride;
		for(long i = 0; i < n; ++i) {
			long in_idx = (long) in_idxs[out_start + i]; // in_idxs like out
			float in_v = in[out_start + in_idx];
			out[out_start + i] = in_v;
		}
		idx += gridDim.x * blockDim.x;
	}
}

DEF_KERNEL
void subtensor_idxs_incr_kernel(float* in, long in_idx_stride, long out_idx_stride, long max_idx, long n, float* in_idxs, float* out) {
	long idx = threadIdx.x + blockDim.x * blockIdx.x;
	while(idx < max_idx) {
		long in_start = idx * in_idx_stride;
		long out_start = idx * out_idx_stride;
		for(long i = 0; i < n; ++i) {
			long in_idx = (long) in_idxs[out_start + i]; // in_idxs like out
			in[out_start + in_idx] += out[out_start + i];
		}
		idx += gridDim.x * blockDim.x;
	}
}

void get_n_max_and_argmax(Ndarray* v, int n, Ndarray* nbest, Ndarray* nbestidx) {
	assert(Ndarray_NDIM(v) == 3);
	long T = Ndarray_DIMS(v)[0];
	long n_batch = Ndarray_DIMS(v)[1];
	long n_dim = Ndarray_DIMS(v)[2];
	assert(Ndarray_NDIM(nbest) == 3);
	assert(Ndarray_NDIM(nbestidx) == 3);
	assert(Ndarray_DIMS(nbest)[0] == T);
	assert(Ndarray_DIMS(nbest)[1] == n_batch);
	assert(Ndarray_DIMS(nbest)[2] == n);
	assert(Ndarray_DIMS(nbestidx)[0] == T);
	assert(Ndarray_DIMS(nbestidx)[1] == n_batch);
	assert(Ndarray_DIMS(nbestidx)[2] == n);

	start_dev_kernel(find_best_n_kernel, (
		/*in*/ Ndarray_DEV_DATA(v),
		/*in_idx_stride*/ n_dim,
		/*out_idx_stride*/ n,
		/*max_idx*/ T * n_batch,
		/*n*/ n,
		/*out*/ Ndarray_DEV_DATA(nbestidx)
	));

	start_dev_kernel(subtensor_idxs_kernel, (
		/*in*/ Ndarray_DEV_DATA(v),
		/*in_idx_stride*/ n_dim,
		/*out_idx_stride*/ n,
		/*max_idx*/ T * n_batch,
		/*n*/ n,
		/*in_idxs*/ Ndarray_DEV_DATA(nbestidx),
		/*out*/ Ndarray_DEV_DATA(nbest)
	));
}

void grad_n_max_and_argmax(Ndarray* out, int n, Ndarray* nbestidx, Ndarray* errsignal) {
	assert(Ndarray_NDIM(out) == 3);
	long T = Ndarray_DIMS(out)[0];
	long n_batch = Ndarray_DIMS(out)[1];
	long n_dim = Ndarray_DIMS(out)[2];
	assert(Ndarray_NDIM(errsignal) == 3);
	assert(Ndarray_NDIM(nbestidx) == 3);
	assert(Ndarray_DIMS(errsignal)[0] == T);
	assert(Ndarray_DIMS(errsignal)[1] == n_batch);
	assert(Ndarray_DIMS(errsignal)[2] == n);
	assert(Ndarray_DIMS(nbestidx)[0] == T);
	assert(Ndarray_DIMS(nbestidx)[1] == n_batch);
	assert(Ndarray_DIMS(nbestidx)[2] == n);

	start_dev_kernel(subtensor_idxs_incr_kernel, (
		/*in*/ Ndarray_DEV_DATA(out),
		/*in_idx_stride*/ n_dim,
		/*out_idx_stride*/ n,
		/*max_idx*/ T * n_batch,
		/*n*/ n,
		/*in_idxs*/ Ndarray_DEV_DATA(nbestidx),
		/*out*/ Ndarray_DEV_DATA(errsignal)
	));
}

DEF_KERNEL
void find_best_path_kernel(float* in, long in_idx_stride, long out_idx_stride, long T, float* out) {
	for(long t = 0; t < T; ++t) {

	}
}

DEF_KERNEL
void levenshtein_kernel(long n_batch,
						float* str1, float* str1index, long str1len,
						float* str2, float* str2index, long str2len,
						float* row0, float* row1) {
	// rows are of length str2len + 1.
	float* original_row1 = row1;

	for(
		long batch_idx = threadIdx.x + blockDim.x * blockIdx.x;
		batch_idx < n_batch;
		batch_idx += gridDim.x * blockDim.x)
	{
		// initial
		for(long j = 0; j <= str2len; ++j)
			row1[j * n_batch + batch_idx] = (float) j;

		// row1 was the last row
		for(long i = 0; i < str1len; ++i) {
			long idx1 = i * n_batch + batch_idx;
			if(str1index[idx1] < 0.1)
				break;

			{ float* tmp = row0; row0 = v1; row1 = tmp; }  // swap row0/row1

			// row1 is the current row, row0 the last
			float str1c = str1[idx1];
			row1[batch_idx] = float(i + 1); // delete (i+1) chars from str1

			for(long j = 0; j < str2len; ++j) {
				long idx2 = j * n_batch + batch_idx;
				if(str2index[idx2] < 0.1) {
					row1[idx2 + n_batch] = row1[idx2];
					continue;
				}
				float cost = (fabs(str1c - str2[idx2]) < 0.1) ? 0 : 1;
				float v = row1[idx2] + 1; // insertion
				float v2 = row0[idx2 + n_batch] + 1; // deletion
				float v3 = row0[idx2] + cost; // substitution
				if(v2 < v) v = v2;
				if(v3 < v) v = v3;
				row1[idx2 + n_batch] = v;
			}
		}
		// row1 was the last row

		if(original_row1 != row1) {
			for(long j = 0; j <= str2len; ++j)
				original_row1[j * n_batch + batch_idx] = row1[j * n_batch + batch_idx];  // we need row1
			row0 = row1;
			row1 = original_row1;
		}
	}
}

void levenshtein(long n_batch,
				 float* str1, float* str1index, long str1len,
				 float* str2, float* str2index, long str2len,
				 float* row0, float* row1) {
	/*
	formally:
	 lev_{a,b}(i,j) =
		max(i,j)  if i == 0 or j == 0, else
		min(lev_{a,b}(i - 1, j) + 1,
			lev_{a,b}(i, j - 1) + 1,
			lev_{a,b}(i - 1, j - 1) + 1_(a_i != b_j))
	If str1/str2 were single strings, we would have:
	 levenshtein(str1, str2) = lev_{str1,str2}(str1len, str2len)
	In this implementation, we support multiple batches.
	 */
	Ndarray_memset(row0, 0, (str2len + 1) * n_batch * sizeof(float)); // previous row
	Ndarray_memset(row1, 0, (str2len + 1) * n_batch * sizeof(float)); // current row

	start_dev_kernel(levenshtein_kernel, (
		n_batch,
		str1, str1index, str1len,
		str2, str2index, str2len,
		row0, row1
	));
}

void levenshtein_loss_estimation(Ndarray* posteriors, Ndarray* loss, Ndarray* error_signal) {

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

	// note: we don't do a real search. we just take the local max in each time-frame.


	// now: which are the possible allophone states regarding these n-best labels?
	// what are possible sequences?
	// what is the best sequence?

}


#ifdef STANDALONE
int main() {

}
#endif
