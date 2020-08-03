import theano
import theano.gradient
import theano.printing
import theano.gof
from theano import gof
from theano.sandbox.cuda.basic_ops import (as_cuda_ndarray_variable, gpu_contiguous)
from .Util import raw_variable
from theano.gof.opt import OpSub
from theano.compile import optdb
import theano.tensor as T
from .Util import get_c_support_code_common, get_c_support_code_mdlstm




class MultiDirectionalTwoDLSTMOpGrad(theano.sandbox.cuda.GpuOp):
  __props__ = ("inplace",)

  def __init__(self, inplace):
    super(MultiDirectionalTwoDLSTMOpGrad, self).__init__()
    self.inplace = inplace
    if inplace:
      #inputs: X, 4xW, 4xV_h, 4xV_v, 4xb, 4xDy, 4xY, 4xH

      #note: outputs [17,18,19,20] are the DYs, but we don't use them here for inplace,
      #as they will usually be aligned with each other (atleast when the sum of the outputs is used)

      #all outputs operate inplace on inputs [26,27,28,29] (which are the Hs)
      #but when the input is marked multiple times, we get an error
      #so we do this workaround
      #anyway theano knows that these inputs will be destroyed, so it should be OK
      self.destroy_map = {4: [26], 5: [27], 6: [28], 7: [29]}

  def make_node(self, X, W1, W2, W3, W4, V_h1, V_h2, V_h3, V_h4, V_v1, V_v2, V_v3, V_v4,
                b1, b2, b3, b4, sizes, DY1, DY2, DY3, DY4, Y1, Y2, Y3, Y4, H1, H2, H3, H4):

    var_names = ["X", "W1", "W2", "W3", "W4", "V_h1", "V_h2", "V_h3", "V_h4",
                 "V_v1", "V_v2", "V_v3", "V_v4", "b1", "b2", "b3", "b4",
                 "DY1", "DY2", "DY3", "DY4", "Y1", "Y2", "Y3", "Y4",
                 "H1", "H2", "H3", "H4"]
    lcl = locals()
    for var_name in var_names:
      lcl[var_name] = gpu_contiguous(as_cuda_ndarray_variable(lcl[var_name]))
      assert lcl[var_name].dtype == "float32"
    #note: sizes lives on the CPU!
    sizes = T.as_tensor_variable(sizes)
    assert sizes.dtype == "float32"
    expected_ndims = [4] + ([2] * 12) + ([1] * 4) + ([4] * 12)
    assert len(var_names) == len(expected_ndims), (len(var_names), len(expected_ndims))
    for var_name, expected_ndim in zip(var_names, expected_ndims):
      assert lcl[var_name].ndim == expected_ndim, \
          (var_name, lcl[var_name].name, lcl[var_name].ndim, expected_ndim)
    assert sizes.ndim == 2
    all_vars_no_sizes = [lcl[var_name] for var_name in var_names]
    all_vars = all_vars_no_sizes[:17] + [sizes] + all_vars_no_sizes[17:]
    inputs_vars = all_vars[:17]

    return theano.Apply(self, all_vars, [v.type() for v in inputs_vars])

  def c_support_code(self):
    return get_c_support_code_common() + get_c_support_code_mdlstm()

  def c_code(self, node, name, input_names, output_names, sub):
    X, W1, W2, W3, W4, V_h1, V_h2, V_h3, V_h4, V_v1, V_v2, V_v3, V_v4, \
        b1, b2, b3, b4, sizes, DY1, DY2, DY3, DY4, Y1, Y2, Y3, Y4, H1, H2, H3, H4 = input_names
    DX, DW1, DW2, DW3, DW4, DV_h1, DV_h2, DV_h3, DV_h4, \
        DV_v1, DV_v2, DV_v3, DV_v4, Db1, Db2, Db3, Db4 = output_names
    fail = sub['fail']
    inplace = "true" if self.inplace else "false"
    return """
    //std::cout << "MultiDirectionalTwoDLSTMOpGrad called" << std::endl;
    if(!%(inplace)s)
    {
      std::cout << "warning, inplace optimization failed, not working inplace" << std::endl;
    }

    if(%(DX)s || %(DW1)s || %(DW2)s || %(DW3)s || %(DW4)s ||
       %(DV_h1)s || %(DV_h2)s || %(DV_h3)s || %(DV_h4)s ||
       %(DV_v1)s || %(DV_v2)s || %(DV_v3)s || %(DV_v4)s ||
       %(Db1)s || %(Db2)s || %(Db3)s || %(Db4)s)
    {
      cout << "output storage already exists" << endl;
      //TODO check if we can reuse it
      Py_XDECREF(%(DX)s);
      Py_XDECREF(%(DW1)s);
      Py_XDECREF(%(DW2)s);
      Py_XDECREF(%(DW3)s);
      Py_XDECREF(%(DW4)s);
      Py_XDECREF(%(DV_h1)s);
      Py_XDECREF(%(DV_h2)s);
      Py_XDECREF(%(DV_h3)s);
      Py_XDECREF(%(DV_h4)s);
      Py_XDECREF(%(DV_v1)s);
      Py_XDECREF(%(DV_v2)s);
      Py_XDECREF(%(DV_v3)s);
      Py_XDECREF(%(DV_v4)s);
      Py_XDECREF(%(Db1)s);
      Py_XDECREF(%(Db2)s);
      Py_XDECREF(%(Db3)s);
      Py_XDECREF(%(Db4)s);
    }

    const int * X_dim = CudaNdarray_HOST_DIMS(%(X)s);
    const int * Y_dim = CudaNdarray_HOST_DIMS(%(Y1)s);
    const int height = X_dim[0];
    const int width = X_dim[1];
    const int n_minibatch = X_dim[2];
    const int n_diags = width + height - 1;
    const int max_diag_size = std::min(Y_dim[0], Y_dim[1]);

    CudaNdarray * delta1 = 0;
    CudaNdarray * delta2 = 0;
    CudaNdarray * delta3 = 0;
    CudaNdarray * delta4 = 0;
    if(%(inplace)s)
    {
      delta1 = %(H1)s;
      delta2 = %(H2)s;
      delta3 = %(H3)s;
      delta4 = %(H4)s;
    }
    else
    {
      delta1 = (CudaNdarray *) CudaNdarray_Copy(%(H1)s);
      delta2 = (CudaNdarray *) CudaNdarray_Copy(%(H2)s);
      delta3 = (CudaNdarray *) CudaNdarray_Copy(%(H3)s);
      delta4 = (CudaNdarray *) CudaNdarray_Copy(%(H4)s);
    }
    CudaNdarray * epsilon1 = (CudaNdarray *) CudaNdarray_Copy(%(DY1)s);
    CudaNdarray * epsilon2 = (CudaNdarray *) CudaNdarray_Copy(%(DY2)s);
    CudaNdarray * epsilon3 = (CudaNdarray *) CudaNdarray_Copy(%(DY3)s);
    CudaNdarray * epsilon4 = (CudaNdarray *) CudaNdarray_Copy(%(DY4)s);

    const int workmem1_dims[] = {2, Y_dim[0], Y_dim[1], Y_dim[2], Y_dim[3]};
    CudaNdarray * workmem1_1 = (CudaNdarray*) MyCudaNdarray_NewDims(5, workmem1_dims);
    assert(workmem1_1);
    CudaNdarray * workmem1_2 = (CudaNdarray*) MyCudaNdarray_NewDims(5, workmem1_dims);
    assert(workmem1_2);
    CudaNdarray * workmem1_3 = (CudaNdarray*) MyCudaNdarray_NewDims(5, workmem1_dims);
    assert(workmem1_3);
    CudaNdarray * workmem1_4 = (CudaNdarray*) MyCudaNdarray_NewDims(5, workmem1_dims);
    assert(workmem1_4);

    //we use floats to store float*'s, as CudaNdarray only supports floats. factor 10 for lstm bwd kernel
    int ptr_storage_dims[] = {4 * 10 * max_diag_size * sizeof(float*) / sizeof(float)};
    CudaNdarray * ptr_storage = (CudaNdarray*) MyCudaNdarray_NewDims(1, ptr_storage_dims);
    assert(ptr_storage);

    //valid: float tensor of 1s and 0s indicating the size of the image
    //4 dirs * max_diag_size * n_minibatch
    int valid_dims[] = {4 * max_diag_size * n_minibatch};
    CudaNdarray * valid_storage = (CudaNdarray*) MyCudaNdarray_NewDims(1, valid_dims);
    assert(valid_storage);

    for(int diag = n_diags-1; diag >= 0; --diag)
    {
      int diag_size = min(diag+1, min(abs(n_diags-diag), min(width, height)));
      int y_high = min(diag, height-1);
      int x_low = max(diag-height+1,0);
      vector<int> ys_h, xs_h, ys_v, xs_v, ys, xs;
      for(int idx = 0; idx < diag_size; ++idx)
      {
        int y = y_high - idx;
        int x = x_low + idx;
        bool rightBorder = (x == X_dim[1]-1);
        if(!rightBorder)
        {
          ys_h.push_back(y);
          xs_h.push_back(x);
        }
        bool botBorder = (y == X_dim[0]-1);
        if(!botBorder)
        {
          ys_v.push_back(y);
          xs_v.push_back(x);
        }
        ys.push_back(y);
        xs.push_back(x);
      }

      affine_y_x_batched_multidir(0, 1, delta1, delta2, delta3, delta4, %(V_h1)s, %(V_h2)s, %(V_h3)s, %(V_h4)s,
        epsilon1, epsilon2, epsilon3, epsilon4, ys_h, xs_h, ptr_storage, height, width, 0, false, true);
      affine_y_x_batched_multidir(1, 0, delta1, delta2, delta3, delta4, %(V_v1)s, %(V_v2)s, %(V_v3)s, %(V_v4)s,
        epsilon1, epsilon2, epsilon3, epsilon4, ys_v, xs_v, ptr_storage, height, width, 0, false, true);

      do_lstm_bwd_batched_multidir(delta1, delta2, delta3, delta4, epsilon1, epsilon2, epsilon3, epsilon4,
        %(Y1)s, %(Y2)s, %(Y3)s, %(Y4)s, workmem1_1, workmem1_2, workmem1_3, workmem1_4,
        X_dim[0], X_dim[1], ys, xs, ptr_storage, valid_storage, %(sizes)s);
    }
    Py_XDECREF(ptr_storage);
    Py_XDECREF(valid_storage);

    Py_XDECREF(workmem1_1);
    Py_XDECREF(workmem1_2);
    Py_XDECREF(workmem1_3);
    Py_XDECREF(workmem1_4);

    %(DX)s = CudaNdarray_uninitialized_like(%(X)s);
    assert(%(DX)s);
    %(DW1)s = CudaNdarray_uninitialized_like(%(W1)s);
    assert(%(DW1)s);
    %(DW2)s = CudaNdarray_uninitialized_like(%(W2)s);
    assert(%(DW2)s);
    %(DW3)s = CudaNdarray_uninitialized_like(%(W3)s);
    assert(%(DW3)s);
    %(DW4)s = CudaNdarray_uninitialized_like(%(W4)s);
    assert(%(DW4)s);

    //DW = X^T * delta
    affine_global(%(X)s, delta1, %(DW1)s, true, false, 0, 0, 0.0f);
    affine_global(%(X)s, delta2, %(DW2)s, true, false, 0, 0, 0.0f);
    affine_global(%(X)s, delta3, %(DW3)s, true, false, 0, 0, 0.0f);
    affine_global(%(X)s, delta4, %(DW4)s, true, false, 0, 0, 0.0f);
    //important! mind the order, first use X, then update DX, which might be aligned to X
    //DX = delta * W^T
    affine_global(delta1, %(W1)s, %(DX)s, false, true, 0, 0, 0.0f);
    affine_global(delta2, %(W2)s, %(DX)s, false, true, 0, 0, 1.0f);
    affine_global(delta3, %(W3)s, %(DX)s, false, true, 0, 0, 1.0f);
    affine_global(delta4, %(W4)s, %(DX)s, false, true, 0, 0, 1.0f);
    //Db = (1 ... 1) * delta
    %(Db1)s = sumOverAllButLastDimensions(delta1);
    %(Db2)s = sumOverAllButLastDimensions(delta2);
    %(Db3)s = sumOverAllButLastDimensions(delta3);
    %(Db4)s = sumOverAllButLastDimensions(delta4);

    //copy left/right part to workmem2 and set to 0 (could be done more efficient, but profiling shows, it's not worth it)
    const int * H_dim = CudaNdarray_HOST_DIMS(%(H1)s);
    const int workmem2_dims[] = {H_dim[0], H_dim[2], H_dim[3]};
    const int block_size = H_dim[2] * H_dim[3];
    CudaNdarray * workmem2_1 = (CudaNdarray*) MyCudaNdarray_NewDims(3, workmem2_dims);
    assert(workmem2_1);
    CudaNdarray * workmem2_2 = (CudaNdarray*) MyCudaNdarray_NewDims(3, workmem2_dims);
    assert(workmem2_2);
    CudaNdarray * workmem2_3 = (CudaNdarray*) MyCudaNdarray_NewDims(3, workmem2_dims);
    assert(workmem2_3);
    CudaNdarray * workmem2_4 = (CudaNdarray*) MyCudaNdarray_NewDims(3, workmem2_dims);
    assert(workmem2_4);
    for(int y = 0; y < Y_dim[0]; ++y)
    {
      float * workmem2_1_data_ptr = CudaNdarray_DEV_DATA(workmem2_1) + y * block_size;
      float * delta1_data_ptr = data_ptr(delta1, y, 0);
      HANDLE_ERROR(cudaMemcpy(workmem2_1_data_ptr, delta1_data_ptr, block_size * sizeof(float), cudaMemcpyDeviceToDevice));
      HANDLE_ERROR(cudaMemset(delta1_data_ptr, 0, sizeof(float) * H_dim[2] * H_dim[3]));

      float * workmem2_2_data_ptr = CudaNdarray_DEV_DATA(workmem2_2) + y * block_size;
      float * delta2_data_ptr = data_ptr(delta2, y, 0);
      HANDLE_ERROR(cudaMemcpy(workmem2_2_data_ptr, delta2_data_ptr, block_size * sizeof(float), cudaMemcpyDeviceToDevice));
      HANDLE_ERROR(cudaMemset(delta2_data_ptr, 0, sizeof(float) * H_dim[2] * H_dim[3]));

      float * workmem2_3_data_ptr = CudaNdarray_DEV_DATA(workmem2_3) + y * block_size;
      float * delta3_data_ptr = data_ptr(delta3, y, width - 1);
      HANDLE_ERROR(cudaMemcpy(workmem2_3_data_ptr, delta3_data_ptr, block_size * sizeof(float), cudaMemcpyDeviceToDevice));
      HANDLE_ERROR(cudaMemset(delta3_data_ptr, 0, sizeof(float) * H_dim[2] * H_dim[3]));

      float * workmem2_4_data_ptr = CudaNdarray_DEV_DATA(workmem2_4) + y * block_size;
      float * delta4_data_ptr = data_ptr(delta4, y, width - 1);
      HANDLE_ERROR(cudaMemcpy(workmem2_4_data_ptr, delta4_data_ptr, block_size * sizeof(float), cudaMemcpyDeviceToDevice));
      HANDLE_ERROR(cudaMemset(delta4_data_ptr, 0, sizeof(float) * H_dim[2] * H_dim[3]));
    }

    %(DV_h1)s = CudaNdarray_uninitialized_like(%(V_h1)s);
    assert(%(DV_h1)s);
    %(DV_h2)s = CudaNdarray_uninitialized_like(%(V_h2)s);
    assert(%(DV_h2)s);
    %(DV_h3)s = CudaNdarray_uninitialized_like(%(V_h3)s);
    assert(%(DV_h3)s);
    %(DV_h4)s = CudaNdarray_uninitialized_like(%(V_h4)s);
    assert(%(DV_h4)s);
    //DV_h = Y[0..end-1]^T * delta[1..end]
    affine_global(%(Y1)s, delta1, %(DV_h1)s, true, false, 0, 1, 0.0f);
    affine_global(%(Y2)s, delta2, %(DV_h2)s, true, false, 0, 1, 0.0f);
    affine_global(%(Y3)s, delta3, %(DV_h3)s, true, false, 1, 0, 0.0f);
    affine_global(%(Y4)s, delta4, %(DV_h4)s, true, false, 1, 0, 0.0f);

    //copy left/right part back
    for(int y = 0; y < Y_dim[0]; ++y)
    {
      float * workmem2_1_data_ptr = CudaNdarray_DEV_DATA(workmem2_1) + y * block_size;
      float * delta1_data_ptr = data_ptr(delta1, y, 0);
      HANDLE_ERROR(cudaMemcpy(delta1_data_ptr, workmem2_1_data_ptr, block_size * sizeof(float), cudaMemcpyDeviceToDevice));

      float * workmem2_2_data_ptr = CudaNdarray_DEV_DATA(workmem2_2) + y * block_size;
      float * delta2_data_ptr = data_ptr(delta2, y, 0);
      HANDLE_ERROR(cudaMemcpy(delta2_data_ptr, workmem2_2_data_ptr, block_size * sizeof(float), cudaMemcpyDeviceToDevice));

      float * workmem2_3_data_ptr = CudaNdarray_DEV_DATA(workmem2_3) + y * block_size;
      float * delta3_data_ptr = data_ptr(delta3, y, width - 1);
      HANDLE_ERROR(cudaMemcpy(delta3_data_ptr, workmem2_3_data_ptr, block_size * sizeof(float), cudaMemcpyDeviceToDevice));

      float * workmem2_4_data_ptr = CudaNdarray_DEV_DATA(workmem2_4) + y * block_size;
      float * delta4_data_ptr = data_ptr(delta4, y, width - 1);
      HANDLE_ERROR(cudaMemcpy(delta4_data_ptr, workmem2_4_data_ptr, block_size * sizeof(float), cudaMemcpyDeviceToDevice));
    }

    Py_XDECREF(workmem2_1);
    Py_XDECREF(workmem2_2);
    Py_XDECREF(workmem2_3);
    Py_XDECREF(workmem2_4);

    %(DV_v1)s = CudaNdarray_uninitialized_like(%(V_v1)s);
    assert(%(DV_v1)s);
    %(DV_v2)s = CudaNdarray_uninitialized_like(%(V_v2)s);
    assert(%(DV_v2)s);
    %(DV_v3)s = CudaNdarray_uninitialized_like(%(V_v3)s);
    assert(%(DV_v3)s);
    %(DV_v4)s = CudaNdarray_uninitialized_like(%(V_v4)s);
    assert(%(DV_v4)s);

    //DV_v = Y[0..end-1]^T * delta[1..end]
    affine_global(%(Y1)s, delta1, %(DV_v1)s, true, false, 0, Y_dim[1], 0.0f);
    affine_global(%(Y2)s, delta2, %(DV_v2)s, true, false, Y_dim[1], 0, 0.0f);
    affine_global(%(Y3)s, delta3, %(DV_v3)s, true, false, 0, Y_dim[1], 0.0f);
    affine_global(%(Y4)s, delta4, %(DV_v4)s, true, false, Y_dim[1], 0, 0.0f);

    //for debugging
    /*cout << "=====delta1====" << endl;
    CudaNdarray_print_part(delta1);
    cout << "=====delta2====" << endl;
    CudaNdarray_print_part(delta2);
    cout << "=====delta3====" << endl;
    CudaNdarray_print_part(delta3);
    cout << "=====delta4====" << endl;
    CudaNdarray_print_part(delta4);
    cout << "===============" << endl;*/

    if(!%(inplace)s)
    {
      Py_XDECREF(delta1);
      Py_XDECREF(delta2);
      Py_XDECREF(delta3);
      Py_XDECREF(delta4);
    }
    Py_XDECREF(epsilon1);
    Py_XDECREF(epsilon2);
    Py_XDECREF(epsilon3);
    Py_XDECREF(epsilon4);

    """ % locals()

  #!!! change this when changing the code!
  def c_code_cache_version(self):
    return 2, 10

MultiDirectionalTwoDLSTMOpGradNoInplaceInstance = MultiDirectionalTwoDLSTMOpGrad(inplace=False)
MultiDirectionalTwoDLSTMOpGradInplaceInstance = MultiDirectionalTwoDLSTMOpGrad(inplace=True)

MultiDirectionalTwoDLSTMOpInplaceOpt = OpSub(MultiDirectionalTwoDLSTMOpGradNoInplaceInstance,
                                             MultiDirectionalTwoDLSTMOpGradInplaceInstance)

#hack to avoid being called twice
if not hasattr(optdb, 'MultiDirectionalTwoDLSTMOpInplaceOpt_registered'):
  optdb.register('MultiDirectionalTwoDLSTMOpInplaceOpt',
                 theano.gof.TopoOptimizer(MultiDirectionalTwoDLSTMOpInplaceOpt, failure_callback=gof.TopoOptimizer.warn_inplace),
                 50.0, 'fast_run', 'inplace', 'gpuarray')
  optdb.MultiDirectionalTwoDLSTMOpInplaceOpt_registered = True


class MultiDirectionalTwoDLSTMOp(theano.sandbox.cuda.GpuOp):
  __props__ = ()

  def __init__(self):
    super(MultiDirectionalTwoDLSTMOp, self).__init__()

  def make_node(self, X, W1, W2, W3, W4, V_h1, V_h2, V_h3, V_h4, V_v1, V_v2, V_v3, V_v4, b1, b2, b3, b4, sizes):
    var_names = ["X", "W1", "W2", "W3", "W4", "V_h1", "V_h2", "V_h3", "V_h4",
                 "V_v1", "V_v2", "V_v3", "V_v4", "b1", "b2", "b3", "b4"]
    lcl = locals()
    for var_name in var_names:
      lcl[var_name] = gpu_contiguous(as_cuda_ndarray_variable(lcl[var_name]))
      assert lcl[var_name].dtype == "float32"
    #note: sizes lives on the CPU!
    sizes = T.as_tensor_variable(sizes)
    assert sizes.dtype == "float32"

    assert lcl["X"].ndim == 4
    assert lcl["W1"].ndim == 2
    assert lcl["W2"].ndim == 2
    assert lcl["W3"].ndim == 2
    assert lcl["W4"].ndim == 2
    assert lcl["V_h1"].ndim == 2
    assert lcl["V_h2"].ndim == 2
    assert lcl["V_h3"].ndim == 2
    assert lcl["V_h4"].ndim == 2
    assert lcl["V_v1"].ndim == 2
    assert lcl["V_v2"].ndim == 2
    assert lcl["V_v3"].ndim == 2
    assert lcl["V_v4"].ndim == 2
    assert lcl["b1"].ndim == 1
    assert lcl["b2"].ndim == 1
    assert lcl["b3"].ndim == 1
    assert lcl["b4"].ndim == 1
    assert sizes.ndim == 2

    all_vars = [lcl[var_name] for var_name in var_names] + [sizes]

    #results: outputs Y1, Y2, Y3, Y4, (gates and cell states) H1, H2, H3, H4
    return theano.Apply(self, all_vars, [lcl["X"].type() for _ in range(8)])

  def c_support_code(self):
    return get_c_support_code_common() + get_c_support_code_mdlstm()

  def c_code(self, node, name, input_names, output_names, sub):
    X, W1, W2, W3, W4, V_h1, V_h2, V_h3, V_h4, V_v1, V_v2, V_v3, V_v4, b1, b2, b3, b4, sizes = input_names
    Y1, Y2, Y3, Y4, H1, H2, H3, H4 = output_names
    fail = sub['fail']
    return """
    //std::cout << "MultiDirectionalTwoDLSTMOp called" << std::endl;
    if(%(Y1)s || %(Y2)s || %(Y3)s || %(Y4)s || %(H1)s || %(H2)s || %(H3)s || %(H4)s)
    {
      //cout << "Ys or Hs already exist" << endl;
      //TODO check if we can reuse it
      Py_XDECREF(%(Y1)s);
      Py_XDECREF(%(Y2)s);
      Py_XDECREF(%(Y3)s);
      Py_XDECREF(%(Y4)s);
      Py_XDECREF(%(H1)s);
      Py_XDECREF(%(H2)s);
      Py_XDECREF(%(H3)s);
      Py_XDECREF(%(H4)s);
    }

    const int * X_dim = CudaNdarray_HOST_DIMS(%(X)s);
    const int * W_dim = CudaNdarray_HOST_DIMS(%(W1)s);
    const int * V_dim = CudaNdarray_HOST_DIMS(%(V_h1)s);
    assert(W_dim[1] %% 5 == 0 && "W has wrong shape");
    assert(5 * V_dim[0] == V_dim[1] && "V has wrong shape");
    assert(W_dim[1] == V_dim[1]);
    assert(W_dim[0] == X_dim[3]);
    const int Y_dim[] = {X_dim[0], X_dim[1], X_dim[2], W_dim[1] / 5};
    const int H_dim[] = {X_dim[0], X_dim[1], X_dim[2], W_dim[1]};
    const int height = X_dim[0];
    const int width = X_dim[1];
    const int n_minibatch = X_dim[2];
    const int max_diag_size = std::min(height, width);
    const int n_diags = width + height - 1;

    //init Ys
    %(Y1)s = (CudaNdarray*) MyCudaNdarray_NewDims(4, Y_dim);
    assert(%(Y1)s);
    %(Y2)s = (CudaNdarray*) MyCudaNdarray_NewDims(4, Y_dim);
    assert(%(Y2)s);
    %(Y3)s = (CudaNdarray*) MyCudaNdarray_NewDims(4, Y_dim);
    assert(%(Y3)s);
    %(Y4)s = (CudaNdarray*) MyCudaNdarray_NewDims(4, Y_dim);
    assert(%(Y4)s);

    //init Hs
    %(H1)s = (CudaNdarray*) MyCudaNdarray_NewDims(4, H_dim);
    assert(%(H1)s);
    %(H2)s = (CudaNdarray*) MyCudaNdarray_NewDims(4, H_dim);
    assert(%(H2)s);
    %(H3)s = (CudaNdarray*) MyCudaNdarray_NewDims(4, H_dim);
    assert(%(H3)s);
    %(H4)s = (CudaNdarray*) MyCudaNdarray_NewDims(4, H_dim);
    assert(%(H4)s);

    //init Hs with bs
    fillmat(%(b1)s, %(H1)s);
    fillmat(%(b2)s, %(H2)s);
    fillmat(%(b3)s, %(H3)s);
    fillmat(%(b4)s, %(H4)s);

    //H+=XW
    affine_global(%(X)s, %(W1)s, %(H1)s);
    affine_global(%(X)s, %(W2)s, %(H2)s);
    affine_global(%(X)s, %(W3)s, %(H3)s);
    affine_global(%(X)s, %(W4)s, %(H4)s);

    //we use floats to store float*'s, as CudaNdarray only supports floats. factor 5 for lstm kernel,
    //additional factor 4 for 4 directions
    int ptr_storage_dims[] = {4 * 5 * max_diag_size * sizeof(float*) / sizeof(float)};
    CudaNdarray * ptr_storage = (CudaNdarray*) MyCudaNdarray_NewDims(1, ptr_storage_dims);
    assert(ptr_storage);

    //valid: float tensor of 1s and 0s indicating the size of the image
    //4 dirs * max_diag_size * n_minibatch
    int valid_dims[] = {4 * max_diag_size * n_minibatch};
    CudaNdarray * valid_storage = (CudaNdarray*) MyCudaNdarray_NewDims(1, valid_dims);
    assert(valid_storage);

    for(int diag = 0; diag < n_diags; ++diag)
    {
      int diag_size = min(diag+1, min(abs(n_diags-diag), min(width, height)));
      int y_high = min(diag, height-1);
      int x_low = max(diag-height+1,0);
      vector<int> ys_h, xs_h, ys_v, xs_v, ys, xs;
      for(int idx = 0; idx < diag_size; ++idx)
      {
        int y = y_high - idx;
        int x = x_low + idx;
        if(x > 0)
        {
          ys_h.push_back(y);
          xs_h.push_back(x);
        }
        if(y > 0)
        {
          ys_v.push_back(y);
          xs_v.push_back(x);
        }
        ys.push_back(y);
        xs.push_back(x);
      }

      affine_y_x_batched_multidir(0, -1,
        %(Y1)s, %(Y2)s, %(Y3)s, %(Y4)s,
        %(V_h1)s, %(V_h2)s, %(V_h3)s, %(V_h4)s,
        %(H1)s, %(H2)s, %(H3)s, %(H4)s,
        ys_h, xs_h, ptr_storage, height, width);
      affine_y_x_batched_multidir(-1, 0,
        %(Y1)s, %(Y2)s, %(Y3)s, %(Y4)s,
        %(V_v1)s, %(V_v2)s, %(V_v3)s, %(V_v4)s,
        %(H1)s, %(H2)s, %(H3)s, %(H4)s,
        ys_v, xs_v, ptr_storage, height, width);

      do_lstm_batched_multidir(%(H1)s, %(H2)s, %(H3)s, %(H4)s,
                                 %(Y1)s, %(Y2)s, %(Y3)s, %(Y4)s,
                                 ys, xs, ptr_storage, valid_storage, %(sizes)s);
    }
    Py_XDECREF(ptr_storage);
    Py_XDECREF(valid_storage);

    """ % locals()

  def grad(self, inputs, output_grads):
    raw_inputs = [raw_variable(v) for v in inputs]
    fwd_results = self(*raw_inputs)
    args = inputs + output_grads[:4] + fwd_results
    grads = MultiDirectionalTwoDLSTMOpGradNoInplaceInstance(*args)
    Dsizes = theano.gradient.grad_undefined(self, len(inputs) - 1, inputs[-1], 'cannot diff w.r.t. sizes')
    return grads + [Dsizes]

  # noinspection PyMethodMayBeStatic
  def infer_shape(self, node, input_shapes):
    Xs, W1s = input_shapes[:2]
    Y_shape = (Xs[0], Xs[1], Xs[2], W1s[1] // 5)
    H_shape = (Xs[0], Xs[1], Xs[2], W1s[1])
    return [Y_shape, Y_shape, Y_shape, Y_shape, H_shape, H_shape, H_shape, H_shape]

  #!!! change this when changing the code!
  def c_code_cache_version(self):
    return 2, 10

MultiDirectionalTwoDLSTMOpInstance = MultiDirectionalTwoDLSTMOp()
