"""
Generic interface which automatically creates:

* CPU and GPU (CUDA) op
* inplace and not inplace
* grad variants

See :mod:`returnn.tf.native_op` and :mod:`returnn.theano.native_op`
for usage in TensorFlow and Theano.

See :ref:`native_ops` for more background.
"""

import copy
import numpy
import typing
from returnn.util.basic import make_hashable, unicode


class NativeOpBaseMixin(object):
    """
    The purpose of having this as a separate base class is to make this independent of any Theano specific
    functionality so that we can also use this base for example for TensorFlow.
    """

    def __init__(
        self,
        in_info,
        out_info,
        c_fw_code,
        c_bw_code=None,
        c_extra_support_code=None,
        code_version=None,
        cpu_support=True,
        grad_input_map=None,
        name=None,
    ):
        """
        :param list[dict(str)] in_info: each dict describes one input var.
          attribs in the dict:
            int ndim: the ndim.
            tuple shape: tuple and can contain None for specific dimensions.
          optional attribs:
            str dtype: "float32" by default.
            bool need_contiguous: false by default.
            int want_inplace: -1 by default. try to optimize to destroy input, on output-index.
              "dummy_out" is a special value which will add another output.
            bool is_inplace: false by default. whether the optimization was applied.
            str gradient: can be "disconnected". see grad().
            bool bw_input: True by default. add this param to the bw input.
          other attribs are just ignored.
        :param list[dict(str)] out_info: like in_info.
          slightly different behavior for:
            shape: we also allow refs to the in_info in the form (in-idx,dim). see infer_shape().
            need_contiguous/want_inplace: used for bw, in case for bw_input == True.
        :param str c_fw_code: C code for forward pass
        :param str|dict[str] c_extra_support_code: C support code (for c_support_code)
        :param str|None c_bw_code: C code for backward pass (for gradient)
        :param tuple[int] code_version: will be returned by c_code_cache_version.
        :param bool cpu_support:
        :param tuple[int]|callable grad_input_map: selection of grad inputs.
          by default, we get all inputs + all outputs + all grad outputs.
        :param str name: name
        """
        assert isinstance(in_info, (list, tuple))
        assert isinstance(out_info, (list, tuple))
        in_info, out_info, num_dummy_outs = self._resolve_want_inplace_dummy(in_info, out_info)
        self.in_info = make_hashable(in_info)
        self.out_info = make_hashable(out_info)
        self.num_dummy_outs = num_dummy_outs
        self.c_fw_code = c_fw_code
        self.c_bw_code = c_bw_code
        self.c_extra_support_code = self._reduce_c_extra_support_code(c_extra_support_code)
        self.code_version = code_version or ()
        self.cpu_support = cpu_support
        self.name = name or "<anonNativeOp>"
        self.grad_input_map = self._convert_grad_input_map(grad_input_map, len(in_info) + len(out_info) * 2)

    @classmethod
    def _resolve_want_inplace_dummy(cls, in_info, out_info):
        in_info = [dict(info) for info in in_info]  # deep copy, don't modify original
        out_info = list(out_info)  # copying list is enough here
        num_dummy_outs = 0
        for in_idx, info in enumerate(in_info):
            if info.get("want_inplace", None) == "dummy_out":
                num_dummy_outs += 1
                dummy_out_idx = len(out_info)
                dummy_out = {
                    "ndim": info["ndim"],
                    "shape": [(in_idx, i) for i in range(info["ndim"])],
                    "dtype": info.get("dtype", "float32"),
                    "name": "dummy_out_%i" % num_dummy_outs,
                }
                out_info += [dummy_out]
                info["want_inplace"] = dummy_out_idx
        return in_info, out_info, num_dummy_outs

    @classmethod
    def _reduce_c_extra_support_code(cls, c):
        if c is None:
            return ""
        if isinstance(c, dict):
            c = [v for (k, v) in sorted(c.items())]
        if isinstance(c, (list, tuple)):
            c = "\n".join([v + "\n\n" for v in c])
        assert isinstance(c, (str, unicode))
        return c

    @classmethod
    def _convert_grad_input_map(cls, gi_map, num_params):
        """
        :param gi_map: see grad_input_map argument for self.__init__
        :param int num_params:
        :return: tuple of int
        :rtype: tuple[int]
        """
        if gi_map is None:
            gi_map = tuple(range(num_params))
        if callable(gi_map):
            gi_map = gi_map(*range(num_params))
        if isinstance(gi_map, list):
            gi_map = tuple(gi_map)
        assert isinstance(gi_map, tuple)
        return gi_map

    def _filter_grad_inputs(self, inputs):
        """
        :param list[T] inputs: inputs + outputs + output_grads. can be either symbolic tensors or info dicts
        :return: filtered list, via self.grad_input_map
        :rtype: list[T]
        """
        assert len(inputs) == len(self.in_info) + len(self.out_info) * 2
        return [inputs[i] for i in self.grad_input_map]

    # noinspection PyUnusedLocal
    def infer_shape(self, node, input_shapes):
        """
        :param node:
        :param input_shapes:
        :rtype: list[tuple[int]]
        """
        assert len(input_shapes) == len(self.in_info)
        out_shapes = []
        for info in self.out_info:
            out_shape = list(info["shape"])
            for idx, s in enumerate(out_shape):
                if isinstance(s, tuple):  # we interpret this as a reference to input shapes
                    assert len(s) == 2, "dim %r invalid in info %r" % (s, info)
                    assert 0 <= s[0] < len(input_shapes), "dim %r invalid in info %r" % (s, info)
                    assert 0 <= s[1] < self.in_info[s[0]]["ndim"], "dim idx %r invalid in input %i %r, info %r" % (
                        s[1],
                        s[0],
                        self.in_info[s[0]],
                        info,
                    )
                    out_shape[idx] = input_shapes[s[0]][s[1]]
            assert not any([s is None for s in out_shape]), "out_shape %r, out_info %r" % (out_shape, self.out_info)
            out_shapes += [tuple(out_shape)]
        return out_shapes

    @classmethod
    def _bw_in_var_info(cls, info):
        """
        :param dict[str] info:
        :return: updated info dict for the gradient (bwd) as input
        :rtype: dict[str]
        """
        if "bw_in_var" in info:
            info = dict(info)
            info.update(info.pop("bw_in_var"))
        return info

    @classmethod
    def _bw_grad_var_info(cls, info):
        """
        :param dict[str] info: backward gradient input for one of our outputs
        :return: updated info dict for the gradient (bwd) as input
        :rtype: dict[str]
        """
        info = dict(info)
        if "bw_grad_var" in info:
            info.update(info.pop("bw_grad_var"))
        if "name" in info:
            info["name"] = "D_" + info["name"]
        return info

    def kwargs_for_grad_op(self):
        """
        :returns: the kwargs for creating a NativeOp for the gradient op. e.g. includes in_info, out_info, etc
        :rtype: dict[str]

        Note: The inputs of the gradient are by default: fwd_op.inputs + fwd_op.outputs + output_grads.
        We filter them via self._filter_grad_inputs.
        """
        # Inputs: inputs + outputs + output_grads, where outputs = op(inputs),
        # i.e. we might reuse some of the calculation.
        in_info = [self._bw_in_var_info(info) for info in self.in_info]
        in_info += [self._bw_in_var_info(info) for info in self.out_info]
        in_info += [self._bw_grad_var_info(info) for info in self.out_info]
        in_info = self._filter_grad_inputs(in_info)
        in_idx_rev = {v: k for (k, v) in enumerate(self.grad_input_map)}
        # Outputs: All like original inputs. Filter our the disconnected.
        out_info = [info.copy() for info in self.in_info]
        for idx, info in enumerate(out_info):
            info.pop("shape")
            if "bw_out_var" in info:
                info.update(info["bw_out_var"])
            if "shape" not in info:
                # Refer to input shapes. See infer_shape().
                info["shape"] = [(in_idx_rev[idx], i) for i in range(info["ndim"])]
        out_info = [info for info in out_info if info.get("gradient", "") != "disconnected"]

        return dict(
            name="GradOf%s" % self.name,
            in_info=in_info,
            out_info=out_info,
            c_fw_code=self.c_bw_code,
            c_extra_support_code=self.c_extra_support_code,
            code_version=self.code_version,
            cpu_support=self.cpu_support,
        )

    def make_results_of_gradient(self, grad_op_outputs, disconnected_type=None):
        """
        :param list[T]|tuple[T] grad_op_outputs: this is already with dummy outputs removed
        :param S disconnected_type:
        :return: gradient for each input of our op
        :rtype: list[T|S]
        """
        if disconnected_type is None:

            def disconnected_type():
                """Dummy"""

        grad_op_outputs = list(grad_op_outputs)
        results = []
        for info in self.in_info:
            if info.get("gradient", "") == "disconnected":
                results += [disconnected_type()]
            else:
                results += grad_op_outputs[:1]
                grad_op_outputs = grad_op_outputs[1:]
        assert len(grad_op_outputs) == 0
        assert len(results) == len(self.in_info)
        return results


class NativeOpGenBase:
    """
    Base interface for op generation.
    See NativeOp.__init__() for attribs.
    """

    in_info = None  # type: typing.Tuple[typing.Dict[str]]
    out_info = None  # type: typing.Tuple[typing.Dict[str]]
    c_fw_code = None  # type: str
    c_bw_code = None  # type: str
    c_extra_support_code = None  # type: typing.Dict[str,str]
    code_version = None  # type: typing.Union[typing.Tuple[int], int]
    grad_input_map = None
    theano_custom_grad = None
    cpu_support = True

    @classmethod
    def map_layer_inputs_to_op(cls, *inputs):
        """
        :param inputs:
        :return: inputs
        """
        return inputs

    @classmethod
    def map_layer_output_from_op(cls, *outputs):
        """
        :param outputs:
        :return: outputs[0]
        """
        return outputs[0]


class LstmGenericBase(NativeOpGenBase):
    # noinspection PyUnresolvedReferences
    """
    inputs:
      :param Z: {input,output,forget} gate + cell state. 3d (time,batch,dim*4)
      :param V_h: recurrent matrix. 2d (dim,dim*4)
      :param c: initial cell state. 2d (batch,dim)
      :param i: index. 2d (time,batch) -> 0 or 1
    outputs:
      :param Y: output. 3d (time,batch,dim)
      :param H: gates and cell state. 3d (time,batch,dim*4)
      :param d: final cell state. 2d (batch,dim)
    """
    in_info = (
        {
            "name": "Z",
            "ndim": 3,
            "shape": (None, None, None),
            "need_contiguous": True,
            "want_inplace": 1,
            "bw_out_var": {"shape": ((2, 0), (2, 1), (0, 1))},
        },  # see grad_input_map() for indices
        {"name": "V_h", "ndim": 2, "shape": (None, None), "need_contiguous": True},
        {"name": "c", "ndim": 2, "shape": (None, None), "need_contiguous": True},
        {"name": "i", "ndim": 2, "shape": (None, None), "need_contiguous": True, "gradient": "disconnected"},
    )
    out_info = (
        {
            "name": "Y",
            "ndim": 3,
            "shape": ((0, 0), (0, 1), (1, 0)),
            "need_contiguous": True,
            "bw_grad_var": {"want_inplace": "dummy_out"},
        },
        {
            "name": "H",
            "ndim": 3,
            "shape": ((0, 0), (0, 1), (0, 2)),
            "need_contiguous": True,
            "bw_in_var": {"want_inplace": 0},
        },
        {"name": "d", "ndim": 2, "shape": ((2, 0), (2, 1)), "need_contiguous": True},
    )

    # noinspection PyPep8Naming,PyUnusedLocal
    @classmethod
    def grad_input_map(cls, Z, V_h, c, i, Y, H, d, DY, DH, Dd):
        """
        Map grads.
        """
        return V_h, c, i, Y, H, DY, Dd

    c_extra_support_code = {
        "lstm_kernel": """
      DEF_KERNEL
      void lstm_kernel(float* data, const float* old_state, bool old_state_strided,
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
    """,
        "lstm_bwd_kernel": """
      DEF_KERNEL
      void lstm_bwd_kernel(
            float* delta, float* epsilon, const float* next_epsilon, const float* old_state,
            bool old_state_strided, const float* Y, int n_cells, int n_batch, const float* i) {
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
      """,
    }

    c_fw_code = """
    // Z*, V_h, c, i = input_names (*: inplace)
    // Y, H, d = output_names
    assert(n_inputs == 4);
    assert(n_outputs == 3);
    Ndarray* V_h = inputs[1];
    Ndarray* c = inputs[2];
    Ndarray* i = inputs[3];
    Ndarray* Y = *outputs[0];
    Ndarray* H = *outputs[1]; // inplace on Z
    Ndarray* d = *outputs[2];

    long T = Ndarray_DIMS(i)[0];
    int n_batch = Ndarray_DIMS(i)[1];
    assert(Ndarray_DIMS(H)[2] %% 4 == 0); // 3 gates + cell
    int n_cells = Ndarray_DIMS(H)[2] / 4;

    assert(T > 0);
    for(int x = 0; x < T; ++x) {
      if(x > 0) {
        //H += Y[x-1]*V_h
        affine_y_x(x-1, Y,  x, V_h,  x, H);
      }

      start_dev_kernel(lstm_kernel, (
        data_ptr(H, x),
        x > 0 ? data_ptr(H, x - 1) : Ndarray_DEV_DATA(c),
        x > 0,
        data_ptr(Y, x),
        (x == T - 1) ? Ndarray_DEV_DATA(d) : 0,
        n_cells,
        n_batch,
        Ndarray_DEV_DATA(i) + x * n_batch
      ));
    }
    HANDLE_LAST_ERROR();
  """

    c_bw_code = """
    // V_h, c, i,   Y, H*,   DY*, Dd = input_names (*: inplace)
    // DZ, DV_h, Dc, tmpDc = output_names
    assert(n_inputs == 7);
    assert(n_outputs == 4);
    Ndarray* V_h = inputs[0];
    Ndarray* c = inputs[1];
    Ndarray* i = inputs[2];
    Ndarray* Y = inputs[3];
    Ndarray* Dd = inputs[6];
    Ndarray* DZ = *outputs[0]; // inplace on H
    Ndarray* DV_h = *outputs[1];
    Ndarray* Dc = *outputs[2];
    Ndarray* tmpDc = *outputs[3]; // (old DY), inplace buffer

    long T = Ndarray_DIMS(i)[0];
    int n_batch = Ndarray_DIMS(i)[1];
    assert(Ndarray_DIMS(DZ)[2] %% 4 == 0); // 3 gates + cell
    int n_cells = Ndarray_DIMS(DZ)[2] / 4;

    assert(T > 0);
    for(int x = T - 1; x >= 0; --x) {
      // add recurrent
      bool rightBorder = (x == T - 1);
      if(!rightBorder)
        affine_y_x(x+1, DZ,  x, V_h,  x, tmpDc,  false, true);

      start_dev_kernel(lstm_bwd_kernel, (
        data_ptr(DZ, x),
        data_ptr(tmpDc, x),
        rightBorder ? Ndarray_DEV_DATA(Dd) : data_ptr(tmpDc, x + 1),
        x > 0 ? data_ptr(DZ, x - 1) : Ndarray_DEV_DATA(c),
        x > 0,
        data_ptr(Y, x),
        n_cells,
        n_batch,
        Ndarray_DEV_DATA(i) + x * n_batch
      ));
    }

    //DV_h = Y[0..end-1]^T * DZ[1..end]
    affine_global(Y, DZ, DV_h, true, false, 1, 0.0f);

    Ndarray_DIMS_Type Dc_dim = Ndarray_HOST_DIMS(Dc);
    Ndarray_memcpy(
      Ndarray_DEV_DATA(Dc), Ndarray_DEV_DATA(tmpDc),
      Dc_dim[0] * Dc_dim[1] * sizeof(float));
    HANDLE_LAST_ERROR();
  """

    code_version = ()


class LstmLowMem(NativeOpGenBase):
    # noinspection PyUnresolvedReferences
    """
    This is designed to require minimal memory during training.
    It only stores the outputs and the cell states,
    i.e. it requires time * cells * 2 floats for memory in total.

    inputs:
      :param X: (time,batch,in_dim)
      :param W: forward+recurrent matrix. 2d (in_dim+dim,dim*4)
      :param b: bias. 1d (dim*4,)
      :param y0: initial output|hidden state. 2d (batch,dim)
      :param c0: initial cell state. 2d (batch,dim)
      :param i: index. 2d (time,batch) -> 0 or 1
      :param start: where to start. must be >=0, default is usually 0. dtype int, scalar.
      :param step: +1 for fwd, -1 for bwd direction. can also be |step|>1 for wider steps. dtype int, scalar.
        for bwd (<0), will start at T-start-1.
    outputs:
      :param Y: output. 3d (time,batch,dim)
      :param C: cell states. 3d (time,batch,dim). gradient ignored!
      :param d: final cell state. 2d (batch,dim)
    """
    in_info = (
        {"name": "X", "ndim": 3, "shape": (None, None, None), "need_contiguous": True},
        {"name": "W", "ndim": 2, "shape": (None, None), "need_contiguous": True},
        {"name": "b", "ndim": 1, "shape": (None,), "need_contiguous": True},
        {"name": "y0", "ndim": 2, "shape": (None, None), "need_contiguous": True},
        {"name": "c0", "ndim": 2, "shape": (None, None), "need_contiguous": True},
        {"name": "i", "ndim": 2, "shape": (None, None), "need_contiguous": True, "gradient": "disconnected"},
        {"name": "start", "ndim": 0, "shape": (), "gradient": "disconnected", "dtype": "int32", "host_memory": True},
        {"name": "step", "ndim": 0, "shape": (), "gradient": "disconnected", "dtype": "int32", "host_memory": True},
    )
    out_info = (
        {"name": "Y", "ndim": 3, "shape": ((0, 0), (0, 1), (4, 1)), "need_contiguous": True},
        {"name": "C", "ndim": 3, "shape": ((0, 0), (0, 1), (4, 1)), "need_contiguous": True},
        {"name": "d", "ndim": 2, "shape": ((0, 1), (4, 1)), "need_contiguous": True},
    )

    # noinspection PyPep8Naming,PyUnusedLocal
    @classmethod
    def grad_input_map(cls, X, W, b, y0, c0, i, start, step, Y, C, d, DY, DC, Dd):
        """
        Map args.
        """
        return X, W, b, y0, c0, i, start, step, Y, C, DY, Dd

    c_extra_support_code = {
        "lstm_kernel": """
      DEF_KERNEL
      void lstm_kernel(
        int n_batch, int n_cells, const float* mask,
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

          float c_b = (prev_c_b * fgtGate + cellIn * inpGate) * mask_b
                      + prev_c_b * (1.f - mask_b);
          c[idx] = c_b;
          y[idx] = tanhf(c_b) * outGate * mask_b;

          idx += gridDim.x * blockDim.x;
        }
      }
      """,
        "lstm_bwd_kernel": """
      DEF_KERNEL
      void lstm_bwd_kernel(
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
      """,
        "add_bias_kernel": """
      DEF_KERNEL
      void add_bias_kernel(int n_batch, int n_dim, float* x, float* b) {
        int idx = threadIdx.x + blockDim.x * blockIdx.x;
        while (idx < n_batch * n_dim) {
          int dim_idx = idx % n_dim;
          x[idx] += b[dim_idx];
          idx += gridDim.x * blockDim.x;
        }
      }
    """,
        "copy_x_h_kernel": """
      DEF_KERNEL
      void copy_x_h_kernel(
        int n_batch, int n_in, int n_cells,
        float* x_h,
        float* x,
        float* h)
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
      """,
        "inv_copy_x_h_kernel": """
    DEF_KERNEL
    void inv_copy_x_h_kernel(
      int n_batch, int n_in, int n_cells,
      float* x_h,
      float* x,
      float* h)
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
    """,
    }

    c_fw_code = """
    // X, W, b, y0, c0, i, start, step = input_names
    // Y, C, d = output_names
    assert(n_inputs == 8);
    assert(n_outputs == 3);
    Ndarray* X = inputs[0];
    Ndarray* W = inputs[1];
    Ndarray* b = inputs[2];
    Ndarray* y0 = inputs[3];
    Ndarray* c0 = inputs[4];
    Ndarray* i = inputs[5];
    assert_cmp(Ndarray_NDIM(inputs[6]), ==, 0);
    assert_cmp(Ndarray_NDIM(inputs[7]), ==, 0);
    int start = Ndarray_DEV_DATA_int32_scalar(inputs[6]);
    int step = Ndarray_DEV_DATA_int32_scalar(inputs[7]);
    Ndarray* Y = *outputs[0];
    Ndarray* C = *outputs[1];
    Ndarray* d = *outputs[2];

    assert_cmp(Ndarray_NDIM(X), ==, 3);
    assert_cmp(Ndarray_NDIM(W), ==, 2);
    assert_cmp(Ndarray_NDIM(b), ==, 1);
    assert_cmp(Ndarray_NDIM(y0), ==, 2);
    assert_cmp(Ndarray_NDIM(c0), ==, 2);
    assert_cmp(Ndarray_NDIM(i), ==, 2);
    assert_cmp(Ndarray_NDIM(Y), ==, 3);
    assert_cmp(Ndarray_NDIM(C), ==, 3);
    assert_cmp(Ndarray_NDIM(d), ==, 2);
    long T = Ndarray_DIMS(i)[0];
    int n_batch = Ndarray_DIMS(i)[1];
    int n_cells = Ndarray_DIMS(y0)[1];
    int n_in = Ndarray_DIMS(X)[2];
    assert_cmp(Ndarray_DIMS(X)[0], ==, T);
    assert_cmp(Ndarray_DIMS(X)[1], ==, n_batch);
    assert_cmp(Ndarray_DIMS(W)[0], ==, n_in + n_cells);
    assert_cmp(Ndarray_DIMS(W)[1], ==, n_cells * 4);
    assert_cmp(Ndarray_DIMS(b)[0], ==, n_cells * 4);
    assert_cmp(Ndarray_DIMS(y0)[0], ==, n_batch);
    assert_cmp(Ndarray_DIMS(y0)[1], ==, n_cells);
    assert_cmp(Ndarray_DIMS(c0)[0], ==, n_batch);
    assert_cmp(Ndarray_DIMS(c0)[1], ==, n_cells);
    assert_cmp(Ndarray_DIMS(Y)[0], ==, T);
    assert_cmp(Ndarray_DIMS(Y)[1], ==, n_batch);
    assert_cmp(Ndarray_DIMS(Y)[2], ==, n_cells);
    assert_cmp(Ndarray_DIMS(C)[0], ==, T);
    assert_cmp(Ndarray_DIMS(C)[1], ==, n_batch);
    assert_cmp(Ndarray_DIMS(C)[2], ==, n_cells);
    assert_cmp(Ndarray_DIMS(d)[0], ==, n_batch);
    assert_cmp(Ndarray_DIMS(d)[1], ==, n_cells);

    float* x_h = (float*) device_malloc(n_batch * (n_in + n_cells) * sizeof(float));
    float* intern = (float*) device_malloc(n_batch * n_cells * 4 * sizeof(float));  // 3 gates + in

    assert_cmp(T, >, 0);
    assert_cmp(start, >=, 0);
    assert_cmp(start, <, T);
    assert_cmp(step, !=, 0);
    int end = T - 1;
    if(step < 0) {
      end = start;
      start = T - start - 1;
    }
    int t = start;
    for(; (step > 0) ? (t <= end) : (t >= end); t += step) {
      // x_h = X[t], Y[t-1]
      start_dev_kernel(copy_x_h_kernel,
        (n_batch, n_in, n_cells, x_h, data_ptr(X, t), (t != start) ? data_ptr(Y, t-step) : Ndarray_DEV_DATA(y0)));
      // intern = x_h * W
      affine_raw(
        x_h, n_batch, n_in + n_cells,
        Ndarray_DEV_DATA(W), n_in + n_cells, n_cells * 4,
        intern, n_batch, n_cells * 4,
        false, false, 0.0);
      // intern += b
      start_dev_kernel(add_bias_kernel, (
        n_batch, n_cells * 4, intern, Ndarray_DEV_DATA(b)));

      start_dev_kernel(lstm_kernel, (
        n_batch,
        n_cells,
        Ndarray_DEV_DATA(i) + t * n_batch,
        intern,
        (t != start) ? data_ptr(C, t-step) : Ndarray_DEV_DATA(c0),
        data_ptr(Y, t),  // out
        data_ptr(C, t)  // out
      ));
    }
    HANDLE_LAST_ERROR();

    device_free(x_h);
    device_free(intern);

    Ndarray_memcpy(Ndarray_DEV_DATA(d), data_ptr(C, t - step), n_batch * n_cells * sizeof(float));
  """

    # language=C++
    c_bw_code = """
    // X, W, b, y0, c0, i, start, step,   Y, C,   DY, Dd = input_names
    // DX, DW, Db, Dh, Dc = output_names
    assert(n_inputs == 12);
    assert(n_outputs == 5);
    Ndarray* X = inputs[0];
    Ndarray* W = inputs[1];
    Ndarray* b = inputs[2];
    Ndarray* y0 = inputs[3];
    Ndarray* c0 = inputs[4];
    Ndarray* i = inputs[5];
    assert_cmp(Ndarray_NDIM(inputs[6]), ==, 0);
    assert_cmp(Ndarray_NDIM(inputs[7]), ==, 0);
    int start = Ndarray_DEV_DATA_int32_scalar(inputs[6]);
    int step = Ndarray_DEV_DATA_int32_scalar(inputs[7]);
    Ndarray* Y = inputs[8];
    Ndarray* C = inputs[9];
    Ndarray* DY = inputs[10];
    Ndarray* Dd = inputs[11];
    Ndarray* DX = *outputs[0];
    Ndarray* DW = *outputs[1];
    Ndarray* Db = *outputs[2];
    Ndarray* Dh = *outputs[3];
    Ndarray* Dc = *outputs[4];

    assert_cmp(Ndarray_NDIM(X), ==, 3);
    assert_cmp(Ndarray_NDIM(W), ==, 2);
    assert_cmp(Ndarray_NDIM(b), ==, 1);
    assert_cmp(Ndarray_NDIM(y0), ==, 2);
    assert_cmp(Ndarray_NDIM(c0), ==, 2);
    assert_cmp(Ndarray_NDIM(i), ==, 2);
    assert_cmp(Ndarray_NDIM(Y), ==, 3);
    assert_cmp(Ndarray_NDIM(C), ==, 3);
    assert_cmp(Ndarray_NDIM(DY), ==, 3);
    assert_cmp(Ndarray_NDIM(Dd), ==, 2);
    assert_cmp(Ndarray_NDIM(DX), ==, 3);
    assert_cmp(Ndarray_NDIM(DW), ==, 2);
    assert_cmp(Ndarray_NDIM(Db), ==, 1);
    assert_cmp(Ndarray_NDIM(Dh), ==, 2);
    assert_cmp(Ndarray_NDIM(Dc), ==, 2);
    long T = Ndarray_DIMS(i)[0];
    int n_batch = Ndarray_DIMS(i)[1];
    int n_cells = Ndarray_DIMS(y0)[1];
    int n_in = Ndarray_DIMS(X)[2];
    assert_cmp(Ndarray_DIMS(X)[0], ==, T);
    assert_cmp(Ndarray_DIMS(X)[1], ==, n_batch);
    assert_cmp(Ndarray_DIMS(W)[0], ==, n_in + n_cells);
    assert_cmp(Ndarray_DIMS(W)[1], ==, n_cells * 4);
    assert_cmp(Ndarray_DIMS(b)[0], ==, n_cells * 4);
    assert_cmp(Ndarray_DIMS(y0)[0], ==, n_batch);
    assert_cmp(Ndarray_DIMS(y0)[1], ==, n_cells);
    assert_cmp(Ndarray_DIMS(c0)[0], ==, n_batch);
    assert_cmp(Ndarray_DIMS(c0)[1], ==, n_cells);
    assert_cmp(Ndarray_DIMS(Y)[0], ==, T);
    assert_cmp(Ndarray_DIMS(Y)[1], ==, n_batch);
    assert_cmp(Ndarray_DIMS(Y)[2], ==, n_cells);
    assert_cmp(Ndarray_DIMS(C)[0], ==, T);
    assert_cmp(Ndarray_DIMS(C)[1], ==, n_batch);
    assert_cmp(Ndarray_DIMS(C)[2], ==, n_cells);
    assert_cmp(Ndarray_DIMS(DY)[0], ==, T);
    assert_cmp(Ndarray_DIMS(DY)[1], ==, n_batch);
    assert_cmp(Ndarray_DIMS(DY)[2], ==, n_cells);
    assert_cmp(Ndarray_DIMS(Dd)[0], ==, n_batch);
    assert_cmp(Ndarray_DIMS(Dd)[1], ==, n_cells);
    assert_cmp(Ndarray_DIMS(DX)[0], ==, T);
    assert_cmp(Ndarray_DIMS(DX)[1], ==, n_batch);
    assert_cmp(Ndarray_DIMS(DX)[2], ==, n_in);
    assert_cmp(Ndarray_DIMS(DW)[0], ==, n_in + n_cells);
    assert_cmp(Ndarray_DIMS(DW)[1], ==, n_cells * 4);
    assert_cmp(Ndarray_DIMS(Db)[0], ==, n_cells * 4);
    assert_cmp(Ndarray_DIMS(Dh)[0], ==, n_batch);
    assert_cmp(Ndarray_DIMS(Dh)[1], ==, n_cells);
    assert_cmp(Ndarray_DIMS(Dc)[0], ==, n_batch);
    assert_cmp(Ndarray_DIMS(Dc)[1], ==, n_cells);

    float* x_h = (float*) device_malloc(n_batch * (n_in + n_cells) * sizeof(float));
    float* intern = (float*) device_malloc(n_batch * n_cells * 4 * sizeof(float));  // 3 gates + in
    float* Dx_h = (float*) device_malloc(n_batch * (n_in + n_cells) * sizeof(float));
    float* Dintern = (float*) device_malloc(n_batch * n_cells * 4 * sizeof(float));  // 3 gates + in

    // We will work inplace on DX/DW/Db.
    Ndarray_memset(Ndarray_DEV_DATA(DX), 0, T * n_batch * n_in * sizeof(float));
    Ndarray_memset(Ndarray_DEV_DATA(DW), 0, (n_in + n_cells) * n_cells * 4 * sizeof(float));
    Ndarray_memset(Ndarray_DEV_DATA(Db), 0, n_cells * 4 * sizeof(float));
    // We will work inplace on Dh.
    Ndarray_memset(Ndarray_DEV_DATA(Dh), 0, n_batch * n_cells * sizeof(float));
    // We will work inplace on Dc, and init it with Dd.
    Ndarray_memcpy(Ndarray_DEV_DATA(Dc), Ndarray_DEV_DATA(Dd), n_batch * n_cells * sizeof(float));

    assert_cmp(T, >, 0);
    assert_cmp(start, >=, 0);
    assert_cmp(start, <, T);
    assert_cmp(step, !=, 0);
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
      start_dev_kernel(copy_x_h_kernel,
        (n_batch, n_in, n_cells,
         x_h, data_ptr(X, t), right ? data_ptr(Y, t-step) : Ndarray_DEV_DATA(y0)));

      // intern = x_h * W
      affine_raw(
        x_h, n_batch, n_in + n_cells,
        Ndarray_DEV_DATA(W), n_in + n_cells, n_cells * 4,
        intern, n_batch, n_cells * 4,
        false, false, 0.0);
      // intern += b
      start_dev_kernel(add_bias_kernel, (
        n_batch, n_cells * 4, intern, Ndarray_DEV_DATA(b)));

      start_dev_kernel(lstm_bwd_kernel, (
        n_batch,
        n_in,
        n_cells,
        Ndarray_DEV_DATA(i) + t * n_batch,
        x_h,
        intern,
        right ? data_ptr(C, t-step) : Ndarray_DEV_DATA(c0),
        data_ptr(Y, t),
        data_ptr(C, t),
        data_ptr(DY, t),
        Ndarray_DEV_DATA(Dh),  // error from prev frame, excluding DY. updated below
        Ndarray_DEV_DATA(Dc),  // in+out, working inplace. also error from prev frame, initially Dd
        Dintern,  // out
        Ndarray_DEV_DATA(Db)  // out
      ));

      // Dx_h = Dintern * W^T
      affine_raw(
        Dintern, n_batch, n_cells * 4,
        Ndarray_DEV_DATA(W), n_in + n_cells, n_cells * 4,
        Dx_h, n_batch, n_in + n_cells,
        false, true, 0.0);

      // DW += x_h^T * Dintern
      affine_raw(
        x_h, n_batch, n_in + n_cells,
        Dintern, n_batch, n_cells * 4,
        Ndarray_DEV_DATA(DW), n_in + n_cells, n_cells * 4,
        true, false);

      // DX[t], Dh = Dx_h
      start_dev_kernel(inv_copy_x_h_kernel,
        (n_batch, n_in, n_cells, Dx_h, data_ptr(DX, t), Ndarray_DEV_DATA(Dh)));
    }
    HANDLE_LAST_ERROR();

    device_free(x_h);
    device_free(intern);
    device_free(Dx_h);
    device_free(Dintern);
  """


class NativeLstm2(NativeOpGenBase):
    # noinspection PyUnresolvedReferences
    """
    Yet another LSTM kernel.
    This kernel is about 27% than NativeLstm,
    and also has some more options (like the direction).
    But it requires time * batch * cells more memory,
    thus time * batch * cells * 6 in total.

    inputs:
      :param X: (time,batch,dim*4)
      :param W: recurrent matrix. 2d (dim,dim*4)
      :param y0: initial output|hidden state. 2d (batch,dim)
      :param c0: initial cell state. 2d (batch,dim)
      :param i: index. 2d (time,batch) -> 0 or 1
      :param start: where to start. must be >=0, default is usually 0. dtype int, scalar.
      :param step: +1 for fwd, -1 for bwd direction. can also be |step|>1 for wider steps. dtype int, scalar.
        for bwd (<0), will start at T-start-1.
    outputs:
      :param Y: output. 3d (time,batch,dim)
      :param C: cell states. 3d (time,batch,dim). gradient ignored!
      :param H: cell-in + gates. 3d (time,batch,dim*4). gradient ignored!
      :param d: final cell state. 2d (batch,dim)
    """
    in_info = (
        {"name": "X", "ndim": 3, "shape": (None, None, None), "need_contiguous": True},
        {"name": "W", "ndim": 2, "shape": (None, None), "need_contiguous": True},
        {"name": "y0", "ndim": 2, "shape": (None, None), "need_contiguous": True},
        {"name": "c0", "ndim": 2, "shape": (None, None), "need_contiguous": True},
        {"name": "i", "ndim": 2, "shape": (None, None), "need_contiguous": True, "gradient": "disconnected"},
        {"name": "start", "ndim": 0, "shape": (), "gradient": "disconnected", "dtype": "int32", "host_memory": True},
        {"name": "step", "ndim": 0, "shape": (), "gradient": "disconnected", "dtype": "int32", "host_memory": True},
    )
    out_info = (
        {"name": "Y", "ndim": 3, "shape": ((0, 0), (0, 1), (1, 0)), "need_contiguous": True},
        {"name": "C", "ndim": 3, "shape": ((0, 0), (0, 1), (1, 0)), "need_contiguous": True},
        {"name": "H", "ndim": 3, "shape": ((0, 0), (0, 1), (1, 1)), "need_contiguous": True},
        {"name": "d", "ndim": 2, "shape": ((0, 1), (1, 0)), "need_contiguous": True},
    )

    # noinspection PyMissingOrEmptyDocstring,PyUnusedLocal,PyPep8Naming
    @classmethod
    def grad_input_map(cls, X, W, y0, c0, i, start, step, Y, C, H, d, DY, DC, DH, Dd):
        # noinspection PyRedundantParentheses
        return (X, W, y0, c0, i, start, step, Y, C, H, DY, Dd)

    c_extra_support_code = {
        # language=C++
        "lstm_kernel": """
      DEF_KERNEL
      void lstm_kernel(
        int n_batch, int n_cells, const float* mask,
        float* h,
        float* prev_y,
        float* prev_c,
        float* y,
        float* c,
        float* y_prev_out)
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

          float c_b = (prev_c_b * fgtGate + cellIn * inpGate) * mask_b
                    + prev_c_b * (1.f - mask_b);
          c[idx] = c_b;
          float y_b = tanhf(c_b) * outGate * mask_b;
          y[idx] = y_b;
          y_prev_out[idx] = y_b + prev_y[idx] * (1.f - mask_b);

          idx += gridDim.x * blockDim.x;
        }
      }
      """,
        # language=C++
        "lstm_bwd_kernel": """
      DEF_KERNEL
      void lstm_bwd_kernel(
        int n_batch, int n_cells, const float* mask,
        float* h,
        float* prev_c,
        float* y,
        float* c,
        float* d_y,
        float* d_h,
        float* d_c,
        float* d_x,
        float* d_x0)
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
      """,
    }

    # language=C++
    c_fw_code = """
    // X, W, y0, c0, i, start, step = input_names
    // Y, C, H, d = output_names
    assert(n_inputs == 7);
    assert(n_outputs == 4);
    Ndarray* X = inputs[0];
    Ndarray* W = inputs[1];
    Ndarray* y0 = inputs[2];
    Ndarray* c0 = inputs[3];
    Ndarray* i = inputs[4];
    assert_cmp(Ndarray_NDIM(inputs[5]), ==, 0);
    assert_cmp(Ndarray_NDIM(inputs[6]), ==, 0);
    int start = Ndarray_DEV_DATA_int32_scalar(inputs[5]);
    int step = Ndarray_DEV_DATA_int32_scalar(inputs[6]);
    Ndarray* Y = *outputs[0];
    Ndarray* C = *outputs[1];
    Ndarray* H = *outputs[2];
    Ndarray* d = *outputs[3];

    assert_cmp(Ndarray_NDIM(X), ==, 3);
    assert_cmp(Ndarray_NDIM(W), ==, 2);
    assert_cmp(Ndarray_NDIM(y0), ==, 2);
    assert_cmp(Ndarray_NDIM(c0), ==, 2);
    assert_cmp(Ndarray_NDIM(i), ==, 2);
    assert_cmp(Ndarray_NDIM(Y), ==, 3);
    assert_cmp(Ndarray_NDIM(C), ==, 3);
    assert_cmp(Ndarray_NDIM(H), ==, 3);
    assert_cmp(Ndarray_NDIM(d), ==, 2);
    long T = Ndarray_DIMS(i)[0];
    int n_batch = Ndarray_DIMS(i)[1];
    int n_cells = Ndarray_DIMS(y0)[1];
    assert_cmp(Ndarray_DIMS(X)[0], ==, T);
    assert_cmp(Ndarray_DIMS(X)[1], ==, n_batch);
    assert_cmp(Ndarray_DIMS(X)[2], ==, n_cells * 4);
    assert_cmp(Ndarray_DIMS(W)[0], ==, n_cells);
    assert_cmp(Ndarray_DIMS(W)[1], ==, n_cells * 4);
    assert_cmp(Ndarray_DIMS(y0)[0], ==, n_batch);
    assert_cmp(Ndarray_DIMS(y0)[1], ==, n_cells);
    assert_cmp(Ndarray_DIMS(c0)[0], ==, n_batch);
    assert_cmp(Ndarray_DIMS(c0)[1], ==, n_cells);
    assert_cmp(Ndarray_DIMS(Y)[0], ==, T);
    assert_cmp(Ndarray_DIMS(Y)[1], ==, n_batch);
    assert_cmp(Ndarray_DIMS(Y)[2], ==, n_cells);
    assert_cmp(Ndarray_DIMS(C)[0], ==, T);
    assert_cmp(Ndarray_DIMS(C)[1], ==, n_batch);
    assert_cmp(Ndarray_DIMS(C)[2], ==, n_cells);
    assert_cmp(Ndarray_DIMS(H)[0], ==, T);
    assert_cmp(Ndarray_DIMS(H)[1], ==, n_batch);
    assert_cmp(Ndarray_DIMS(H)[2], ==, n_cells * 4);
    assert_cmp(Ndarray_DIMS(d)[0], ==, n_batch);
    assert_cmp(Ndarray_DIMS(d)[1], ==, n_cells);

    if(T == 0) {
      Ndarray_memcpy(Ndarray_DEV_DATA(d), Ndarray_DEV_DATA(c0), n_batch * n_cells * sizeof(float));

    } else {  // T > 0
      // It makes the backprop with step<0 easier to implement,
      // esp. the DW = Y[0..T-2]^T * DX[1..T-1] calculation,
      // if we can have Y[t] = 0 where mask[t] = 0.
      // That is why we need to keep track of Y[t-1] explicitly.
      float* y_prev = (float*) device_malloc(n_batch * n_cells * sizeof(float));

      // H = X
      Ndarray_memcpy(Ndarray_DEV_DATA(H), Ndarray_DEV_DATA(X), T * n_batch * n_cells * 4 * sizeof(float));

      assert_cmp(T, >, 0);
      assert_cmp(start, >=, 0);
      assert_cmp(start, <, T);
      assert_cmp(step, !=, 0);
      int end = T - 1;
      if(step < 0) {
        end = 0;
        start = T - start - 1;
      }
      int t = start;
      for(; (step > 0) ? (t <= end) : (t >= end); t += step) {
        // H[t] += Y[t-1] * W
        affine_raw(
          (t != start) ? y_prev : Ndarray_DEV_DATA(y0), n_batch, n_cells,
          Ndarray_DEV_DATA(W), n_cells, n_cells * 4,
          data_ptr(H, t), n_batch, n_cells * 4,
          false, false);

        start_dev_kernel(lstm_kernel, (
          n_batch,
          n_cells,
          Ndarray_DEV_DATA(i) + t * n_batch,
          data_ptr(H, t),  // inplace
          (t != start) ? y_prev : Ndarray_DEV_DATA(y0),
          (t != start) ? data_ptr(C, t-step) : Ndarray_DEV_DATA(c0),
          data_ptr(Y, t),  // out
          data_ptr(C, t),  // out
          y_prev  // out
        ));
      }
      HANDLE_LAST_ERROR();

      Ndarray_memcpy(Ndarray_DEV_DATA(d), data_ptr(C, t - step), n_batch * n_cells * sizeof(float));

      device_free(y_prev);
    }
  """

    # language=C++
    c_bw_code = """
    // X, W, y0, c0, i, start, step,   Y, C, H,   DY, Dd = input_names
    // DX, DW, Dy0, Dc0 = output_names
    assert(n_inputs == 12);
    assert(n_outputs == 4);
    Ndarray* X = inputs[0];
    Ndarray* W = inputs[1];
    Ndarray* y0 = inputs[2];
    Ndarray* c0 = inputs[3];
    Ndarray* i = inputs[4];
    assert_cmp(Ndarray_NDIM(inputs[5]), ==, 0);
    assert_cmp(Ndarray_NDIM(inputs[6]), ==, 0);
    int start = Ndarray_DEV_DATA_int32_scalar(inputs[5]);
    int step = Ndarray_DEV_DATA_int32_scalar(inputs[6]);
    Ndarray* Y = inputs[7];
    Ndarray* C = inputs[8];
    Ndarray* H = inputs[9];
    Ndarray* DY = inputs[10];
    Ndarray* Dd = inputs[11];
    Ndarray* DX = *outputs[0];
    Ndarray* DW = *outputs[1];
    Ndarray* Dy0 = *outputs[2];
    Ndarray* Dc0 = *outputs[3];

    assert_cmp(Ndarray_NDIM(X), ==, 3);
    assert_cmp(Ndarray_NDIM(W), ==, 2);
    assert_cmp(Ndarray_NDIM(y0), ==, 2);
    assert_cmp(Ndarray_NDIM(c0), ==, 2);
    assert_cmp(Ndarray_NDIM(i), ==, 2);
    assert_cmp(Ndarray_NDIM(Y), ==, 3);
    assert_cmp(Ndarray_NDIM(C), ==, 3);
    assert_cmp(Ndarray_NDIM(H), ==, 3);
    assert_cmp(Ndarray_NDIM(DY), ==, 3);
    assert_cmp(Ndarray_NDIM(Dd), ==, 2);
    assert_cmp(Ndarray_NDIM(DX), ==, 3);
    assert_cmp(Ndarray_NDIM(DW), ==, 2);
    assert_cmp(Ndarray_NDIM(Dy0), ==, 2);
    assert_cmp(Ndarray_NDIM(Dc0), ==, 2);
    long T = Ndarray_DIMS(i)[0];
    int n_batch = Ndarray_DIMS(i)[1];
    int n_cells = Ndarray_DIMS(y0)[1];
    assert_cmp(Ndarray_DIMS(X)[0], ==, T);
    assert_cmp(Ndarray_DIMS(X)[1], ==, n_batch);
    assert_cmp(Ndarray_DIMS(X)[2], ==, n_cells * 4);
    assert_cmp(Ndarray_DIMS(W)[0], ==, n_cells);
    assert_cmp(Ndarray_DIMS(W)[1], ==, n_cells * 4);
    assert_cmp(Ndarray_DIMS(y0)[0], ==, n_batch);
    assert_cmp(Ndarray_DIMS(y0)[1], ==, n_cells);
    assert_cmp(Ndarray_DIMS(c0)[0], ==, n_batch);
    assert_cmp(Ndarray_DIMS(c0)[1], ==, n_cells);
    assert_cmp(Ndarray_DIMS(Y)[0], ==, T);
    assert_cmp(Ndarray_DIMS(Y)[1], ==, n_batch);
    assert_cmp(Ndarray_DIMS(Y)[2], ==, n_cells);
    assert_cmp(Ndarray_DIMS(C)[0], ==, T);
    assert_cmp(Ndarray_DIMS(C)[1], ==, n_batch);
    assert_cmp(Ndarray_DIMS(C)[2], ==, n_cells);
    assert_cmp(Ndarray_DIMS(H)[0], ==, T);
    assert_cmp(Ndarray_DIMS(H)[1], ==, n_batch);
    assert_cmp(Ndarray_DIMS(H)[2], ==, n_cells * 4);
    assert_cmp(Ndarray_DIMS(DY)[0], ==, T);
    assert_cmp(Ndarray_DIMS(DY)[1], ==, n_batch);
    assert_cmp(Ndarray_DIMS(DY)[2], ==, n_cells);
    assert_cmp(Ndarray_DIMS(Dd)[0], ==, n_batch);
    assert_cmp(Ndarray_DIMS(Dd)[1], ==, n_cells);
    assert_cmp(Ndarray_DIMS(DX)[0], ==, T);
    assert_cmp(Ndarray_DIMS(DX)[1], ==, n_batch);
    assert_cmp(Ndarray_DIMS(DX)[2], ==, n_cells * 4);
    assert_cmp(Ndarray_DIMS(DW)[0], ==, n_cells);
    assert_cmp(Ndarray_DIMS(DW)[1], ==, n_cells * 4);
    assert_cmp(Ndarray_DIMS(Dy0)[0], ==, n_batch);
    assert_cmp(Ndarray_DIMS(Dy0)[1], ==, n_cells);
    assert_cmp(Ndarray_DIMS(Dc0)[0], ==, n_batch);
    assert_cmp(Ndarray_DIMS(Dc0)[1], ==, n_cells);

    // We will work inplace on DW.
    Ndarray_memset(Ndarray_DEV_DATA(DW), 0, n_cells * n_cells * 4 * sizeof(float));
    // We will work inplace on (Dy0) DY[t], initially 0.
    Ndarray_memset(Ndarray_DEV_DATA(Dy0), 0, n_batch * n_cells * sizeof(float));
    // We will work inplace on (Dc0) DC[t], and init it with Dd.
    Ndarray_memcpy(Ndarray_DEV_DATA(Dc0), Ndarray_DEV_DATA(Dd), n_batch * n_cells * sizeof(float));

    if(T == 0) {
      // just do nothing. at least do not crash

    } else {
      // Need to keep track of (logical) DX[0], which in practice (masking, step<0)
      // can be different from data_ptr(DX, start).
      float* dx0 = (float*) device_malloc(n_batch * n_cells * 4 * sizeof(float));
      Ndarray_memset(dx0, 0, n_batch * n_cells * 4 * sizeof(float));

      assert_cmp(T, >, 0);
      assert_cmp(start, >=, 0);
      assert_cmp(start, <, T);
      assert_cmp(step, !=, 0);
      int abs_step = std::abs(step);

      if(abs_step > 1 || start > 0)
        // Normally the kernel would visit and reset all DX.
        // But with abs_step>1 or start>0, we will not visit all. Reset now.
        Ndarray_memset(Ndarray_DEV_DATA(DX), 0, T * n_batch * n_cells * 4 * sizeof(float));

      // e.g.:
      // step=1, start=0, T=10 -> num_steps=10=T
      // step=5, start=0, T=10 -> num_steps=2=T/step
      // step=5, start=0, T=9  -> num_steps=2=(T+step-1)/step
      // step=5, start=0, T=6  -> num_steps=2=(T+step-1)/step
      // step=5, start=0, T=5  -> num_steps=1=(T+step-1)/step
      // step=5, start=4, T=10 -> num_steps=2=(T-start+step-1)/step
      // step=-5, start=0, T=10 -> num_steps=2=T/abs_step
      // step=-5, start=0, T=9  -> num_steps=2=(T+abs_step-1)/abs_step
      // step=-5, start=4, T=10 -> num_steps=2=(T-start+abs_step-1)/abs_step
      int num_steps = (T - start + abs_step - 1) / abs_step;
      assert_cmp(num_steps, >, 0);
      if(step < 0)
        start = T - start - 1;
      int end = start + (num_steps - 1) * step;  // inclusive
      assert_cmp(end, >=, 0);
      assert_cmp(end, <, T);
      int t = end;  // go backwards
      for(; (step > 0) ? (t >= start) : (t <= start); t -= step) {
        bool right = (step > 0) ? (t - step >= start) : (t - step <= start);

        start_dev_kernel(lstm_bwd_kernel, (
          n_batch,
          n_cells,
          Ndarray_DEV_DATA(i) + t * n_batch,
          data_ptr(H, t),
          right ? data_ptr(C, t-step) : Ndarray_DEV_DATA(c0),
          data_ptr(Y, t),
          data_ptr(C, t),
          data_ptr(DY, t),
          Ndarray_DEV_DATA(Dy0),  // in+out, error from prev frame, excluding DY. reset here, updated below
          Ndarray_DEV_DATA(Dc0),  // in+out, working inplace. also error from prev frame, initially Dd
          data_ptr(DX, t),  // out
          dx0  // out
        ));

        // (Dy0) DY[t-1] += DX[t] * W^T
        affine_raw(
          data_ptr(DX, t), n_batch, n_cells * 4,
          Ndarray_DEV_DATA(W), n_cells, n_cells * 4,
          Ndarray_DEV_DATA(Dy0), n_batch, n_cells,
          false, true);
      }

      //DW = Y[0..T-2]^T * DX[1..T-1]  (if step==1)
      if(num_steps > 1) {
        if(abs_step == 1) {
          affine_raw(
            data_ptr(Y, std::min(start, end) + std::max(0, -step)), (num_steps - 1) * n_batch, n_cells,
            data_ptr(DX, std::min(start, end) + std::max(0, step)), (num_steps - 1) * n_batch, n_cells * 4,
            Ndarray_DEV_DATA(DW), n_cells, n_cells * 4,
            true, false, 0.0f, 1.0f);
        } else {
          // Unfortunately we cannot do efficient striding. Thus loop again.
          t = end - step;  // one before
          for(; (step > 0) ? (t >= start) : (t <= start); t -= step) {
            affine_raw(
              data_ptr(Y, t), n_batch, n_cells,
              data_ptr(DX, t + step), n_batch, n_cells * 4,
              Ndarray_DEV_DATA(DW), n_cells, n_cells * 4,
              true, false);
          }
        }
      }
      HANDLE_LAST_ERROR();

      //DW += y0^T * DX[0]
      affine_raw(
        Ndarray_DEV_DATA(y0), n_batch, n_cells,
        dx0, n_batch, n_cells * 4,
        Ndarray_DEV_DATA(DW), n_cells, n_cells * 4,
        true, false);

      device_free(dx0);
    }
  """


class TwoDLSTM(NativeOpGenBase):
    # noinspection PyUnresolvedReferences
    """
    inputs:
      :param X: {input,output,forget,lambda} gate + cell state. 3d (timeT,timeS,batch,dim*5) // dim*5 or dim*1 ?
      :param V_h: recurrent matrix. 2d (dim,dim*5)
      :param V_v: recurrent matrix. 2d (dim,dim*5)
      :param W: recurrent matrix. 2d (dim,dim*5)
      :param b: bias. 2d (batch,dim)
      :param ptr_storage: ptr_storage. 1d (1 * 5 * max_diag_size * sizeof(float*) / sizeof(float))
      :param valid: used internally to store which cells are valid (have to be computed).
        1d (1 * max_diag_size * n_minibatch)
      :param workmem2: used internally. 3d (H[0], H[2], H[3])
      :param sizes: height (target) x width (source) of the unpadded sentences. 2d (batch, 2)
    outputs:
      :param CompleteY: output. 4d (timeS,timeT,batch,dim)
      :param H: gates and cell state. 4d (timeS,timeT,batch,dim*5) ?
      :param d: final cell state. 3d (timeT,batch,dim)
    """
    in_info = (
        {
            "name": "X",
            "ndim": 4,
            "shape": (None, None, None, None),
            "need_contiguous": True,
            "bw_out_var": {"shape": ((0, 0), (0, 1), (0, 2), (0, 3))},
        },  # see grad_input_map() for indices
        {"name": "V_h", "ndim": 2, "shape": (None, None), "need_contiguous": True},
        {"name": "V_v", "ndim": 2, "shape": (None, None), "need_contiguous": True},
        {"name": "W", "ndim": 2, "shape": (None, None), "need_contiguous": True},
        {"name": "b", "ndim": 1, "shape": (None,), "need_contiguous": True},
        {"name": "ptr_storage_fwd", "ndim": 1, "shape": (None,), "need_contiguous": True, "gradient": "disconnected"},
        {"name": "ptr_storage_bwd", "ndim": 1, "shape": (None,), "need_contiguous": True, "gradient": "disconnected"},
        {"name": "valid", "ndim": 1, "shape": (None,), "need_contiguous": True, "gradient": "disconnected"},
        {
            "name": "workmem",
            "ndim": 5,
            "shape": (None, None, None, None, None),
            "need_contiguous": True,
            "gradient": "disconnected",
        },
        {
            "name": "workmem2",
            "ndim": 3,
            "shape": (None, None, None),
            "need_contiguous": True,
            "gradient": "disconnected",
        },
        {"name": "sizes", "ndim": 2, "shape": (None, None), "need_contiguous": True, "gradient": "disconnected"},
        {
            "name": "DYDummy",
            "ndim": 4,
            "shape": (None, None, None, None),
            "need_contiguous": True,
            "gradient": "disconnected",
        },
        {
            "name": "initialState",
            "ndim": 4,
            "shape": (None, None, None, None),
            "need_contiguous": True,
            "gradient": "disconnected",
        },
        {
            "name": "initialOutput",
            "ndim": 4,
            "shape": (None, None, None, None),
            "need_contiguous": True,
            "gradient": "disconnected",
        },
        {"name": "iteration", "ndim": 1, "shape": (None,), "need_contiguous": True, "gradient": "disconnected"},
    )
    out_info = (
        {
            "name": "CompleteY",
            "ndim": 4,
            "shape": ((0, 0), (0, 1), (0, 2), (1, 0)),
            "need_contiguous": True,
        },  # "bw_grad_var": {"want_inplace": "dummy_out"}},
        {
            "name": "H",
            "ndim": 4,
            "shape": ((0, 0), (0, 1), (0, 2), (3, 1)),
            "need_contiguous": True,
            # (timeT, timeS, batch, dim*5)
        },  # "bw_in_var": {"want_inplace": "dummy_out"}},
    )

    # noinspection PyMissingOrEmptyDocstring,PyPep8Naming
    @classmethod
    def grad_input_map(
        cls,
        X,
        V_h,
        V_v,
        W,
        b,
        ptr_storage_fwd,
        ptr_storage_bwd,
        valid,
        workmem,
        workmem2,
        sizes,
        DYDummy,
        initialState,
        initialOutput,
        iteration,
        CompleteY,
        H,
        DCompleteY,
        DH,
    ):
        return (
            X,
            V_h,
            V_v,
            W,
            b,
            ptr_storage_fwd,
            ptr_storage_bwd,
            valid,
            workmem,
            workmem2,
            sizes,
            DYDummy,
            initialState,
            initialOutput,
            iteration,
            CompleteY,
            H,
            DCompleteY,
            DH,
        )

    # noinspection PyMissingOrEmptyDocstring,PyPep8Naming
    @classmethod
    def map_layer_inputs_to_op(cls, Zs, Zt, V_h, V_v, W, b, ptr_storage):
        assert False  # no support for Theano

    c_extra_support_code = {
        "01_repvec": """
      DEF_KERNEL
      void repvec(const float * v, int vlen, int nCopies, float * dest)
      {
        int idx = threadIdx.x + blockDim.x * blockIdx.x;
        while (idx < vlen * nCopies)
        {
          dest[idx] = v[idx % vlen];
          idx += gridDim.x * blockDim.x;
        }
      }
    """,
        "02_fillmat": """
      void fillmat(OpKernelContext* context, const Ndarray * b, Ndarray * dst)
      {
        const float * data_b = Ndarray_DEV_DATA(b);
        float * data_dst = Ndarray_DEV_DATA(dst);
        Ndarray_DIMS_Type dims_b = Ndarray_HOST_DIMS(b);
        int dims_dst[2];
        lastTwoDims(dst, dims_dst);
        assert(dims_b[0] == dims_dst[1]);
        start_dev_kernel(repvec, (
          data_b,
          dims_dst[1],
          Ndarray_SIZE(dst)/dims_dst[1],
          data_dst
        ));
      }
    """,
        "03_data_ptr": """
      // if nd is 2 then assume a weight matrix and just return beginning of data
      // else nd should be 3 and we pick the x part
      float* data_ptr(const Ndarray* a, int y, int x, int outer_dim=0) {
          assert(Ndarray_NDIM(a) == 2 || Ndarray_NDIM(a) == 4 || Ndarray_NDIM(a) == 5);
          if(Ndarray_NDIM(a) == 2)
              return Ndarray_DEV_DATA(a);
          else if(Ndarray_NDIM(a) == 4) {
              Ndarray_DIMS_Type dims = Ndarray_HOST_DIMS(a);
              return Ndarray_DEV_DATA(a)
                + y * dims[1] * dims[2] * dims[3]
                + x * dims[2] * dims[3]; // row-major or minor?
          }
          else {
              Ndarray_DIMS_Type dims = Ndarray_HOST_DIMS(a);
              return Ndarray_DEV_DATA(a)
                + outer_dim * dims[1] * dims[2] * dims[3] * dims[4]
                + y * dims[2] * dims[3] * dims[4]
                + x * dims[3] * dims[4];
          }
      }

      float * data_ptr(Ndarray * a, int y, int x, int outer_dim=0)
      {
        const Ndarray * ca = a;
        return const_cast<float *>(data_ptr(ca, y, x, outer_dim));
      }
    """,
        "04_affine_y_x_batched_onedir": """
      // ys and xs: base indices, offset by y_A, x_A (-1,0,1)
      void affine_y_x_batched_onedir(OpKernelContext* context, int y_A, int x_A,
        const Ndarray * A1,
        const Ndarray * B1,
        Ndarray * C1,
        const std::vector<int>& ys, const std::vector<int>& xs, Ndarray * ptr_storage, int height, int width,
        cudaStream_t stream = 0, bool transpose_A=false, bool transpose_B=false)
      {
        const int batch_size = ys.size();
        if(batch_size == 0)
        {
          return;
        }
        std::vector<const float*> ABC_ptrs(3 * 1 * batch_size); //content layout: 3x1xbatch_size (3: A,B,C, 1: dirs)

        for(int i = 0; i < batch_size; ++i)
        {
          //A
          //y not flipped, x not flipped
          ABC_ptrs[0 * 1 * batch_size + 0 * batch_size + i] = data_ptr(A1, y_A + ys[i], x_A + xs[i]);

          //B
          //index doesent matter here, as B is only 2dimensional
          ABC_ptrs[1 * 1 * batch_size + 0 * batch_size + i] = data_ptr(B1, 0, 0);

          //we write the result (C) in the same destination (y,x) as the source (A), so we don't need to flip later
          //C
          //y not flipped, x not flipped
          ABC_ptrs[2 * 1 * batch_size + 0 * batch_size + i] = data_ptr(C1, ys[i], xs[i]);
        }
        const float ** ptr_storage_data = reinterpret_cast<const float**>(&(ABC_ptrs[0]));
        const float ** A_ptrs_data = (const float**) ptr_storage_data + 0 * 1 * batch_size;
        const float ** B_ptrs_data = (const float**) ptr_storage_data + 1 * 1 * batch_size;
        const float ** C_ptrs_data = ptr_storage_data + 2 * 1 * batch_size;

        int A_dim[2], B_dim[2];
        lastTwoDims(A1, A_dim);
        lastTwoDims(B1, B_dim);
        int ldB = B_dim[1];
        int ldA = A_dim[1];
        char transA = transpose_A ? 'T' : 'N';
        char transB = transpose_B ? 'T' : 'N';
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

        Ndarray_sgemm_batched(
          transB, transA, B_dim[1], A_dim[0], A_dim[1], &alpha,
          B_ptrs_data, ldB, A_ptrs_data, ldA, &beta,
          C_ptrs_data, B_dim[1], 1 * batch_size, batch_size == 1);
      }
    """,
        "05_lstm_stable_cell_kernel_batched": """
      DEF_KERNEL
      void lstm_stable_cell_kernel_batched(float ** datas, const float ** old_state_ys, const float ** old_state_xs,
       float ** outputs, const float ** valids, int n_outer_batch, int n_cells, int n_minibatch)
      {
        //layout (for every outer batch):
        //data[0*n_cells..1*n_cells-1] : input gate
        //data[1*n_cells..2*n_cells-1] : forget gate
        //data[2*n_cells..3*n_cells-1] : lambda gate
        //data[3*n_cells..4*n_cells-1] : output gate
        //data[5*n_cells..6*n_cells-1] : cell state
        //output[0*n_cells..1*n_cells-1]: cell output
        //valids: either 1.0 or 0.0, indicating if the current (y,x) position
        //  is still inside the image in this minibatch
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
    """,
        "06_do_lstm_batched_onedir": """
      // H, CompleteY, ys, xs, ptr_storage
      void do_lstm_batched_onedir(
       OpKernelContext* context, Ndarray* H, Ndarray* initialState, float iteration, Ndarray* completeOut,
       const std::vector<int>& ys, const std::vector<int>& xs,
       Ndarray* ptr_storage, Ndarray* valid_storage, Ndarray* sizes)
      {
        int n_outer_batch = ys.size();
        Ndarray_DIMS_Type H_dims = Ndarray_HOST_DIMS(H);
        int height = H_dims[0];
        int width = H_dims[1];
        int n_minibatch = H_dims[2];
        int n_cells = H_dims[3] / 5;
        assert(H_dims[3] % 5 == 0); //4 gates + cell

        std::vector<float*> ptrs(1 * 5 * n_outer_batch); //1 dirs * 5 arrays
        std::vector<float> valid(1 * n_minibatch * n_outer_batch, 1.0f);

        float* h_sizes; // the sizes array is stored on the GPU, we have to copy it to the CPU
        int dsize =
          (n_outer_batch) * (n_minibatch) * sizeof(float) * 2; // (*2), because we have 2 (height, width) numbers
        h_sizes = (float*)malloc(dsize);
        HANDLE_ERROR(cudaMemcpy(h_sizes, Ndarray_DEV_DATA(sizes), dsize, cudaMemcpyDeviceToHost));

        for(int i = 0; i < n_outer_batch; ++i)
        {
          int y = ys[i];
          int x = xs[i];

          //fill valid
          for(int n = 0; n < n_minibatch; ++n) // iterates through all examples in the current batch
          {
            float img_height = *(h_sizes + 2*n);
            float img_width = *(h_sizes + 2*n +1);

            valid[i * 1 * n_minibatch + 0 * n_minibatch + n] = float(y < img_height && x < img_width);
          }

          //y not flipped, x not flipped
          float * data_H = data_ptr(H, y, x);

          //y not flipped, x not flipped
          float * data_old_state_y;
          data_old_state_y = y > 0 ? data_ptr(H, y - 1, x) + 4 * n_cells : data_ptr(initialState, 0, x) + 4 * n_cells;

          //y not flipped, x not flipped
          float * data_old_state_x = x > 0 ? data_ptr(H, y, x - 1) + 4 * n_cells : 0;

          //y not flipped, x not flipped
          float * data_out = data_ptr(completeOut, y, x);

          float * valid = Ndarray_DEV_DATA(valid_storage) + i * 1 * n_minibatch + 0 * n_minibatch;

          ptrs[0 * 1 * n_outer_batch + 0 * n_outer_batch + i] = data_H;
          ptrs[1 * 1 * n_outer_batch + 0 * n_outer_batch + i] = data_old_state_y;
          ptrs[2 * 1 * n_outer_batch + 0 * n_outer_batch + i] = data_old_state_x;
          ptrs[3 * 1 * n_outer_batch + 0 * n_outer_batch + i] = data_out;
          ptrs[4 * 1 * n_outer_batch + 0 * n_outer_batch + i] = valid;
        }

        free(h_sizes);

        HANDLE_ERROR(cudaMemcpy(Ndarray_DEV_DATA(valid_storage), valid.data(),
          valid.size() * sizeof(float), cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpy(Ndarray_DEV_DATA(ptr_storage), ptrs.data(),
          ptrs.size() * sizeof(float*), cudaMemcpyHostToDevice));
        float ** ptr_storage_data = reinterpret_cast<float**>(Ndarray_DEV_DATA(ptr_storage));
        float ** data_Hs = ptr_storage_data + 0 * 1 * n_outer_batch;
        const float ** data_old_state_ys = (const float**) ptr_storage_data + 1 * 1 * n_outer_batch;
        const float ** data_old_state_xs = (const float**) ptr_storage_data + 2 * 1 * n_outer_batch;
        float ** data_outs = ptr_storage_data + 3 * 1 * n_outer_batch;
        const float ** data_valids = (const float**) (ptr_storage_data + 4 * 1 * n_outer_batch);

        start_dev_kernel(lstm_stable_cell_kernel_batched, (
          data_Hs,
          data_old_state_ys,
          data_old_state_xs,
          data_outs,
          data_valids,
          1 * n_outer_batch,
          n_cells,
          n_minibatch
        ));
      }
    """,
        "07_lstm_bwd_stable_cell_kernel_batched": """
      DEF_KERNEL
      void lstm_bwd_stable_cell_kernel_batched(float ** deltas, const float ** epsilons,
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
        //next_epsilon_x[0*n_cells..1*n_cells-1]:
        //  cell state derivative * forget_gate * (-1*)lambda_gate (of next timestep)
        //epsilon_y[0*n_cells..1*n_cells-1]:
        //  cell state derivative * forget_gate * lambda_gate (of current timestep, as output)
        //epsilon_x[0*n_cells..1*n_cells-1]:
        //  cell state derivative * forget_gate * (1-lambda_gate) (of current timestep, as output)
        //valids: either 1.0 or 0.0, indicating if the current (y,x) position
        //  is still inside the image in this minibatch
        //repeated for every mini-batch

        float near_zero = 0.00000000001f;

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
          if (outGate < -near_zero || outGate > near_zero)
          {
            gc = Y[inner_idx] / outGate;
          }

          if (inpGate < -near_zero || inpGate > near_zero)
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
    """,
        "08_do_lstm_bwd_batched_onedir": """
      //epsilon are the derivates w.r.t. Z, delta stores the gate and cell activations
      //  and will store the derivatives later
      void do_lstm_bwd_batched_onedir(OpKernelContext* context, Ndarray * delta1, Ndarray * epsilon1,
       const Ndarray* CompleteY, Ndarray * workmem1,
       int height, int width, const std::vector<int>& ys, const std::vector<int>& xs,
       Ndarray * ptr_storage, Ndarray * valid_storage, Ndarray*  sizes, int iteration, cudaStream_t stream=0)
      {
        int n_outer_batch = ys.size();
        int dims[2];
        lastTwoDims(delta1, dims);
        assert(dims[1] % 5 == 0); //4 gates + cell
        int n_cells = dims[1] / 5;
        int n_minibatch = dims[0];

        std::vector<const float*> ptrs(1 * 10 * n_outer_batch); //1 dirs * 10 arrays
        std::vector<float> valid(1 * n_minibatch * n_outer_batch, 1.0f);

        float* h_sizes; // the sizes array is stored on the GPU, we have to copy it to the CPU
        int dsize =
          (n_outer_batch) * (n_minibatch) * sizeof(float) * 2; // (*2), because we have 2 (height, width) numbers
        h_sizes = (float*)malloc(dsize);
        HANDLE_ERROR(cudaMemcpy(h_sizes, Ndarray_DEV_DATA(sizes), dsize, cudaMemcpyDeviceToHost));

        for(int i = 0; i < n_outer_batch; ++i)
        {
          int y = ys[i];
          int x = xs[i];
          //fill valid
          for(int n = 0; n < n_minibatch; ++n)
          {
            //these are the sizes of a single image in the batch, while height/width are the maximum sizes in the batch
            float img_height = *(h_sizes + 2*n);
            float img_width = *(h_sizes + 2*n +1);
            valid[i * 1 * n_minibatch + 0 * n_minibatch + n] = float(y < img_height && x < img_width);
          }

          bool botBorder = (y == height-1);
          bool rightBorder = (x == width-1);
          int yp1 = y + 1;
          int xp1 = x + 1;
          int ym1 = y - 1;
          int xm1 = x - 1;

          float * data_delta1 = data_ptr(delta1, y, x);
          const float * data_epsilon1 = data_ptr(epsilon1, y, x);
          const float * data_next_epsilon_y1 = botBorder ? 0 : data_ptr(workmem1, (iteration-1)%2, x, 0);
          const float * data_next_epsilon_x1 = rightBorder ? 0 : data_ptr(workmem1, (iteration-1)%2, xp1, 1);
          float * data_epsilon_y1 = data_ptr(workmem1, iteration%2, x, 0);
          float * data_epsilon_x1 = data_ptr(workmem1, iteration%2, x, 1);
          const float * data_last_state_y1 = y > 0 ? data_ptr(delta1, ym1, x) + 4 * n_cells : 0;
          const float * data_last_state_x1 = x > 0 ? data_ptr(delta1, y, xm1) + 4 * n_cells : 0;
          const float * data_Y1 = data_ptr(CompleteY, y, x);
          float * valid1 = Ndarray_DEV_DATA(valid_storage) + i * 1 * n_minibatch + 0 * n_minibatch;

          ptrs[0 * 1 * n_outer_batch + 0 * n_outer_batch + i] = data_delta1;
          ptrs[1 * 1 * n_outer_batch + 0 * n_outer_batch + i] = data_epsilon1;
          ptrs[2 * 1 * n_outer_batch + 0 * n_outer_batch + i] = data_next_epsilon_y1;
          ptrs[3 * 1 * n_outer_batch + 0 * n_outer_batch + i] = data_next_epsilon_x1;
          ptrs[4 * 1 * n_outer_batch + 0 * n_outer_batch + i] = data_epsilon_y1;
          ptrs[5 * 1 * n_outer_batch + 0 * n_outer_batch + i] = data_epsilon_x1;
          ptrs[6 * 1 * n_outer_batch + 0 * n_outer_batch + i] = data_last_state_y1;
          ptrs[7 * 1 * n_outer_batch + 0 * n_outer_batch + i] = data_last_state_x1;
          ptrs[8 * 1 * n_outer_batch + 0 * n_outer_batch + i] = data_Y1;
          ptrs[9 * 1 * n_outer_batch + 0 * n_outer_batch + i] = valid1;
        }

        free(h_sizes);

        HANDLE_ERROR(cudaMemcpy(Ndarray_DEV_DATA(valid_storage), valid.data(),
          valid.size() * sizeof(float), cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpy(Ndarray_DEV_DATA(ptr_storage), ptrs.data(),
          ptrs.size() * sizeof(float*), cudaMemcpyHostToDevice));
        float ** ptr_storage_data = reinterpret_cast<float**>(Ndarray_DEV_DATA(ptr_storage));
        float ** data_deltas = ptr_storage_data + 0 * 1 * n_outer_batch;
        const float ** data_epsilons = (const float**) ptr_storage_data + 1 * 1 * n_outer_batch;
        const float ** data_next_epsilon_ys = (const float**) ptr_storage_data + 2 * 1 * n_outer_batch;
        const float ** data_next_epsilon_xs = (const float**) ptr_storage_data + 3 * 1 * n_outer_batch;
        float ** data_epsilon_ys = ptr_storage_data + 4 * 1 * n_outer_batch;
        float ** data_epsilon_xs = ptr_storage_data + 5 * 1 * n_outer_batch;
        const float ** data_last_state_ys = (const float**) ptr_storage_data + 6 * 1 * n_outer_batch;
        const float ** data_last_state_xs = (const float**) ptr_storage_data + 7 * 1 * n_outer_batch;
        const float ** data_Ys = (const float**) ptr_storage_data + 8 * 1 * n_outer_batch;
        const float ** data_valids = (const float**) (ptr_storage_data + 9 * 1 * n_outer_batch);

        start_dev_kernel(lstm_bwd_stable_cell_kernel_batched, (
          data_deltas,
          data_epsilons,
          data_next_epsilon_ys,
          data_next_epsilon_xs,
          data_epsilon_ys,
          data_epsilon_xs,
          data_last_state_ys,
          data_last_state_xs,
          data_Ys,
          data_valids,
          1 * n_outer_batch,
          n_cells,
          n_minibatch
        ));
      }
    """,
    }

    c_fw_code = """
    // X*, V_h, V_v, W, b, ptr_storage_fwd, ptr_storage_bwd, valid, workmem, sizes, DYDummy,
    //   initialState, initialOutput, iteration = input_names (*: inplace)
    // CompleteY, H = output_names

    assert(n_inputs == 15);
    assert(n_outputs == 2);

    Ndarray* X = inputs[0];
    Ndarray* V_h = inputs[1];
    Ndarray* V_v = inputs[2];
    Ndarray* W = inputs[3];
    Ndarray* b = inputs[4];
    Ndarray* ptr_storage_fwd = inputs[5];
    Ndarray* ptr_storage_bwd = inputs[6]; // not used in fwd
    Ndarray* valid = inputs[7];
    Ndarray* workmem = inputs[8]; // not used in fwd
    Ndarray* workmem2 = inputs[9]; // not used in fwd
    Ndarray* sizes = inputs[10];
    Ndarray* DYDummy = inputs[11]; // not used in fwd
    Ndarray* initialState = inputs[12];
    Ndarray* initialOutput = inputs[13];
    Ndarray* iteration = inputs[14];

    assert(sizeof(float) == 4 && "ptr_storage has wrong size if sizeof(float) != 4");
    assert(sizeof(float*) == 8 && "ptr_storage has wrong size if sizeof(float*) != 8");

    Ndarray* CompleteY = *outputs[0];
    Ndarray* H = *outputs[1];

    Ndarray_DIMS_Type X_dim = Ndarray_DIMS(X);
    Ndarray_DIMS_Type W_dim = Ndarray_DIMS(W);
    Ndarray_DIMS_Type V_dim = Ndarray_DIMS(V_h);
    assert(W_dim[1] %% 5 == 0 && "W has wrong shape");
    assert(5 * V_dim[0] == V_dim[1] && "V has wrong shape");
    assert(W_dim[1] == V_dim[1]);
    assert(W_dim[0] == X_dim[3]);
    const long long Y_dim[] = {X_dim[0], X_dim[1], X_dim[2], W_dim[1] / 5};
    const long long H_dim[] = {X_dim[0], X_dim[1], X_dim[2], W_dim[1]};
    const long long height = X_dim[0];
    const long long width = X_dim[1];
    const long long n_minibatch = X_dim[2];
    const long long max_diag_size = std::min(height, width);
    const long long n_diags = width + height - 1;

    //H = XW (+ b, currently always 0)
    fillmat(context, b, H);
    affine_global(X, W, H);

    // The iteration is stored on the GPU, but we need it on the CPU to controll the programm flow (use explicitly
    // provided previous state/output on first iteration). Maybe this could be optimized by storing the tensor
    // directly on the CPU?
    // We only look at the first value of the tensor with shape (batch,), as every entry has the same value by design
    float h_iteration;
    HANDLE_ERROR(cudaMemcpy(&h_iteration, Ndarray_DEV_DATA(iteration), 1*sizeof(float), cudaMemcpyDeviceToHost));

    for(long long diag = 0; diag < n_diags; ++diag)
    {
      int diag_size = min(diag+1, min((long long) abs(n_diags-diag), min(width, height)));
      int y_high = min(diag, height-1);
      int x_low = max(diag-height+1,(long long) 0);
      std::vector<int> ys_h, xs_h, ys_v, xs_v, ys, xs;
      for(int idx = 0; idx < diag_size; ++idx)
      {
        int y = y_high - idx;
        int x = x_low + idx;
        if(x > 0)
        {
          ys_h.push_back(y);
          xs_h.push_back(x);
        }
        if(y > 0 || h_iteration >= 1) {
          ys_v.push_back(y);
          xs_v.push_back(x);
        }
        ys.push_back(y);
        xs.push_back(x);
      }

      affine_y_x_batched_onedir(context, 0, -1,
        CompleteY, V_h, H, ys_h, xs_h, ptr_storage_fwd, height, width);

      // If it's not the first iteration, we need to use the explicitly provided initial output
      if(h_iteration >= 1) {
        assert(ys_v.size() == 1); // Otherwise, the target length would be != 1, we don't support that yet.
        affine_y_x_batched_onedir(context, 0, 0,
          initialOutput, V_v, H, ys_v, xs_v, ptr_storage_fwd, height, width);
      }
      else {
        affine_y_x_batched_onedir(context, -1, 0,
          CompleteY, V_v, H, ys_v, xs_v, ptr_storage_fwd, height, width);
      }

      do_lstm_batched_onedir(context, H, initialState, h_iteration, CompleteY, ys, xs, ptr_storage_fwd, valid, sizes);
    }
    """

    c_bw_code = """
    // X, V_h, V_v, W, b, ptr_storage_fwd, ptr_storage_bwd, valid, workmem, workmem2, sizes, DYDummy, initialState,
    //   initialOutput, iteration, CompleteY, H, DCompleteY, DH = inputs
    // DX, DV_h, DV_v, DW, Db = outputs

    assert(n_inputs == 19);
    assert(n_outputs == 5);

    Ndarray* X = inputs[0];
    Ndarray* V_h = inputs[1];
    Ndarray* V_v = inputs[2];
    Ndarray* W = inputs[3];
    Ndarray* b = inputs[4];
    Ndarray* ptr_storage_fwd = inputs[5]; // not used in bwd
    Ndarray* ptr_storage_bwd = inputs[6];
    Ndarray* valid_storage = inputs[7];
    Ndarray* workmem = inputs[8];
    Ndarray* workmem2 = inputs[9];
    Ndarray* sizes = inputs[10];
    Ndarray* DYDummy = inputs[11];
    Ndarray* initialState = inputs[12];
    Ndarray* initialOutput = inputs[13];
    Ndarray* iteration = inputs[14]; // not used in bwd (only for asserting it's == 0)
    Ndarray* CompleteY = inputs[15];
    Ndarray* H = inputs[16];
    Ndarray* DCompleteY = inputs[17];
    Ndarray* DH = inputs[18];

    Ndarray* DX = *outputs[0];
    Ndarray* DV_h = *outputs[1];
    Ndarray* DV_v = *outputs[2];
    Ndarray* DW = *outputs[3];
    Ndarray* Db = *outputs[4];

    Ndarray_DIMS_Type X_dim = Ndarray_HOST_DIMS(X);
    Ndarray_DIMS_Type Y_dim = Ndarray_HOST_DIMS(CompleteY);
    Ndarray_DIMS_Type Vh_dim = Ndarray_HOST_DIMS(V_h);
    const int height = X_dim[0];
    const int width = X_dim[1];
    const int n_minibatch = X_dim[2];
    const int n_diags = width + height - 1;
    const int max_diag_size = std::min(Y_dim[0], Y_dim[1]);

    Ndarray * delta1 = H;
    Ndarray * epsilon = DYDummy;

    int size = X_dim[0] * X_dim[1] * X_dim[2] * Vh_dim[0] * sizeof(float);
    HANDLE_ERROR(cudaMemcpy(Ndarray_DEV_DATA(epsilon), Ndarray_DEV_DATA(DCompleteY), size, cudaMemcpyDeviceToDevice));

    for(int diag = n_diags-1; diag >= 0; --diag)
    {
      int diag_size = std::min(diag+1, std::min(std::abs(n_diags-diag), std::min(width, height)));
      int y_high = std::min(diag, height-1);
      int x_low = std::max(diag-height+1,0);
      std::vector<int> ys_h, xs_h, ys_v, xs_v, ys, xs;
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

      affine_y_x_batched_onedir(context, 0, 1, delta1, V_h,
        epsilon, ys_h, xs_h, ptr_storage_bwd, height, width, 0, false, true);
      affine_y_x_batched_onedir(context, 1, 0, delta1, V_v,
        epsilon, ys_v, xs_v, ptr_storage_bwd, height, width, 0, false, true);

      do_lstm_bwd_batched_onedir(
        context, delta1, epsilon, CompleteY, workmem,
        X_dim[0], X_dim[2], ys, xs, ptr_storage_bwd, valid_storage, sizes, diag+1);
    }

    //DW = X^T * delta
    affine_global(X, delta1, DW, true, false, 0, 0.0f);
    //important! mind the order, first use X, then update DX, which might be aligned to X
    //DX = delta * W^T
    affine_global(delta1, W, DX, false, true, 0, 0.0f);

    // Currently, the bias is not trained
    //Db = (1 ... 1) * delta

    //copy left/right part to workmem2 and set to 0
    // (could be done more efficient, but profiling shows, it's not worth it)
    Ndarray_DIMS_Type H_dim = Ndarray_HOST_DIMS(H);
    const int block_size = H_dim[2] * H_dim[3];
    for(int y = 0; y < Y_dim[0]; ++y)
    {
      float * workmem2_1_data_ptr = Ndarray_DEV_DATA(workmem2) + y * block_size;
      float * delta1_data_ptr = data_ptr(delta1, y, 0);
      HANDLE_ERROR(cudaMemcpy(
        workmem2_1_data_ptr, delta1_data_ptr, block_size * sizeof(float), cudaMemcpyDeviceToDevice));
      HANDLE_ERROR(cudaMemset(delta1_data_ptr, 0, sizeof(float) * H_dim[2] * H_dim[3]));
    }

    //DV_h = Y[0..end-1]^T * delta[1..end]
    affine_global(CompleteY, delta1, DV_h, true, false, 1, 0.0f);

    //copy left/right part back
    for(int y = 0; y < Y_dim[0]; ++y)
    {
      float * workmem2_1_data_ptr = Ndarray_DEV_DATA(workmem2) + y * block_size;
      float * delta1_data_ptr = data_ptr(delta1, y, 0);
      HANDLE_ERROR(cudaMemcpy(
        delta1_data_ptr, workmem2_1_data_ptr, block_size * sizeof(float), cudaMemcpyDeviceToDevice));
    }

    //DV_v = Y[0..end-1]^T * delta[1..end]
    affine_global(CompleteY, delta1, DV_v, true, false, Y_dim[1], 0.0f);
  """

    cpu_support = False
    code_version = ()


class Chunking(NativeOpGenBase):
    """
    Given an input in 3d (n_time,n_batch,n_dim), we chunk up the time dimension
    in chunks of size chunk_size, every chunk_step frames.
    This results in an 3d output (chunk_size, n_batch * n_chunks, n_dim)
    where n_chunks = floor( max(n_time - chunk_size + chunk_step - 1, 0) / chunk_step ) + 1.
    Examples:
      n_time=1,   chunk_size=50, chunk_step=10 -> n_chunks=1
      n_time=49,  chunk_size=50, chunk_step=10 -> n_chunks=1
      n_time=50,  chunk_size=50, chunk_step=10 -> n_chunks=1
      n_time=51,  chunk_size=50, chunk_step=10 -> n_chunks=2
      n_time=60,  chunk_size=50, chunk_step=10 -> n_chunks=2
      n_time=61,  chunk_size=50, chunk_step=10 -> n_chunks=3
      n_time=99,  chunk_size=50, chunk_step=10 -> n_chunks=6
      n_time=100, chunk_size=50, chunk_step=10 -> n_chunks=6
      n_time=101, chunk_size=50, chunk_step=10 -> n_chunks=7
    """

    in_info = (
        {"name": "input", "ndim": 3, "shape": (None, None, None)},
        {"name": "index", "ndim": 2, "shape": (None, None), "gradient": "disconnected"},
        {
            "name": "output_buffer",
            "ndim": 3,
            "shape": (None, None, None),
            "want_inplace": 0,
            "gradient": "disconnected",
        },
        {"name": "oindex_buffer", "ndim": 2, "shape": (None, None), "want_inplace": 1, "gradient": "disconnected"},
        {
            "name": "chunk_params",  # chunk_size, chunk_step
            "ndim": 1,
            "shape": (2,),
            "need_contiguous": True,
            "gradient": "disconnected",
        },  # (chunk_size, chunk_step)
    )
    out_info = (
        {"name": "output", "ndim": 3, "shape": ((2, 0), (2, 1), (2, 2))},
        {"name": "oindex", "ndim": 2, "shape": ((3, 0), (3, 1))},
    )

    c_extra_support_code = {
        "copy_kernel": """
    DEF_KERNEL
    void copy_kernel(
      float* chunk_params,
      float* input, long in_dim0, long in_dim1, long in_dim2, long in_stride0, long in_stride1, long in_stride2,
      float* index, long idx_stride0, long idx_stride1,
      float* output, long out_dim0, long out_dim1, long out_stride0, long out_stride1, long out_stride2,
      float* oindex, long oidx_stride0, long oidx_stride1
    ) {
      assert_cmp(out_dim1 % in_dim1, ==, 0);
      const long n_chunks = out_dim1 / in_dim1;
      assert_cmp(n_chunks, >, 0);
      const long chunk_size = out_dim0;
      assert_cmp(long(chunk_params[0]), ==, chunk_size);
      const long chunk_step = long(chunk_params[1]);
      assert_cmp(chunk_step, >, 0);
      assert_cmp(chunk_step * (n_chunks - 1) + chunk_size, >=, in_dim0);
      assert_cmp(chunk_step * (n_chunks - 1), <, in_dim0);

      // Iterate over output (chunked) x/y coordinates.
      // In an inner loop, we will loop over z.
      const long max_idx = out_dim0 * out_dim1;
      for(
        long idx = threadIdx.x + blockDim.x * blockIdx.x;
        idx < max_idx;
        idx += gridDim.x * blockDim.x)
      {
        long out_x = idx % out_dim0;  // time
        long out_y = idx / out_dim0;  // batch

        long chunk_idx = out_y % n_chunks;
        long in_y =      out_y / n_chunks;

        long in_x = chunk_step * chunk_idx + out_x;

        if(in_x < in_dim0 && index[in_x * idx_stride0 + in_y * idx_stride1] > 0.1) {
          for(long z = 0; z < in_dim2; ++z)
            output[out_x * out_stride0 + out_y * out_stride1 + z * out_stride2] =
              input[in_x * in_stride0 + in_y * in_stride1 + z * in_stride2];
          oindex[out_x * oidx_stride0 + out_y * oidx_stride1] = 1;
        }
        else {
          for(long z = 0; z < in_dim2; ++z)
            output[out_x * out_stride0 + out_y * out_stride1 + z * out_stride2] = 0;
          oindex[out_x * oidx_stride0 + out_y * oidx_stride1] = 0;
        }
      }
    }
    """
    }

    c_fw_code = """
    assert_cmp(n_inputs, ==, 5);
    assert_cmp(n_outputs, ==, 2);
    Ndarray* input = inputs[0];
    Ndarray* index = inputs[1];
    Ndarray* chunk_params = inputs[4];
    Ndarray* output = *outputs[0];
    Ndarray* oindex = *outputs[1];

    assert_cmp(Ndarray_NDIM(input), ==, 3);
    assert_cmp(Ndarray_NDIM(index), ==, 2);
    assert_cmp(Ndarray_DIMS(input)[0], ==, Ndarray_DIMS(index)[0]);
    assert_cmp(Ndarray_DIMS(input)[1], ==, Ndarray_DIMS(index)[1]);
    assert_cmp(Ndarray_NDIM(chunk_params), ==, 1);
    assert_cmp(Ndarray_DIMS(chunk_params)[0], ==, 2);
    assert_cmp(Ndarray_NDIM(output), ==, 3);
    assert_cmp(Ndarray_NDIM(oindex), ==, 2);
    assert_cmp(Ndarray_DIMS(output)[0], ==, Ndarray_DIMS(oindex)[0]);
    assert_cmp(Ndarray_DIMS(output)[1], ==, Ndarray_DIMS(oindex)[1]);
    assert_cmp(Ndarray_DIMS(output)[2], ==, Ndarray_DIMS(input)[2]);

    start_dev_kernel(copy_kernel, (
      Ndarray_DEV_DATA(chunk_params),
      Ndarray_DEV_DATA(input),
        Ndarray_DIMS(input)[0],
        Ndarray_DIMS(input)[1],
        Ndarray_DIMS(input)[2],
        Ndarray_STRIDE(input, 0),
        Ndarray_STRIDE(input, 1),
        Ndarray_STRIDE(input, 2),
      Ndarray_DEV_DATA(index),
        Ndarray_STRIDE(index, 0),
        Ndarray_STRIDE(index, 1),
      Ndarray_DEV_DATA(output),
        Ndarray_DIMS(output)[0],
        Ndarray_DIMS(output)[1],
        Ndarray_STRIDE(output, 0),
        Ndarray_STRIDE(output, 1),
        Ndarray_STRIDE(output, 2),
      Ndarray_DEV_DATA(oindex),
        Ndarray_STRIDE(oindex, 0),
        Ndarray_STRIDE(oindex, 1)
    ));
    HANDLE_LAST_ERROR();
  """

    code_version = ()

    @staticmethod
    def naive_chunk_start_frames(n_time, chunk_size, chunk_step):
        """
        This is just for documentation / demonstration. Also used by testing code.
        """
        t = 0
        chunk_start_frames = []
        while True:
            chunk_start_frames.append(t)
            if t + chunk_size >= n_time:
                break
            t += chunk_step
        return chunk_start_frames


class UnChunking(NativeOpGenBase):
    """
    This reverses the output from `Chunking`, i.e. chunking the time dimension.
    We get a 3d input (chunk_size, n_batch * n_chunks, n_dim)
    and return an 3d output (n_time, n_batch, n_dim)
    where the chunks are of size chunk_size, every chunk_step frames.
    Because of overlaps, we have to combine the overlapping chunks somehow.
    We will do that with a uniform distribution, i.e. take the mean of all overlaps per frame.
    """

    in_info = (
        {"name": "input", "ndim": 3, "shape": (None, None, None)},
        {"name": "index", "ndim": 2, "shape": (None, None), "gradient": "disconnected"},
        {
            "name": "output_buffer",
            "ndim": 3,
            "shape": (None, None, None),
            "want_inplace": 0,
            "gradient": "disconnected",
        },
        {"name": "oindex_buffer", "ndim": 2, "shape": (None, None), "want_inplace": 1, "gradient": "disconnected"},
        {"name": "ofactors_buffer", "ndim": 2, "shape": (None, None), "want_inplace": 2, "gradient": "disconnected"},
        {
            "name": "chunk_params",
            "ndim": 1,
            "shape": (2,),
            "need_contiguous": True,
            "gradient": "disconnected",
        },  # (chunk_size, chunk_step)
    )
    out_info = (
        {"name": "output", "ndim": 3, "shape": ((2, 0), (2, 1), (2, 2))},
        {"name": "oindex", "ndim": 2, "shape": ((3, 0), (3, 1))},
        {"name": "ofactors", "ndim": 2, "shape": ((4, 0), (4, 1))},
    )

    c_extra_support_code = {
        "unchunk_kernel": """
    DEF_KERNEL
    void unchunk_kernel(
      float* chunk_params,
      float* input, long in_dim0, long in_dim1, long in_dim2, long in_stride0, long in_stride1, long in_stride2,
      float* index, long idx_stride0, long idx_stride1,
      float* output, long out_dim0, long out_dim1, long out_stride0, long out_stride1, long out_stride2,
      float* oindex, long oidx_stride0, long oidx_stride1,
      float* ofactors, long ofac_stride0, long ofac_stride1
    ) {
      assert_cmp(in_dim1 % out_dim1, ==, 0);
      const long n_chunks = in_dim1 / out_dim1;
      assert_cmp(n_chunks, >, 0);
      const long chunk_size = in_dim0;
      assert_cmp(long(chunk_params[0]), ==, chunk_size);
      const long chunk_step = long(chunk_params[1]);
      assert_cmp(chunk_step, >, 0);
      assert_cmp(chunk_step * (n_chunks - 1) + chunk_size, >=, out_dim0);
      assert_cmp(chunk_step * (n_chunks - 1), <, out_dim0);

      // Iterate over output (unchunked) x/y coordinates.
      // In an inner loop, we will loop over z.
      const long max_idx = out_dim0 * out_dim1;
      for(
        long idx = threadIdx.x + blockDim.x * blockIdx.x;
        idx < max_idx;
        idx += gridDim.x * blockDim.x)
      {
        long out_x = idx % out_dim0;  // time
        long out_y = idx / out_dim0;  // batch

        float c = 0;
        for(long z = 0; z < in_dim2; ++z)
          output[out_x * out_stride0 + out_y * out_stride1 + z * out_stride2] = 0;

        // in_x = out_x - chunk_step * chunk_idx,
        // thus in_x < 0           when chunk_idx * chunk_step >  out_x,
        // and  in_x >= chunk_size when chunk_idx * chunk_step <= out_x - chunk_size,
        // thus we need chunk_idx <= out_x / chunk_step,
        // and          chunk_idx > (out_x - chunk_size) / chunk_step.
        // Examples:
        //   out_x=0,  chunk_size=10, chunk_step=4 -> chunk_idx_start,end=0,1
        //   out_x=3,  chunk_size=10, chunk_step=4 -> chunk_idx_start,end=0,1
        //   out_x=4,  chunk_size=10, chunk_step=4 -> chunk_idx_start,end=0,2
        //   out_x=7,  chunk_size=10, chunk_step=4 -> chunk_idx_start,end=0,2
        //   out_x=8,  chunk_size=10, chunk_step=4 -> chunk_idx_start,end=0,3
        //   out_x=9,  chunk_size=10, chunk_step=4 -> chunk_idx_start,end=0,3
        //   out_x=10, chunk_size=10, chunk_step=4 -> chunk_idx_start,end=1,3
        //   out_x=11, chunk_size=10, chunk_step=4 -> chunk_idx_start,end=1,3
        //   out_x=12, chunk_size=10, chunk_step=4 -> chunk_idx_start,end=1,4
        //   out_x=13, chunk_size=10, chunk_step=4 -> chunk_idx_start,end=1,4
        //   out_x=14, chunk_size=10, chunk_step=4 -> chunk_idx_start,end=2,4
        long chunk_idx_start = (out_x - chunk_size + chunk_step) / chunk_step;
        if(chunk_idx_start < 0) chunk_idx_start = 0;
        long chunk_idx_end = out_x / chunk_step + 1;
        if(chunk_idx_end > n_chunks) chunk_idx_end = n_chunks;
        assert_cmp(chunk_idx_start, <, chunk_idx_end);
        for(long chunk_idx = chunk_idx_start; chunk_idx < chunk_idx_end; ++chunk_idx) {
          long in_y = out_y * n_chunks + chunk_idx;
          long in_x = out_x - chunk_step * chunk_idx;
          assert_cmp(in_x, >=, 0);
          assert_cmp(in_x, <, chunk_size);
          if(index[in_x * idx_stride0 + in_y * idx_stride1] > 0.1) {
            c += 1;
            for(long z = 0; z < in_dim2; ++z)
              output[out_x * out_stride0 + out_y * out_stride1 + z * out_stride2] +=
                input[in_x * in_stride0 + in_y * in_stride1 + z * in_stride2];
          }
        }

        if(c > 0.1) {
          for(long z = 0; z < in_dim2; ++z)
            output[out_x * out_stride0 + out_y * out_stride1 + z * out_stride2] /= c;
          oindex[out_x * oidx_stride0 + out_y * oidx_stride1] = 1;
          ofactors[out_x * ofac_stride0 + out_y * ofac_stride1] = 1.0 / c;
        } else {
          oindex[out_x * oidx_stride0 + out_y * oidx_stride1] = 0;
          ofactors[out_x * ofac_stride0 + out_y * ofac_stride1] = 1.0;
        }
      }
    }
    """
    }

    c_fw_code = """
    assert_cmp(n_inputs, ==, 6);
    assert_cmp(n_outputs, ==, 3);
    Ndarray* input = inputs[0];
    Ndarray* index = inputs[1];
    Ndarray* chunk_params = inputs[5];
    Ndarray* output = *outputs[0];
    Ndarray* oindex = *outputs[1];
    Ndarray* ofactors = *outputs[2];

    assert_cmp(Ndarray_NDIM(input), ==, 3);
    assert_cmp(Ndarray_NDIM(index), ==, 2);
    assert_cmp(Ndarray_DIMS(input)[0], ==, Ndarray_DIMS(index)[0]);
    assert_cmp(Ndarray_DIMS(input)[1], ==, Ndarray_DIMS(index)[1]);
    assert_cmp(Ndarray_NDIM(chunk_params), ==, 1);
    assert_cmp(Ndarray_DIMS(chunk_params)[0], ==, 2);
    assert_cmp(Ndarray_NDIM(output), ==, 3);
    assert_cmp(Ndarray_NDIM(oindex), ==, 2);
    assert_cmp(Ndarray_NDIM(ofactors), ==, 2);
    assert_cmp(Ndarray_DIMS(output)[0], ==, Ndarray_DIMS(oindex)[0]);
    assert_cmp(Ndarray_DIMS(output)[1], ==, Ndarray_DIMS(oindex)[1]);
    assert_cmp(Ndarray_DIMS(output)[2], ==, Ndarray_DIMS(input)[2]);
    assert_cmp(Ndarray_DIMS(oindex)[0], ==, Ndarray_DIMS(ofactors)[0]);
    assert_cmp(Ndarray_DIMS(oindex)[1], ==, Ndarray_DIMS(ofactors)[1]);

    start_dev_kernel(unchunk_kernel, (
      Ndarray_DEV_DATA(chunk_params),
      Ndarray_DEV_DATA(input),
        Ndarray_DIMS(input)[0],
        Ndarray_DIMS(input)[1],
        Ndarray_DIMS(input)[2],
        Ndarray_STRIDE(input, 0),
        Ndarray_STRIDE(input, 1),
        Ndarray_STRIDE(input, 2),
      Ndarray_DEV_DATA(index),
        Ndarray_STRIDE(index, 0),
        Ndarray_STRIDE(index, 1),
      Ndarray_DEV_DATA(output),
        Ndarray_DIMS(output)[0],
        Ndarray_DIMS(output)[1],
        Ndarray_STRIDE(output, 0),
        Ndarray_STRIDE(output, 1),
        Ndarray_STRIDE(output, 2),
      Ndarray_DEV_DATA(oindex),
        Ndarray_STRIDE(oindex, 0),
        Ndarray_STRIDE(oindex, 1),
      Ndarray_DEV_DATA(ofactors),
        Ndarray_STRIDE(ofactors, 0),
        Ndarray_STRIDE(ofactors, 1)
    ));
    HANDLE_LAST_ERROR();
  """

    code_version = ()


class SubtensorBatchedIndex(NativeOpGenBase):
    """
    Consider you have:
      idx: 2d (n_time, n_batch) -> idx (in [0..n_dim-1])
      x: 3d (n_time, n_batch, n_dim)
    Then, this op will calculate:
      x[..., idx[...]]: 2d (n_time, n_batch)
    """

    in_info = (
        {"name": "x", "ndim": 3, "shape": (None, None, None), "bw_in_var": {"want_inplace": 0}},
        {"name": "idx", "ndim": 2, "shape": (None, None), "gradient": "disconnected"},
    )
    out_info = ({"name": "y", "ndim": 2, "shape": ((0, 0), (0, 1))},)

    # noinspection PyUnusedLocal,PyPep8Naming
    @classmethod
    def grad_input_map(cls, x, idx, y, DY):
        """
        Map.
        """
        return x, idx, DY

    c_extra_support_code = {
        "select_kernel": """
    DEF_KERNEL
    void select_kernel(
      float* x, long x_dim0, long x_dim1, long x_dim2, long x_stride0, long x_stride1, long x_stride2,
      float* index, long idx_stride0, long idx_stride1,
      float* y, long y_stride0, long y_stride1
    ) {
      const long max_idx = x_dim0 * x_dim1;
      for(
        long idx = threadIdx.x + blockDim.x * blockIdx.x;
        idx < max_idx;
        idx += gridDim.x * blockDim.x)
      {
        long d0 = idx % x_dim0;
        long d1 = idx / x_dim0;
        long d2 = long(index[d0 * idx_stride0 + d1 * idx_stride1]);
        if(d2 < 0) d2 = 0;
        if(d2 >= x_dim2) d2 = x_dim2 - 1;
        y[d0 * y_stride0 + d1 * y_stride1] = x[d0 * x_stride0 + d1 * x_stride1 + d2 * x_stride2];
      }
    }
    """,
        "select_bw_kernel": """
    DEF_KERNEL
    void select_bw_kernel(
      float* Dx, long Dx_dim0, long Dx_dim1, long Dx_dim2, long Dx_stride0, long Dx_stride1, long Dx_stride2,
      float* index, long idx_stride0, long idx_stride1,
      float* Dy, long Dy_stride0, long Dy_stride1
    ) {
      const long max_idx = Dx_dim0 * Dx_dim1;
      for(
        long idx = threadIdx.x + blockDim.x * blockIdx.x;
        idx < max_idx;
        idx += gridDim.x * blockDim.x)
      {
        long d0 = idx % Dx_dim0;
        long d1 = idx / Dx_dim0;
        long d2 = long(index[d0 * idx_stride0 + d1 * idx_stride1]);
        if(d2 < 0) d2 = 0;
        if(d2 >= Dx_dim2) d2 = Dx_dim2 - 1;
        Dx[d0 * Dx_stride0 + d1 * Dx_stride1 + d2 * Dx_stride2] = Dy[d0 * Dy_stride0 + d1 * Dy_stride1];
      }
    }
    """,
    }

    c_fw_code = """
    assert_cmp(n_inputs, ==, 2);
    assert_cmp(n_outputs, ==, 1);
    Ndarray* x = inputs[0];
    Ndarray* idx = inputs[1];
    Ndarray* y = *outputs[0];

    assert_cmp(Ndarray_NDIM(x), ==, 3);
    assert_cmp(Ndarray_NDIM(idx), ==, 2);
    assert_cmp(Ndarray_DIMS(x)[0], ==, Ndarray_DIMS(idx)[0]);
    assert_cmp(Ndarray_DIMS(x)[1], ==, Ndarray_DIMS(idx)[1]);
    assert_cmp(Ndarray_NDIM(y), ==, 2);
    assert_cmp(Ndarray_DIMS(y)[0], ==, Ndarray_DIMS(idx)[0]);
    assert_cmp(Ndarray_DIMS(y)[1], ==, Ndarray_DIMS(idx)[1]);

    start_dev_kernel(select_kernel, (
      Ndarray_DEV_DATA(x),
        Ndarray_DIMS(x)[0],
        Ndarray_DIMS(x)[1],
        Ndarray_DIMS(x)[2],
        Ndarray_STRIDE(x, 0),
        Ndarray_STRIDE(x, 1),
        Ndarray_STRIDE(x, 2),
      Ndarray_DEV_DATA(idx),
        Ndarray_STRIDE(idx, 0),
        Ndarray_STRIDE(idx, 1),
      Ndarray_DEV_DATA(y),
        Ndarray_STRIDE(y, 0),
        Ndarray_STRIDE(y, 1)
    ));
    HANDLE_LAST_ERROR();
  """

    c_bw_code = """
    assert_cmp(n_inputs, ==, 3);
    assert_cmp(n_outputs, ==, 1);
    Ndarray* x = inputs[0];
    Ndarray* idx = inputs[1];
    Ndarray* Dy = inputs[2];
    Ndarray* Dx = *outputs[0];  // inplace on x

    assert_cmp(Ndarray_NDIM(x), ==, 3);
    assert_cmp(Ndarray_NDIM(idx), ==, 2);
    assert_cmp(Ndarray_DIMS(x)[0], ==, Ndarray_DIMS(idx)[0]);
    assert_cmp(Ndarray_DIMS(x)[1], ==, Ndarray_DIMS(idx)[1]);
    assert_cmp(Ndarray_NDIM(Dy), ==, 2);
    assert_cmp(Ndarray_DIMS(Dy)[0], ==, Ndarray_DIMS(idx)[0]);
    assert_cmp(Ndarray_DIMS(Dy)[1], ==, Ndarray_DIMS(idx)[1]);
    assert_cmp(Ndarray_NDIM(Dx), ==, 3);
    assert_cmp(Ndarray_DIMS(Dx)[0], ==, Ndarray_DIMS(x)[0]);
    assert_cmp(Ndarray_DIMS(Dx)[1], ==, Ndarray_DIMS(x)[1]);
    assert_cmp(Ndarray_DIMS(Dx)[2], ==, Ndarray_DIMS(x)[2]);

    Ndarray_set_zero(Dx);
    start_dev_kernel(select_bw_kernel, (
      Ndarray_DEV_DATA(Dx),
        Ndarray_DIMS(Dx)[0],
        Ndarray_DIMS(Dx)[1],
        Ndarray_DIMS(Dx)[2],
        Ndarray_STRIDE(Dx, 0),
        Ndarray_STRIDE(Dx, 1),
        Ndarray_STRIDE(Dx, 2),
      Ndarray_DEV_DATA(idx),
        Ndarray_STRIDE(idx, 0),
        Ndarray_STRIDE(idx, 1),
      Ndarray_DEV_DATA(Dy),
        Ndarray_STRIDE(Dy, 0),
        Ndarray_STRIDE(Dy, 1)
    ));
    HANDLE_LAST_ERROR();
  """


class SparseToDense(NativeOpGenBase):
    """
    Expects a sparse matrix in COOrdinate format,
    where W[s0[i,b],b,s1[i]] = weight[i,b] for all i, and all batches b.
    Will return W (time,batch,dim).
    """

    in_info = (
        {"name": "_initial_W", "ndim": 3, "shape": (None, None, None), "need_contiguous": True, "want_inplace": 0},
        {"name": "s0", "ndim": 2, "shape": (None, None), "need_contiguous": True},
        {"name": "s1", "ndim": 2, "shape": (None, None), "need_contiguous": True},
        {"name": "weight", "ndim": 2, "shape": (None, None), "need_contiguous": True},
        {"name": "mask", "ndim": 2, "shape": (None, None), "need_contiguous": True},
    )
    out_info = ({"name": "W", "ndim": 3, "shape": ((0, 0), (0, 1), (0, 2))},)

    c_extra_support_code = {
        "assign_kernel": """
    DEF_KERNEL
    void assign_kernel(
      float* out, float* s0, float* s1, float* w, float* mask,
      long n_sparse_idx, long n_time, long n_batch, long n_dim)
    {
      long max_idx = n_batch * n_sparse_idx;
      for(
        long idx = threadIdx.x + blockDim.x * blockIdx.x;
        idx < max_idx;
        idx += gridDim.x * blockDim.x)
      {
        if(mask[idx] < 0.1) continue;
        long batch = idx % n_batch;
        long t = (long) s0[idx];
        long j = (long) s1[idx];
        float y = w[idx];
        if(t < 0 || t >= n_time) continue;  // error somehow?
        if(j < 0 || j >= n_dim) continue;  // error somehow?
        long out_idx = t * n_batch * n_dim + batch * n_dim + j;
        out[out_idx] += y;
      }
    }
    """
    }

    c_fw_code = """
    assert(n_inputs == 5);
    assert(n_outputs == 1);
    Ndarray* s0 = inputs[1];
    Ndarray* s1 = inputs[2];
    Ndarray* weight = inputs[3];
    Ndarray* mask = inputs[4];
    Ndarray* out_W = *outputs[0];

    assert(Ndarray_NDIM(s0) == 2);
    assert(Ndarray_NDIM(s1) == 2);
    assert(Ndarray_NDIM(weight) == 2);
    assert(Ndarray_NDIM(mask) == 2);
    assert(Ndarray_NDIM(out_W) == 3);
    int n_sparse_idx = Ndarray_DIMS(s0)[0];
    assert(n_sparse_idx == Ndarray_DIMS(s1)[0]);
    assert(n_sparse_idx == Ndarray_DIMS(weight)[0]);
    assert(n_sparse_idx == Ndarray_DIMS(mask)[0]);
    int n_batch = Ndarray_DIMS(s0)[1];
    assert(n_batch == Ndarray_DIMS(s1)[1]);
    assert(n_batch == Ndarray_DIMS(weight)[1]);
    assert(n_batch == Ndarray_DIMS(mask)[1]);
    assert(n_batch == Ndarray_DIMS(out_W)[1]);
    int n_time = Ndarray_DIMS(out_W)[0];
    int n_dim = Ndarray_DIMS(out_W)[2];

    start_dev_kernel(assign_kernel, (
      Ndarray_DEV_DATA(out_W),
      Ndarray_DEV_DATA(s0),
      Ndarray_DEV_DATA(s1),
      Ndarray_DEV_DATA(weight),
      Ndarray_DEV_DATA(mask),
      n_sparse_idx, n_time, n_batch, n_dim
    ));
    HANDLE_LAST_ERROR();
  """


class MaxAndArgmaxSparse(NativeOpGenBase):
    """
    Expects a sparse matrix in COOrdinate format,
    where W[s0[i,b],s1[i],b] = weight[i,b] for all i, and all batches b.
    It will return the max and argmax for all W[:,:,b]
    over the second axis.
    """

    in_info = (
        {"name": "s0", "ndim": 2, "shape": (None, None), "need_contiguous": True, "gradient": "disconnected"},
        {"name": "s1", "ndim": 2, "shape": (None, None), "need_contiguous": True, "gradient": "disconnected"},
        {"name": "weight", "ndim": 2, "shape": (None, None), "need_contiguous": True},
        {"name": "mask", "ndim": 2, "shape": (None, None), "need_contiguous": True, "gradient": "disconnected"},
        {
            "name": "_out_max",
            "ndim": 2,
            "shape": (None, None),
            "need_contiguous": True,
            "want_inplace": 0,
            "gradient": "disconnected",
        },
        {
            "name": "_out_arg",
            "ndim": 2,
            "shape": (None, None),
            "need_contiguous": True,
            "want_inplace": 1,
            "gradient": "disconnected",
        },
    )
    out_info = (
        {"name": "out_max", "ndim": 2, "shape": ((4, 0), (4, 1))},
        {"name": "out_arg", "ndim": 2, "shape": ((5, 0), (5, 1))},
    )

    c_extra_support_code = {
        "doit_kernel": """
    DEF_KERNEL
    void doit_kernel(
        long n_batch, long n_in_time, long n_out_time,
        float* s0, float* s1, float* weight, float* mask,
        float* out_max, float* out_arg) {
      long batch_idx = threadIdx.x + blockDim.x * blockIdx.x;
      while(batch_idx < n_batch) {
        for(long i = 0; i < n_in_time; ++i) {
          long idx = i * n_batch + batch_idx;
          if(mask[idx] < 0.1) continue;
          long t = (long) s0[idx];
          long j = (long) s1[idx];
          float w = weight[idx];
          if(t < 0 || t >= n_out_time) continue;  // error somehow?
          long out_idx = t * n_batch + batch_idx;
          if(w > out_max[out_idx]) {
            out_max[out_idx] = w;
            out_arg[out_idx] = (float) j;
          }
        }
        batch_idx += gridDim.x * blockDim.x;
      }
    }
    """
    }

    c_fw_code = """
    assert(n_inputs == 6);
    assert(n_outputs == 2);
    Ndarray* s0 = inputs[0];
    Ndarray* s1 = inputs[1];
    Ndarray* weight = inputs[2];
    Ndarray* mask = inputs[3];
    Ndarray* out_max = *outputs[0];
    Ndarray* out_arg = *outputs[1];

    assert(Ndarray_NDIM(s0) == 2);
    assert(Ndarray_NDIM(s1) == 2);
    assert(Ndarray_NDIM(weight) == 2);
    assert(Ndarray_NDIM(mask) == 2);
    assert(Ndarray_NDIM(out_max) == 2);
    assert(Ndarray_NDIM(out_arg) == 2);
    int n_in_time = Ndarray_DIMS(s0)[0];
    assert(n_in_time == Ndarray_DIMS(s1)[0]);
    assert(n_in_time == Ndarray_DIMS(weight)[0]);
    assert(n_in_time == Ndarray_DIMS(mask)[0]);
    int n_batch = Ndarray_DIMS(s0)[1];
    assert(n_batch == Ndarray_DIMS(s1)[1]);
    assert(n_batch == Ndarray_DIMS(weight)[1]);
    assert(n_batch == Ndarray_DIMS(mask)[1]);
    assert(n_batch == Ndarray_DIMS(out_arg)[1]);
    assert(n_batch == Ndarray_DIMS(out_max)[1]);
    int n_out_time = Ndarray_DIMS(out_arg)[0];
    assert(n_out_time == Ndarray_DIMS(out_max)[0]);
    assert(out_max != out_arg);  // earlier bug in NativeOp

    start_dev_kernel(doit_kernel, (
      n_batch, n_in_time, n_out_time,
      Ndarray_DEV_DATA(s0),
      Ndarray_DEV_DATA(s1),
      Ndarray_DEV_DATA(weight),
      Ndarray_DEV_DATA(mask),
      Ndarray_DEV_DATA(out_max),
      Ndarray_DEV_DATA(out_arg)
    ));
    HANDLE_LAST_ERROR();
  """

    code_version = ()


class CrossEntropySoftmaxAndGradientZSparse(NativeOpGenBase):
    """
    y_target is given in sparse COOrdinate format.
    We will calculate CE[t,b] = \\sum_i y_target[t,b,i] * log(softmax(z[t,b])[i]),
    for any timeframe t and batch b,
    and grad(CE[t,b], z[t,b]) = softmax(z[t,b]) - y_target[t,b].
    We also support an index-mask for z, i.e. for the possible [t,b].
    """

    in_info = (
        {"name": "z", "ndim": 3, "shape": (None, None, None), "need_contiguous": True},
        {"name": "z_mask", "ndim": 2, "shape": (None, None), "need_contiguous": True},
        {"name": "y_target_t", "ndim": 2, "shape": (None, None), "need_contiguous": True},
        {"name": "y_target_i", "ndim": 2, "shape": (None, None), "need_contiguous": True},
        {"name": "y_target_w", "ndim": 2, "shape": (None, None), "need_contiguous": True},
        {"name": "y_target_mask", "ndim": 2, "shape": (None, None), "need_contiguous": True},
    )
    out_info = (
        {"name": "out_ce", "ndim": 2, "shape": ((0, 0), (0, 1))},
        {"name": "out_grad_z", "ndim": 3, "shape": ((0, 0), (0, 1), (0, 2))},
        {"name": "_out_max_z", "ndim": 2, "shape": ((0, 0), (0, 1))},
    )

    c_extra_support_code = {
        "max_kernel": """
    DEF_KERNEL
    void max_kernel(float* out, float* v, float* mask, long stride, long max_idx) {
      for(
        long idx = threadIdx.x + blockDim.x * blockIdx.x;
        idx < max_idx;
        idx += gridDim.x * blockDim.x)
      {
        if(mask[idx] < 0.1)
          continue;
        long start = idx * stride;
        float last_max = v[start];
        out[idx] = last_max;
        for(long i = 1; i < stride; ++i) {
          float cur = v[start + i];
          if(cur > last_max) {
            last_max = cur;
            out[idx] = cur;
          }
        }
      }
    }
    """,
        "softmax_kernel": """
    DEF_KERNEL
    void softmax_kernel(
      float* out_softmax,
      float* z, float* max_z, float* mask,
      long stride, long max_idx)
    {
      for(
        long idx = threadIdx.x + blockDim.x * blockIdx.x;
        idx < max_idx;
        idx += gridDim.x * blockDim.x)
      {
        long start = idx * stride;
        float s = 0;
        for(long i = 0; i < stride; ++i) {
          s += exp(z[start + i] - max_z[idx]);
        }
        if(s < 1e-16) s = 1e-16;
        for(long i = 0; i < stride; ++i) {
          float y = exp(z[start + i] - max_z[idx]) / s;
          out_softmax[start + i] = (mask[idx] > 0.5) ? y : 0;
        }
      }
    }
    """,
        "ce_sm_grad_kernel": """
    DEF_KERNEL
    void ce_sm_grad_kernel(
      float* out_ce, float* out_grad_z,
      float* z, float* max_z, float* z_mask,
      float* s0, float* s1, float* w, float* s_mask,
      long n_time, long n_batch, long n_dim, long n_sparse_index)
    {
      long max_idx = n_batch * n_sparse_index;
      for(
        long idx = threadIdx.x + blockDim.x * blockIdx.x;
        idx < max_idx;
        idx += gridDim.x * blockDim.x)
      {
        if(s_mask[idx] < 0.1) continue;
        long batch = idx % n_batch;
        long t = (long) s0[idx];
        long j = (long) s1[idx];
        float y_target = w[idx];
        if(t < 0 || t >= n_time) continue;  // error somehow?
        if(j < 0 || j >= n_dim) continue;  // error somehow?
        long out_ce_idx = t * n_batch + batch;
        long out_y_idx = t * n_batch * n_dim + batch * n_dim + j;
        // This assumes that out_grad_z is still softmax(z).
        // This also assumes that every [t,j] is only represented once in the sparse data.
        out_ce[out_ce_idx] -= y_target * log(fmax(out_grad_z[out_y_idx], 1e-30f));
        out_grad_z[out_y_idx] -= y_target;
      }
    }
    """,
    }

    c_fw_code = """
    assert(n_inputs == 6);
    assert(n_outputs == 3);
    Ndarray* z = inputs[0];
    Ndarray* z_mask = inputs[1];
    Ndarray* s0 = inputs[2];
    Ndarray* s1 = inputs[3];
    Ndarray* w = inputs[4];
    Ndarray* s_mask = inputs[5];
    Ndarray* out_ce = *outputs[0];
    Ndarray* out_grad_z = *outputs[1];
    Ndarray* out_max_z = *outputs[2];

    assert(Ndarray_NDIM(z) == 3);
    assert(Ndarray_NDIM(z_mask) == 2);
    assert(Ndarray_NDIM(out_ce) == 2);
    assert(Ndarray_NDIM(out_grad_z) == 3);
    assert(Ndarray_NDIM(out_max_z) == 2);
    assert(Ndarray_NDIM(s0) == 2);
    assert(Ndarray_NDIM(s1) == 2);
    assert(Ndarray_NDIM(w) == 2);
    assert(Ndarray_NDIM(out_ce) == 2);
    int n_time = Ndarray_DIMS(z)[0];
    int n_batch = Ndarray_DIMS(z)[1];
    int n_dim = Ndarray_DIMS(z)[2];
    assert(n_time == Ndarray_DIMS(z_mask)[0]);
    assert(n_time == Ndarray_DIMS(out_ce)[0]);
    assert(n_time == Ndarray_DIMS(out_grad_z)[0]);
    assert(n_time == Ndarray_DIMS(out_max_z)[0]);
    assert(n_batch == Ndarray_DIMS(z_mask)[1]);
    assert(n_batch == Ndarray_DIMS(out_ce)[1]);
    assert(n_batch == Ndarray_DIMS(out_grad_z)[1]);
    assert(n_batch == Ndarray_DIMS(out_max_z)[1]);
    assert(n_batch == Ndarray_DIMS(s0)[1]);
    assert(n_batch == Ndarray_DIMS(s1)[1]);
    assert(n_batch == Ndarray_DIMS(w)[1]);
    assert(n_batch == Ndarray_DIMS(s_mask)[1]);
    assert(n_dim == Ndarray_DIMS(out_grad_z)[2]);
    int n_sparse_index = Ndarray_DIMS(s0)[0];
    assert(n_sparse_index == Ndarray_DIMS(s1)[0]);
    assert(n_sparse_index == Ndarray_DIMS(w)[0]);
    assert(n_sparse_index == Ndarray_DIMS(s_mask)[0]);

    start_dev_kernel(max_kernel, (
      Ndarray_DEV_DATA(out_max_z), Ndarray_DEV_DATA(z), Ndarray_DEV_DATA(z_mask),
      n_dim, n_time * n_batch
    ));
    HANDLE_LAST_ERROR();
    Ndarray_set_zero(out_ce);
    start_dev_kernel(softmax_kernel, (
      Ndarray_DEV_DATA(out_grad_z),
      Ndarray_DEV_DATA(z), Ndarray_DEV_DATA(out_max_z), Ndarray_DEV_DATA(z_mask),
      n_dim, n_time * n_batch
    ));
    HANDLE_LAST_ERROR();
    start_dev_kernel(ce_sm_grad_kernel, (
      Ndarray_DEV_DATA(out_ce), Ndarray_DEV_DATA(out_grad_z),
      Ndarray_DEV_DATA(z), Ndarray_DEV_DATA(out_max_z), Ndarray_DEV_DATA(z_mask),
      Ndarray_DEV_DATA(s0), Ndarray_DEV_DATA(s1), Ndarray_DEV_DATA(w), Ndarray_DEV_DATA(s_mask),
      n_time, n_batch, n_dim, n_sparse_index
    ));
    HANDLE_LAST_ERROR();
  """


common_fast_bw_kernels = {
    "001_set_start_states": """
    DEF_KERNEL
    void set_start_states(float* states, unsigned* start_states) {
      unsigned state_idx = start_states[blockIdx.x * blockDim.x + threadIdx.x];
      states[state_idx] = 0.0;
    }
  """,
    "010_fill_array": """
    DEF_KERNEL
    void fill_array(float* array, float value, unsigned size) {
      unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
      if (idx < size) {
        array[idx] = value;
      }
    }
  """,
    "011_remove_inf": """
  DEF_KERNEL
  void remove_inf(float* array, unsigned size) {
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
      array[idx] = fminf(array[idx], 1e32);
    }
  }
  """,
    "012_prob_add": """
    DEV_FUNC
    float prob_add(float a, float b) {
      float diff = a - b;
      if (isnan(diff)) {
        return INF_F;
      }
      else {
        return -log1pf(expf(-fabsf(diff))) + fminf(a, b);
      }
    }
  """,
    "013_atomic_prob_add": """
    DEV_FUNC
    void atomic_prob_add(float* a, float b) {
      int* addr = (int*)a;
      int old   = float_as_int(*a);
      int assumed;
      do {
        assumed = old;
        old     = elem_atomic_cas(addr, assumed, float_as_int(prob_add(int_as_float(old), b)));
      } while (old != assumed);
    }
  """,
    "020_dump_to_file": """
    template<typename T>
    void dump_to_file_1d(T* d_mem, unsigned n_d1, std::string const& path) {
      std::vector<T> buffer(n_d1);
      //cudaMemcpy(buffer.data(), d_mem, buffer.size() * sizeof(T), cudaMemcpyDeviceToHost);

      std::ofstream output(path.c_str(), std::ios::trunc | std::ios::out);
      for (size_t i1 = 0ul; i1 < n_d1; i1++) {
        T val = buffer[i1];
        if (!std::numeric_limits<T>::has_infinity or !isinf(val)) {
          output << i1 << ' ' << val << '\\n';
        }
      }
    }

    template<typename T>
    void dump_to_file_2d(T* d_mem, unsigned n_d1, unsigned n_d2, std::string const& path) {
      std::vector<T> buffer(n_d1 * n_d2);
      //cudaMemcpy(buffer.data(), d_mem, buffer.size() * sizeof(T), cudaMemcpyDeviceToHost);

      std::ofstream output(path.c_str(), std::ios::trunc | std::ios::out);
      for (size_t i1 = 0ul; i1 < n_d1; i1++) {
        for (size_t i2 = 0ul; i2 < n_d2; i2++) {
          T val = buffer[i1 * n_d2 + i2];
          if (!std::numeric_limits<T>::has_infinity or !isinf(val)) {
            output << i1 << ' ' << i2 << ' ' << val << '\\n';
          }
        }
      }
    }

    template<typename T>
    void dump_to_file_3d(T* d_mem, unsigned n_d1, unsigned n_d2, unsigned n_d3, std::string const& path) {
      std::vector<T> buffer(n_d1 * n_d2 * n_d3);
      //cudaMemcpy(buffer.data(), d_mem, buffer.size() * sizeof(T), cudaMemcpyDeviceToHost);

      std::ofstream output(path.c_str(), std::ios::trunc | std::ios::out);
      for (size_t i1 = 0ul; i1 < n_d1; i1++) {
        for (size_t i2 = 0ul; i2 < n_d2; i2++) {
          for (size_t i3 = 0ul; i3 < n_d3; i3++) {
            T val = buffer[i1 * n_d2 * n_d3 + i2 * n_d3 + i3];
            if (!std::numeric_limits<T>::has_infinity or !isinf(val)) {
              output << i1 << ' ' << i2 << ' ' << i3 << ' ' << val << '\\n';
            }
          }
        }
      }
    }
  """,
}


class FastBaumWelchOp(NativeOpGenBase):
    # noinspection PyUnresolvedReferences
    """
    inputs:
      :param am_scores: scores in -log space. 3d (time,batch,dim)
      :param edges: edges of the graph (from,to,emission_idx,sequence_idx)
      :param weights: weights of the edges
    outputs:
      :param output: Baum-Welch alignment, scores in -log space. 3d (time,batch,dim), like am_scores
    """
    in_info = (
        {
            "name": "am_scores",
            "ndim": 3,
            "shape": (None, None, None),
            "need_contiguous": True,
            "gradient": "disconnected",
        },
        {
            "name": "edges",
            "ndim": 2,
            "shape": (None, None),
            "dtype": "int32",
            "need_contiguous": True,
            "gradient": "disconnected",
        },
        {"name": "weights", "ndim": 1, "shape": (None,), "need_contiguous": True, "gradient": "disconnected"},
        {
            "name": "start_end_states",
            "ndim": 2,
            "shape": (2, None),
            "dtype": "int32",
            "need_contiguous": True,
            "gradient": "disconnected",
        },
        {"name": "index", "ndim": 2, "shape": ((0, 0), (0, 1)), "need_contiguous": True, "gradient": "disconnected"},
        {"name": "state_buffer", "ndim": 2, "shape": (2, None), "need_contiguous": True, "gradient": "disconnected"},
    )
    out_info = (
        {"name": "output", "ndim": 3, "shape": ((0, 0), (0, 1), (0, 2)), "need_contiguous": True},
        {"name": "sums", "ndim": 2, "shape": ((0, 0), (0, 1)), "need_contiguous": True},
    )

    c_extra_support_code = copy.copy(common_fast_bw_kernels)
    c_extra_support_code.update(
        {
            "100_init_bwd_state_buffer": """
      DEF_KERNEL
      void init_bwd_state_buffer(
          float* states, unsigned* end_states, unsigned t, unsigned max_t, float* index, unsigned index_stride) {
        unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (index[t * index_stride + idx] == 1.0 && (t == max_t || index[(t + 1) * index_stride + idx] == 0.0)) {
          unsigned state_idx = end_states[idx];
          states[state_idx] = 0.0;
        }
      }
    """,
            "101_next_frame": """
      DEF_KERNEL
      void next_frame(bool fwd, unsigned num_edges, unsigned  num_emissions,
                      unsigned* sequence_idxs, unsigned* from_buffer, unsigned* to_buffer, float* weight_buffer,
                      unsigned* emission_idxs,
                      float* prev_frame, float* next_frame, float* am_scores, float* edge_buffer) {
        unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_edges) {
          return;
        }

        unsigned from     = from_buffer  [idx];
        float    prev_val = prev_frame[from];
        if (isinf(prev_val)) {
          edge_buffer[idx] = INF_F;
          return;
        }

        unsigned to           = to_buffer    [idx];
        unsigned emission_idx = emission_idxs[idx];
        float    edge_weight  = weight_buffer[idx];
        unsigned sequence_idx = sequence_idxs[idx];

        float val = prev_val + edge_weight + am_scores[sequence_idx * num_emissions + emission_idx];

        if (fwd) {
          edge_buffer[idx] += val;
        }
        else {
          edge_buffer[idx] += prev_val;
        }
        atomic_prob_add(next_frame + to, val);
      }
    """,
            "102_normalize": """
      DEF_KERNEL
      void normalize(float* buffer, unsigned* sequence_idxs, unsigned num_edges, unsigned num_seqs, float* sum_output) {
        DEF_SHARED(float, sum);

        buffer += blockIdx.x * num_edges;

        for (unsigned s = 0u; s < num_seqs; s++) {
          sum[s] = INF_F;
        }

        for (unsigned e = 0u; e < num_edges; e++) {
          unsigned s = sequence_idxs[e];
          sum[s] = prob_add(sum[s], buffer[e]);
        }

        for (unsigned s = 0ul; s < num_seqs; s++) {
          if (isinf(sum[s])) {
            // if the frame is empty (happens due to batching of seqs with unequal length), set it to 0
            sum_output[blockIdx.x * num_seqs + s] = 0.0;
          }
          else {
            sum_output[blockIdx.x * num_seqs + s] = sum[s];
          }
        }

        for (unsigned e = 0u; e < num_edges; e++) {
          unsigned s = sequence_idxs[e];
          buffer[e] -= sum[s];
        }
      }
    """,
            "103_compute_result": """
      DEF_KERNEL
      void compute_result(float* edge_buffer, float* out, unsigned* emission_idxs, unsigned* sequence_idxs,
                          unsigned frame_stride, unsigned seq_stride,
                          unsigned num_frames, unsigned num_seqs, unsigned num_edges) {
        unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_frames * num_edges) {
          return;
        }

        unsigned e_idx        = idx % num_edges;
        unsigned frame        = idx / num_edges;
        unsigned emission_idx = emission_idxs[e_idx];
        unsigned seq_idx      = sequence_idxs[e_idx];
        float    score        = edge_buffer[idx];

        atomic_prob_add(out + frame * frame_stride + seq_idx * seq_stride + emission_idx, score);
      }
    """,
            "110_write_alignment_to_file": """
      void write_alignment_to_file(float* d_state_buffer, float* d_index, unsigned index_stride,
                                   unsigned* d_start_states, unsigned* d_end_states,
                                   float pruning, unsigned n_frames, unsigned n_seqs, unsigned n_states,
                                   unsigned batch_idx) {
        std::vector<float>    state_buffer((n_frames + 1u) * n_states);
        std::vector<float>    index       (n_frames * index_stride);
        std::vector<unsigned> start_states(n_seqs);
        std::vector<unsigned> end_states  (n_seqs);

        //HANDLE_ERROR(cudaMemcpy(
        //  state_buffer.data(), d_state_buffer, state_buffer.size() * sizeof(float), cudaMemcpyDeviceToHost));
        //HANDLE_ERROR(cudaMemcpy(
        //  index.data(),        d_index,        index.size()        * sizeof(float), cudaMemcpyDeviceToHost));
        //HANDLE_ERROR(cudaMemcpy(
        //  start_states.data(), d_start_states, start_states.size() * sizeof(float), cudaMemcpyDeviceToHost));
        //HANDLE_ERROR(cudaMemcpy(
        //  end_states.data(),   d_end_states,   end_states.size()   * sizeof(float), cudaMemcpyDeviceToHost));

        for (unsigned seq = 0u; seq < n_seqs; seq++) {
          std::stringstream filename;
          filename << "alignment.dump." << batch_idx << '.' << seq;
          std::ofstream out(filename.str().c_str(), std::ios::out | std::ios::trunc);
          for (unsigned t = 0u; t < n_frames; t++) {
            if (t > 0u && index[seq * index_stride + t] <= 0.0) {
              break;
            }
            float sum = std::numeric_limits<float>::infinity();
            for (unsigned s = start_states[seq]; s <= end_states[seq]; s++) {
              const float val = state_buffer[t * n_states + s];
              float diff = val - sum;
              if (!isnan(diff)) {
                sum = -log1p(exp(-abs(diff))) + fminf(sum, val);
              }
            }
            for (unsigned s = start_states[seq]; s <= end_states[seq]; s++) {
              const float val = state_buffer[t * n_states + s] - sum;
              if (val <= pruning) {
                out << t << ' ' << (s - start_states[seq]) << ' ' << val << '\\n';
              }
            }
          }
        }
      }
    """,
            "111_write_output_to_file": """
      void write_output_to_file(float* d_out, float* d_index, unsigned index_stride,
                                float pruning, unsigned n_frames, unsigned n_seqs, unsigned n_emissions,
                                unsigned batch_idx) {
        std::vector<float> buffer(n_frames * n_seqs * n_emissions);
        std::vector<float> index (n_frames * index_stride);

        //HANDLE_ERROR(cudaMemcpy(buffer.data(), d_out,   buffer.size() * sizeof(float), cudaMemcpyDeviceToHost));
        //HANDLE_ERROR(cudaMemcpy(index.data(),  d_index, index.size()  * sizeof(float), cudaMemcpyDeviceToHost));

        for (unsigned seq = 0u; seq < n_seqs; seq++) {
          std::stringstream filename;
          filename << "target.dump." << batch_idx << '.' << seq;
          std::ofstream out(filename.str().c_str(), std::ios::out | std::ios::trunc);
          for (unsigned t = 0u; t < n_frames; t++) {
            if (t > 0u && index[seq * index_stride + t] <= 0.0) {
              break;
            }
            for (unsigned e = 0u; e < n_emissions; e++) {
              const float val = buffer[t * n_seqs * n_emissions + seq * n_emissions + e];
              if (val <= pruning) {
                out << t << ' ' << e << ' ' << val << '\\n';
              }
            }
          }
        }
      }
    """,
        }
    )

    c_fw_code = """
    // am_scores, edges, weights, start_end_states, index, state_buffer* = input_names (*: inplace)
    // output = output_names
    assert(n_inputs  == 6);
    assert(n_outputs == 2);
    Ndarray* am_scores        = inputs[0];
    Ndarray* edges            = inputs[1];
    Ndarray* weights          = inputs[2];
    Ndarray* start_end_states = inputs[3];
    Ndarray* index            = inputs[4];
    Ndarray* state_buffer     = inputs[5];
    Ndarray* out              = *outputs[0];
    Ndarray* sum_output       = *outputs[1];

    /*
    debug_print(context, am_scores, "am_scores");
    debug_print(context, edges, "edges");
    debug_print(context, weights, "weights");
    debug_print(context, start_end_states, "start_end_states");
    debug_print(context, index, "index");
    debug_print(context, state_buffer, "state_buffer");
    */

    assert_cmp(Ndarray_DIMS(am_scores)[0], ==, Ndarray_DIMS(out)[0]);
    assert_cmp(Ndarray_DIMS(am_scores)[1], ==, Ndarray_DIMS(out)[1]);
    assert_cmp(Ndarray_DIMS(am_scores)[2], ==, Ndarray_DIMS(out)[2]);
    assert_cmp(Ndarray_DIMS(am_scores)[1], ==, Ndarray_DIMS(start_end_states)[1]);

    assert_cmp(Ndarray_DIMS(sum_output)[0], ==, Ndarray_DIMS(am_scores)[0]);
    assert_cmp(Ndarray_DIMS(sum_output)[1], ==, Ndarray_DIMS(am_scores)[1]);

    bool            dump_alignment = false;
    bool            dump_output    = false;
    unsigned        dump_every = 40u;
    static unsigned batch_idx  = 0u;
    float           pruning    = 10.f;

    unsigned* d_from = reinterpret_cast<unsigned*>(Ndarray_DEV_DATA_int32(edges)
      + 0 * Ndarray_STRIDE(edges, 0));
    unsigned* d_to = reinterpret_cast<unsigned*>(Ndarray_DEV_DATA_int32(edges)
      + 1 * Ndarray_STRIDE(edges, 0));
    unsigned* d_emission_idxs = reinterpret_cast<unsigned*>(Ndarray_DEV_DATA_int32(edges)
      + 2 * Ndarray_STRIDE(edges, 0));
    unsigned* d_sequence_idxs = reinterpret_cast<unsigned*>(Ndarray_DEV_DATA_int32(edges)
      + 3 * Ndarray_STRIDE(edges, 0));
    float*    d_weights = Ndarray_DEV_DATA(weights);
    float*    d_am_scores = Ndarray_DEV_DATA(am_scores);
    unsigned* d_start_states = reinterpret_cast<unsigned*>(Ndarray_DEV_DATA_int32(start_end_states)
      + 0 * Ndarray_STRIDE(start_end_states, 0));
    unsigned* d_end_states = reinterpret_cast<unsigned*>(Ndarray_DEV_DATA_int32(start_end_states)
      + 1 * Ndarray_STRIDE(start_end_states, 0));
    float*    d_index             = Ndarray_DEV_DATA(index);
    float*    d_state_buffer_prev = Ndarray_DEV_DATA(state_buffer) + 0 * Ndarray_STRIDE(state_buffer, 0);
    float*    d_state_buffer_next = Ndarray_DEV_DATA(state_buffer) + 1 * Ndarray_STRIDE(state_buffer, 0);
    float*    d_out               = Ndarray_DEV_DATA(out);
    float*    d_sum_output        = Ndarray_DEV_DATA(sum_output);

    unsigned n_frames    = Ndarray_DIMS(am_scores)[0];
    unsigned n_seqs      = Ndarray_DIMS(am_scores)[1];
    unsigned n_emissions = Ndarray_DIMS(am_scores)[2];
    unsigned n_states    = Ndarray_DIMS(state_buffer)[1];
    unsigned n_edges     = Ndarray_DIMS(edges)[1];
    unsigned n_threads   = 1024u;
    unsigned n_blocks    = (n_edges + n_threads - 1) / n_threads;

    unsigned frame_stride    = Ndarray_STRIDE(am_scores, 0);
    unsigned sequence_stride = Ndarray_STRIDE(am_scores, 1);
    unsigned index_stride    = Ndarray_STRIDE(index, 0);

    assert_cmp(n_frames, >, 0);
    assert_cmp(n_states, >, 0);
    //std::cerr << "n_frames: "    << n_frames    << std::endl;
    //std::cerr << "n_seqs: "      << n_seqs      << std::endl;
    //std::cerr << "n_emissions: " << n_emissions << std::endl;
    //std::cerr << "n_states: "    << n_states    << std::endl;
    //std::cerr << "n_edges: "     << n_edges     << std::endl;
    //std::cerr << "n_threads: "   << n_threads   << std::endl;
    //std::cerr << "n_blocks: "    << n_blocks    << std::endl;

    //std::cerr << "frame_stride: "     << frame_stride    << std::endl;
    //std::cerr << "sequnence_stride: " << sequence_stride << std::endl;
    //std::cerr << "index_stride: "     << index_stride    << std::endl;

    // initialize edge buffer
    float* d_edge_buffer = reinterpret_cast<float*>(device_malloc(n_edges * n_frames * sizeof(float)));
    if(!d_edge_buffer) { HANDLE_LAST_ERROR(); abort(); }  // error should have been set in device_malloc
    unsigned n_fill_blocks = (n_edges * n_frames + n_threads - 1u) / n_threads;
    start_dev_kernel2(fill_array, n_fill_blocks, n_threads, 0, (d_edge_buffer, 0.0, n_edges * n_frames));
    HANDLE_LAST_ERROR();

    // initialize the state buffer
    n_fill_blocks = (n_states + n_threads - 1u) / n_threads;
    start_dev_kernel2(
      fill_array, n_fill_blocks, n_threads, 0,
      (d_state_buffer_prev, std::numeric_limits<float>::infinity(), n_states));
    HANDLE_LAST_ERROR();
    start_dev_kernel2(set_start_states, 1, n_seqs, 0, (d_state_buffer_prev, d_start_states));
    HANDLE_LAST_ERROR();

    // initialize full state buffer (only used to dump the alignment)
    float* d_state_buffer_all = NULL;
    if (dump_alignment && batch_idx %% dump_every == 0) {
      d_state_buffer_all = reinterpret_cast<float*>(device_malloc(n_states * (n_frames + 1u) * sizeof(float)));
      if(!d_state_buffer_all) { HANDLE_LAST_ERROR(); abort(); }  // error should have been set in device_malloc
      Ndarray_memcpy(d_state_buffer_all, d_state_buffer_prev, n_states * sizeof(float));
      HANDLE_LAST_ERROR();
    }

    // fwd pass
    for (unsigned t = 0u; t < n_frames; t++) {
      start_dev_kernel2(
        fill_array, n_fill_blocks, n_threads, 0,
        (d_state_buffer_next, std::numeric_limits<float>::infinity(), n_states));
      HANDLE_LAST_ERROR();
      start_dev_kernel2(next_frame, n_blocks, n_threads, 0,
        (true, n_edges, sequence_stride,
         d_sequence_idxs, d_from, d_to, d_weights, d_emission_idxs,
         d_state_buffer_prev, d_state_buffer_next, d_am_scores + t * frame_stride, d_edge_buffer + t * n_edges));
      HANDLE_LAST_ERROR();
      if (dump_alignment && batch_idx %% dump_every == 0) {
        Ndarray_memcpy(d_state_buffer_all + (t + 1u) * n_states, d_state_buffer_next, n_states * sizeof(float));
        HANDLE_LAST_ERROR();
      }
      std::swap(d_state_buffer_prev, d_state_buffer_next);
    }

    // bwd pass
    start_dev_kernel2(
      fill_array, n_fill_blocks, n_threads, 0,
      (d_state_buffer_prev, std::numeric_limits<float>::infinity(), n_states));
    HANDLE_LAST_ERROR();
    for (unsigned t = n_frames; t > 0; t--) {
      start_dev_kernel2(init_bwd_state_buffer, 1, n_seqs, 0,
        (d_state_buffer_prev, d_end_states, t - 1, n_frames - 1, d_index, index_stride));
      HANDLE_LAST_ERROR();
      if (dump_alignment && batch_idx %% dump_every == 0) {
        float alpha = 1.0f;
        //HANDLE_ERROR(cublasSaxpy(
        //  handle, n_states, &alpha, d_state_buffer_prev, 1, d_state_buffer_all + t * n_states, 1));
      }
      start_dev_kernel2(
        fill_array, n_fill_blocks, n_threads, 0,
        (d_state_buffer_next, std::numeric_limits<float>::infinity(), n_states));
      HANDLE_LAST_ERROR();
      start_dev_kernel2(next_frame, n_blocks, n_threads, 0,
        (false, n_edges, sequence_stride,
         d_sequence_idxs, d_to, d_from, d_weights, d_emission_idxs,
         d_state_buffer_prev, d_state_buffer_next, d_am_scores + (t - 1) * frame_stride,
         d_edge_buffer + (t - 1) * n_edges));
      HANDLE_LAST_ERROR();
      std::swap(d_state_buffer_prev, d_state_buffer_next);
    }
    if (dump_alignment && batch_idx %% dump_every == 0) {
      float alpha = 1.0f;
      //HANDLE_ERROR(cublasSaxpy(handle, n_states, &alpha, d_state_buffer_prev, 1, d_state_buffer_all, 1));
    }

    // normalize at each time frame
    start_dev_kernel2(normalize, n_frames, 1, n_seqs * sizeof(float),
      (d_edge_buffer, d_sequence_idxs, n_edges, n_seqs, d_sum_output));
    HANDLE_LAST_ERROR();

    // dump alignment
    if (dump_alignment && batch_idx %% dump_every == 0) {
      write_alignment_to_file(d_state_buffer_all, d_index, index_stride, d_start_states, d_end_states,
                              pruning, n_frames, n_seqs, n_states, batch_idx);
    }

    n_fill_blocks = (n_frames * n_seqs * n_emissions + n_threads - 1u) / n_threads;
    start_dev_kernel2(
      fill_array, n_fill_blocks, n_threads, 0,
      (d_out, std::numeric_limits<float>::infinity(), n_frames * n_seqs * n_emissions));
    HANDLE_LAST_ERROR();

    frame_stride    = Ndarray_STRIDE(out, 0);
    sequence_stride = Ndarray_STRIDE(out, 1);
    n_blocks        = (n_frames * n_edges + n_threads - 1u) / n_threads;
    start_dev_kernel2(compute_result, n_blocks, n_threads, 0,
      (d_edge_buffer, d_out, d_emission_idxs, d_sequence_idxs,
       frame_stride, sequence_stride, n_frames, n_seqs, n_edges));
    HANDLE_LAST_ERROR();

    #if TENSORFLOW
    // Certain TensorFlow code doesn't like inf, even if it is just the CheckNumerics,
    // which is helpful for debugging.
    // We replace it by a very high number, so that tf.exp(-out) will still result in 0.0.
    n_blocks = (n_frames * n_seqs * n_emissions + n_threads - 1u) / n_threads;
    start_dev_kernel2(remove_inf, n_blocks, n_threads, 0, (d_out, n_frames * n_seqs * n_emissions));
    //debug_print(context, out, "out");
    #endif
    if (dump_output && batch_idx %% dump_every == 0) {
      write_output_to_file(d_out, d_index, index_stride, pruning, n_frames, n_seqs, n_emissions, batch_idx);
    }

    device_free(d_edge_buffer);
    if (d_state_buffer_all != NULL) {
      device_free(d_state_buffer_all);
    }
    batch_idx++;
  """

    c_bw_code = None


class MultiEndFastBaumWelchOp(NativeOpGenBase):
    # noinspection PyUnresolvedReferences
    """
    inputs:
      :param am_scores: scores in -log space. 3d (time,batch,dim)
      :param edges: edges of the graph (from,to,emission_idx,sequence_idx)
      :param weights: weights of the edges
    outputs:
      :param output: Baum-Welch alignment, scores in -log space. 3d (time,batch,dim), like am_scores
    """
    in_info = (
        {
            "name": "am_scores",
            "ndim": 3,
            "shape": (None, None, None),
            "need_contiguous": True,
            "gradient": "disconnected",
        },
        {
            "name": "edges",
            "ndim": 2,
            "shape": (None, None),
            "need_contiguous": True,
            "gradient": "disconnected",
            "dtype": "int32",
        },
        {"name": "weights", "ndim": 1, "shape": (None,), "need_contiguous": True, "gradient": "disconnected"},
        {
            "name": "start_states",
            "ndim": 1,
            "shape": (None,),
            "need_contiguous": True,
            "gradient": "disconnected",
            "dtype": "int32",
        },
        {
            "name": "end_states",
            "ndim": 2,
            "shape": (None, 2),
            "need_contiguous": True,
            "gradient": "disconnected",
            "dtype": "int32",
        },
        {
            "name": "end_state_weights",
            "ndim": 1,
            "shape": ((4, 0),),
            "need_contiguous": True,
            "gradient": "disconnected",
        },
        {"name": "index", "ndim": 2, "shape": ((0, 0), (0, 1)), "need_contiguous": True, "gradient": "disconnected"},
        {"name": "state_buffer", "ndim": 2, "shape": (2, None), "need_contiguous": True, "gradient": "disconnected"},
    )
    out_info = (
        {"name": "output", "ndim": 3, "shape": ((0, 0), (0, 1), (0, 2)), "need_contiguous": True},
        {"name": "sums", "ndim": 2, "shape": ((0, 0), (0, 1)), "need_contiguous": True},
    )

    c_extra_support_code = copy.copy(FastBaumWelchOp.c_extra_support_code)
    c_extra_support_code.update(
        {
            "100_init_bwd_state_buffer": """
      __global__
      void init_bwd_state_buffer(unsigned t, unsigned max_t, unsigned num_endstates, unsigned index_stride,
                                 float* states, unsigned const* end_states, float const* end_state_weights,
                                 float const* index) {
        unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_endstates) {
          return;
        }

        unsigned seq_idx = end_states[idx * 2u + 0u];
        if (index[t * index_stride + seq_idx] == 1.0
            && (t == max_t || index[(t + 1) * index_stride + seq_idx] == 0.0)) {
          unsigned state_idx = end_states[idx * 2u + 1u];
          float    weight    = end_state_weights[idx];
          states[state_idx] = weight;
        }
      }
    """
        }
    )

    c_fw_code = """
    // am_scores, edges, weights, start_states, end_states, end_state_weights,
    //   index, state_buffer* = input_names (*: inplace)
    // output = output_names
    assert(n_inputs  == 8);
    assert(n_outputs == 2);
    Ndarray* am_scores         = inputs[0];
    Ndarray* edges             = inputs[1];
    Ndarray* weights           = inputs[2];
    Ndarray* start_states      = inputs[3];
    Ndarray* end_states        = inputs[4];
    Ndarray* end_state_weights = inputs[5];
    Ndarray* index             = inputs[6];
    Ndarray* state_buffer      = inputs[7];
    Ndarray* out               = *outputs[0];
    Ndarray* sum_output        = *outputs[1];

    assert(Ndarray_DIMS(am_scores)[0] == Ndarray_DIMS(out)[0]);
    assert(Ndarray_DIMS(am_scores)[1] == Ndarray_DIMS(out)[1]);
    assert(Ndarray_DIMS(am_scores)[2] == Ndarray_DIMS(out)[2]);
//    assert(Ndarray_DIMS(am_scores)[1] == Ndarray_DIMS(end_states)[0]);

    assert(Ndarray_DIMS(sum_output)[0] == Ndarray_DIMS(am_scores)[0]);
    assert(Ndarray_DIMS(sum_output)[1] == Ndarray_DIMS(am_scores)[1]);

    bool            dump_alignment = false;
    bool            dump_output    = false;
    unsigned        dump_every = 40u;
    static unsigned batch_idx  = 0u;
    float           pruning    = 10.f;

    unsigned* d_from = reinterpret_cast<unsigned*>(Ndarray_DEV_DATA_int32(edges)
      + 0 * Ndarray_STRIDE(edges, 0));
    unsigned* d_to = reinterpret_cast<unsigned*>(Ndarray_DEV_DATA_int32(edges)
      + 1 * Ndarray_STRIDE(edges, 0));
    unsigned* d_emission_idxs = reinterpret_cast<unsigned*>(Ndarray_DEV_DATA_int32(edges)
      + 2 * Ndarray_STRIDE(edges, 0));
    unsigned* d_sequence_idxs = reinterpret_cast<unsigned*>(Ndarray_DEV_DATA_int32(edges)
      + 3 * Ndarray_STRIDE(edges, 0));
    float*    d_weights           = Ndarray_DEV_DATA(weights);
    float*    d_am_scores         = Ndarray_DEV_DATA(am_scores);
    unsigned* d_start_states      = reinterpret_cast<unsigned*>(Ndarray_DEV_DATA_int32(start_states));
    unsigned* d_end_states        = reinterpret_cast<unsigned*>(Ndarray_DEV_DATA_int32(end_states));
    float*    d_end_state_weights = Ndarray_DEV_DATA(end_state_weights);
    float*    d_index             = Ndarray_DEV_DATA(index);
    float*    d_state_buffer_prev = Ndarray_DEV_DATA(state_buffer) + 0 * Ndarray_STRIDE(state_buffer, 0);
    float*    d_state_buffer_next = Ndarray_DEV_DATA(state_buffer) + 1 * Ndarray_STRIDE(state_buffer, 0);
    float*    d_out               = Ndarray_DEV_DATA(out);
    float*    d_sum_output        = Ndarray_DEV_DATA(sum_output);

    unsigned n_frames       = Ndarray_DIMS(am_scores)[0];
    unsigned n_seqs         = Ndarray_DIMS(am_scores)[1];
    unsigned n_emissions    = Ndarray_DIMS(am_scores)[2];
    unsigned n_states       = Ndarray_DIMS(state_buffer)[1];
    unsigned n_edges        = Ndarray_DIMS(edges)[1];
    unsigned n_start_states = Ndarray_DIMS(start_states)[0];
    unsigned n_end_states   = Ndarray_DIMS(end_states)[0];
    unsigned n_threads      = 1024u;
    unsigned n_blocks       = (n_edges + n_threads - 1) / n_threads;

    unsigned frame_stride    = Ndarray_STRIDE(am_scores, 0);
    unsigned sequence_stride = Ndarray_STRIDE(am_scores, 1);
    unsigned index_stride    = Ndarray_STRIDE(index, 0);

    assert(n_frames > 0);

//    std::cerr << "n_frames: "       << n_frames       << std::endl;
//    std::cerr << "n_seqs: "         << n_seqs         << std::endl;
//    std::cerr << "n_emissions: "    << n_emissions    << std::endl;
//    std::cerr << "n_states: "       << n_states       << std::endl;
//    std::cerr << "n_edges: "        << n_edges        << std::endl;
//    std::cerr << "n_start_states: " << n_start_states << std::endl;
//    std::cerr << "n_end_states: "   << n_end_states   << std::endl;
//    std::cerr << "n_threads: "      << n_threads      << std::endl;
//    std::cerr << "n_blocks: "       << n_blocks       << std::endl;

//    std::cerr << "frame_stride: "     << frame_stride    << std::endl;
//    std::cerr << "sequence_stride: "  << sequence_stride << std::endl;
//    std::cerr << "index_stride: "     << index_stride    << std::endl;

    // initialize edge buffer
    float* d_edge_buffer = reinterpret_cast<float*>(device_malloc(n_edges * n_frames * sizeof(float)));
//    cudaDeviceSynchronize();
//    HANDLE_LAST_ERROR();
    unsigned n_fill_blocks = (n_edges * n_frames + n_threads - 1u) / n_threads;
    fill_array<<<n_fill_blocks, n_threads>>>(d_edge_buffer, 0.0, n_edges * n_frames);
//    cudaDeviceSynchronize();
//    HANDLE_LAST_ERROR();

    // initialize the state buffer
    n_fill_blocks = (n_states + n_threads - 1u) / n_threads;
    fill_array<<<n_fill_blocks, n_threads>>>(d_state_buffer_prev, std::numeric_limits<float>::infinity(), n_states);
//    cudaDeviceSynchronize();
//    HANDLE_LAST_ERROR();
    set_start_states<<<1, n_start_states>>>(d_state_buffer_prev, d_start_states);
//    cudaDeviceSynchronize();
//    HANDLE_LAST_ERROR();

    // initialize full state buffer (only used to dump the alignment)
    float* d_state_buffer_all = NULL;
    if (dump_alignment and batch_idx %% dump_every == 0) {
      d_state_buffer_all = reinterpret_cast<float*>(device_malloc(n_states * (n_frames + 1u) * sizeof(float)));
//      cudaDeviceSynchronize();
//      HANDLE_LAST_ERROR();
      cudaMemcpy(d_state_buffer_all, d_state_buffer_prev, n_states * sizeof(float), cudaMemcpyDeviceToDevice);
//      HANDLE_LAST_ERROR();
    }

    // fwd pass
    for (unsigned t = 0u; t < n_frames; t++) {
      fill_array<<<n_fill_blocks, n_threads>>>(d_state_buffer_next, std::numeric_limits<float>::infinity(), n_states);
//      cudaDeviceSynchronize();
//      HANDLE_LAST_ERROR();
//      std::cerr << "frame " << t << std::endl;
      next_frame<<<n_blocks, n_threads>>>(true, n_edges, sequence_stride,
                                          d_sequence_idxs, d_from, d_to, d_weights, d_emission_idxs,
                                          d_state_buffer_prev, d_state_buffer_next, d_am_scores + t * frame_stride,
                                          d_edge_buffer + t * n_edges);
//      cudaDeviceSynchronize();
//      HANDLE_LAST_ERROR();
      if (dump_alignment and batch_idx %% dump_every == 0) {
        cudaMemcpy(
          d_state_buffer_all + (t + 1u) * n_states, d_state_buffer_next, n_states * sizeof(float),
          cudaMemcpyDeviceToDevice);
      }
      std::swap(d_state_buffer_prev, d_state_buffer_next);
    }

    // bwd pass
    const unsigned n_end_state_blocks = (n_end_states + n_threads - 1u) / n_threads;
    const unsigned n_end_state_threads = min(n_threads, n_end_states);
    fill_array<<<n_fill_blocks, n_threads>>>(d_state_buffer_prev, std::numeric_limits<float>::infinity(), n_states);
//    cudaDeviceSynchronize();
//    HANDLE_LAST_ERROR();
    for (unsigned t = n_frames; t > 0; t--) {
      init_bwd_state_buffer<<<n_end_state_blocks, n_end_state_threads>>>(
        t - 1, n_frames - 1, n_end_states, index_stride,
        d_state_buffer_prev, d_end_states, d_end_state_weights,  d_index);
//      cudaDeviceSynchronize();
//      HANDLE_LAST_ERROR();
      if (dump_alignment and batch_idx %% dump_every == 0) {
        float alpha = 1.0f;
//        HANDLE_ERROR(cublasSaxpy(
//          handle, n_states, &alpha, d_state_buffer_prev, 1, d_state_buffer_all + t * n_states, 1));
      }
      fill_array<<<n_fill_blocks, n_threads>>>(d_state_buffer_next, std::numeric_limits<float>::infinity(), n_states);
//      cudaDeviceSynchronize();
//      HANDLE_LAST_ERROR();
      next_frame<<<n_blocks, n_threads>>>(false, n_edges, sequence_stride,
                                          d_sequence_idxs, d_to, d_from, d_weights, d_emission_idxs,
                                          d_state_buffer_prev, d_state_buffer_next,
                                          d_am_scores + (t - 1) * frame_stride,
                                          d_edge_buffer + (t - 1) * n_edges);
//      cudaDeviceSynchronize();
//      HANDLE_LAST_ERROR();
      std::swap(d_state_buffer_prev, d_state_buffer_next);
    }
    if (dump_alignment and batch_idx %% dump_every == 0) {
      float alpha = 1.0f;
//      HANDLE_ERROR(cublasSaxpy(handle, n_states, &alpha, d_state_buffer_prev, 1, d_state_buffer_all, 1));
    }

    // normalize at each time frame
    normalize<<<n_frames, 1, n_seqs * sizeof(float)>>>(d_edge_buffer, d_sequence_idxs, n_edges, n_seqs, d_sum_output);
//    cudaDeviceSynchronize();
//    HANDLE_LAST_ERROR();

    // dump alignment
    if (dump_alignment and batch_idx %% dump_every == 0) {
      write_alignment_to_file(d_state_buffer_all, d_index, index_stride, d_start_states, d_end_states,
                              pruning, n_frames, n_seqs, n_states, batch_idx);
    }

    n_fill_blocks = (n_frames * n_seqs * n_emissions + n_threads - 1u) / n_threads;
    fill_array<<<n_fill_blocks, n_threads>>>(
      d_out, std::numeric_limits<float>::infinity(), n_frames * n_seqs * n_emissions);
//    cudaDeviceSynchronize();
//    HANDLE_LAST_ERROR();

    frame_stride    = Ndarray_STRIDE(out, 0);
    sequence_stride = Ndarray_STRIDE(out, 1);
    n_blocks        = (n_frames * n_edges + n_threads - 1u) / n_threads;
    compute_result<<<n_blocks, n_threads>>>(d_edge_buffer, d_out, d_emission_idxs, d_sequence_idxs,
                                            frame_stride, sequence_stride, n_frames, n_seqs, n_edges);
//    cudaDeviceSynchronize();
//    HANDLE_LAST_ERROR();

    #if TENSORFLOW
    // Certain TensorFlow code doesn't like inf, even if it is just the CheckNumerics,
    // which is helpful for debugging.
    // We replace it by a very high number, so that tf.exp(-out) will still result in 0.0.
    n_blocks = (n_frames * n_seqs * n_emissions + n_threads - 1u) / n_threads;
    remove_inf<<<n_blocks, n_threads>>>(d_out, n_frames * n_seqs * n_emissions);
    //debug_print(context, out, "out");
    #endif
    if (dump_output and batch_idx %% dump_every == 0) {
      write_output_to_file(d_out, d_index, index_stride, pruning, n_frames, n_seqs, n_emissions, batch_idx);
    }

    device_free(d_edge_buffer);
    if (d_state_buffer_all != NULL) {
      device_free(d_state_buffer_all);
    }
    batch_idx++;
  """

    c_bw_code = None

    cpu_support = False  # TODO: fix CPU support...


class SegmentFastBaumWelchOp(NativeOpGenBase):
    """
    Segmental Baum-Welch...
    """

    in_info = (
        {
            "name": "am_scores",
            "ndim": 3,
            "shape": (None, None, None),
            "need_contiguous": True,
            "gradient": "disconnected",
        },
        {"name": "batch_idxs", "ndim": 2, "shape": (None, None), "need_contiguous": True, "gradient": "disconnected"},
        {"name": "edges", "ndim": 2, "shape": (None, None), "need_contiguous": True, "gradient": "disconnected"},
        {"name": "weights", "ndim": 1, "shape": ((2, 1),), "need_contiguous": True, "gradient": "disconnected"},
        {
            "name": "length_models",
            "ndim": 2,
            "shape": (None, (0, 0)),
            "need_contiguous": True,
            "gradient": "disconnected",
        },
        {
            "name": "start_end_states",
            "ndim": 2,
            "shape": (2, None),
            "need_contiguous": True,
            "gradient": "disconnected",
        },
        {"name": "index", "ndim": 2, "shape": ((0, 0), (0, 1)), "need_contiguous": True, "gradient": "disconnected"},
        {"name": "am_score_scales", "ndim": 1, "shape": (None,), "need_contiguous": True, "gradient": "disconnected"},
        {"name": "epoch", "ndim": 0, "shape": (), "need_contiguous": True, "gradient": "disconnected"},
    )
    out_info = (
        {"name": "output", "ndim": 3, "shape": ((0, 0), (0, 1), (0, 2)), "need_contiguous": True},
        {"name": "normalization_factors", "ndim": 2, "shape": ((0, 0), (0, 1)), "need_contiguous": True},
        {"name": "posterior_weigths", "ndim": 2, "shape": ((0, 0), (0, 1)), "need_contiguous": True},
    )

    c_extra_support_code = copy.copy(common_fast_bw_kernels)
    c_extra_support_code.update(
        {
            "100_get_batch_idx": """
      __device__
      int get_batch_idx(int const* batch_idxs, unsigned num_seqs, unsigned t, unsigned seq_idx) {
        if (NEW_BATCH_IDX_FORMAT) {
          int res = batch_idxs[seq_idx] + t;
          if (res >= batch_idxs[seq_idx + 1]) {
            return -1;
          }
          return res;
        }
        else {
          return batch_idxs[t * num_seqs + seq_idx];
        }
      }
    """,
            "101_init_bwd_state_buffer": """
      __global__
      void init_bwd_state_buffer(unsigned t, unsigned num_batches, unsigned num_seqs,
                                 int* batch_idxs, float* index, float* states, unsigned* end_states) {
        unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
        int batch_idx = get_batch_idx(batch_idxs, num_seqs, t, idx);
        if (batch_idx < 0) {
          return;
        }
        float* batch_first_frame = index + batch_idx;
        //if (*batch_first_frame != 0.0 && (t == max_t || *(batch_first_frame + 1) == 0.0)) {
        if (batch_first_frame[0] != 0.0 && batch_first_frame[num_batches] == 0.0) {
          unsigned state_idx = end_states[idx];
          states[state_idx] = 0.0;
        }
      }
    """,
            "102_next_frame_fwd": """
      __global__
      void next_frame_fwd(unsigned time, unsigned num_states, unsigned num_edges, unsigned num_emissions,
                          unsigned num_seg_frames,
                          unsigned num_tot_frames, unsigned num_seqs, unsigned num_am_score_scales,
                          unsigned const* sequence_idxs, unsigned const* from_buffer, unsigned const* to_buffer,
                          float const* weight_buffer,
                          unsigned const* emission_idxs, unsigned const* lenmod_idxs, int const* batch_idxs,
                          float const* am_scores, float const* length_models, float const* am_score_scales,
                          float const* epoch,
                          float* state_buffer, float* edge_buffer) {
        const unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_edges) {
          return;
        }

        const unsigned num_ringbuffer_frames = num_seg_frames + 1;
        const unsigned max_seg_frames        = min(num_seg_frames, num_tot_frames - time);

        const unsigned prev_frame_idx   = time % num_ringbuffer_frames;
        const unsigned prev_frame_start = prev_frame_idx * num_states;

        const unsigned from     = from_buffer [idx];
        const float    prev_val = state_buffer[prev_frame_start + from];
        if (isinf(prev_val)) {
          return;
        }

        const unsigned sequence_idx = sequence_idxs[idx];
        const int      batch_idx    = get_batch_idx(batch_idxs, num_seqs, time, sequence_idx);
        if (batch_idx == -1) {
          return;
        }

        const unsigned amss_idx       = min(static_cast<unsigned>(*epoch), num_am_score_scales - 1);
        const float    am_score_scale = am_score_scales[amss_idx];

        const unsigned to             = to_buffer    [idx];
        const unsigned emission_idx   = emission_idxs[idx];
        const unsigned lenmod_idx     = lenmod_idxs  [idx];
        const float    edge_weight    = weight_buffer[idx];
        const float    prev_plus_edge = prev_val + edge_weight;

        float const* am_buffer_in    = am_scores     + batch_idx  * num_seg_frames * num_emissions + emission_idx;
        float const* length_scores   = length_models + lenmod_idx * num_seg_frames;
        float*       edge_buffer_out = edge_buffer   + idx;

        for (unsigned i = 0u; i < max_seg_frames; i++) {
          const float val = prev_plus_edge + am_score_scale * am_buffer_in[i * num_emissions] + length_scores[i];
          edge_buffer_out[i * num_edges] = val;
          const unsigned next_frame = (prev_frame_idx + 1 + i) % num_ringbuffer_frames;
          atomic_prob_add(state_buffer + (next_frame * num_states + to), val);
        }
      }
    """,
            "103_next_frame_bwd": """
      __global__
      void next_frame_bwd(unsigned time, unsigned num_states, unsigned num_edges, unsigned num_emissions,
                          unsigned num_seg_frames,
                          unsigned num_tot_frames, unsigned num_seqs, unsigned num_am_score_scales,
                          unsigned const* sequence_idxs, unsigned const* from_buffer, unsigned const* to_buffer,
                          float const* weight_buffer,
                          unsigned const* emission_idxs, unsigned const* lenmod_idxs, int const* batch_idxs,
                          float const* am_scores, float const* length_models, float const* am_score_scales,
                          float const* epoch,
                          float* state_buffer, float* edge_buffer) {
        const unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_edges) {
          return;
        }

        const unsigned num_ringbuffer_frames = num_seg_frames + 1;
        const unsigned max_seg_frames        = min(num_seg_frames, num_tot_frames - time);

        const unsigned sequence_idx = sequence_idxs[idx];
        const int      batch_idx    = get_batch_idx(batch_idxs, num_seqs, time, sequence_idx);
        if (batch_idx == -1) {
          return;
        }

        const unsigned amss_idx       = min(static_cast<unsigned>(*epoch), num_am_score_scales - 1);
        const float    am_score_scale = am_score_scales[amss_idx];

        const unsigned from           = from_buffer  [idx];
        const unsigned to             = to_buffer    [idx];
        const unsigned emission_idx   = emission_idxs[idx];
        const unsigned lenmod_idx     = lenmod_idxs  [idx];
        const float    edge_weight    = weight_buffer[idx];
        const unsigned next_frame_idx = time % num_ringbuffer_frames;

        float const*   am_buffer_in    = am_scores     + batch_idx  * num_seg_frames * num_emissions + emission_idx;
        float const*   length_scores   = length_models + lenmod_idx * num_seg_frames;
        float*         edge_buffer_out = edge_buffer   + idx;

        float acc_val = CUDART_INF_F;

        for (unsigned i = 0u; i < max_seg_frames; i++) {
          const unsigned prev_frame_idx = (next_frame_idx + i + 1) % num_ringbuffer_frames;
          const float    prev_val       = state_buffer[prev_frame_idx * num_states + from];
          if (isinf(prev_val)) {
            edge_buffer_out[i * num_edges] = CUDART_INF_F;
          }
          else {
            const float val =
              prev_val + edge_weight + am_score_scale * am_buffer_in[i * num_emissions] + length_scores[i];
            edge_buffer_out[i * num_edges] += prev_val;
            acc_val = prob_add(acc_val, val);
          }
        }

        atomic_prob_add(state_buffer + next_frame_idx * num_states + to, acc_val);
      }
    """,
            "104_compute_framewise_sum": """
      __global__
      void compute_framewise_sum(unsigned num_tot_frames, unsigned num_seqs, unsigned num_seg_frames,
                                 unsigned num_batches, unsigned num_edges,
                                 unsigned const* sequence_idxs, int const* batch_idxs, float const* index,
                                 float const* edge_buffer,
                                 float* output_buffer) {
        extern __shared__ float sum[];

        const unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_tot_frames * num_seg_frames) {
          return;
        }

        float* sum_buffer = sum + threadIdx.x * num_seqs;
        edge_buffer += idx * num_edges;

        for (unsigned s = 0u; s < num_seqs; s++) {
          sum_buffer[s] = CUDART_INF_F;
        }

        for (unsigned i = 0; i < num_edges; i++) {
          const unsigned seq_idx = sequence_idxs[i];
          sum_buffer[seq_idx] = prob_add(sum_buffer[seq_idx], edge_buffer[i]);
        }

        const unsigned time     = idx / num_seg_frames;
        const unsigned seg_size = idx % num_seg_frames;
        for (unsigned s = 0u; s < num_seqs; s++) {
          const int batch_idx = get_batch_idx(batch_idxs, num_seqs, time, s);
          if (batch_idx >= 0) {
            const unsigned output_idx = seg_size * num_batches + batch_idx;
            if (isinf(sum_buffer[s]) or index[output_idx] == 0.0) {
              output_buffer[output_idx] = 0.0;
            }
            else {
              output_buffer[output_idx] = sum_buffer[s];
            }
          }
        }
      }
    """,
            "105_merge_framewise_sums": """
      __global__
      void merge_framewise_sum(unsigned num_seg_frames, unsigned num_batches, float const* index, float* sum_buffer) {
        const unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_batches) {
          return;
        }

        sum_buffer += idx;
        index += idx;

        float sum = sum_buffer[0];
        for (unsigned s = 1; s < num_seg_frames; s++) {
          if (index[s * num_batches] != 0.0f) {
            sum = prob_add(sum, sum_buffer[s * num_batches]);
          }
        }

        for (unsigned s = 0; s < num_seg_frames; s++) {
          if (index[s * num_batches] != 0.0f) {
            sum_buffer[s * num_batches] = sum;
          }
        }
      }
    """,
            "106_compute_targets": """
      __global__
      void compute_targets(unsigned num_tot_frames, unsigned num_seg_frames, unsigned num_edges, unsigned num_batches,
                           unsigned num_seqs, unsigned num_emissions,
                           unsigned const* sequence_idxs, unsigned const* emission_idxs, int const* batch_idxs,
                           float const* index,
                           float const* edge_buffer, float const* normalization_buffer, float* output_buffer) {
        const unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_tot_frames * num_seg_frames * num_edges) {
          return;
        }

        const unsigned edge_idx  = idx % num_edges;
        const unsigned time      = idx / (num_edges * num_seg_frames);
        const unsigned seq_idx   = sequence_idxs[edge_idx];
        const int      batch_idx = get_batch_idx(batch_idxs, num_seqs, time, seq_idx);

        if (batch_idx < 0) {
          return;
        }

        const unsigned seg_length = (idx / num_edges) % num_seg_frames;

        if (index[seg_length * num_batches + batch_idx] == 0.0) {
          return;
        }

        const unsigned emission_idx  = emission_idxs[edge_idx];
        const float    normalization = normalization_buffer[seg_length * num_batches + batch_idx];

        atomic_prob_add(
          output_buffer + seg_length * num_batches * num_emissions + batch_idx * num_emissions + emission_idx,
          edge_buffer[idx] - normalization);
      }
    """,
            "107_compute_posterior_weights": """
    __global__
    void compute_posterior_weights(unsigned num_tot_frames, unsigned num_seg_frames, unsigned num_seqs,
                                   unsigned num_batches,
                                   float const* state_buffer, unsigned const* start_states, int const* batch_idxs,
                                   float const* index, float const* normalization_factors, float* posterior_weigths) {
        const unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_tot_frames * num_seqs) {
          return;
        }

        const unsigned time    = idx / num_seqs;
        const unsigned seq_idx = idx % num_seqs;

        const int batch_idx = get_batch_idx(batch_idxs, num_seqs, time, seq_idx);
        if (batch_idx < 0) {
          return;
        }

        const float seq_sum = state_buffer[start_states[seq_idx]];
        for (unsigned s = 0u; s < num_seg_frames; s++) {
          const unsigned i = s * num_batches + batch_idx;
          if (index[i] == 0.0) {
            return;
          }
          posterior_weigths[i] = exp(-(normalization_factors[i] - seq_sum));
        }
    }
    """,
        }
    )

    c_fw_code = """
    // inputs:  am_scores, batch_idxs, edges, weights, length_models, start_end_states, index, am_score_scales, epoch
    // outputs: output, normalization_factors, posterior_weigths
    assert(n_inputs  == 9);
    assert(n_outputs == 3);
    Ndarray* ary_am_scores         = inputs[0];
    Ndarray* ary_batch_idxs        = inputs[1];
    Ndarray* ary_edges             = inputs[2];
    Ndarray* ary_weights           = inputs[3];
    Ndarray* ary_start_end_states  = inputs[4];
    Ndarray* ary_length_models     = inputs[5];
    Ndarray* ary_index             = inputs[6];
    Ndarray* ary_am_score_scales   = inputs[7];
    Ndarray* ary_epoch             = inputs[8];
    Ndarray* ary_out               = *outputs[0];
    Ndarray* ary_norm_factors      = *outputs[1];
    Ndarray* ary_posterior_weights = *outputs[2];

    assert(Ndarray_DIMS(ary_edges)[1] == Ndarray_DIMS(ary_weights)[0]);

    static unsigned iter = 0u; // used for debug output

    float*    d_am_scores         = Ndarray_DEV_DATA(ary_am_scores);
    int*      d_batch_idxs        = reinterpret_cast<int*>(Ndarray_DEV_DATA(ary_batch_idxs));
    unsigned* d_from              =
      reinterpret_cast<unsigned*>(Ndarray_DEV_DATA(ary_edges) + 0 * Ndarray_STRIDE(ary_edges, 0));
    unsigned* d_to                =
      reinterpret_cast<unsigned*>(Ndarray_DEV_DATA(ary_edges) + 1 * Ndarray_STRIDE(ary_edges, 0));
    unsigned* d_emission_idxs     =
      reinterpret_cast<unsigned*>(Ndarray_DEV_DATA(ary_edges) + 2 * Ndarray_STRIDE(ary_edges, 0));
    unsigned* d_lenmod_idxs       =
      reinterpret_cast<unsigned*>(Ndarray_DEV_DATA(ary_edges) + 3 * Ndarray_STRIDE(ary_edges, 0));
    unsigned* d_sequence_idxs     =
      reinterpret_cast<unsigned*>(Ndarray_DEV_DATA(ary_edges) + 4 * Ndarray_STRIDE(ary_edges, 0));
    float*    d_weights           = Ndarray_DEV_DATA(ary_weights);
    float*    d_length_models     = Ndarray_DEV_DATA(ary_length_models);
    unsigned* d_start_states      =
      reinterpret_cast<unsigned*>(Ndarray_DEV_DATA(ary_start_end_states) + 0 * Ndarray_STRIDE(ary_start_end_states, 0));
    unsigned* d_end_states        =
      reinterpret_cast<unsigned*>(Ndarray_DEV_DATA(ary_start_end_states) + 1 * Ndarray_STRIDE(ary_start_end_states, 0));
    float*    d_index             = Ndarray_DEV_DATA(ary_index);
    float*    d_am_score_scales   = Ndarray_DEV_DATA(ary_am_score_scales);
    float*    d_epoch             = Ndarray_DEV_DATA(ary_epoch);
    float*    d_out               = Ndarray_DEV_DATA(ary_out);
    float*    d_norm_factors      = Ndarray_DEV_DATA(ary_norm_factors);
    float*    d_posterior_weights = Ndarray_DEV_DATA(ary_posterior_weights);

    std::vector<int> seq_lengths;
    if (NEW_BATCH_IDX_FORMAT) {
      seq_lengths.resize(Ndarray_DIMS(ary_batch_idxs)[0]);
      HANDLE_ERROR(cudaMemcpy(
        seq_lengths.data(), d_batch_idxs, seq_lengths.size() * sizeof(int), cudaMemcpyDeviceToHost));
    }

    const unsigned n_seg_frames      = Ndarray_DIMS(ary_am_scores)[0];
    const unsigned n_batches         = Ndarray_DIMS(ary_am_scores)[1];
    const unsigned n_emissions       = Ndarray_DIMS(ary_am_scores)[2];
    const unsigned n_seqs            =
      NEW_BATCH_IDX_FORMAT ? (Ndarray_DIMS(ary_batch_idxs)[0] - 1) : Ndarray_DIMS(ary_batch_idxs)[1];
    const unsigned n_tot_frames      =
      NEW_BATCH_IDX_FORMAT ? seq_lengths.back()                     : Ndarray_DIMS(ary_batch_idxs)[0];
    const unsigned n_edges           = Ndarray_DIMS(ary_edges)[1];
    const unsigned n_length_models   = Ndarray_DIMS(ary_length_models)[1];
    const unsigned n_am_score_scales = Ndarray_DIMS(ary_am_score_scales)[0];
    const unsigned n_threads         = 1024u;
    unsigned       n_blocks          = (n_edges + n_threads - 1) / n_threads;

    unsigned tmp;
    HANDLE_ERROR(cudaMemcpy(&tmp, d_end_states + n_seqs - 1, sizeof(float), cudaMemcpyDeviceToHost));

    const unsigned n_states = tmp + 1;

    /*std::cerr << "seg frames: "    << n_seg_frames    << std::endl;
    std::cerr << "batches: "       << n_batches       << std::endl;
    std::cerr << "emissions: "     << n_emissions     << std::endl;
    std::cerr << "tot frames: "    << n_tot_frames    << std::endl;
    std::cerr << "seqs: "          << n_seqs          << std::endl;
    std::cerr << "edges: "         << n_edges         << std::endl;
    std::cerr << "length models: " << n_length_models << std::endl;
    std::cerr << "threads: "       << n_threads       << std::endl;
    std::cerr << "blocks: "        << n_blocks        << std::endl;
    std::cerr << "num states: "    << n_states        << std::endl;*/

    // initialize edge buffer
    const unsigned edge_buffer_size = n_tot_frames * n_seg_frames * n_edges;
    float* d_edge_buffer  = reinterpret_cast<float*>(device_malloc(edge_buffer_size * sizeof(float)));
    HANDLE_LAST_ERROR();
    unsigned n_fill_blocks = (edge_buffer_size + n_threads - 1u) / n_threads;
    fill_array<<<n_fill_blocks, n_threads>>>(d_edge_buffer, std::numeric_limits<float>::infinity(), edge_buffer_size);
    HANDLE_LAST_ERROR();

    // initialize the state buffer
    const unsigned n_ringbuffer_frames = n_seg_frames + 1;
    float* d_state_buffer = reinterpret_cast<float*>(device_malloc(n_states * n_ringbuffer_frames * sizeof(float)));
    HANDLE_LAST_ERROR();
    n_fill_blocks = (n_states * n_ringbuffer_frames + n_threads - 1u) / n_threads;
    fill_array<<<n_fill_blocks, n_threads>>>(
      d_state_buffer, std::numeric_limits<float>::infinity(), n_states * n_ringbuffer_frames);
    HANDLE_LAST_ERROR();

    // initialize sum buffer and posterior weigths
    n_fill_blocks = (n_batches * n_seg_frames + n_threads - 1u) / n_threads;
    fill_array<<<n_fill_blocks, n_threads>>>(d_norm_factors, 0.0f, n_batches * n_seg_frames);
    HANDLE_LAST_ERROR();
    fill_array<<<n_fill_blocks, n_threads>>>(d_posterior_weights, 0.0f, n_batches * n_seg_frames);
    HANDLE_LAST_ERROR();

    set_start_states<<<1, n_seqs>>>(d_state_buffer, d_start_states);
    HANDLE_LAST_ERROR();

    // fwd pass
    for (unsigned t = 0u; t < n_tot_frames; t++) {
      //std::cerr << "fwd t: " << t << " " << n_tot_frames << std::endl;
      float* d_state_buffer_prev = d_state_buffer + ((t - 1) %% n_ringbuffer_frames) * n_states;
      fill_array<<<n_fill_blocks, n_threads>>>(d_state_buffer_prev, std::numeric_limits<float>::infinity(), n_states);
      HANDLE_LAST_ERROR();
      next_frame_fwd<<<n_blocks, n_threads>>>(t, n_states, n_edges, n_emissions, n_seg_frames, n_tot_frames, n_seqs,
                                              n_am_score_scales,
                                              d_sequence_idxs, d_from, d_to, d_weights, d_emission_idxs, d_lenmod_idxs,
                                              d_batch_idxs,
                                              d_am_scores, d_length_models, d_am_score_scales, d_epoch,
                                              d_state_buffer, d_edge_buffer + t * n_seg_frames * n_edges);
      HANDLE_LAST_ERROR();

      //std::stringstream ss;
      //ss << "dump/fwd_state_buffer." << t << ".dump";
      //dump_to_file_2d(d_state_buffer, n_ringbuffer_frames, n_states, ss.str());
    }

    //dump_to_file_3d(d_edge_buffer, n_tot_frames, n_seg_frames, n_edges, "dump/fwd_edges.dump");

    // bwd pass
    n_fill_blocks = (n_states * n_ringbuffer_frames + n_threads - 1u) / n_threads;
    fill_array<<<n_fill_blocks, n_threads>>>(
      d_state_buffer, std::numeric_limits<float>::infinity(), n_states * n_ringbuffer_frames);
    HANDLE_LAST_ERROR();
    n_fill_blocks = (n_states + n_threads - 1u) / n_threads;
    for (unsigned t = n_tot_frames; t > 0; t--) {
      //std::cerr <<
      //"bwd t: " << t << " " << n_tot_frames << " buffer next: " << ((t-1) %% n_ringbuffer_frames) << std::endl;
      float* d_state_buffer_next = d_state_buffer + ((t - 1) %% n_ringbuffer_frames) * n_states;
      float* d_state_buffer_prev = d_state_buffer + ( t      %% n_ringbuffer_frames) * n_states;
      fill_array<<<n_fill_blocks, n_threads>>>(d_state_buffer_next, std::numeric_limits<float>::infinity(), n_states);
      HANDLE_LAST_ERROR();
      init_bwd_state_buffer<<<1, n_seqs>>>(
        t - 1, n_batches, n_seqs, d_batch_idxs, d_index, d_state_buffer_prev, d_end_states);
      HANDLE_LAST_ERROR();
      next_frame_bwd<<<n_blocks, n_threads>>>(
        t - 1, n_states, n_edges, n_emissions, n_seg_frames, n_tot_frames, n_seqs, n_am_score_scales,
        d_sequence_idxs, d_to, d_from, d_weights, d_emission_idxs, d_lenmod_idxs, d_batch_idxs,
        d_am_scores, d_length_models, d_am_score_scales, d_epoch,
        d_state_buffer, d_edge_buffer + (t - 1) * n_seg_frames * n_edges);
      HANDLE_LAST_ERROR();

      //std::stringstream ss;
      //ss << "dump/bwd_state_buffer." << t << ".dump";
      //dump_to_file_2d(d_state_buffer, n_ringbuffer_frames, n_states, ss.str());
    }

    n_blocks = (n_tot_frames * n_seg_frames + n_threads - 1) / n_threads;
    compute_framewise_sum<<<n_blocks, n_threads, n_threads * n_seqs * sizeof(float)>>>(
      n_tot_frames, n_seqs, n_seg_frames, n_batches, n_edges,
      d_sequence_idxs, d_batch_idxs,
      d_index, d_edge_buffer, d_norm_factors);
    HANDLE_LAST_ERROR();

    //dump_to_file_2d(d_norm_factors, n_seg_frames, n_batches, "dump/norm_factors_1.dump");

    if (segmentwise_normalization) {
      n_blocks = (n_batches + n_threads - 1) / n_threads;
      merge_framewise_sum<<<n_blocks, n_threads>>>(n_seg_frames, n_batches, d_index, d_norm_factors);
      HANDLE_LAST_ERROR();
    }

    //dump_to_file_2d(d_norm_factors, n_seg_frames, n_batches, "dump/norm_factors_2.dump");

    n_blocks = (n_tot_frames * n_seqs + n_threads - 1) / n_threads;
    compute_posterior_weights<<<n_blocks, n_threads>>>(n_tot_frames, n_seg_frames, n_seqs, n_batches, d_state_buffer,
                                                       d_start_states, d_batch_idxs, d_index, d_norm_factors,
                                                       d_posterior_weights);
    HANDLE_LAST_ERROR();

    n_fill_blocks = (n_batches * n_seg_frames * n_emissions + n_threads - 1u) / n_threads;
    fill_array<<<n_fill_blocks, n_threads>>>(
      d_out, std::numeric_limits<float>::infinity(), n_batches * n_seg_frames * n_emissions);
    HANDLE_LAST_ERROR();

    n_blocks = (n_tot_frames * n_seg_frames * n_edges + n_threads - 1) / n_threads;
    compute_targets<<<n_blocks, n_threads>>>(n_tot_frames, n_seg_frames, n_edges, n_batches, n_seqs, n_emissions,
                                             d_sequence_idxs, d_emission_idxs, d_batch_idxs, d_index, d_edge_buffer,
                                             d_norm_factors, d_out);
    HANDLE_LAST_ERROR();

    //dump_to_file_1d(d_weights,       n_edges, "dump/edge_weights.dump");
    //dump_to_file_1d(d_sequence_idxs, n_edges, "dump/sequence_idxs.dump");
    //dump_to_file_2d(d_state_buffer,  n_ringbuffer_frames, n_states,  "dump/state_buffer.dump");
    //dump_to_file_2d(d_batch_idxs,    n_tot_frames,        n_seqs,    "dump/batch_idxs.dump");
    //dump_to_file_2d(d_index,         n_seg_frames,        n_batches, "dump/index.dump");
    //dump_to_file_3d(d_edge_buffer,   n_tot_frames,        n_seg_frames, n_edges,     "dump/edges.dump");
    //dump_to_file_3d(d_am_scores,     n_seg_frames,        n_batches,    n_emissions, "dump/am_scores.dump");
    //dump_to_file_3d(d_out,           n_seg_frames,        n_batches,    n_emissions, "dump/targets.dump");

    if (dump_targets and iter %% dump_targets_interval == 0) {
      std::stringstream ss;
      ss << "dump/targets_" << iter << ".dump";
      dump_to_file_3d(d_out, n_seg_frames, n_batches, n_emissions, ss.str());
      ss.str("");
      ss.clear();
      ss << "dump/norm_factors_" << iter << ".dump";
      dump_to_file_2d(d_norm_factors, n_seg_frames, n_batches, ss.str());
      ss.str("");
      ss.clear();
      ss << "dump/posterior_weights_" << iter << ".dump";
      dump_to_file_2d(d_posterior_weights, n_seg_frames, n_batches, ss.str());
    }

    iter += 1;

    device_free(d_state_buffer);
    device_free(d_edge_buffer);
  """

    cpu_support = False  # TODO: fix CPU support...

    def __init__(self, segmentwise_normalization=False, dump_targets_interval=None, new_batch_idxs_format=False):
        # the new_buffer_idx_format flag can be used to change the format of buffer_idxs parameter, if set to false
        # the code expects a two-dimensional array that stores the batch index (within the given am_scores) for any
        # timeframe and sequence index. if the flag is true we expect a list of offsets (as given by a cumulative sum
        # of the sequence lengths).
        def _to_cpp_bool(v):
            return "true" if v else "false"

        extra_lines = [
            "const bool segmentwise_normalization = %s;" % _to_cpp_bool(segmentwise_normalization),
            "const bool dump_targets = %s;" % _to_cpp_bool(dump_targets_interval is not None),
            "const unsigned dump_targets_interval = %d;"
            % (0 if dump_targets_interval is None else dump_targets_interval),
        ]

        self.c_extra_support_code = dict(**self.c_extra_support_code)
        self.c_extra_support_code["000_batch_format"] = "#define NEW_BATCH_IDX_FORMAT %s\n" % _to_cpp_bool(
            new_batch_idxs_format
        )
        if new_batch_idxs_format:
            in_info = list(self.in_info)  # copy class member to instance
            in_info[1] = {
                "name": "batch_idxs",
                "ndim": 1,
                "shape": (None,),
                "need_contiguous": True,
                "gradient": "disconnected",
            }
            self.in_info = tuple(in_info)

        self.c_fw_code = "\n".join(extra_lines) + "\n" + self.c_fw_code


class FastViterbiOp(NativeOpGenBase):
    # noinspection PyUnresolvedReferences
    """
    inputs:
      :param am_scores: scores in +log space. 3d (time,batch,dim)
      :param am_seq_len: (batch,)
      :param edges: edges of the graph (from,to,emission_idx,sequence_idx), i.e. (4, n_edges)
      :param weights: weights of the edges (n_edges,)
      :param start_end_states: (2, batch)
      :param n_states: scalar, int32
    outputs:
      :param output: Viterbi (hard) alignment, scores in +log space. 2d (time,batch)
      :param scores: (batch,)
    """
    in_info = (
        {
            "name": "am_scores",
            "ndim": 3,
            "shape": (None, None, None),
            "need_contiguous": True,
            "gradient": "disconnected",
        },
        {
            "name": "am_seq_len",
            "ndim": 1,
            "shape": ((0, 0),),
            "dtype": "int32",
            "need_contiguous": True,
            "gradient": "disconnected",
        },
        {
            "name": "edges",
            "ndim": 2,
            "shape": (4, None),
            "dtype": "int32",
            "need_contiguous": True,
            "gradient": "disconnected",
        },
        {"name": "weights", "ndim": 1, "shape": ((3, 1),), "need_contiguous": True, "gradient": "disconnected"},
        {
            "name": "start_end_states",
            "ndim": 2,
            "shape": (2, (0, 0)),
            "dtype": "int32",
            "need_contiguous": True,
            "gradient": "disconnected",
        },
        {
            "name": "n_states",
            "ndim": 0,
            "shape": (),
            "dtype": "int32",
            "need_contiguous": True,
            "gradient": "disconnected",
            "host_memory": True,
        },
    )
    out_info = (
        {"name": "output", "ndim": 2, "shape": ((0, 0), (0, 1)), "dtype": "int32", "need_contiguous": True},
        {"name": "scores", "ndim": 1, "shape": ((0, 1),), "need_contiguous": True},
    )

    c_extra_support_code = {
        "01_IdxAndVal": """
      struct __attribute__((__packed__)) IdxAndVal {
        int idx;
        float val;
      };
    """,
        "04_select_max": """
      DEV_FUNC
      void select_max(IdxAndVal* a, IdxAndVal b) {
        // fast path
        if(b.val < a->val)
          return;
        // Maybe we could use double compare-and-swap (https://stackoverflow.com/questions/55941382/).
        // But not sure how.
        // So instead, we use double-wide compare-and-swap.
        union U {
          IdxAndVal s;
          unsigned long long int v64;
        };
        while(true) {
          U prev;
          prev.s = *a;
          if(b.val < prev.s.val)
            return;
          if(b.val == prev.s.val && b.idx >= prev.s.idx)
            return;
          U updated;
          updated.s = b;

          U old;
          old.v64 = elem_atomic_cas((unsigned long long int*) a, prev.v64, updated.v64);
          if(old.v64 == prev.v64)
            return;
          // Not the same, so repeat.
        }
      }
    """,
        "05_init_buffer": """
      DEF_KERNEL
      void init_buffer
      (
        int n_time,
        int n_states, // for the whole batch
        IdxAndVal* buffer // (time+1,n_states), states for the whole batch
      )
      {
        int idx = threadIdx.x + blockDim.x * blockIdx.x;
        while(idx < (n_time + 1) * n_states) {
          buffer[idx].val = -INF_F;
          buffer[idx].idx = -1;
          idx += gridDim.x * blockDim.x;
        }
      }
    """,
        "06_init_first_frame": """
      DEF_KERNEL
      void init_first_frame
      (
        int n_batch,
        int n_states, // for the whole batch
        IdxAndVal* frame, // (n_states,), states for the whole batch
        const int32_t* d_start_states // (n_batch,)
      )
      {
        int idx = threadIdx.x + blockDim.x * blockIdx.x;
        while(idx < n_batch) {
          int state_idx = d_start_states[idx];
          frame[state_idx].val = 0;
          idx += gridDim.x * blockDim.x;
        }
      }
    """,
        "08_next_frame": """
      DEF_KERNEL
      void next_frame
      (
        int n_time,
        int n_states,
        int n_edges,
        int n_classes,
        int t,
        const float* d_am_scores,
        const int32_t* d_am_seq_len,
        const IdxAndVal* prev_frame,
        IdxAndVal* frame,
        const int32_t* d_edge_from,
        const int32_t* d_edge_to,
        const int32_t* d_edge_emission_idx,
        const int32_t* d_edge_seq_idx,
        const float* d_edge_weights,
        const int32_t* d_end_states // (n_batch,)
      )
      {
        int idx = threadIdx.x + blockDim.x * blockIdx.x;
        while(idx < n_edges) {
          int from_idx = d_edge_from[idx];
          //assert_cmp(0, <=, from_idx); assert_cmp(from_idx, <, n_states);

          int seq_idx = d_edge_seq_idx[idx];
          if(t < d_am_seq_len[seq_idx]) {
            float prev_val = prev_frame[from_idx].val;
            int emission_idx = d_edge_emission_idx[idx];
            //assert_cmp(0, <=, emission_idx); assert_cmp(emission_idx, <, n_classes);
            int to_idx = d_edge_to[idx];
            //assert_cmp(0, <=, to_idx); assert_cmp(to_idx, <, n_states);
            IdxAndVal candidate;
            candidate.val = prev_val + d_edge_weights[idx] + d_am_scores[seq_idx * n_classes + emission_idx];
            candidate.idx = idx;
            select_max(&frame[to_idx], candidate);
          }

          idx += gridDim.x * blockDim.x;
        }
      }
    """,
        "11_select_scores": """
      DEF_KERNEL
      void select_scores
      (
        int n_batch,
        int n_states,
        int buffer_stride,
        const IdxAndVal* buffer,
        const int32_t* d_am_seq_len, // (n_batch,)
        const int32_t* d_end_states, // (n_batch,)
        float* d_score // (n_batch,)
      )
      {
        int idx = threadIdx.x + blockDim.x * blockIdx.x;
        while(idx < n_batch) {
          const IdxAndVal* last_frame = buffer + d_am_seq_len[idx] * buffer_stride;
          int end_state_idx = d_end_states[idx];
          d_score[idx] = last_frame[end_state_idx].val;

          idx += gridDim.x * blockDim.x;
        }
      }
    """,
        "13_select_best_path": """
      DEF_KERNEL
      void select_best_path
      (
        int n_batch,
        int n_states,
        int n_edges,
        int t,
        int32* cur_state, // (n_batch,)
        const IdxAndVal* frame,
        const int32_t* d_am_seq_len,
        const int32_t* d_edge_from,
        const int32_t* d_edge_to,
        const int32_t* d_edge_emission_idx,
        int32_t* output
      )
      {
        int idx = threadIdx.x + blockDim.x * blockIdx.x;
        while(idx < n_batch) {
          if(t < d_am_seq_len[idx]) {
            int state_idx = cur_state[idx];
            //assert_cmp(0, <=, state_idx); assert_cmp(state_idx, <, n_states);
            int edge_idx = frame[state_idx].idx;
            if(edge_idx >= 0) {
              //assert_cmp(0, <=, edge_idx); assert_cmp(edge_idx, <, n_edges);
              //assert_cmp(state_idx, ==, d_edge_to[edge_idx]);
              cur_state[idx] = d_edge_from[edge_idx];
              output[idx] = d_edge_emission_idx[edge_idx];
            }
            else  // no path found
              output[idx] = 0;
          }
          else {
            output[idx] = 0;
          }
          idx += gridDim.x * blockDim.x;
        }
      }
    """,
    }

    c_fw_code = """
    using namespace std;
    // am_scores, am_seq_len, edges, weights, start_end_states, n_states = input_names
    // output, scores = output_names
    assert(n_inputs == 6);
    assert(n_outputs == 2);
    Ndarray* am_scores = inputs[0];
    Ndarray* am_seq_len = inputs[1];
    Ndarray* edges = inputs[2];
    Ndarray* weights = inputs[3];
    Ndarray* start_end_states = inputs[4];
    Ndarray* n_states_ref = inputs[5];
    Ndarray* output = *outputs[0];
    Ndarray* score = *outputs[1];

    assert_cmp(Ndarray_NDIM(am_scores), ==, 3);
    assert_cmp(Ndarray_NDIM(am_seq_len), ==, 1);
    assert_cmp(Ndarray_NDIM(edges), ==, 2);
    assert_cmp(Ndarray_NDIM(weights), ==, 1);
    assert_cmp(Ndarray_NDIM(start_end_states), ==, 2);
    assert_cmp(Ndarray_NDIM(n_states_ref), ==, 0);
    assert_cmp(Ndarray_NDIM(output), ==, 2);
    assert_cmp(Ndarray_NDIM(score), ==, 1);
    int n_time = Ndarray_DIMS(am_scores)[0];
    int n_batch = Ndarray_DIMS(am_scores)[1];
    int n_classes = Ndarray_DIMS(am_scores)[2];
    assert_cmp(Ndarray_DIMS(am_scores)[0], ==, n_time);
    assert_cmp(Ndarray_DIMS(am_scores)[1], ==, n_batch);
    assert_cmp(Ndarray_DIMS(am_scores)[2], ==, n_classes);
    assert_cmp(Ndarray_DIMS(am_seq_len)[0], ==, n_batch);
    int n_edges = Ndarray_DIMS(edges)[1];
    assert_cmp(Ndarray_DIMS(edges)[0], ==, 4);
    assert_cmp(Ndarray_DIMS(edges)[1], ==, n_edges);
    assert_cmp(Ndarray_DIMS(weights)[0], ==, n_edges);
    assert_cmp(Ndarray_DIMS(start_end_states)[0], ==, 2);
    assert_cmp(Ndarray_DIMS(start_end_states)[1], ==, n_batch);
    int n_states = Ndarray_DEV_DATA_int32_scalar(n_states_ref);
    assert_cmp(Ndarray_DIMS(output)[0], ==, n_time);
    assert_cmp(Ndarray_DIMS(output)[1], ==, n_batch);
    assert_cmp(Ndarray_DIMS(score)[0], ==, n_batch);

    int32_t* d_edge_from = Ndarray_DEV_DATA_int32(edges) + 0 * Ndarray_STRIDE(edges, 0);
    int32_t* d_edge_to = Ndarray_DEV_DATA_int32(edges) + 1 * Ndarray_STRIDE(edges, 0);
    int32_t* d_edge_emission_idx = Ndarray_DEV_DATA_int32(edges) + 2 * Ndarray_STRIDE(edges, 0);
    int32_t* d_edge_seq_idx = Ndarray_DEV_DATA_int32(edges) + 3 * Ndarray_STRIDE(edges, 0);
    float* d_edge_weights = Ndarray_DEV_DATA(weights);
    float* d_am_scores = Ndarray_DEV_DATA(am_scores);
    int am_scores_stride = Ndarray_STRIDE(am_scores, 0);
    int32_t* d_am_seq_len = Ndarray_DEV_DATA_int32(am_seq_len);
    int32_t* d_start_states = Ndarray_DEV_DATA_int32(start_end_states) + 0 * Ndarray_STRIDE(start_end_states, 0);
    int32_t* d_end_states = Ndarray_DEV_DATA_int32(start_end_states) + 1 * Ndarray_STRIDE(start_end_states, 0);
    int32_t* d_output = Ndarray_DEV_DATA_int32(output);
    int output_stride = Ndarray_STRIDE(output, 0);
    float* d_score = Ndarray_DEV_DATA(score);

    IdxAndVal* d_buffer = (IdxAndVal*) device_malloc((n_time + 1) * n_states * sizeof(IdxAndVal));
    int buffer_stride = n_states;
    start_dev_kernel(init_buffer, (n_time, n_states, d_buffer));
    start_dev_kernel(init_first_frame, (n_batch, n_states, d_buffer, d_start_states));
    HANDLE_LAST_ERROR();

    for(int t = 0; t < n_time; ++t) {
      start_dev_kernel(next_frame, (
        n_time,
        n_states,
        n_edges,
        n_classes,
        t,
        d_am_scores + t * am_scores_stride,
        d_am_seq_len,
        d_buffer + t * buffer_stride,
        d_buffer + (t + 1) * buffer_stride,
        d_edge_from,
        d_edge_to,
        d_edge_emission_idx,
        d_edge_seq_idx,
        d_edge_weights,
        d_end_states
      ));
    }
    HANDLE_LAST_ERROR();

    start_dev_kernel(select_scores, (
      n_batch,
      n_states,
      buffer_stride,
      d_buffer,
      d_am_seq_len,
      d_end_states,
      d_score // out
    ));

    int32_t* d_cur_state = (int32_t*) device_malloc(n_batch * sizeof(int32_t));
    Ndarray_memcpy(d_cur_state, d_end_states, n_batch * sizeof(int32_t));

    for(int t = n_time - 1; t >= 0; --t) {
      start_dev_kernel(select_best_path, (
        n_batch,
        n_states,
        n_edges,
        t,
        d_cur_state,
        d_buffer + (t + 1) * buffer_stride,
        d_am_seq_len,
        d_edge_from,
        d_edge_to,
        d_edge_emission_idx,
        d_output + t * output_stride // out
      ));
    }
    HANDLE_LAST_ERROR();

    device_free(d_cur_state);
    device_free(d_buffer);
  """

    c_bw_code = None


class GetCtcFsaFastBwOp(NativeOpGenBase):
    # noinspection PyUnresolvedReferences
    """
    This implements :func:`Fsa.get_ctc_fsa_fast_bw` as a native op.
    This is for constructing a FSA with a CTC topology.
    The output format is compatible to the FastBaumWelch native op.

    inputs:
      :param targets: shape (batch,time), int32
      :param seq_lens: shape (batch), int32
      :param blank_idx: scalar, int32
      :param weights: shape (num_edges,), float32 (not used, except for target shape)
      :param label_loop: scalar, int32 (casted from bool). True -> normal CTC; False -> RNA-like
    outputs:
      :param edges: (4,num_edges), int32, edges of the graph (from,to,emission_idx,sequence_idx)
      :param start_end_states: (2,batch), int32, (start,end) state idx in FSA

    To construct `weights` (for FastBaumWelch), `weights` should be just `tf.zeros((num_edges,))`.
    `num_edges` should be `n_batch * (5 * (n_time - 1) + 10)`
      (see construction in kernel why that number).
    """
    in_info = (
        {
            "name": "targets",
            "ndim": 2,
            "shape": (None, None),
            "dtype": "int32",
            "need_contiguous": True,
            "gradient": "disconnected",
        },
        {
            "name": "seq_lens",
            "ndim": 1,
            "shape": (None,),
            "dtype": "int32",
            "need_contiguous": True,
            "gradient": "disconnected",
        },
        {
            "name": "blank_idx",
            "ndim": 0,
            "shape": (),
            "dtype": "int32",
            "need_contiguous": True,
            "gradient": "disconnected",
            "host_memory": True,
        },
        {
            "name": "weights",
            "ndim": 1,
            "shape": (None,),
            "dtype": "float32",
            "need_contiguous": True,
            "gradient": "disconnected",
        },
        {
            "name": "label_loop",
            "ndim": 0,
            "shape": (),
            "dtype": "int32",
            "need_contiguous": True,
            "gradient": "disconnected",
            "host_memory": True,
        },
    )
    out_info = (
        {"name": "edges", "ndim": 2, "shape": (4, (3, 0)), "dtype": "int32", "need_contiguous": True},
        {"name": "start_end_states", "ndim": 2, "shape": (2, (1, 0)), "dtype": "int32", "need_contiguous": True},
    )

    c_extra_support_code = {
        "01_kernel": """
      template<bool label_loop>
      DEF_KERNEL
      void construct_kernel
        (
        int n_batch, int n_time, int n_edges,
        const int32_t* targets, const int32_t* seq_lens,
        int32_t blank_idx,
        int32_t* edges, int32_t* start_end_states
        )
      {
        int idx = threadIdx.x + blockDim.x * blockIdx.x;
        // n_edges should be n_batch * (5 * (n_time - 1) + 10).
        assert(n_edges % n_batch == 0);
        while(idx < n_edges) {
          int batch_idx = idx / (n_edges / n_batch);
          int rel_edge_idx = idx % (n_edges / n_batch);
          int32_t seq_len = seq_lens[batch_idx];
          // state_idx: 0 b, 1 l, 2 b, 3 l, ..., (T-1)*2 b, T*2-1 l, T*2 b, T*2+1 dummy, T*2+2 end
          // i.e. T*2+3 states per seq.
          int state_idx_offset = (n_time * 2 + 3) * batch_idx;
          int t = -1; // pos in targets
          int srel_edge_idx = -1; // state relative edge
          // (seq_len * 2) - 1 is last label state idx. seq_len * 2 is last blank state idx.
          int32_t dummy_state_idx = seq_len * 2 + 1;
          int32_t end_state_idx = seq_len * 2 + 2;
          int32_t state_idx = dummy_state_idx;
          int32_t to_state_idx = dummy_state_idx;
          if(rel_edge_idx == 0) {
            start_end_states[0 * n_batch + batch_idx] = state_idx_offset; // start
            start_end_states[1 * n_batch + batch_idx] = state_idx_offset + end_state_idx; // end
          }
          int32_t emission_idx = blank_idx;
          int32_t label_idx = -1, next_label_idx = -1;
          if(seq_len == 0) {
            t = -1;
            emission_idx = blank_idx;
            // 1 single blank loop
            if(rel_edge_idx == 0) {
              state_idx = 0;
              to_state_idx = 0;
              srel_edge_idx = 0;
            }
            else if(rel_edge_idx == 1) {
              state_idx = 0;
              to_state_idx = end_state_idx;
              srel_edge_idx = 1;
            }
            else {
              state_idx = dummy_state_idx;
              srel_edge_idx = -1;
            }
          }
          else if(seq_len == 1) {
            label_idx = targets[batch_idx * n_time + 0];
            // 3 edges for first / prev last blank
            if(rel_edge_idx < 3) {
              t = 0;
              state_idx = 0;
              srel_edge_idx = rel_edge_idx;
              if(srel_edge_idx == 0) {
                to_state_idx = state_idx;
                emission_idx = blank_idx;
              }
              else if(srel_edge_idx == 1) {
                to_state_idx = state_idx + 1;
                emission_idx = label_idx;
              }
              else if(srel_edge_idx == 2) {
                to_state_idx = end_state_idx;
                emission_idx = label_idx;
              }
            }
            // 4 edges for first / last label
            else if(rel_edge_idx < 7) {
              t = 0;
              state_idx = 1;
              srel_edge_idx = rel_edge_idx - 3;
              if(srel_edge_idx == 0) {
                to_state_idx = label_loop ? state_idx : dummy_state_idx;
                emission_idx = label_idx;
              }
              else if(srel_edge_idx == 1) {
                to_state_idx = state_idx + 1;
                emission_idx = blank_idx;
              }
              else if(srel_edge_idx == 2) {
                to_state_idx = label_loop ? end_state_idx : dummy_state_idx;
                emission_idx = label_idx;
              }
              else if(srel_edge_idx == 3) {
                to_state_idx = end_state_idx;
                emission_idx = blank_idx;
              }
            }
            // 2 edges for last blank
            else if(rel_edge_idx < 9) {
              t = -1;
              emission_idx = blank_idx;
              state_idx = 2;
              srel_edge_idx = rel_edge_idx - 7;
              if(srel_edge_idx == 0)
                to_state_idx = state_idx;
              else
                to_state_idx = end_state_idx;
            }
            else {
              t = -1;
              state_idx = dummy_state_idx;
              srel_edge_idx = -1;
            }
          }
          else { // seq_len >= 2
            // 2 edges for each blank, 3 for each label. up to prev last.
            if(rel_edge_idx < 5 * (seq_len - 1)) {
              t = rel_edge_idx / 5;
              label_idx = targets[batch_idx * n_time + t];
              next_label_idx = targets[batch_idx * n_time + t + 1];
              state_idx = 2 * (rel_edge_idx / 5);
              srel_edge_idx = rel_edge_idx % 5;
              if(srel_edge_idx >= 2) {
                srel_edge_idx -= 2;
                state_idx += 1;
              }
              if(state_idx % 2 == 0) { // blank loop state
                if(srel_edge_idx == 0) {
                  to_state_idx = state_idx;
                  emission_idx = blank_idx;
                }
                else if(srel_edge_idx == 1) {
                  to_state_idx = state_idx + 1;
                  emission_idx = label_idx;
                }
              }
              else { // label loop state
                if(srel_edge_idx == 0) {
                  to_state_idx = label_loop ? state_idx : dummy_state_idx;
                  emission_idx = label_idx;
                }
                else if(srel_edge_idx == 1) {
                  to_state_idx = state_idx + 1;
                  emission_idx = blank_idx;
                }
                else if(srel_edge_idx == 2) {
                  // skip over blank to next label (if allowed <=> next label is different)
                  if(label_idx != next_label_idx || !label_loop) {
                    to_state_idx = state_idx + 2;
                    emission_idx = next_label_idx;
                  }
                }
              }
            }
            // 1 more edge for prev last label
            else if(rel_edge_idx == 5 * (seq_len - 1)) {
              t = seq_len - 2;
              label_idx = targets[batch_idx * n_time + t];
              next_label_idx = targets[batch_idx * n_time + t + 1];
              state_idx = (seq_len - 2) * 2 + 1;
              srel_edge_idx = 3;
              // skip over blank to next label / end state (if allowed <=> next label is different)
              if(label_idx != next_label_idx || !label_loop) {
                to_state_idx = end_state_idx;
                emission_idx = next_label_idx;
              }
            }
            // 3 edges for prev last blank
            else if(rel_edge_idx <= 5 * (seq_len - 1) + 3) {
              t = seq_len - 1;
              label_idx = targets[batch_idx * n_time + t];
              state_idx = (seq_len - 1) * 2;
              srel_edge_idx = rel_edge_idx - (5 * (seq_len - 1) + 1);
              if(srel_edge_idx == 0) {
                to_state_idx = state_idx;
                emission_idx = blank_idx;
              }
              else if(srel_edge_idx == 1) {
                to_state_idx = state_idx + 1;
                emission_idx = label_idx;
              }
              else if(srel_edge_idx == 2) {
                to_state_idx = end_state_idx;
                emission_idx = label_idx;
              }
            }
            // 4 edges for last label
            else if(rel_edge_idx <= 5 * (seq_len - 1) + 7) {
              t = seq_len - 1;
              label_idx = targets[batch_idx * n_time + t];
              state_idx = (seq_len - 1) * 2 + 1;
              srel_edge_idx = rel_edge_idx - (5 * (seq_len - 1) + 4);
              if(srel_edge_idx == 0) {
                to_state_idx = label_loop ? state_idx : dummy_state_idx;
                emission_idx = label_idx;
              }
              else if(srel_edge_idx == 1) {
                to_state_idx = state_idx + 1;
                emission_idx = blank_idx;
              }
              else if(srel_edge_idx == 2) {
                to_state_idx = label_loop ? end_state_idx : dummy_state_idx;
                emission_idx = label_idx;
              }
              else if(srel_edge_idx == 3) {
                to_state_idx = end_state_idx;
                emission_idx = blank_idx;
              }
            }
            // 2 edges for last blank
            else if(rel_edge_idx <= 5 * (seq_len - 1) + 9) {
              t = -1;
              emission_idx = blank_idx;
              state_idx = (seq_len - 1) * 2 + 2;
              srel_edge_idx = rel_edge_idx - (5 * (seq_len - 1) + 8);
              if(srel_edge_idx == 0)
                to_state_idx = state_idx;
              else
                to_state_idx = end_state_idx;
            }
            else {
              t = -1;
              state_idx = dummy_state_idx;
              srel_edge_idx = -1;
            }
          }

          edges[0 * n_edges + idx] = state_idx_offset + state_idx; // from
          edges[1 * n_edges + idx] = state_idx_offset + to_state_idx; // to
          edges[2 * n_edges + idx] = emission_idx; // emission
          edges[3 * n_edges + idx] = batch_idx; // batch

          idx += gridDim.x * blockDim.x;
        }
      }
    """
    }

    c_fw_code = """
    assert(n_inputs == 5);
    assert(n_outputs == 2);
    Ndarray* targets = inputs[0];
    Ndarray* seq_lens = inputs[1];
    Ndarray* blank_idx_ref = inputs[2];
    Ndarray* weights = inputs[3];
    bool label_loop = (bool) Ndarray_DEV_DATA_int32_scalar(inputs[4]);
    Ndarray* edges = *outputs[0];
    Ndarray* start_end_states = *outputs[1];
    assert_cmp(Ndarray_NDIM(targets), ==, 2);
    assert_cmp(Ndarray_NDIM(seq_lens), ==, 1);
    assert_cmp(Ndarray_NDIM(blank_idx_ref), ==, 0);
    assert_cmp(Ndarray_NDIM(weights), ==, 1);
    assert_cmp(Ndarray_NDIM(edges), ==, 2);
    assert_cmp(Ndarray_NDIM(start_end_states), ==, 2);
    int n_batch = Ndarray_DIMS(seq_lens)[0];
    assert_cmp(Ndarray_DIMS(targets)[0], ==, n_batch);
    assert_cmp(Ndarray_DIMS(seq_lens)[0], ==, n_batch);
    assert_cmp(Ndarray_DIMS(start_end_states)[1], ==, n_batch);
    int n_time = Ndarray_DIMS(targets)[1];
    int n_edges = Ndarray_DIMS(weights)[0];
    assert_cmp(Ndarray_DIMS(start_end_states)[0], ==, 2);
    assert_cmp(Ndarray_DIMS(edges)[0], ==, 4);
    assert_cmp(Ndarray_DIMS(edges)[1], ==, n_edges);

    assert_cmp(n_edges, ==, n_batch * (5 * (n_time - 1) + 10));

    Ndarray_memset(Ndarray_DEV_DATA_int32(edges), 255, 4 * n_edges * sizeof(int32_t));
    Ndarray_memset(Ndarray_DEV_DATA_int32(start_end_states), 255, 2 * n_batch * sizeof(int32_t));
    int32_t blank_idx = Ndarray_DEV_DATA_int32_scalar(blank_idx_ref);

    if(label_loop) {
      start_dev_kernel(construct_kernel<true>, (
        n_batch, n_time, n_edges,
        Ndarray_DEV_DATA_int32(targets), Ndarray_DEV_DATA_int32(seq_lens),
        blank_idx,
        Ndarray_DEV_DATA_int32(edges), Ndarray_DEV_DATA_int32(start_end_states)
      ));
    } else {
      start_dev_kernel(construct_kernel<false>, (
        n_batch, n_time, n_edges,
        Ndarray_DEV_DATA_int32(targets), Ndarray_DEV_DATA_int32(seq_lens),
        blank_idx,
        Ndarray_DEV_DATA_int32(edges), Ndarray_DEV_DATA_int32(start_end_states)
      ));
    }
    HANDLE_LAST_ERROR();
  """


class EditDistanceOp(NativeOpGenBase):
    # noinspection PyUnresolvedReferences
    """
    Similar to :func:`tf.edit_distance`.
    Calculates the `edit distance / Levenshtein distance <https://en.wikipedia.org/wiki/Levenshtein_distance>`__.

    The naive implementation either goes over ``a`` and then ``b``, thus results in O(|a|*|b|) time complexity.
    To calculate a new entry in the table (over then length of ``a`` and ``b``),
    it depends on the prev symbol in ``a`` (left) (deletion error),
    the prev symbol in ``b`` (up) (insertion error),
    and the left-up diagonal (substitution error, or no error).

    To take advantage of the parallelism of the GPU, we follow a diagonal iteration scheme, such that
    in every iteration, all entries on the diagonal can be computed in parallel, as they do not depend on each other.
    After implementing this, we found that this algorithm is described here::

      Using GPUs to Speed-Up Levenshtein Edit Distance Computation, Balhaf et al, 2016,
      https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7476090&tag=1

    inputs:
      :param a: symbols. 2d (batch,time), int32
      :param a_len: 1d (batch,), int32
      :param b: symbols. 2d (batch,time), int32
      :param b_len: 1d (batch,), int32
    outputs:
      :param output: 1d (batch,), int32, unnormalized edit distance
    """
    in_info = (
        {
            "name": "a",
            "ndim": 2,
            "shape": (None, None),
            "dtype": "int32",
            "need_contiguous": True,
            "gradient": "disconnected",
        },
        {
            "name": "a_len",
            "ndim": 1,
            "shape": (None,),
            "dtype": "int32",
            "need_contiguous": True,
            "gradient": "disconnected",
        },
        {
            "name": "b",
            "ndim": 2,
            "shape": (None, None),
            "dtype": "int32",
            "need_contiguous": True,
            "gradient": "disconnected",
        },
        {
            "name": "b_len",
            "ndim": 1,
            "shape": (None,),
            "dtype": "int32",
            "need_contiguous": True,
            "gradient": "disconnected",
        },
    )
    out_info = ({"name": "output", "ndim": 1, "shape": ((0, 0),), "dtype": "int32", "need_contiguous": True},)

    c_extra_support_code = {
        "001_next_step": """
      DEF_KERNEL
      void next_step_kernel(
            int n_batch, int n_a_max_len, int n_b_max_len,
            int diag_idx,
            const int32_t* a, const int32_t* b,
            const int32_t* a_len, const int32_t* b_len,
            const int32_t* last1_dist, const int32_t* last2_dist, int32_t* cur_dist,
            int32_t* result) {
        int idx = threadIdx.x + blockDim.x * blockIdx.x;
        // We are going diagonal!
        int num_entries;
        if(diag_idx <= n_a_max_len) {
          num_entries = diag_idx + 1;
          if(num_entries > n_b_max_len + 1)
            num_entries = n_b_max_len + 1;
        } else {
          num_entries = n_b_max_len + 1 - (diag_idx - n_a_max_len);
          if(num_entries > n_a_max_len + 1)
            num_entries = n_a_max_len + 1;
        }
        int max_num_entries = n_a_max_len + 1;
        if(max_num_entries > n_b_max_len + 1)
          max_num_entries = n_b_max_len + 1;
        while(idx < n_batch * num_entries) {
          int batch_idx = idx / num_entries;
          int entry_idx = idx % num_entries;
          int dist_idx = batch_idx * max_num_entries + entry_idx;

          int t_a, t_b;
          if(diag_idx <= n_a_max_len) {
            t_a = diag_idx - entry_idx;
            t_b = entry_idx;
          } else {
            t_a = n_a_max_len - entry_idx;
            t_b = diag_idx - n_a_max_len + entry_idx;
          }

          if(t_a == 0)
            cur_dist[dist_idx] = t_b;  // distance == how much to delete from b
          else if(t_b == 0)
            cur_dist[dist_idx] = t_a;  // distance == how much to delete from a
          else {
            // last1 is with diag_idx - 2. Needed for substitution cost.
            // last2 is with diag_idx - 1. Needed for insertion or deletion cost.
            // last2 refers to the first, for deletion. last2_idx + 1 is for insertion.
            int last1_idx, last2_idx;
            if(diag_idx - 1 < n_a_max_len)
              last1_idx = dist_idx - 1;
            else if(diag_idx - 1 == n_a_max_len)
              last1_idx = dist_idx;
            else
              last1_idx = dist_idx + 1;
            if(diag_idx <= n_a_max_len)
              last2_idx = dist_idx - 1;
            else
              last2_idx = dist_idx;

            int del_cost, ins_cost, sub_cost;
            del_cost = last2_dist[last2_idx] + 1;
            ins_cost = last2_dist[last2_idx + 1] + 1;
            sub_cost = last1_dist[last1_idx];
            if(a[batch_idx * n_a_max_len + t_a - 1] != b[batch_idx * n_b_max_len + t_b - 1])
              ++sub_cost;
            //printf("t_a %i, t_b %i, del %i, ins %i, sub %i\\n", t_a, t_b, del_cost, ins_cost, sub_cost);
            int min_cost = del_cost;
            if(min_cost > ins_cost) min_cost = ins_cost;
            if(min_cost > sub_cost) min_cost = sub_cost;
            cur_dist[dist_idx] = min_cost;
          }
          //printf("t_a %i, t_b %i, dist %i\\n", t_a, t_b, cur_dist[dist_idx]);

          if(t_a == a_len[batch_idx] && t_b == b_len[batch_idx])
            result[batch_idx] = cur_dist[dist_idx];

          idx += gridDim.x * blockDim.x;
        }
      }
    """
    }

    c_fw_code = """
    assert(n_inputs == 4);
    assert(n_outputs == 1);
    Ndarray* a = inputs[0];
    Ndarray* a_len = inputs[1];
    Ndarray* b = inputs[2];
    Ndarray* b_len = inputs[3];
    Ndarray* out = *outputs[0];
    assert_cmp(Ndarray_NDIM(a), ==, 2);
    assert_cmp(Ndarray_NDIM(a_len), ==, 1);
    assert_cmp(Ndarray_NDIM(b), ==, 2);
    assert_cmp(Ndarray_NDIM(b_len), ==, 1);
    assert_cmp(Ndarray_NDIM(out), ==, 1);
    int n_batch = Ndarray_DIMS(out)[0];
    assert_cmp(Ndarray_DIMS(a)[0], ==, n_batch);
    assert_cmp(Ndarray_DIMS(a_len)[0], ==, n_batch);
    assert_cmp(Ndarray_DIMS(b)[0], ==, n_batch);
    assert_cmp(Ndarray_DIMS(b_len)[0], ==, n_batch);
    int n_a_max_len = Ndarray_DIMS(a)[1];
    int n_b_max_len = Ndarray_DIMS(b)[1];
    Ndarray_memset(Ndarray_DEV_DATA_int32(out), 255, n_batch * sizeof(int32_t));

    // Working buffer.
    int max_num_entries = std::min(n_a_max_len + 1, n_b_max_len + 1);
    int32_t* buffer = (int32_t*) device_malloc(3 * n_batch * max_num_entries * sizeof(int32_t));
    int32_t* last1_dist = buffer;
    int32_t* last2_dist = buffer + n_batch * max_num_entries;
    int32_t* cur_dist = buffer + 2 * n_batch * max_num_entries;

    int num_diag = n_a_max_len + n_b_max_len + 1;
    for(int diag_idx = 0; diag_idx < num_diag; ++diag_idx) {
      start_dev_kernel(next_step_kernel, (
        n_batch, n_a_max_len, n_b_max_len,
        diag_idx,
        Ndarray_DEV_DATA_int32(a), Ndarray_DEV_DATA_int32(b),
        Ndarray_DEV_DATA_int32(a_len), Ndarray_DEV_DATA_int32(b_len),
        last1_dist, last2_dist, cur_dist,
        Ndarray_DEV_DATA_int32(out)));
      // Rotate. last1_dist not needed anymore.
      int32_t* tmp = last1_dist;
      last1_dist = last2_dist;
      last2_dist = cur_dist;
      cur_dist = tmp;
    }
    HANDLE_LAST_ERROR();

    device_free(buffer);
  """

    c_bw_code = None


class OptimalCompletionEditDistanceOp(NativeOpGenBase):
    # noinspection PyUnresolvedReferences
    """
    Given some prefix ``a``, what is the minimum possible edit distance to ``b`` with any possible suffix on ``a`` ?
    This is described in `Optimal Completion Distillation (OCD) <https://arxiv.org/abs/1810.01398>`__.
    The implementation is derived from :class:`EditDistanceOp`.

    inputs:
      :param a: symbols. 2d (batch,time), int32. prefix.
      :param a_len: 1d (batch,), int32
      :param b: symbols. 2d (batch,time), int32
      :param b_len: 1d (batch,), int32
    outputs:
      :param output: 1d (batch,), int32, unnormalized edit distance
    """
    in_info = (
        {
            "name": "a",
            "ndim": 2,
            "shape": (None, None),
            "dtype": "int32",
            "need_contiguous": True,
            "gradient": "disconnected",
        },
        {
            "name": "a_len",
            "ndim": 1,
            "shape": (None,),
            "dtype": "int32",
            "need_contiguous": True,
            "gradient": "disconnected",
        },
        {
            "name": "b",
            "ndim": 2,
            "shape": (None, None),
            "dtype": "int32",
            "need_contiguous": True,
            "gradient": "disconnected",
        },
        {
            "name": "b_len",
            "ndim": 1,
            "shape": (None,),
            "dtype": "int32",
            "need_contiguous": True,
            "gradient": "disconnected",
        },
    )
    out_info = ({"name": "output", "ndim": 1, "shape": ((0, 0),), "dtype": "int32", "need_contiguous": True},)

    c_extra_support_code = {
        "001_init_result": """
      DEF_KERNEL
      void init_result_kernel(int n_batch, int32_t* result) {
        int idx = threadIdx.x + blockDim.x * blockIdx.x;
        while(idx < n_batch) {
          result[idx] = 2147483647;  // biggest int32
          idx += gridDim.x * blockDim.x;
        }
      }
    """,
        "002_next_step": """
      DEF_KERNEL
      void next_step_kernel(
            int n_batch, int n_a_max_len, int n_b_max_len,
            int diag_idx,
            const int32_t* a, const int32_t* b,
            const int32_t* a_len, const int32_t* b_len,
            const int32_t* last1_dist, const int32_t* last2_dist, int32_t* cur_dist,
            int32_t* result) {
        int idx = threadIdx.x + blockDim.x * blockIdx.x;
        // We are going diagonal!
        int num_entries;
        if(diag_idx <= n_a_max_len) {
          num_entries = diag_idx + 1;
          if(num_entries > n_b_max_len + 1)
            num_entries = n_b_max_len + 1;
        } else {
          num_entries = n_b_max_len + 1 - (diag_idx - n_a_max_len);
          if(num_entries > n_a_max_len + 1)
            num_entries = n_a_max_len + 1;
        }
        int max_num_entries = n_a_max_len + 1;
        if(max_num_entries > n_b_max_len + 1)
          max_num_entries = n_b_max_len + 1;
        while(idx < n_batch * num_entries) {
          int batch_idx = idx / num_entries;
          int entry_idx = idx % num_entries;
          int dist_idx = batch_idx * max_num_entries + entry_idx;

          int t_a, t_b;
          if(diag_idx <= n_a_max_len) {
            t_a = diag_idx - entry_idx;
            t_b = entry_idx;
          } else {
            t_a = n_a_max_len - entry_idx;
            t_b = diag_idx - n_a_max_len + entry_idx;
          }

          if(t_a == 0)
            cur_dist[dist_idx] = t_b;  // distance == how much to delete from b
          else if(t_b == 0)
            cur_dist[dist_idx] = t_a;  // distance == how much to delete from a
          else {
            // last1 is with diag_idx - 2. Needed for substitution cost.
            // last2 is with diag_idx - 1. Needed for insertion or deletion cost.
            // last2 refers to the first, for deletion. last2_idx + 1 is for insertion.
            int last1_idx, last2_idx;
            if(diag_idx - 1 < n_a_max_len)
              last1_idx = dist_idx - 1;
            else if(diag_idx - 1 == n_a_max_len)
              last1_idx = dist_idx;
            else
              last1_idx = dist_idx + 1;
            if(diag_idx <= n_a_max_len)
              last2_idx = dist_idx - 1;
            else
              last2_idx = dist_idx;

            int del_cost, ins_cost, sub_cost;
            del_cost = last2_dist[last2_idx] + 1;
            ins_cost = last2_dist[last2_idx + 1] + 1;
            sub_cost = last1_dist[last1_idx];
            if(a[batch_idx * n_a_max_len + t_a - 1] != b[batch_idx * n_b_max_len + t_b - 1])
              ++sub_cost;
            int min_cost = del_cost;
            if(min_cost > ins_cost) min_cost = ins_cost;
            if(min_cost > sub_cost) min_cost = sub_cost;
            cur_dist[dist_idx] = min_cost;
          }

          if(t_a == a_len[batch_idx] && t_b <= b_len[batch_idx])
            elem_atomic_min(&result[batch_idx], cur_dist[dist_idx]);

          idx += gridDim.x * blockDim.x;
        }
      }
    """,
    }

    c_fw_code = """
    assert(n_inputs == 4);
    assert(n_outputs == 1);
    Ndarray* a = inputs[0];
    Ndarray* a_len = inputs[1];
    Ndarray* b = inputs[2];
    Ndarray* b_len = inputs[3];
    Ndarray* out = *outputs[0];
    assert_cmp(Ndarray_NDIM(a), ==, 2);
    assert_cmp(Ndarray_NDIM(a_len), ==, 1);
    assert_cmp(Ndarray_NDIM(b), ==, 2);
    assert_cmp(Ndarray_NDIM(b_len), ==, 1);
    assert_cmp(Ndarray_NDIM(out), ==, 1);
    int n_batch = Ndarray_DIMS(out)[0];
    assert_cmp(Ndarray_DIMS(a)[0], ==, n_batch);
    assert_cmp(Ndarray_DIMS(a_len)[0], ==, n_batch);
    assert_cmp(Ndarray_DIMS(b)[0], ==, n_batch);
    assert_cmp(Ndarray_DIMS(b_len)[0], ==, n_batch);
    int n_a_max_len = Ndarray_DIMS(a)[1];
    int n_b_max_len = Ndarray_DIMS(b)[1];
    start_dev_kernel(init_result_kernel, (n_batch, Ndarray_DEV_DATA_int32(out)));

    // Working buffer.
    int max_num_entries = std::min(n_a_max_len + 1, n_b_max_len + 1);
    int32_t* buffer = (int32_t*) device_malloc(3 * n_batch * max_num_entries * sizeof(int32_t));
    int32_t* last1_dist = buffer;
    int32_t* last2_dist = buffer + n_batch * max_num_entries;
    int32_t* cur_dist = buffer + 2 * n_batch * max_num_entries;

    int num_diag = n_a_max_len + n_b_max_len + 1;
    for(int diag_idx = 0; diag_idx < num_diag; ++diag_idx) {
      start_dev_kernel(next_step_kernel, (
        n_batch, n_a_max_len, n_b_max_len,
        diag_idx,
        Ndarray_DEV_DATA_int32(a), Ndarray_DEV_DATA_int32(b),
        Ndarray_DEV_DATA_int32(a_len), Ndarray_DEV_DATA_int32(b_len),
        last1_dist, last2_dist, cur_dist,
        Ndarray_DEV_DATA_int32(out)));
      // Rotate. last1_dist not needed anymore.
      int32_t* tmp = last1_dist;
      last1_dist = last2_dist;
      last2_dist = cur_dist;
      cur_dist = tmp;
    }
    HANDLE_LAST_ERROR();

    device_free(buffer);
  """

    c_bw_code = None


class OptimalCompletionEditDistancePerSuccessorOp(NativeOpGenBase):
    # noinspection PyUnresolvedReferences
    """
    Given some prefix ``a`` + successor,
    what is the minimum possible edit distance to ``b`` with any possible suffix on ``a`` + successor,
    for successor in ``successors``.
    This is described in `Optimal Completion Distillation (OCD) <https://arxiv.org/abs/1810.01398>`__.
    The implementation is derived from :class:`OptimalCompletionEditDistanceOp`.

    inputs:
      :param a: symbols. 2d (batch,time), int32. prefix.
      :param a_len: 1d (batch,), int32
      :param b: symbols. 2d (batch,time), int32
      :param b_len: 1d (batch,), int32
      :param successors: 1d (num_labels,), int32
    outputs:
      :param output: 2d (batch,num_labels), int32, unnormalized edit distance
    """
    in_info = (
        {
            "name": "a",
            "ndim": 2,
            "shape": (None, None),
            "dtype": "int32",
            "need_contiguous": True,
            "gradient": "disconnected",
        },
        {
            "name": "a_len",
            "ndim": 1,
            "shape": (None,),
            "dtype": "int32",
            "need_contiguous": True,
            "gradient": "disconnected",
        },
        {
            "name": "b",
            "ndim": 2,
            "shape": (None, None),
            "dtype": "int32",
            "need_contiguous": True,
            "gradient": "disconnected",
        },
        {
            "name": "b_len",
            "ndim": 1,
            "shape": (None,),
            "dtype": "int32",
            "need_contiguous": True,
            "gradient": "disconnected",
        },
        {
            "name": "successors",
            "ndim": 1,
            "shape": (None,),
            "dtype": "int32",
            "need_contiguous": True,
            "gradient": "disconnected",
        },
    )
    out_info = ({"name": "output", "ndim": 2, "shape": ((0, 0), (4, 0)), "dtype": "int32", "need_contiguous": True},)

    c_extra_support_code = {
        "001_next_step": """
      DEF_KERNEL
      void next_step_kernel(
            int n_batch, int n_a_max_len, int n_b_max_len,
            int diag_idx,
            const int32_t* a, const int32_t* b,
            const int32_t* a_len, const int32_t* b_len,
            const int32_t* last1_dist, const int32_t* last2_dist, int32_t* cur_dist, int32_t* a_last_row) {
        int idx = threadIdx.x + blockDim.x * blockIdx.x;
        // We are going diagonal!
        int num_entries;
        if(diag_idx <= n_a_max_len) {
          num_entries = diag_idx + 1;
          if(num_entries > n_b_max_len + 1)
            num_entries = n_b_max_len + 1;
        } else {
          num_entries = n_b_max_len + 1 - (diag_idx - n_a_max_len);
          if(num_entries > n_a_max_len + 1)
            num_entries = n_a_max_len + 1;
        }
        int max_num_entries = n_a_max_len + 1;
        if(max_num_entries > n_b_max_len + 1)
          max_num_entries = n_b_max_len + 1;
        while(idx < n_batch * num_entries) {
          int batch_idx = idx / num_entries;
          int entry_idx = idx % num_entries;
          int dist_idx = batch_idx * max_num_entries + entry_idx;

          int t_a, t_b;
          if(diag_idx <= n_a_max_len) {
            t_a = diag_idx - entry_idx;
            t_b = entry_idx;
          } else {
            t_a = n_a_max_len - entry_idx;
            t_b = diag_idx - n_a_max_len + entry_idx;
          }

          if(t_a == 0)
            cur_dist[dist_idx] = t_b;  // distance == how much to delete from b
          else if(t_b == 0)
            cur_dist[dist_idx] = t_a;  // distance == how much to delete from a
          else {
            // last1 is with diag_idx - 2. Needed for substitution cost.
            // last2 is with diag_idx - 1. Needed for insertion or deletion cost.
            // last2 refers to the first, for deletion. last2_idx + 1 is for insertion.
            int last1_idx, last2_idx;
            if(diag_idx - 1 < n_a_max_len)
              last1_idx = dist_idx - 1;
            else if(diag_idx - 1 == n_a_max_len)
              last1_idx = dist_idx;
            else
              last1_idx = dist_idx + 1;
            if(diag_idx <= n_a_max_len)
              last2_idx = dist_idx - 1;
            else
              last2_idx = dist_idx;

            int del_cost, ins_cost, sub_cost;
            del_cost = last2_dist[last2_idx] + 1;
            ins_cost = last2_dist[last2_idx + 1] + 1;
            sub_cost = last1_dist[last1_idx];
            if(a[batch_idx * n_a_max_len + t_a - 1] != b[batch_idx * n_b_max_len + t_b - 1])
              ++sub_cost;
            int min_cost = del_cost;
            if(min_cost > ins_cost) min_cost = ins_cost;
            if(min_cost > sub_cost) min_cost = sub_cost;
            cur_dist[dist_idx] = min_cost;
          }

          if(t_a == a_len[batch_idx] && t_b <= b_len[batch_idx])
            a_last_row[batch_idx * (n_b_max_len + 1) + t_b] = cur_dist[dist_idx];

          idx += gridDim.x * blockDim.x;
        }
      }
    """,
        "002_init_result": """
      DEF_KERNEL
      void init_result_kernel(
            int n_batch, int n_b_max_len, int n_labels,
            const int32_t* a_len, const int32_t* b_len,
            const int32_t* a_last_row,
            int32_t* result
      ) {
        int idx = threadIdx.x + blockDim.x * blockIdx.x;
        while(idx < n_batch * n_labels) {
          int batch_idx = idx / n_labels;
          int successor_idx = idx % n_labels;

          // Initial insertion, last deletion.
          int t_a = a_len[batch_idx] + 1;
          int min_cost = t_a;
          int last_del_cost = a_last_row[batch_idx * (n_b_max_len + 1) + b_len[batch_idx]] + 1;
          if(min_cost > last_del_cost) min_cost = last_del_cost;
          result[batch_idx * n_labels + successor_idx] = min_cost;

          idx += gridDim.x * blockDim.x;
        }
      }
    """,
        "003_expand": """
      DEF_KERNEL
      void expand_kernel(
            int n_batch, int n_b_max_len, int n_labels,
            const int32_t* b,
            const int32_t* b_len,
            const int32_t* a_last_row,
            const int32_t* successors,
            int32_t* result
      ) {
        int idx = threadIdx.x + blockDim.x * blockIdx.x;
        while(idx < n_batch * n_labels * n_b_max_len) {
          int batch_idx = idx / n_b_max_len / n_labels;
          int successor_idx = (idx / n_b_max_len) % n_labels;
          int t_b = idx % n_b_max_len;
          int successor = successors[successor_idx];

          if(t_b < b_len[batch_idx]) {
            // We can ignore insertion/deletion
            // (except initial insertion / last deletion, see init_result_kernel).
            int sub_cost = a_last_row[batch_idx * (n_b_max_len + 1) + t_b];
            if(successor != b[batch_idx * n_b_max_len + t_b])
              ++sub_cost;
            elem_atomic_min(&result[batch_idx * n_labels + successor_idx], sub_cost);
          }

          idx += gridDim.x * blockDim.x;
        }
      }
    """,
    }

    c_fw_code = """
    assert(n_inputs == 5);
    assert(n_outputs == 1);
    Ndarray* a = inputs[0];
    Ndarray* a_len = inputs[1];
    Ndarray* b = inputs[2];
    Ndarray* b_len = inputs[3];
    Ndarray* successors = inputs[4];
    Ndarray* out = *outputs[0];
    assert_cmp(Ndarray_NDIM(a), ==, 2);
    assert_cmp(Ndarray_NDIM(a_len), ==, 1);
    assert_cmp(Ndarray_NDIM(b), ==, 2);
    assert_cmp(Ndarray_NDIM(b_len), ==, 1);
    assert_cmp(Ndarray_NDIM(successors), ==, 1);
    assert_cmp(Ndarray_NDIM(out), ==, 2);
    int n_batch = Ndarray_DIMS(out)[0];
    int n_labels = Ndarray_DIMS(successors)[0];
    assert_cmp(Ndarray_DIMS(out)[1], ==, n_labels);
    assert_cmp(Ndarray_DIMS(a)[0], ==, n_batch);
    assert_cmp(Ndarray_DIMS(a_len)[0], ==, n_batch);
    assert_cmp(Ndarray_DIMS(b)[0], ==, n_batch);
    assert_cmp(Ndarray_DIMS(b_len)[0], ==, n_batch);
    int n_a_max_len = Ndarray_DIMS(a)[1];
    int n_b_max_len = Ndarray_DIMS(b)[1];
    Ndarray_memset(Ndarray_DEV_DATA_int32(out), 255, n_batch * n_labels * sizeof(int32_t));

    // Working buffer.
    int max_num_entries = std::min(n_a_max_len + 1, n_b_max_len + 1);
    int32_t* buffer = (int32_t*) device_malloc(3 * n_batch * max_num_entries * sizeof(int32_t));
    int32_t* last1_dist = buffer;
    int32_t* last2_dist = buffer + n_batch * max_num_entries;
    int32_t* cur_dist = buffer + 2 * n_batch * max_num_entries;
    int32_t* a_last_row = (int32_t*) device_malloc(n_batch * (n_b_max_len + 1) * sizeof(int32_t));

    int num_diag = n_a_max_len + n_b_max_len + 1;
    for(int diag_idx = 0; diag_idx < num_diag; ++diag_idx) {
      start_dev_kernel(next_step_kernel, (
        n_batch, n_a_max_len, n_b_max_len,
        diag_idx,
        Ndarray_DEV_DATA_int32(a), Ndarray_DEV_DATA_int32(b),
        Ndarray_DEV_DATA_int32(a_len), Ndarray_DEV_DATA_int32(b_len),
        last1_dist, last2_dist, cur_dist, a_last_row
      ));
      // Rotate. last1_dist not needed anymore.
      int32_t* tmp = last1_dist;
      last1_dist = last2_dist;
      last2_dist = cur_dist;
      cur_dist = tmp;
    }
    HANDLE_LAST_ERROR();

    start_dev_kernel(init_result_kernel, (
      n_batch, n_b_max_len, n_labels,
      Ndarray_DEV_DATA_int32(a_len), Ndarray_DEV_DATA_int32(b_len),
      a_last_row,
      Ndarray_DEV_DATA_int32(out)
    ));
    HANDLE_LAST_ERROR();

    start_dev_kernel(expand_kernel, (
      n_batch, n_b_max_len, n_labels,
      Ndarray_DEV_DATA_int32(b),
      Ndarray_DEV_DATA_int32(b_len),
      a_last_row,
      Ndarray_DEV_DATA_int32(successors),
      Ndarray_DEV_DATA_int32(out)
    ));
    HANDLE_LAST_ERROR();

    device_free(buffer);
    device_free(a_last_row);
  """

    c_bw_code = None


class NextEditDistanceRowOp(NativeOpGenBase):
    # noinspection PyUnresolvedReferences
    """
    This does a single step in calculating the edit distance table, going over the symbols in ``a``.
    Note that when you have the full sequence ``a`` in advance, :class:`EditDistanceOp` should be faster.
    However, this iterative op is useful when ``a`` is constructed step by step.

    inputs:
      :param last_row: 2d (batch,b_time + 1), int32. last edit distances
      :param a: symbols. 1d (batch,), int32. current.
      :param a_n: (batch,), int32. current position
      :param a_ended: 1d (batch,), int32 (casted from bool, because int32 easier to handle)
      :param b: symbols. 2d (batch,b_time), int32
      :param b_len: 1d (batch,), int32
    outputs:
      :param output: 2d (batch,b_time + 1), int32, next (unnormalized) edit distance row
    """
    in_info = (
        {
            "name": "last_row",
            "ndim": 2,
            "shape": (None, None),
            "dtype": "int32",
            "need_contiguous": True,
            "gradient": "disconnected",
        },
        {
            "name": "a",
            "ndim": 1,
            "shape": (None,),
            "dtype": "int32",
            "need_contiguous": True,
            "gradient": "disconnected",
        },
        {
            "name": "a_n",
            "ndim": 1,
            "shape": (None,),
            "dtype": "int32",
            "need_contiguous": True,
            "gradient": "disconnected",
        },
        {
            "name": "a_ended",
            "ndim": 1,
            "shape": (None,),
            "dtype": "int32",
            "need_contiguous": True,
            "gradient": "disconnected",
        },
        {
            "name": "b",
            "ndim": 2,
            "shape": (None, None),
            "dtype": "int32",
            "need_contiguous": True,
            "gradient": "disconnected",
        },
        {
            "name": "b_len",
            "ndim": 1,
            "shape": (None,),
            "dtype": "int32",
            "need_contiguous": True,
            "gradient": "disconnected",
        },
    )
    out_info = ({"name": "output", "ndim": 2, "shape": ((0, 0), (0, 1)), "dtype": "int32", "need_contiguous": True},)

    c_extra_support_code = {
        "001_next_row": """
      DEF_KERNEL
      void next_row_kernel(
            int n_batch, int n_b_max_len,
            const int32_t* last_row,
            const int32_t* a, const int32_t* a_n, const int32_t* a_ended,
            const int32_t* b, const int32_t* b_len,
            int32_t* next_row
      ) {
        int idx = threadIdx.x + blockDim.x * blockIdx.x;
        while(idx < n_batch) {
          int batch_idx = idx;

          int last_dist;
          if(!a_ended[batch_idx]) {
            last_dist = a_n[batch_idx] + 1;  // Initial deletion error.
            next_row[batch_idx * (n_b_max_len + 1)] = last_dist;
            for(int t_b = 1; t_b <= b_len[batch_idx]; ++t_b) {
              int ins_error = last_row[batch_idx * (n_b_max_len + 1) + t_b] + 1;
              int del_error = last_dist + 1;
              int sub_error = last_row[batch_idx * (n_b_max_len + 1) + t_b - 1];
              if(a[batch_idx] != b[batch_idx * n_b_max_len + t_b - 1])
                ++sub_error;
              last_dist = ins_error;
              if(last_dist > del_error) last_dist = del_error;
              if(last_dist > sub_error) last_dist = sub_error;
              next_row[batch_idx * (n_b_max_len + 1) + t_b] = last_dist;
            }
          }
          else {  // a ended
            // Just copy over.
            for(int t_b = 0; t_b <= b_len[batch_idx]; ++t_b) {
              last_dist = last_row[batch_idx * (n_b_max_len + 1) + t_b];
              next_row[batch_idx * (n_b_max_len + 1) + t_b] = last_dist;
            }
          }
          // Repeat last entry.
          for(int t_b = b_len[batch_idx] + 1; t_b < n_b_max_len + 1; ++t_b)
            next_row[batch_idx * (n_b_max_len + 1) + t_b] = last_dist;

          idx += gridDim.x * blockDim.x;
        }
      }
    """
    }

    c_fw_code = """
    assert(n_inputs == 6);
    assert(n_outputs == 1);
    Ndarray* last_row = inputs[0];
    Ndarray* a = inputs[1];
    Ndarray* a_n = inputs[2];
    Ndarray* a_ended = inputs[3];
    Ndarray* b = inputs[4];
    Ndarray* b_len = inputs[5];
    Ndarray* out = *outputs[0];
    assert_cmp(Ndarray_NDIM(last_row), ==, 2);
    assert_cmp(Ndarray_NDIM(a), ==, 1);
    assert_cmp(Ndarray_NDIM(a_n), ==, 1);
    assert_cmp(Ndarray_NDIM(a_ended), ==, 1);
    assert_cmp(Ndarray_NDIM(b), ==, 2);
    assert_cmp(Ndarray_NDIM(b_len), ==, 1);
    assert_cmp(Ndarray_NDIM(out), ==, 2);
    int n_batch = Ndarray_DIMS(out)[0];
    int n_b_max_len = Ndarray_DIMS(b)[1];
    assert_cmp(Ndarray_DIMS(out)[0], ==, n_batch);
    assert_cmp(Ndarray_DIMS(out)[1], ==, n_b_max_len + 1);
    assert_cmp(Ndarray_DIMS(last_row)[0], ==, n_batch);
    assert_cmp(Ndarray_DIMS(last_row)[1], ==, n_b_max_len + 1);
    assert_cmp(Ndarray_DIMS(a)[0], ==, n_batch);
    assert_cmp(Ndarray_DIMS(a_n)[0], ==, n_batch);
    assert_cmp(Ndarray_DIMS(a_ended)[0], ==, n_batch);
    assert_cmp(Ndarray_DIMS(b)[0], ==, n_batch);
    assert_cmp(Ndarray_DIMS(b)[1], ==, n_b_max_len);
    assert_cmp(Ndarray_DIMS(b_len)[0], ==, n_batch);

    start_dev_kernel(next_row_kernel, (
      n_batch, n_b_max_len,
      Ndarray_DEV_DATA_int32(last_row),
      Ndarray_DEV_DATA_int32(a), Ndarray_DEV_DATA_int32(a_n), Ndarray_DEV_DATA_int32(a_ended),
      Ndarray_DEV_DATA_int32(b), Ndarray_DEV_DATA_int32(b_len),
      Ndarray_DEV_DATA_int32(out)
    ));
    HANDLE_LAST_ERROR();
  """

    c_bw_code = None


class NextEditDistanceReduceOp(NativeOpGenBase):
    # noinspection PyUnresolvedReferences
    """
    Code derived from :class:`NextEditDistanceRowOp`.

    inputs:
      :param last_row: 2d (batch,b_time + 1), int32. last edit distances
      :param a: symbols. 2d (batch|1,n_labels), int32. current.
      :param a_n: 1d (batch,), int32. current position
      :param a_ended: 1d (batch,), int32 (casted from bool, because int32 easier to handle)
      :param b: symbols. 2d (batch,b_time), int32
      :param b_len: 1d (batch,), int32
      :param optimal_completion: scalar, int32 (casted from bool). True -> reduce_min over row; False -> last of row
      :param a_blank_idx: scalar, int32. use -1 to not use
    outputs:
      :param output: 2d (batch,n_labels), int32, next (unnormalized) (maybe optional) edit distance
    """
    in_info = (
        {
            "name": "last_row",
            "ndim": 2,
            "shape": (None, None),
            "dtype": "int32",
            "need_contiguous": True,
            "gradient": "disconnected",
        },
        {
            "name": "a",
            "ndim": 2,
            "shape": (None, None),
            "dtype": "int32",
            "need_contiguous": True,
            "gradient": "disconnected",
        },
        {
            "name": "a_n",
            "ndim": 1,
            "shape": (None,),
            "dtype": "int32",
            "need_contiguous": True,
            "gradient": "disconnected",
        },
        {
            "name": "a_ended",
            "ndim": 1,
            "shape": (None,),
            "dtype": "int32",
            "need_contiguous": True,
            "gradient": "disconnected",
        },
        {
            "name": "b",
            "ndim": 2,
            "shape": (None, None),
            "dtype": "int32",
            "need_contiguous": True,
            "gradient": "disconnected",
        },
        {
            "name": "b_len",
            "ndim": 1,
            "shape": (None,),
            "dtype": "int32",
            "need_contiguous": True,
            "gradient": "disconnected",
        },
        {
            "name": "optimal_completion",
            "ndim": 0,
            "shape": (),
            "dtype": "int32",
            "gradient": "disconnected",
            "host_memory": True,
        },
        {
            "name": "a_blank_idx",
            "ndim": 0,
            "shape": (),
            "dtype": "int32",
            "gradient": "disconnected",
            "host_memory": True,
        },
    )
    out_info = ({"name": "output", "ndim": 2, "shape": ((0, 0), (1, 1)), "dtype": "int32", "need_contiguous": True},)

    c_extra_support_code = {
        "001_calc_result": """
      DEF_KERNEL
      void calc_result_kernel(
            int n_batch, int n_b_max_len, int n_labels,
            const int32_t* last_row,
            const int32_t* a, const int32_t* a_n, const int32_t* a_ended,
            const int32_t* b, const int32_t* b_len,
            int32_t* result,
            bool optimal_completion,
            bool a_broadcast_batch,
            int32_t a_blank_idx
      ) {
        int idx = threadIdx.x + blockDim.x * blockIdx.x;
        while(idx < n_batch * n_labels) {
          int batch_idx = idx / n_labels;
          int label_idx = idx % n_labels;
          int a_label = a[(a_broadcast_batch ? 0 : batch_idx) * n_labels + label_idx];

          int total_min_error;
          int last_dist;
          if(!a_ended[batch_idx] && a_label != a_blank_idx) {
            last_dist = a_n[batch_idx] + 1;  // Initial deletion error.
            total_min_error = last_dist;
            for(int t_b = 1; t_b <= b_len[batch_idx]; ++t_b) {
              int ins_error = last_row[batch_idx * (n_b_max_len + 1) + t_b] + 1;
              int del_error = last_dist + 1;
              int sub_error = last_row[batch_idx * (n_b_max_len + 1) + t_b - 1];
              if(a_label != b[batch_idx * n_b_max_len + t_b - 1])
                ++sub_error;
              int min_error = ins_error;
              if(min_error > del_error) min_error = del_error;
              if(min_error > sub_error) min_error = sub_error;
              last_dist = min_error;
              if(total_min_error > last_dist) total_min_error = last_dist;
            }
          }
          else {  // a ended or blank
            // Just copy over.
            total_min_error = last_row[batch_idx * (n_b_max_len + 1)];
            for(int t_b = 0; t_b <= b_len[batch_idx]; ++t_b) {
              last_dist = last_row[batch_idx * (n_b_max_len + 1) + t_b];
              if(total_min_error > last_dist) total_min_error = last_dist;
            }
          }

          result[batch_idx * n_labels + label_idx] = optimal_completion ? total_min_error : last_dist;

          idx += gridDim.x * blockDim.x;
        }
      }
    """
    }

    c_fw_code = """
    assert(n_inputs == 8);
    assert(n_outputs == 1);
    Ndarray* last_row = inputs[0];
    Ndarray* a = inputs[1];
    Ndarray* a_n = inputs[2];
    Ndarray* a_ended = inputs[3];
    Ndarray* b = inputs[4];
    Ndarray* b_len = inputs[5];
    bool optimal_completion = (bool) Ndarray_DEV_DATA_int32_scalar(inputs[6]);
    int32_t a_blank_idx = Ndarray_DEV_DATA_int32_scalar(inputs[7]);
    Ndarray* out = *outputs[0];
    assert_cmp(Ndarray_NDIM(last_row), ==, 2);
    assert_cmp(Ndarray_NDIM(a), ==, 2);
    assert_cmp(Ndarray_NDIM(a_n), ==, 1);
    assert_cmp(Ndarray_NDIM(a_ended), ==, 1);
    assert_cmp(Ndarray_NDIM(b), ==, 2);
    assert_cmp(Ndarray_NDIM(b_len), ==, 1);
    assert_cmp(Ndarray_NDIM(out), ==, 2);
    int n_batch = Ndarray_DIMS(out)[0];
    int n_labels = Ndarray_DIMS(out)[1];
    int n_b_max_len = Ndarray_DIMS(b)[1];
    assert_cmp(Ndarray_DIMS(out)[0], ==, n_batch);
    assert_cmp(Ndarray_DIMS(out)[1], ==, n_labels);
    assert_cmp(Ndarray_DIMS(last_row)[0], ==, n_batch);
    assert_cmp(Ndarray_DIMS(last_row)[1], ==, n_b_max_len + 1);
    bool a_broadcast_batch = Ndarray_DIMS(a)[0] == 1;
    if(!a_broadcast_batch)
      assert_cmp(Ndarray_DIMS(a)[0], ==, n_batch);
    assert_cmp(Ndarray_DIMS(a)[1], ==, n_labels);
    assert_cmp(Ndarray_DIMS(a_n)[0], ==, n_batch);
    assert_cmp(Ndarray_DIMS(a_ended)[0], ==, n_batch);
    assert_cmp(Ndarray_DIMS(b)[0], ==, n_batch);
    assert_cmp(Ndarray_DIMS(b)[1], ==, n_b_max_len);
    assert_cmp(Ndarray_DIMS(b_len)[0], ==, n_batch);

    start_dev_kernel(calc_result_kernel, (
      n_batch, n_b_max_len, n_labels,
      Ndarray_DEV_DATA_int32(last_row),
      Ndarray_DEV_DATA_int32(a), Ndarray_DEV_DATA_int32(a_n), Ndarray_DEV_DATA_int32(a_ended),
      Ndarray_DEV_DATA_int32(b), Ndarray_DEV_DATA_int32(b_len),
      Ndarray_DEV_DATA_int32(out),
      optimal_completion,
      a_broadcast_batch, a_blank_idx
    ));
    HANDLE_LAST_ERROR();
  """

    c_bw_code = None


def sparse_splice_offset_numpy(s0, idx):
    """
    Like sparse_slice_offset().
    """
    mask = s0 < idx
    return numpy.sum(mask)
