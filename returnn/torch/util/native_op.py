"""
Native ops for Torch, similar to :mod:`returnn.tf.native_op`.
"""

from __future__ import annotations
from typing import Optional, Any, Tuple, Dict
import os
import sys
from textwrap import dedent
from threading import RLock

import torch

from returnn import native_op
from .native_op_code_compiler import OpCodeCompiler


_base_dir = os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
_base_dir = os.path.realpath(_base_dir)  # Make canonical path-name.


class OpDescription(native_op.NativeOpBaseMixin):
    """
    Meta-info about an op, used by :class:`OpMaker`.
    """

    @classmethod
    def from_gen_base(cls, gen_base):
        """
        :param returnn.native_op.NativeOpGenBase|type[returnn.native_op.NativeOpGenBase] gen_base:
        :rtype: OpDescription
        """
        name = gen_base.__name__
        assert gen_base.in_info is not None
        assert gen_base.out_info is not None
        assert gen_base.c_fw_code is not None
        return OpDescription(
            in_info=gen_base.in_info,
            out_info=gen_base.out_info,
            c_fw_code=gen_base.c_fw_code,
            c_bw_code=gen_base.c_bw_code,
            c_extra_support_code=gen_base.c_extra_support_code,
            cpu_support=gen_base.cpu_support,
            grad_input_map=gen_base.grad_input_map,
            name=name,
        )

    @property
    def is_grad_defined(self) -> bool:
        """
        :return: whether the gradient is defined
        """
        return bool(self.c_bw_code)

    def grad(self) -> Optional[OpDescription]:
        """
        :rtype: OpDescription|None
        """
        if not self.is_grad_defined:
            return None
        kwargs = self.kwargs_for_grad_op()
        return OpDescription(**kwargs)


class OpMaker:
    """
    https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html
    https://docs.pytorch.org/cppdocs/
    """

    with_cuda: Optional[bool] = None
    global_lock = RLock()
    mod_cache = {}  # cache_key -> mod
    op_cache = {}  # cache_key -> op
    log_stream = sys.stdout

    def __init__(
        self,
        description: OpDescription,
        *,
        compiler_opts: Optional[Dict[str, str]] = None,
        with_cuda: Optional[bool] = None,
    ):
        """
        :param description:
        :param compiler_opts: passed on to OpCodeCompiler as kwargs
        :param with_cuda: override auto-detection of CUDA availability
        """
        if with_cuda is not None:
            self.with_cuda = with_cuda
        else:
            self._cls_init_with_cuda()
        self.description = description
        self.name = description.name
        self.compiler_opts = compiler_opts or {}

    @classmethod
    def _cls_init_with_cuda(cls):
        if cls.with_cuda is None:
            cls.with_cuda = torch.cuda.is_available()

    @property
    def op_name(self) -> str:
        """op name"""
        return self.name

    @property
    def cache_key(self) -> str:
        """cache key"""
        return self.name

    @property
    def support_native_op_cpp_filename(self) -> str:
        """
        :return: filename of NativeOp.cpp
        """
        support_native_op_cpp_filename = "%s/native_op.cpp" % _base_dir
        assert os.path.exists(support_native_op_cpp_filename)
        return support_native_op_cpp_filename

    def _make_code(self):
        # https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html
        # https://docs.pytorch.org/cppdocs/

        # We also include NativeOp.cpp.

        # noinspection PyProtectedMember
        in_info, out_info, _ = native_op.NativeOpBaseMixin._resolve_want_inplace_dummy(
            in_info=self.description.in_info, out_info=self.description.out_info
        )

        # noinspection PyShadowingNames
        def map_name(v: Dict[str, Any], is_out: bool = False) -> str:
            """name"""
            name = v["name"].lower()
            if is_out:
                name = "_out_%s" % name
            else:
                name = "_in_%s" % name
            return name

        # noinspection PyShadowingNames,PyUnusedLocal
        def map_type(v: Dict[str, Any]) -> str:
            """dtype"""
            t = v.get("dtype", "float32")
            if t == "float32":
                return "at::kFloat"
            elif t == "int32":
                return "at::kInt"
            else:
                raise NotImplementedError("unsupported dtype %r" % t)

        def make_compute_code(*, cuda: bool = False) -> str:
            """compute code"""

            code_device_specific = ""
            if cuda:
                tensor_vars = [v for v in in_info if _schema_type_str(v) == "Tensor"]
                if tensor_vars:
                    code_device_specific += dedent(f"""\
                        at::Device _device = {map_name(tensor_vars[0])}.device();
                        at::OptionalDeviceGuard _device_guard(_device);
                        """)
                else:
                    code_device_specific += "at::Device _device = at::kCUDA;\n"

            code_forward_io = ""
            out_is_ref = {}  # output vars which are inplace, out_name -> in_idx
            # want_inplace: output-index which this input should operate on
            # Unlike the Theano variant, we always do it inplace,
            # so the user has to make a copy if this is not the intention.
            for in_idx, v in enumerate(in_info):
                out_idx = v.get("want_inplace", -1)
                if out_idx >= 0:
                    code_forward_io += dedent(f"""\
                        torch::Tensor {map_name(out_info[out_idx], is_out=True)} = {map_name(v)};  // inplace
                        """)
                    out_name = out_info[out_idx]["name"]
                    assert out_name not in out_is_ref
                    out_is_ref[out_name] = in_idx

            code_set_io = ""
            for in_idx, v in enumerate(in_info):
                if _schema_type_str(v) != "Tensor":
                    code_set_io += dedent(f"""\
                        torch::Tensor {map_name(v)}_tensor = torch::tensor({map_name(v)});
                        """)
                    continue  # scalar input
                ndim = len(v["shape"])
                code_set_io += dedent(f"""\
                    TORCH_CHECK(
                        {map_name(v)}.dim() == {ndim},
                        "{v["name"]} shape ndim is not {ndim}, got shape ", {map_name(v)}.sizes());
                    """)
                for axis, d in enumerate(v["shape"]):
                    if isinstance(d, int):
                        code_set_io += dedent(f"""\
                            TORCH_CHECK(
                                {map_name(v)}.size({axis}) == {d},
                                "{v["name"]} shape[{axis}] != {d}, got shape ", {map_name(v)}.sizes());
                            """)

            for out_idx, v in enumerate(out_info):
                out_name = out_info[out_idx]["name"]
                if out_name in out_is_ref:  # is ref on input
                    pass
                else:  # no ref
                    cshape = "{%s}" % ", ".join(
                        [
                            str(dim) if isinstance(dim, int) else f"{map_name(in_info[dim[0]])}.size({dim[1]})"
                            for dim in v["shape"]
                        ]
                    )
                    code_set_io += dedent(f"""\
                        torch::Tensor {map_name(v, is_out=True)}
                            = torch::zeros({cshape}, torch::dtype({map_type(v)}){".device(_device)" if cuda else ""});
                        """)

            code_set_contiguous = ""
            for v in in_info:
                if v.get("want_contiguous", False):
                    code_set_contiguous += dedent(f"""\
                        if(!{map_name(v)}.is_contiguous()) {{
                            {map_name(v)} = {map_name(v)}.contiguous();
                        }}
                        """)

            # The user code uses inputs and outputs arrays.
            _code_wrap_io_input_vars_list = [
                f"&{map_name(v)}" + ("" if _schema_type_str(v) == "Tensor" else "_tensor") for v in in_info
            ]
            code_wrap_io = dedent(f"""\
                static const int n_inputs = {len(in_info)}, n_outputs = {len(out_info)};
                torch::Tensor* inputs[n_inputs] = {{
                    {", ".join(_code_wrap_io_input_vars_list)} }};
                torch::Tensor* _outputs_ptr[n_outputs] = {{
                    {", ".join(f"&{map_name(v, is_out=True)}" for v in out_info)} }};
                torch::Tensor** outputs[n_outputs] = {{
                    {", ".join(f"&_outputs_ptr[{i}]" for i in range(len(out_info)))} }};
                """)

            code_user = self.description.c_fw_code % {"fail": "assert(false);"}

            code_return = "return std::make_tuple(%s);\n" % ", ".join([map_name(v, is_out=True) for v in out_info])

            code_compute = "\n\n".join(
                [
                    code_device_specific,
                    code_forward_io,
                    code_set_io,
                    code_set_contiguous,
                    code_wrap_io,
                    code_user,
                    code_return,
                ]
            )

            return code_compute

        code_header = ""
        code_header += dedent("""\
            #include <torch/extension.h>
            #include <torch/types.h>
            #include <c10/core/CPUAllocator.h>
            """)
        if self.with_cuda:
            code_header += dedent("""\
              #include <cuda.h>
              #include <cuda_runtime.h>
              #include <math_constants.h>
              #include <ATen/cuda/CUDAContext.h>
              #include <c10/cuda/CUDACachingAllocator.h>
              """)

        def _schema_type_str(v: Dict[str, Any], *, c: bool = False) -> str:
            if v.get("host_memory", False):
                assert v["ndim"] == 0  # not supported otherwise...
                dtype = v.get("dtype", "float32")
                if dtype == "float32":
                    if c:
                        return "float32_t"
                    return "float"
                elif dtype == "int32":
                    if c:
                        return "int64_t"  # int8_t, int64_t and bool are supported as an integral argument type
                    return "int"
                else:
                    raise NotImplementedError("unsupported dtype %r" % dtype)
            if c:
                return "torch::Tensor"
            return "Tensor"

        # https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/README.md#func
        func_schema_str = "(%s)" % ", ".join(
            f"{_schema_type_str(v)} {v['name']}" for v in in_info
        ) + " -> (%s)" % ", ".join(_schema_type_str(v) for v in out_info)

        code_header += dedent(
            f"""\
            #define _ns  // so _ns::something will use the root namespace
            #define TORCH 1
            #define CUDA 0
            #include "{self.support_native_op_cpp_filename}"

            TORCH_LIBRARY({self.op_name}, m) {{
                m.def("{self.op_name}{func_schema_str}");
            }}

            """
        )

        if self.description.cpu_support:
            # noinspection PyProtectedMember
            code_cpu_op = self.description._reduce_c_extra_support_code(self.description.c_extra_support_code)
            code_cpu_op += dedent(
                f"""\

                std::tuple<{", ".join(["torch::Tensor"] * len(out_info))}>
                {self.op_name}_cpu(
                    {", ".join(f"{_schema_type_str(v, c=True)} {map_name(v)}" for v in in_info)}
                ) {{
                """
            )
            code_cpu_op += make_compute_code()
            code_cpu_op += dedent(
                f"""\
                }}

                TORCH_LIBRARY_IMPL({self.op_name}, CPU, m) {{
                    m.impl("{self.op_name}", &{self.op_name}_cpu);
                }}
                """
            )
        else:
            code_cpu_op = ""

        if self.with_cuda:
            # noinspection PyProtectedMember
            code_cuda_op = dedent(f"""\
                namespace _cuda_impl {{

                    #ifdef _ns
                      #undef _ns
                      #define _ns _ns
                    #endif
                    namespace _ns = ::_cuda_impl;
                    #undef Ndarray_memcpy
                    #undef Ndarray_memset
                    #undef Ndarray_sgemm
                    #undef Ndarray_sgemv
                    #undef Ndarray_sgemm_batched
                    #undef DEF_KERNEL
                    #undef start_dev_kernel
                    #undef assert_cmp
                    #undef threadIdx
                    #undef blockIdx
                    #undef blockDim
                    #undef gridDim
                    #undef DEF_SHARED
                    #undef DEV_FUNC
                    #undef HANDLE_LAST_ERROR
                    #undef HOST_FUNC
                    #undef INF_F
                    #undef NAN_F
                    #undef elem_atomic_add
                    #undef elem_atomic_cas
                    #undef elem_atomic_min
                    #undef float_as_int
                    #undef int_as_float
                    #undef start_dev_kernel2
                    #undef CHECK_WITH_MSG

                    #undef CUDA
                    #define CUDA 1

                    #include "{self.support_native_op_cpp_filename}"

                    #undef CUDA  // name collision in Torch code below
                """)
            # noinspection PyProtectedMember
            code_cuda_op += self.description._reduce_c_extra_support_code(self.description.c_extra_support_code)
            code_cuda_op += dedent(f"""\

                    std::tuple<{", ".join(["torch::Tensor"] * len(out_info))}>
                    {self.op_name}_cuda(
                        {", ".join(f"{_schema_type_str(v, c=True)} {map_name(v)}" for v in in_info)}
                    ) {{
                """)
            code_cuda_op += make_compute_code(cuda=True)
            code_cuda_op += dedent(f"""\
                    }}

                    TORCH_LIBRARY_IMPL({self.op_name}, CUDA, m) {{
                        m.impl("{self.op_name}", &{self.op_name}_cuda);
                    }}
                }}  // namespace _cuda_impl
                """)
        else:
            code_cuda_op = ""

        return code_header + code_cpu_op + code_cuda_op

    def _make_mod(self):
        if self.cache_key in self.mod_cache:
            return self.mod_cache[self.cache_key]

        comp = OpCodeCompiler(
            base_name=self.name,
            code_version=self.description.code_version,
            code=self._make_code(),
            include_deps=[self.support_native_op_cpp_filename],
            use_cuda_if_available=self.with_cuda,
            log_stream=self.log_stream,
            **dict(self.compiler_opts),
        )
        mod = comp.load_module()
        mod._op_compiler = comp
        self.mod_cache[self.cache_key] = mod
        return mod

    def make_op(self):
        """
        :return: op
        """
        with self.global_lock:
            if self.cache_key in self.op_cache:
                return self.op_cache[self.cache_key]
            mod = self._make_mod()
            op = getattr(mod, self.op_name)
            op._op_maker = self
            op._op_module = mod
            self.op_cache[self.cache_key] = op

            if self.description.is_grad_defined:
                pass  # not implemented yet...

        return op


def make_op(cls, **kwargs):
    """
    :param type[returnn.native_op.NativeOpGenBase] cls:
    :param kwargs: passed to OpMaker
    :return: op
    :rtype: (torch.Tensor) -> tuple[torch.Tensor]
    """
    maker = OpMaker(OpDescription.from_gen_base(cls), **kwargs)
    return maker.make_op()


def ctc_loss(
    *,
    logits: torch.Tensor,
    logits_seq_lens: torch.Tensor,
    targets: torch.Tensor,
    targets_seq_lens: torch.Tensor,
    ctc_merge_repeated: bool = True,
    logits_time_major: bool = False,
    logits_normalize: bool = True,
    blank_index: int = -1,
) -> torch.Tensor:
    """
    Similar to :func:`tf.nn.ctc_loss`.
    We use our :func:`fast_baum_welch`.
    Also see :class:`FastBaumWelchLoss`.

    :param logits: (time,batch,dim) or (batch,time,dim). unnormalized (before softmax)
    :param logits_seq_lens: shape (batch,) of int32|int64
    :param logits_time_major:
    :param targets: batch-major, [batch,time]
    :param targets_seq_lens: (batch,)
    :param ctc_merge_repeated:
    :param logits_normalize: apply log_softmax on logits (default).
      if False, you might also set grad_wrt_softmax_in=False
    :param blank_index: vocab index of the blank symbol
    :return: loss, shape (batch,)
    """
    from .array_ import sequence_mask_time_major

    assert logits.ndim == 3
    dim = logits.shape[-1]
    if not logits_time_major:
        logits = torch.transpose(logits, 0, 1)  # (time,batch,dim)

    if blank_index < 0:
        blank_index += dim
    assert 0 <= blank_index < dim
    edges, weights, start_end_states = get_ctc_fsa_fast_bw(
        targets=targets, seq_lens=targets_seq_lens, blank_idx=blank_index, label_loop=ctc_merge_repeated
    )

    seq_mask = sequence_mask_time_major(logits_seq_lens)

    loss = _FastBaumWelchScoresAutogradFunc.apply(logits, logits_normalize, seq_mask, edges, weights, start_end_states)
    return loss


# noinspection PyMethodOverriding,PyAbstractClass,PyMissingOrEmptyDocstring
class _FastBaumWelchScoresAutogradFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        logits: torch.Tensor,
        logits_normalize: bool,
        seq_mask: torch.Tensor,
        edges: torch.Tensor,
        weights: torch.Tensor,
        start_end_states: torch.Tensor,
        state_buffer: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if logits_normalize:
            log_sm = torch.log_softmax(logits, dim=-1)  # (time,batch,dim)
        else:
            log_sm = logits
        fwdbwd, obs_scores = fast_baum_welch(
            am_scores=-log_sm,
            seq_mask=seq_mask,
            edges=edges,
            weights=weights,
            start_end_states=start_end_states,
            state_buffer=state_buffer,
        )
        loss = obs_scores[0]  # (batch,)
        ctx.grad_wrt_softmax_in = logits_normalize
        if logits_normalize:
            ctx.save_for_backward(log_sm, seq_mask, fwdbwd)
        else:
            ctx.save_for_backward(seq_mask, fwdbwd)
        return loss

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:
        if ctx.grad_wrt_softmax_in:
            log_sm, seq_mask, fwdbwd = ctx.saved_tensors
        else:
            log_sm = None
            seq_mask, fwdbwd = ctx.saved_tensors
        bw = torch.exp(-fwdbwd)  # (time,batch,dim)
        if ctx.grad_wrt_softmax_in:
            grad_x = torch.exp(log_sm) - bw  # (time,batch,dim)
        else:
            grad_x = -bw  # (time,batch,dim)
        grad_x = torch.where(seq_mask[:, None, :], grad_x, 0.0)
        grad_x *= grad_output[None, :, None]
        return grad_x, None, None, None, None, None, None, None


def get_ctc_fsa_fast_bw(
    *, targets: torch.Tensor, seq_lens: torch.Tensor, blank_idx: int, label_loop: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    See :class:`NativeOp.GetCtcFsaFastBwOp`.
    Generates a FSA with CTC topology. The output format is compatible to :func:`fast_baum_welch`.

    :param targets: shape (batch,time), int32
    :param seq_lens: shape (batch), int32
    :param blank_idx: vocab index of the blank symbol
    :param label_loop: True -> normal CTC; False -> RNA-like
    :return: edges, weights, start_end_states;
        edges is (4,num_edges), int32, edges of the graph (from,to,emission_idx,sequence_idx).
        weights is (num_edges,), float32. all zero.
        start_end_states is (2,batch), int32, (start,end) state idx in FSA.
    """
    assert targets.ndim == 2
    targets = targets.to(torch.int32)
    n_batch, n_time = targets.shape

    from .assert_ import assert_

    # The check on the seq lens is important
    # because invalid seq lens might not directly lead to an error here
    # but it might just return an invalid FSA.
    # An invalid FSA can however later cause a crash in the FastBaumWelchOp.
    assert_(seq_lens.max() == n_time, "get_ctc_fsa_fast_bw seq_lens invalid")

    n_edges = n_batch * (5 * (n_time - 1) + 10)  # see op documentation
    weights = torch.zeros((n_edges,), device=targets.device)
    maker = OpMaker(OpDescription.from_gen_base(native_op.GetCtcFsaFastBwOp))
    op = maker.make_op()
    edges, start_end_states = op(targets, seq_lens, blank_idx, weights, label_loop)

    return edges, weights, start_end_states


def fast_baum_welch(
    *,
    am_scores: torch.Tensor,
    seq_mask: torch.Tensor,
    edges: torch.Tensor,
    weights: torch.Tensor,
    start_end_states: torch.Tensor,
    state_buffer: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    :param am_scores: (time, batch, dim), in -log space
    :param seq_mask: (time, batch) -> 0 or 1 (index mask, via seq lens)
    :param edges: (4,num_edges), edges of the graph (from,to,emission_idx,sequence_idx)
    :param weights: (num_edges,), weights of the edges
    :param start_end_states: (2, batch), (start,end) state idx in automaton.
        there is only one single automaton.
    :param state_buffer: (2, num_states)
    :return: (fwdbwd, obs_scores), fwdbwd is (time, batch, dim), obs_scores is (time, batch), in -log space
    """
    from .assert_ import assert_

    # edges, weights, start_end_states, state_buffer = SprintAlignmentAutomataOp(self.sprint_opts)(self.network.tags)
    op = make_fast_baum_welch_op()
    float_idx = seq_mask.float()
    if state_buffer is None:
        last_state_idx = start_end_states[1].max()  # see get_automata_for_batch
        assert_(last_state_idx >= 0, "fast_baum_welch last_state_idx must be >= 0")
        state_buffer = torch.zeros((2, last_state_idx + 1))
    fwdbwd, obs_scores = op(am_scores, edges, weights, start_end_states, float_idx, state_buffer)  # noqa
    return fwdbwd, obs_scores


def make_fast_baum_welch_op(**kwargs):
    """
    :return: op
    :rtype: (torch.Tensor) -> tuple[torch.Tensor]
    """
    maker = OpMaker(OpDescription.from_gen_base(native_op.FastBaumWelchOp), **kwargs)
    return maker.make_op()


def fast_viterbi(
    *,
    am_scores: torch.Tensor,
    am_seq_len: torch.Tensor,
    edges: torch.Tensor,
    weights: torch.Tensor,
    start_end_states: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    :param am_scores: (time, batch, dim), in +log space, already normalized / just used as-is
    :param am_seq_len: (batch,), int32
    :param edges: (4,num_edges), edges of the graph (from,to,emission_idx,sequence_idx)
    :param weights: (num_edges,), weights of the edges
    :param start_end_states: (2, batch), (start,end) state idx in automaton.
        there is only one single automaton.
    :return: (alignment, scores), alignment is (time, batch), scores is (batch,), in +log space
    """
    last_state_idx = start_end_states[1].max()
    n_states = last_state_idx + 1
    maker = OpMaker(OpDescription.from_gen_base(native_op.FastViterbiOp))
    op = maker.make_op()
    alignment, scores = op(am_scores, am_seq_len, edges, weights, start_end_states, n_states)
    return alignment, scores
