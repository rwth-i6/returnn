"""
Native ops for the JAX backend.

Same ops as the TF and PyTorch backends (:mod:`returnn.native_op` describes them, and
``native_op.cpp`` holds the shared support code) -- what differs is only how they are wired in.
JAX has no operator registry: an op is an XLA FFI custom call, so this module generates a C++
handler per op, compiles it to a shared library, registers it with ``jax.ffi``, and calls it
through ``jax.ffi.ffi_call``.

The op's own ``c_fw_code`` is used unchanged. It addresses its arguments through the
``inputs``/``outputs`` arrays of ``Ndarray*``, and the JAX branch of ``native_op.cpp`` defines
``Ndarray`` as a pointer plus a shape -- which is exactly what XLA hands to an FFI handler.

Gradients are :func:`jax.custom_vjp`, in place of the ``torch.autograd.Function`` the PyTorch
backend uses; the math is the same.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence, Tuple

import os
from functools import partial
from textwrap import dedent, indent
from threading import RLock

import numpy

import jax
import jax.numpy as jnp

from returnn import native_op

from returnn.native_op import OpDescription
from returnn.util.native_code_compiler import NativeCodeCompiler
from returnn.util.cuda_env import CudaEnv


class OpCodeCompiler(NativeCodeCompiler):
    """
    Compiles an op with nvcc.

    Not :class:`returnn.torch.util.native_op_code_compiler.OpCodeCompiler`: that one locates CUDA
    from the libcudart TORCH has mapped into the process, which a JAX run never loads.
    The generic :class:`CudaEnv` finds it from CUDA_HOME / the usual paths instead.
    """

    def __init__(self, *, with_cuda: bool = True, **kwargs):
        # before super().__init__: it builds the info dict, which asks for the compiler bin
        self._with_cuda = with_cuda
        self._cuda_env = CudaEnv.get_instance() if with_cuda else None
        super().__init__(**kwargs)

    def _get_compiler_bin(self) -> str:
        if not self._with_cuda:
            return super()._get_compiler_bin()
        return self._cuda_env.get_compiler_bin()

    def _transform_compiler_opts(self, opts: List[str]) -> List[str]:
        if not self._with_cuda:
            return opts + ["-std=c++17"]  # the XLA FFI headers need it
        # nvcc takes host-compiler flags only via -Xcompiler.
        # Drop CudaEnv's -std: the XLA FFI headers need C++17, and nvcc rejects two -std.
        res = [o for o in self._cuda_env.get_compiler_opts() if not o.startswith("-std=")]
        for opt in opts:
            res += ["-Xcompiler", opt]
        # the file is named .cc but holds CUDA launch syntax
        res += ["-x", "cu", "-std=c++17", "--expt-relaxed-constexpr"]
        return res


__all__ = ["OpMaker", "make_op", "fast_baum_welch", "get_ctc_fsa_fast_bw", "ctc_loss"]


_base_dir = os.path.dirname(os.path.abspath(native_op.__file__))


def _ffi_dtype(v: Dict[str, Any]) -> str:
    """
    :param v: in_info / out_info entry
    :return: the xla::ffi data-type tag
    """
    t = v.get("dtype", "float32")
    if t == "float32":
        return "ffi::F32"
    if t == "int32":
        return "ffi::S32"
    raise NotImplementedError(f"RF JaxBackend native op: unsupported dtype {t!r}")


class OpMaker:
    """
    Compiles one native op into an XLA FFI custom call and registers it with JAX.

    Counterpart of :class:`returnn.torch.util.native_op.OpMaker`.
    """

    global_lock = RLock()
    mod_cache: Dict[str, Any] = {}  # cache_key -> compiled+registered target name

    def __init__(self, description: native_op.NativeOpBaseMixin, *, compiler_opts: Optional[Dict[str, Any]] = None):
        """
        :param description: the op, from :mod:`returnn.native_op`
        :param compiler_opts: passed on to :class:`NativeCodeCompiler`
        """
        self.description = description
        self.name = description.name
        self.compiler_opts = compiler_opts or {}

    @property
    def target_name(self) -> str:
        """:return: the name the custom call is registered under"""
        return f"returnn_{self.name}"

    def _make_code(self, *, cuda: bool = True) -> str:
        """
        :param cuda: emit the CUDA variant; False gives the CPU one
        :return: the full C++/CUDA translation unit for this op
        """
        # noinspection PyProtectedMember
        in_info, out_info, _ = native_op.NativeOpBaseMixin._resolve_want_inplace_dummy(
            in_info=self.description.in_info, out_info=self.description.out_info
        )

        args = []  # the Bind() chain
        unpack = []  # Ndarray structs over the FFI buffers
        self._attr_names = []
        for i, v in enumerate(in_info):
            if v.get("host_memory", False):
                # A host_memory scalar must be READABLE ON THE HOST, and an FFI buffer never is
                # (it is a device pointer). XLA's equivalent is an attribute, passed by value.
                assert v["ndim"] == 0, f"{self.name}: host_memory input {v['name']!r} must be scalar"
                name = v["name"]
                self._attr_names.append(name)
                # the C++ parameter is prefixed: c_fw_code declares its own local of the same name
                args.append(f'.Attr<int32_t>("{name}")')
                unpack.append(f"int32_t _attrval_{name} = _attr_{name};")
                unpack.append(f"Ndarray _in_{i} = {{(void*)&_attrval_{name}, nullptr, 0}};")
                continue
            args.append(f".Arg<ffi::Buffer<{_ffi_dtype(v)}>>()")
            unpack.append(
                f"Ndarray _in_{i} = {{(void*)in{i}.untyped_data(), in{i}.dimensions().begin(), "
                f"(int)in{i}.dimensions().size()}};"
            )
        for i, v in enumerate(out_info):
            args.append(f".Ret<ffi::Buffer<{_ffi_dtype(v)}>>()")
            unpack.append(
                f"Ndarray _out_{i} = {{(void*)out{i}->untyped_data(), out{i}->dimensions().begin(), "
                f"(int)out{i}->dimensions().size()}};"
            )

        params = ", ".join(
            [
                (f"int32_t _attr_{v['name']}" if v.get("host_memory", False) else f"ffi::Buffer<{_ffi_dtype(v)}> in{i}")
                for i, v in enumerate(in_info)
            ]
            + [f"ffi::Result<ffi::Buffer<{_ffi_dtype(v)}>> out{i}" for i, v in enumerate(out_info)]
        )

        code_wrap_io = dedent(f"""\
            static const int n_inputs = {len(in_info)}, n_outputs = {len(out_info)};
            Ndarray* inputs[n_inputs] = {{ {", ".join(f"&_in_{i}" for i in range(len(in_info)))} }};
            Ndarray* _outputs_ptr[n_outputs] = {{ {", ".join(f"&_out_{i}" for i in range(len(out_info)))} }};
            Ndarray** outputs[n_outputs] = {{ {", ".join(f"&_outputs_ptr[{i}]" for i in range(len(out_info)))} }};
            """)

        code_user = self.description.c_fw_code % {"fail": "assert(false);"}

        cuda_headers = dedent("""\
            #include <cuda.h>
            #include <cuda_runtime.h>
            #include <math_constants.h>
            """)
        # the stream and the scratch allocator are CUDA-only context; on CPU there is neither
        cuda_ctx_decl = dedent("""\
            // set per call, read by CUDA_CUR_STREAM and _malloc in native_op.cpp
            __thread cudaStream_t _returnn_jax_stream = nullptr;
            __thread ffi::ScratchAllocator* _returnn_jax_scratch = nullptr;
            """)
        # noinspection PyProtectedMember
        return dedent(f"""\
            #define JAX 1
            #define CUDA {1 if cuda else 0}

            {cuda_headers if cuda else ""}
            #include <cstdio>
            #include <cstdlib>
            #include <cassert>
            #include "xla/ffi/api/ffi.h"

            namespace ffi = xla::ffi;

            {cuda_ctx_decl if cuda else ""}

            #undef RETURNN_CUDA
            #define RETURNN_CUDA {1 if cuda else 0}  // survives the CUDA undef, see the torch OpMaker

            #include "native_op.cpp"

            // the op's own kernels (fill_array, next_frame, normalize, ...), used by c_fw_code
            {self.description._reduce_c_extra_support_code(self.description.c_extra_support_code)}

            static ffi::Error _{self.name}_impl(
                    {"cudaStream_t stream, ffi::ScratchAllocator scratch, " if cuda else ""}{params}) {{
                {"_returnn_jax_stream = stream; _returnn_jax_scratch = &scratch;" if cuda else ""}
            {indent(chr(10).join(unpack), "    ")}
            {indent(code_wrap_io, "    ")}
            {indent(code_user, "    ")}
                return ffi::Error::Success();
            }}

            XLA_FFI_DEFINE_HANDLER_SYMBOL(
                {self.target_name}, _{self.name}_impl,
                ffi::Ffi::Bind()
                    {".Ctx<ffi::PlatformStream<cudaStream_t>>()" if cuda else ""}
                    {".Ctx<ffi::ScratchAllocator>()" if cuda else ""}
                    {chr(10).join("    " + a for a in args)});
            """)

    @property
    def attr_names(self) -> List[str]:
        """:return: the host_memory inputs, which are passed as FFI attributes, in order"""
        self._make_code()  # fills it
        return list(self._attr_names)

    def make(self) -> str:
        """
        Compile and register the op, for every platform it can be built for.

        The same source serves both: ``native_op.cpp`` and the ops themselves branch on ``CUDA``,
        so the CPU variant is the same translation unit with ``CUDA=0``. Registering both means a
        JAX program picks the right one by device, as the TF and PyTorch backends do.

        :return: the registered custom-call target name (the same for every platform)
        """
        with self.global_lock:
            if self.name in self.mod_cache:
                return self.mod_cache[self.name]
            import ctypes

            # "cpu" is the name XLA canonicalizes the host platform to; "CPU" is not matched
            for platform, cuda in (("CUDA", True), ("cpu", False)):
                if cuda and not CudaEnv.get_instance().is_available():
                    continue
                comp = OpCodeCompiler(
                    base_name=f"{self.target_name}_{'cuda' if cuda else 'cpu'}",
                    code_version=self.description.code_version,
                    code=self._make_code(cuda=cuda),
                    include_paths=(_base_dir, jax.ffi.include_dir()),
                    c_macro_defines={"JAX": 1, "CUDA": 1 if cuda else 0},
                    use_cxx11_abi=True,  # jaxlib's headers/libs are built with it
                    with_cuda=cuda,
                    **self.compiler_opts,
                )
                lib = ctypes.cdll.LoadLibrary(comp.get_lib_filename())
                jax.ffi.register_ffi_target(
                    self.target_name, jax.ffi.pycapsule(getattr(lib, self.target_name)), platform=platform
                )
            self.mod_cache[self.name] = self.target_name
            return self.target_name


def make_op(cls, **kwargs) -> str:
    """
    :param cls: e.g. :class:`native_op.FastBaumWelchOp`
    :param kwargs: for :class:`OpMaker`
    :return: registered custom-call target name
    """
    maker = OpMaker(description=OpDescription.from_gen_base(cls), **kwargs)
    return maker.make()


def _call(target_name: str, out_specs: Sequence[jax.ShapeDtypeStruct], args, attrs) -> Tuple[jax.Array, ...]:
    """
    :param target_name: as registered by :meth:`OpMaker.make`
    :param out_specs: shape+dtype per output
    :param args: buffer inputs, in the op's declared order (host_memory ones excluded)
    :param attrs: the host_memory inputs, by name
    :return: the outputs
    """
    res = jax.ffi.ffi_call(target_name, list(out_specs))(*args, **attrs)
    return tuple(res) if isinstance(res, (list, tuple)) else (res,)


def get_ctc_fsa_fast_bw(
    *, targets: jax.Array, seq_lens: jax.Array, blank_idx: int, label_loop: bool = True
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """
    CTC-topology FSA, in the format :func:`fast_baum_welch` takes.

    :param targets: (batch,time), int32
    :param seq_lens: (batch,), int32
    :param blank_idx: vocab index of blank
    :param label_loop: True = CTC, False = RNA-like
    :return: edges (4,n_edges) int32, weights (n_edges,) float32, start_end_states (2,batch) int32
    """
    target = make_op(native_op.GetCtcFsaFastBwOp)
    n_batch, n_time = targets.shape
    n_edges = n_batch * (5 * (n_time - 1) + 10)  # see the op's docs
    # weights is an INPUT: the op fills it, and takes the edge count from its shape
    weights = jnp.zeros((n_edges,), dtype=jnp.float32)
    edges, start_end_states = _call(
        target,
        [jax.ShapeDtypeStruct((4, n_edges), jnp.int32), jax.ShapeDtypeStruct((2, n_batch), jnp.int32)],
        (targets.astype(jnp.int32), seq_lens.astype(jnp.int32), weights),
        {"blank_idx": numpy.int32(blank_idx), "label_loop": numpy.int32(1 if label_loop else 0)},
    )
    return edges, weights, start_end_states


def fast_baum_welch(
    *,
    am_scores: jax.Array,
    seq_mask: jax.Array,
    edges: jax.Array,
    weights: jax.Array,
    start_end_states: jax.Array,
    n_states: Optional[int] = None,
) -> Tuple[jax.Array, jax.Array]:
    """
    :param am_scores: (time,batch,dim), in -log space
    :param seq_mask: (time,batch) -> 0 or 1
    :param edges: (4,n_edges), (from,to,emission_idx,sequence_idx)
    :param weights: (n_edges,)
    :param start_end_states: (2,batch), (start,end) state idx
    :param n_states: total number of FSA states. The op sizes its own state buffer from this, and
        it is a host_memory scalar, so it must be a Python int -- a traced value cannot reach it.
        Take the CTC bound: 2*S+2 states per seq over the batch, from the DECLARED targets width.
    :return: (fwdbwd, obs_scores), (time,batch,dim) and (time,batch), in -log space
    """
    target = make_op(native_op.FastBaumWelchOp)
    n_time, n_batch, n_dim = am_scores.shape
    if n_states is None:
        raise ValueError("fast_baum_welch: n_states is required (the op cannot derive it)")
    return _call(
        target,
        [
            jax.ShapeDtypeStruct((n_time, n_batch, n_dim), jnp.float32),  # fwdbwd
            jax.ShapeDtypeStruct((n_time, n_batch), jnp.float32),  # obs_scores
        ],
        (
            am_scores.astype(jnp.float32),
            edges.astype(jnp.int32),
            weights.astype(jnp.float32),
            start_end_states.astype(jnp.int32),
            seq_mask.astype(jnp.float32),  # "index"
        ),
        {"n_states": numpy.int32(n_states)},
    )


@partial(jax.custom_vjp, nondiff_argnums=(1, 6))
def _fast_bw_loss(logits, logits_normalize, seq_mask, edges, weights, start_end_states, n_states):
    """
    Full-sum (Baum-Welch) score. Counterpart of the PyTorch ``_FastBaumWelchScoresAutogradFunc``.

    :return: loss per seq (batch,)
    """
    return _fast_bw_loss_fwd(logits, logits_normalize, seq_mask, edges, weights, start_end_states, n_states)[0]


def _fast_bw_loss_fwd(logits, logits_normalize, seq_mask, edges, weights, start_end_states, n_states):
    log_sm = jax.nn.log_softmax(logits, axis=-1) if logits_normalize else logits  # (time,batch,dim)
    fwdbwd, obs_scores = fast_baum_welch(
        am_scores=-log_sm,
        seq_mask=seq_mask,
        edges=edges,
        weights=weights,
        start_end_states=start_end_states,
        n_states=n_states,
    )
    return obs_scores[0], (log_sm if logits_normalize else None, seq_mask, fwdbwd)


def _fast_bw_loss_bwd(logits_normalize, n_states, res, grad_output):
    del n_states
    log_sm, seq_mask, fwdbwd = res
    bw = jnp.exp(-fwdbwd)  # (time,batch,dim), the soft alignment
    grad_x = (jnp.exp(log_sm) - bw) if logits_normalize else -bw
    grad_x = jnp.where(seq_mask[:, :, None], grad_x, 0.0)
    grad_x = grad_x * grad_output[None, :, None]
    return grad_x, None, None, None, None


_fast_bw_loss.defvjp(_fast_bw_loss_fwd, _fast_bw_loss_bwd)


def ctc_loss(
    *,
    logits: jax.Array,
    logits_seq_lens: jax.Array,
    targets: jax.Array,
    targets_seq_lens: jax.Array,
    label_loop: bool = True,
    logits_time_major: bool = False,
    logits_normalize: bool = True,
    blank_index: int = -1,
    zero_infinity: bool = True,
) -> jax.Array:
    """
    CTC loss via the native fast-Baum-Welch op. Mirrors the PyTorch backend's ``ctc_loss``.

    :param logits: (time,batch,dim) or (batch,time,dim), unnormalized
    :param logits_seq_lens: (batch,)
    :param targets: (batch,time)
    :param targets_seq_lens: (batch,)
    :param label_loop: True = CTC, False = RNA-like
    :param logits_time_major:
    :param logits_normalize: apply log_softmax on the logits
    :param blank_index: vocab index of blank; negative counts from the end
    :param zero_infinity: a seq whose targets do not fit its input has no valid path, so its loss
        is +inf; left in, it poisons the batch mean and every gradient through it
    :return: loss, (batch,)
    """
    assert logits.ndim == 3
    dim = logits.shape[-1]
    if not logits_time_major:
        logits = jnp.transpose(logits, (1, 0, 2))  # (time,batch,dim)
    if blank_index < 0:
        blank_index += dim
    assert 0 <= blank_index < dim

    edges, weights, start_end_states = get_ctc_fsa_fast_bw(
        targets=targets, seq_lens=targets_seq_lens, blank_idx=blank_index, label_loop=label_loop
    )
    n_batch, n_targets = targets.shape
    # The FSA numbers states per seq at a fixed stride over the targets BUFFER width:
    # "state_idx: 0 b, 1 l, ..., T*2 b, T*2+1 dummy, T*2+2 end, i.e. T*2+3 states per seq",
    # with state_idx_offset = (n_time*2+3)*batch_idx (see GetCtcFsaFastBwOp's construct_kernel).
    # One less than this and the op writes past its state buffer.
    n_states = n_batch * (2 * n_targets + 3)
    n_time = logits.shape[0]
    seq_mask = jnp.arange(n_time)[:, None] < logits_seq_lens[None, :]  # (time,batch)

    loss = _fast_bw_loss(logits, logits_normalize, seq_mask, edges, weights, start_end_states, n_states)
    if zero_infinity:
        loss = jnp.where(jnp.isfinite(loss), loss, 0.0)
    return loss
