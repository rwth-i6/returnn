"""
Benchmark for RETURNN frontend (RF) with PyTorch (PT) backend.

You can run this file directly (then it will just call `returnn.__main__.main`)
or use as a RETURNN config file.

This trains a standard AED model
on some dummy data.
"""

from __future__ import annotations

import subprocess
from typing import Optional, Any, Tuple, Dict, Sequence, List
import os
import sys
import tree
import numpy
import argparse
import tempfile
import time
import multiprocessing

if __name__ == "__main__":
    import _setup_returnn_env  # noqa

from returnn.tensor import Tensor, TensorDict, Dim, single_step_dim, batch_dim
import returnn.frontend as rf
from returnn.frontend.tensor_array import TensorArray
from returnn.frontend.encoder.conformer import ConformerEncoder, ConformerConvSubsample
from returnn import __main__

_my_file = os.path.abspath(__file__)
_my_dir = os.path.dirname(_my_file)
_returnn_root_dir = os.path.dirname(_my_dir)

config = dict(
    task="train",
    backend="torch",
    torch_dataloader_opts=dict(num_workers=1),
    # batching="laplace:.1000",  -- dummy dataset does not support this
    batch_size=15_000,
    max_seqs=200,
    max_seq_length_default_target=75,
    accum_grad_multiple_step=2,
    # gradient_clip=0,
    # gradient_clip_global_norm = 1.0
    optimizer={"class": "nadam", "epsilon": 1e-8},
    # gradient_noise=0.0,
    learning_rate=0.0005,
    learning_rates=(
        # matching pretraining
        list(numpy.linspace(0.0001, 0.001, num=10)) * 3
        + list(numpy.linspace(0.0001, 0.0005, num=10))
        + [0.0005] * 20
        + list(numpy.linspace(0.0005, 0.001, num=20))
    ),
    min_learning_rate=0.001 / 50,
    learning_rate_control="newbob_multi_epoch",
    learning_rate_control_relative_error_relative_lr=True,
    relative_error_div_by_old=True,
    use_learning_rate_control_always=True,
    newbob_multi_update_interval=1,
    learning_rate_control_min_num_epochs_per_new_lr=1,
    learning_rate_decay=0.9,
    newbob_relative_error_threshold=-0.01,
    num_epochs=1000,
    log_verbosity=[5],
)


# depends on dataset
extern_data_inputs_name = "data"
extern_data_targets_name = "classes"
data_spatial_dim = Dim(None, name="in_spatial")
data_feature_dim = Dim(2, name="in_feature")
targets_spatial_dim = Dim(None, name="out_spatial")
targets_dim = Dim(10, name="vocab")
extern_data = {
    extern_data_inputs_name: dict(
        dims=[
            batch_dim,
            data_spatial_dim,
        ],
        sparse_dim=data_feature_dim,
        dtype="int32",
        available_for_inference=True,
    ),
    extern_data_targets_name: dict(
        dims=[
            batch_dim,
            targets_spatial_dim,
        ],
        sparse_dim=targets_dim,
        dtype="int32",
        available_for_inference=False,
    ),
}


def _get_dataset_opts(name: str):
    opts = {
        "class": "TaskNumberBaseConvertDataset",
        "input_base": data_feature_dim.dimension,
        "output_base": targets_dim.dimension,
    }
    if name == "train":
        opts["num_seqs"] = 10_000
    else:
        opts["num_seqs"] = 1_000
        opts["fixed_random_seed"] = sum(map(ord, name))
    return opts


train = _get_dataset_opts("train")
dev = _get_dataset_opts("dev")

targets_ext_dim = targets_dim + 1  # for BOS/EOS
targets_eos_idx = targets_ext_dim.dimension - 1
encoder_in_dim = Dim(80, name="enc_in_feature")


def get_model(**_kwargs):
    """get model, RETURNN config callback"""
    return Model(
        in_dim=data_feature_dim,
        encoder_in_dim=encoder_in_dim,
        num_enc_layers=12,
        enc_model_dim=Dim(name="enc", dimension=512, kind=Dim.Types.Feature),
        enc_ff_dim=Dim(name="enc-ff", dimension=2048, kind=Dim.Types.Feature),
        enc_att_num_heads=8,
        enc_conformer_layer_opts=dict(
            conv_norm_opts=dict(use_mask=True),
            self_att_opts=dict(
                # Shawn et al. 2018 style, old RETURNN way.
                with_bias=False,
                with_linear_pos=False,
                with_pos_bias=False,
                learnable_pos_emb=True,
                separate_pos_emb_per_head=False,
            ),
            ff_activation=lambda x: rf.relu(x) ** 2.0,
        ),
        target_dim=targets_ext_dim,
        bos_idx=targets_eos_idx,
        eos_idx=targets_eos_idx,
    )


class Model(rf.Module):
    """Model definition"""

    # noinspection PyShadowingNames
    def __init__(
        self,
        in_dim: Dim,
        encoder_in_dim: Dim,
        *,
        num_enc_layers: int = 12,
        target_dim: Dim,
        eos_idx: int,
        bos_idx: int,
        enc_model_dim: Dim = Dim(name="enc", dimension=512),
        enc_ff_dim: Dim = Dim(name="enc-ff", dimension=2048),
        enc_att_num_heads: int = 4,
        enc_conformer_layer_opts: Optional[Dict[str, Any]] = None,
        enc_key_total_dim: Dim = Dim(name="enc_key_total_dim", dimension=1024),
        att_num_heads: Dim = Dim(name="att_num_heads", dimension=1),
        att_dropout: float = 0.1,
        enc_dropout: float = 0.1,
        enc_att_dropout: float = 0.1,
        l2: float = 0.0001,
    ):
        super(Model, self).__init__()
        self.encoder_in_dim = encoder_in_dim
        self.embedding = rf.Embedding(in_dim, encoder_in_dim)
        self.encoder = ConformerEncoder(
            encoder_in_dim,
            enc_model_dim,
            ff_dim=enc_ff_dim,
            input_layer=ConformerConvSubsample(
                encoder_in_dim,
                out_dims=[Dim(32, name="conv1"), Dim(64, name="conv2"), Dim(64, name="conv3")],
                filter_sizes=[(3, 3), (3, 3), (3, 3)],
                pool_sizes=[(1, 2)],
                strides=[(1, 1), (3, 1), (2, 1)],
            ),
            encoder_layer_opts=enc_conformer_layer_opts,
            num_layers=num_enc_layers,
            num_heads=enc_att_num_heads,
            dropout=enc_dropout,
            att_dropout=enc_att_dropout,
        )

        self.target_dim = target_dim
        self.eos_idx = eos_idx
        self.bos_idx = bos_idx  # for non-blank labels; for with-blank labels, we use bos_idx=blank_idx

        self.enc_key_total_dim = enc_key_total_dim
        self.enc_key_per_head_dim = enc_key_total_dim.div_left(att_num_heads)
        self.att_num_heads = att_num_heads
        self.att_dropout = att_dropout
        self.dropout_broadcast = rf.dropout_broadcast_default()

        # https://github.com/rwth-i6/returnn-experiments/blob/master/2020-rnn-transducer/configs/base2.conv2l.specaug4a.ctc.devtrain.config

        self.enc_ctx = rf.Linear(self.encoder.out_dim, enc_key_total_dim)
        self.enc_ctx_dropout = 0.2
        self.enc_win_dim = Dim(name="enc_win_dim", dimension=5)

        self.inv_fertility = rf.Linear(self.encoder.out_dim, att_num_heads, with_bias=False)

        self.target_embed = rf.Embedding(target_dim, Dim(name="target_embed", dimension=640))

        self.s = rf.ZoneoutLSTM(
            self.target_embed.out_dim + att_num_heads * self.encoder.out_dim,
            Dim(name="lstm", dimension=1024),
            zoneout_factor_cell=0.15,
            zoneout_factor_output=0.05,
            use_zoneout_output=False,  # like RETURNN/TF ZoneoutLSTM old default
            # parts_order="icfo",  # like RETURNN/TF ZoneoutLSTM
            # parts_order="ifco",
            parts_order="jifo",  # NativeLSTM (the code above converts it...)
            forget_bias=0.0,  # the code above already adds it during conversion
        )

        self.weight_feedback = rf.Linear(att_num_heads, enc_key_total_dim, with_bias=False)
        self.s_transformed = rf.Linear(self.s.out_dim, enc_key_total_dim, with_bias=False)
        self.energy = rf.Linear(enc_key_total_dim, att_num_heads, with_bias=False)
        self.readout_in = rf.Linear(
            self.s.out_dim + self.target_embed.out_dim + att_num_heads * self.encoder.out_dim,
            Dim(name="readout", dimension=1024),
        )
        self.output_prob = rf.Linear(self.readout_in.out_dim // 2, target_dim)

        for p in self.parameters():
            p.weight_decay = l2

    def encode(
        self,
        source: Tensor,
        *,
        in_spatial_dim: Dim,
        collected_outputs: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Dict[str, Tensor], Dim]:
        """encode, and extend the encoder output for things we need in the decoder"""
        source = self.embedding(source)
        # SpecAugment
        source = rf.audio.specaugment(source, spatial_dim=in_spatial_dim, feature_dim=self.encoder_in_dim)
        # Encoder including convolutional frontend
        enc, enc_spatial_dim = self.encoder(source, in_spatial_dim=in_spatial_dim, collected_outputs=collected_outputs)
        enc_ctx = self.enc_ctx(enc)
        inv_fertility = rf.sigmoid(self.inv_fertility(enc))
        return dict(enc=enc, enc_ctx=enc_ctx, inv_fertility=inv_fertility), enc_spatial_dim

    def decoder_default_initial_state(self, *, batch_dims: Sequence[Dim], enc_spatial_dim: Dim) -> rf.State:
        """Default initial state"""
        state = rf.State(
            s=self.s.default_initial_state(batch_dims=batch_dims),
            att=rf.zeros(list(batch_dims) + [self.att_num_heads * self.encoder.out_dim]),
            accum_att_weights=rf.zeros(
                list(batch_dims) + [enc_spatial_dim, self.att_num_heads], feature_dim=self.att_num_heads
            ),
        )
        state.att.feature_dim_axis = len(state.att.dims) - 1
        return state

    def loop_step_output_templates(self, batch_dims: List[Dim]) -> Dict[str, Tensor]:
        """loop step out"""
        return {
            "s": Tensor(
                "s", dims=batch_dims + [self.s.out_dim], dtype=rf.get_default_float_dtype(), feature_dim_axis=-1
            ),
            "att": Tensor(
                "att",
                dims=batch_dims + [self.att_num_heads * self.encoder.out_dim],
                dtype=rf.get_default_float_dtype(),
                feature_dim_axis=-1,
            ),
        }

    def loop_step(
        self,
        *,
        enc: rf.Tensor,
        enc_ctx: rf.Tensor,
        inv_fertility: rf.Tensor,
        enc_spatial_dim: Dim,
        input_embed: rf.Tensor,
        state: Optional[rf.State] = None,
    ) -> Tuple[Dict[str, rf.Tensor], rf.State]:
        """step of the inner loop"""
        if state is None:
            batch_dims = enc.remaining_dims(
                remove=(enc.feature_dim, enc_spatial_dim) if enc_spatial_dim != single_step_dim else (enc.feature_dim,)
            )
            state = self.decoder_default_initial_state(batch_dims=batch_dims, enc_spatial_dim=enc_spatial_dim)
        state_ = rf.State()

        prev_att = state.att

        s, state_.s = self.s(rf.concat_features(input_embed, prev_att), state=state.s, spatial_dim=single_step_dim)

        weight_feedback = self.weight_feedback(state.accum_att_weights)
        s_transformed = self.s_transformed(s)
        energy_in = enc_ctx + weight_feedback + s_transformed
        energy = self.energy(rf.tanh(energy_in))
        att_weights = rf.softmax(energy, axis=enc_spatial_dim)
        state_.accum_att_weights = state.accum_att_weights + att_weights * inv_fertility * 0.5
        att0 = rf.dot(att_weights, enc, reduce=enc_spatial_dim, use_mask=False)
        att0.feature_dim = self.encoder.out_dim
        att, _ = rf.merge_dims(att0, dims=(self.att_num_heads, self.encoder.out_dim))
        state_.att = att

        return {"s": s, "att": att}, state_

    def decode_logits(self, *, s: Tensor, input_embed: Tensor, att: Tensor) -> Tensor:
        """logits for the decoder"""
        readout_in = self.readout_in(rf.concat_features(s, input_embed, att))
        readout = rf.reduce_out(readout_in, mode="max", num_pieces=2, out_dim=self.output_prob.in_dim)
        readout = rf.dropout(readout, drop_prob=0.3, axis=self.dropout_broadcast and readout.feature_dim)
        logits = self.output_prob(readout)
        return logits


# noinspection PyShadowingNames
def train_step(*, model: Model, extern_data: TensorDict, **_kwargs):
    """Function is run within RETURNN."""
    data = extern_data[extern_data_inputs_name]
    targets = extern_data[extern_data_targets_name]
    enc_args, enc_spatial_dim = model.encode(data, in_spatial_dim=data_spatial_dim)

    batch_dims = data.remaining_dims(data_spatial_dim)
    targets = targets.copy()
    targets.sparse_dim = targets_ext_dim
    targets = rf.shift_right(targets, axis=targets_spatial_dim, pad_value=targets_eos_idx)
    input_embeddings = model.target_embed(targets)

    def _body(input_embed: Tensor, state: rf.State):
        new_state = rf.State()
        loop_out_, new_state.decoder = model.loop_step(
            **enc_args,
            enc_spatial_dim=enc_spatial_dim,
            input_embed=input_embed,
            state=state.decoder,
        )
        return loop_out_, new_state

    loop_out, _, _ = rf.scan(
        spatial_dim=targets_spatial_dim,
        xs=input_embeddings,
        ys=model.loop_step_output_templates(batch_dims=batch_dims),
        initial=rf.State(
            decoder=model.decoder_default_initial_state(batch_dims=batch_dims, enc_spatial_dim=enc_spatial_dim),
        ),
        body=_body,
    )

    logits = model.decode_logits(input_embed=input_embeddings, **loop_out)
    logits_packed, pack_dim = rf.pack_padded(logits, dims=batch_dims + [targets_spatial_dim], enforce_sorted=False)
    targets_packed, _ = rf.pack_padded(
        targets, dims=batch_dims + [targets_spatial_dim], enforce_sorted=False, out_dim=pack_dim
    )

    log_prob = rf.log_softmax(logits_packed, axis=model.target_dim)
    log_prob = rf.label_smoothed_log_prob_gradient(log_prob, 0.1, axis=model.target_dim)
    loss = rf.cross_entropy(
        target=targets_packed, estimated=log_prob, estimated_type="log-probs", axis=model.target_dim
    )
    loss.mark_as_loss("ce")

    best = rf.reduce_argmax(logits_packed, axis=model.target_dim)
    frame_error = best != targets_packed
    frame_error.mark_as_loss(name="fer", as_error=True)


# noinspection PyShadowingNames
def forward_step(*, model: Model, extern_data: TensorDict, **_kwargs) -> Tuple[Tensor, Tensor, Dim, Dim]:
    """
    Function is run within RETURNN.

    Earlier we used the generic beam_search function,
    but now we just directly perform the search here,
    as this is overall simpler and shorter.

    :return:
        recog results including beam {batch, beam, out_spatial},
        log probs {batch, beam},
        out_spatial_dim,
        final beam_dim
    """
    data = extern_data[extern_data_inputs_name]
    batch_dims = data.remaining_dims((data_spatial_dim, data.feature_dim))
    enc_args, enc_spatial_dim = model.encode(data, in_spatial_dim=data_spatial_dim)
    beam_size = 12
    length_normalization_exponent = 1.0
    max_seq_len = enc_spatial_dim.get_size_tensor()
    print("** max seq len:", max_seq_len.raw_tensor)

    # Eager-mode implementation of beam search.
    # Initial state.
    beam_dim = Dim(1, name="initial-beam")
    batch_dims_ = [beam_dim] + batch_dims
    decoder_state = model.decoder_default_initial_state(batch_dims=batch_dims_, enc_spatial_dim=enc_spatial_dim)
    target = rf.constant(model.bos_idx, dims=batch_dims_, sparse_dim=model.target_dim)
    ended = rf.constant(False, dims=batch_dims_)
    out_seq_len = rf.constant(0, dims=batch_dims_)
    seq_log_prob = rf.constant(0.0, dims=batch_dims_)

    i = 0
    seq_targets = []
    seq_backrefs = []
    while True:
        input_embed = model.target_embed(target)
        step_out, decoder_state = model.loop_step(
            **enc_args,
            enc_spatial_dim=enc_spatial_dim,
            input_embed=input_embed,
            state=decoder_state,
        )
        logits = model.decode_logits(input_embed=input_embed, **step_out)
        label_log_prob = rf.log_softmax(logits, axis=model.target_dim)
        # Filter out finished beams
        label_log_prob = rf.where(
            ended,
            rf.sparse_to_dense(model.eos_idx, axis=model.target_dim, label_value=0.0, other_value=-1.0e30),
            label_log_prob,
        )
        seq_log_prob = seq_log_prob + label_log_prob  # Batch, InBeam, Vocab
        seq_log_prob, (backrefs, target), beam_dim = rf.top_k(
            seq_log_prob, k_dim=Dim(beam_size, name=f"dec-step{i}-beam"), axis=[beam_dim, model.target_dim]
        )  # seq_log_prob, backrefs, target: Batch, Beam
        seq_targets.append(target)
        seq_backrefs.append(backrefs)
        decoder_state = tree.map_structure(lambda s: rf.gather(s, indices=backrefs), decoder_state)
        ended = rf.gather(ended, indices=backrefs)
        out_seq_len = rf.gather(out_seq_len, indices=backrefs)
        i += 1

        ended = rf.logical_or(ended, target == model.eos_idx)
        ended = rf.logical_or(ended, rf.copy_to_device(i >= max_seq_len))
        if bool(rf.reduce_all(ended, axis=ended.dims).raw_tensor):
            break
        out_seq_len = out_seq_len + rf.where(ended, 0, 1)

        if i > 1 and length_normalization_exponent != 0:
            # Length-normalized scores, so we evaluate score_t/len.
            # If seq ended, score_i/i == score_{i-1}/(i-1), thus score_i = score_{i-1}*(i/(i-1))
            # Because we count with EOS symbol, shifted by one.
            seq_log_prob *= rf.where(
                ended,
                (i / (i - 1)) ** length_normalization_exponent,
                1.0,
            )

    if i > 0 and length_normalization_exponent != 0:
        seq_log_prob *= (1 / i) ** length_normalization_exponent

    # Backtrack via backrefs, resolve beams.
    seq_targets_ = []
    indices = rf.range_over_dim(beam_dim)  # FinalBeam -> FinalBeam
    for backrefs, target in zip(seq_backrefs[::-1], seq_targets[::-1]):
        # indices: FinalBeam -> Beam
        # backrefs: Beam -> PrevBeam
        seq_targets_.insert(0, rf.gather(target, indices=indices))
        indices = rf.gather(backrefs, indices=indices)  # FinalBeam -> PrevBeam

    seq_targets__ = TensorArray(seq_targets_[0])
    for target in seq_targets_:
        seq_targets__ = seq_targets__.push_back(target)
    out_spatial_dim = Dim(out_seq_len, name="out-spatial")
    seq_targets = seq_targets__.stack(axis=out_spatial_dim)

    return seq_targets, seq_log_prob, out_spatial_dim, beam_dim


# See: https://github.com/rwth-i6/returnn/issues/1402
_interesting_commits = [
    "a776227",  # -- some baseline before those first optimizations
    "f09222e",
    "fa9818c",
    "01d0653",
    "dc14a2c",
    "361e238",
    "49b69ed",
    "07078b9",
    "2e104f5",
]


def main():
    """main"""
    # pass ++num_epochs 1 ++device cpu or so...
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--bench-action", choices=("run", "multi-run", "profile"), default="run")
    args, remaining_args = arg_parser.parse_known_args()

    try:
        # https://youtrack.jetbrains.com/issue/PY-63226/PyCharm-debugger-hangs-when-process-is-forking
        multiprocessing.set_start_method("spawn")
    except Exception as exc:
        print("multiprocessing.set_start_method 'spawn' exception:", exc)
        print("Ignoring this...")

    if args.bench_action == "run":
        __main__.main(sys.argv[:1] + [_my_file] + remaining_args)
    elif args.bench_action == "profile":
        _custom_loop(remaining_args)
    elif args.bench_action == "multi-run":
        with tempfile.TemporaryDirectory(prefix="returnn-tmp-checkout-") as returnn_tmp_dir:
            _subproc_check_call("git", "clone", "--shared", _returnn_root_dir, returnn_tmp_dir)
            os.chdir(returnn_tmp_dir)
            _subproc_check_call("git", "config", "--local", "advice.detachedHead", "false")
            os.environ["PYTHONUNBUFFERED"] = "1"
            for commit in _interesting_commits:
                _subproc_check_call("git", "checkout", commit)
                _subproc_check_call_filter_returnn_out(sys.executable, "rnn.py", _my_file, *remaining_args)
    else:
        raise ValueError(f"invalid --bench-action {args.bench_action!r}")


def _custom_loop(argv):
    from returnn.log import log
    from returnn.util.basic import hms
    from returnn.datasets import init_dataset
    from returnn.torch.data import pipeline as data_pipeline
    from returnn.torch.data import returnn_dataset_wrapper
    from returnn.torch.data import extern_data as extern_data_util
    from returnn.torch.engine import get_device_from_config_opt
    from returnn.torch.frontend.bridge import rf_module_to_pt_module

    import torch
    from torch.utils.data import DataLoader

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--device", default=None)
    arg_parser.add_argument("--tb-dir", default="tb-log")
    args = arg_parser.parse_args(argv)

    rf.select_backend_torch()

    extern_data_template = extern_data_util.extern_data_template_from_config_opts(extern_data)

    device_with_reason = get_device_from_config_opt(args.device)
    device = device_with_reason.result
    print("Using device:", device, f"({device_with_reason.reason})", file=log.v2)

    model = get_model()
    pt_model = rf_module_to_pt_module(model)
    pt_model.to(device)
    pt_model.train()

    optimizer = torch.optim.Adam(pt_model.parameters(), lr=config["learning_rate"])

    dataset = init_dataset(train)
    wrapped_dataset = returnn_dataset_wrapper.ReturnnDatasetIterDataPipe(dataset)
    batch_size = config["batch_size"]
    max_seqs = config["max_seqs"]
    batches_dataset = data_pipeline.BatchingIterDataPipe(wrapped_dataset, batch_size=batch_size, max_seqs=max_seqs)
    data_loader = DataLoader(batches_dataset, batch_size=None, collate_fn=data_pipeline.collate_batch)
    data_iter = iter(data_loader)

    # noinspection PyUnresolvedReferences,PyProtectedMember
    with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=4, active=3, repeat=2),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(args.tb_dir),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        # https://github.com/pytorch/pytorch/issues/100253
        experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True),
    ) as prof:
        step_idx = 0
        epoch_start_time = time.time()
        elapsed_computation_time = 0

        while True:
            with torch.no_grad():
                extern_data_raw = next(data_iter, None)
            if extern_data_raw is None:
                break

            step_begin_time = time.time()

            optimizer.zero_grad()

            extern_data_ = extern_data_util.raw_dict_to_extern_data(
                extern_data_raw, extern_data_template=extern_data_template, device=device
            )
            rf.init_train_step_run_ctx(train_flag=True, step=step_idx)
            with rf.set_default_device_ctx(device):
                train_step(model=model, extern_data=extern_data_)

            train_ctx = rf.get_run_ctx()
            total_loss = train_ctx.total_loss()
            total_loss.raw_tensor.backward()

            optimizer.step()

            print("step %i, loss %f" % (step_idx, total_loss.raw_tensor.detach().cpu()), file=log.v3)

            elapsed_computation_time += time.time() - step_begin_time
            step_idx += 1
            prof.step()

        elapsed = time.time() - epoch_start_time
        elapsed_computation_percentage = elapsed_computation_time / elapsed
        print(
            "Trained %i steps, %s elapsed (%.1f%% computing time)"
            % (step_idx, hms(elapsed), (elapsed_computation_percentage * 100.0)),
            file=log.v3,
        )


def _subproc_check_call(*args):
    print("$", *args)
    subprocess.check_call(args)


def _subproc_check_call_filter_returnn_out(*args):
    print("$", *args)
    line_count = 0
    need_newline = False
    with subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as p:
        while True:
            line = p.stdout.readline()
            if not line:
                break
            if line_count < 5:
                sys.stdout.buffer.write(line)
            # "Trained %i steps, %s elapsed (%.1f%% computing time)"
            # "elapsed: ..."
            elif b"elapsed" in line:
                if need_newline:
                    sys.stdout.buffer.write(b"\n")
                sys.stdout.buffer.write(line)
            else:
                sys.stdout.buffer.write(b".")
                need_newline = True
            sys.stdout.buffer.flush()
            line_count += 1
    if need_newline:
        sys.stdout.buffer.write(b"\n")
        sys.stdout.buffer.flush()
    if p.returncode != 0:
        raise subprocess.CalledProcessError(p.returncode, args)


if __name__ == "__main__":
    main()

else:
    globals().update(config)
