#!/usr/bin/env python3

"""
This script can import a Blocks MT model into Returnn.
It currently assumes a specific Returnn network topology with specific layer names.
Example Returnn network topology:

.. code-block:: python

    network = {
    "source_embed": {"class": "linear", "activation": None, "with_bias": False, "n_out": 620},

    "lstm0_fw" : { "class": "rec", "unit": "standardlstm", "unit_opts": {"use_peepholes": True, "forget_bias": 0.0}, "initial_state": "var", "n_out" : 1000, "direction": 1, "from": ["source_embed"] },
    "lstm0_bw" : { "class": "rec", "unit": "standardlstm", "unit_opts": {"use_peepholes": True, "forget_bias": 0.0}, "initial_state": "var", "n_out" : 1000, "direction": -1, "from": ["source_embed"] },

    "lstm1_fw" : { "class": "rec", "unit": "standardlstm", "unit_opts": {"use_peepholes": True, "forget_bias": 0.0}, "initial_state": "var", "n_out" : 1000, "direction": 1, "from": ["lstm0_fw", "lstm0_bw"] },
    "lstm1_bw" : { "class": "rec", "unit": "standardlstm", "unit_opts": {"use_peepholes": True, "forget_bias": 0.0}, "initial_state": "var", "n_out" : 1000, "direction": -1, "from": ["lstm0_fw", "lstm0_bw"] },

    "encoder": {"class": "copy", "from": ["lstm1_fw", "lstm1_bw"]},
    "enc_ctx": {"class": "linear", "activation": None, "with_bias": True, "from": ["encoder"], "n_out": 1000},  # preprocessed_attended in Blocks
    "fertility": {"class": "linear", "activation": "sigmoid", "with_bias": False, "from": ["encoder"], "n_out": 1},

    "output": {"class": "rec", "from": [], "unit": {
        'output': {'class': 'choice', 'target': 'classes', 'beam_size': beam_size, 'from': ["output_prob"], "initial_output": 0, "length_normalization": False},
        "end": {"class": "compare", "from": ["output"], "value": 0},
        'target_embed': {'class': 'linear', 'activation': None, "with_bias": False, 'from': ['output'], "n_out": 621, "initial_output": 0},  # feedback_input
        "weight_feedback": {"class": "linear", "activation": None, "with_bias": False, "from": ["prev:accum_att_weights"], "n_out": 1000},
        "prev_s_state": {"class": "get_last_hidden_state", "from": ["prev:s"], "n_out": 2000},
        "prev_s_transformed": {"class": "linear", "activation": None, "with_bias": False, "from": ["prev_s_state"], "n_out": 1000},
        "energy_in": {"class": "combine", "kind": "add", "from": ["base:enc_ctx", "weight_feedback", "prev_s_transformed"], "n_out": 1000},
        "energy_tanh": {"class": "activation", "activation": "tanh", "from": ["energy_in"]},
        "energy": {"class": "linear", "activation": None, "with_bias": False, "from": ["energy_tanh"], "n_out": 1},  # (B, enc-T, 1)
        "att_weights": {"class": "softmax_over_spatial", "from": ["energy"]},  # (B, enc-T, 1)
        "accum_att_weights": {"class": "eval", "from": ["prev:accum_att_weights", "att_weights", "base:fertility"],
            "eval": "source(0) + source(1) / (2.0 * source(2))", "out_type": {"dim": 1, "shape": (None, 1)}},
        "att": {"class": "generic_attention", "weights": "att_weights", "base": "base:encoder"},
        "s": {"class": "rnn_cell", "unit": "standardlstm", "unit_opts": {"use_peepholes": True, "forget_bias": 0.0},
            "initial_state": "var", "from": ["target_embed", "att"], "n_out": 1000},  # transform
        "readout_in": {"class": "linear", "from": ["prev:s", "prev:target_embed", "att"], "activation": None, "n_out": 1000},  # merge + post_merge bias
        "readout": {"class": "reduce_out", "mode": "max", "num_pieces": 2, "from": ["readout_in"]},
        "output_prob": {"class": "softmax", "from": ["readout"], "target": "classes", "loss": "ce"}
    }, "target": "classes", "max_seq_len": 75},

    "decision": {
        "class": "decide", "from": ["output"], "length_normalization": True,
        "loss": "edit_distance", "target": "classes",
        "loss_opts": {
            #"debug_print": True
            }
        }
    }

"""

from __future__ import annotations

import os
import sys
import numpy
import re
from pprint import pprint
from numpy.testing import assert_almost_equal
import tensorflow as tf
import pickle

my_dir = os.path.dirname(os.path.abspath(__file__))
returnn_dir = os.path.dirname(my_dir)
sys.path.insert(0, returnn_dir)

from returnn.util import better_exchook
import returnn.__main__ as rnn
import returnn.util.basic as util
from returnn.tf.network import TFNetwork
from returnn.tf.layers.basic import SourceLayer, LayerBase, LinearLayer
from returnn.tf.layers.rec import ChoiceLayer


def get_network():
    """
    :rtype: TFNetwork
    """
    return rnn.engine.network


def get_input_layers():
    """
    :rtype: list[LayerBase]
    """
    ls = []
    for layer in get_network().layers.values():
        if len(layer.sources) != 1:
            continue
        if isinstance(layer.sources[0], SourceLayer):
            ls.append(layer)
    return ls


def find_our_input_embed_layer():
    """
    :rtype: LinearLayer
    """
    input_layers = get_input_layers()
    assert len(input_layers) == 1
    layer = input_layers[0]
    assert isinstance(layer, LinearLayer)
    return layer


def get_in_hierarchy(name, hierarchy):
    """
    :param str name: e.g. "decoder/sequencegenerator"
    :param dict[str,dict[str]] hierarchy: nested hierarchy
    :rtype: dict[str,dict[str]]
    """
    if "/" in name:
        name, rest = name.split("/", 1)
    else:
        rest = None
    if rest is None:
        return hierarchy[name]
    else:
        return get_in_hierarchy(rest, hierarchy[name])


def main():
    rnn.init(
        command_line_options=sys.argv[1:],
        config_updates={
            "task": "nop",
            "log": None,
            "device": "cpu",
            "allow_random_model_init": True,
            "debug_add_check_numerics_on_output": False,
        },
        extra_greeting="Import Blocks MT model.",
    )
    assert util.BackendEngine.is_tensorflow_selected()
    config = rnn.config

    # Load Blocks MT model params.
    if not config.has("blocks_mt_model"):
        print("Please provide the option blocks_mt_model.")
        sys.exit(1)
    blocks_mt_model_fn = config.value("blocks_mt_model", "")
    assert blocks_mt_model_fn
    assert os.path.exists(blocks_mt_model_fn)
    if os.path.isdir(blocks_mt_model_fn):
        blocks_mt_model_fn += "/params.npz"
        assert os.path.exists(blocks_mt_model_fn)

    dry_run = config.bool("dry_run", False)
    if dry_run:
        our_model_fn = None
        print("Dry-run, will not save model.")
    else:
        our_model_fn = config.value("model", "returnn-model") + ".imported"
        print("Will save Returnn model as %s." % our_model_fn)
        assert os.path.exists(os.path.dirname(our_model_fn) or "."), "model-dir does not exist"
        assert not os.path.exists(our_model_fn + util.get_model_filename_postfix()), "model-file already exists"

    blocks_mt_model = numpy.load(blocks_mt_model_fn)
    assert isinstance(blocks_mt_model, numpy.lib.npyio.NpzFile), "did not expect type %r in file %r" % (
        type(blocks_mt_model),
        blocks_mt_model_fn,
    )
    print("Params found in Blocks model:")
    blocks_params = {}  # type: dict[str,numpy.ndarray]
    blocks_params_hierarchy = {}  # type: dict[str,dict[str]]
    blocks_total_num_params = 0
    for key in sorted(blocks_mt_model.keys()):
        value = blocks_mt_model[key]
        key = key.replace("-", "/")
        assert key[0] == "/"
        key = key[1:]
        blocks_params[key] = value
        print("  %s: %s, %s" % (key, value.shape, value.dtype))
        blocks_total_num_params += numpy.prod(value.shape)
        d = blocks_params_hierarchy
        for part in key.split("/"):
            d = d.setdefault(part, {})
    print("Blocks total num params: %i" % blocks_total_num_params)

    # Init our network structure.
    from returnn.tf.layers.rec import _SubnetworkRecCell

    _SubnetworkRecCell._debug_out = []  # enable for debugging intermediate values below
    ChoiceLayer._debug_out = []  # also for debug outputs of search
    rnn.engine.use_search_flag = True  # construct the net as in search
    rnn.engine.init_network_from_config()
    print("Our network model params:")
    our_params = {}  # type: dict[str,tf.Variable]
    our_total_num_params = 0
    for v in rnn.engine.network.get_params_list():
        key = v.name[:-2]
        our_params[key] = v
        print("  %s: %s, %s" % (key, v.shape, v.dtype.base_dtype.name))
        our_total_num_params += numpy.prod(v.shape.as_list())
    print("Our total num params: %i" % our_total_num_params)

    # Now matching...
    blocks_used_params = set()  # type: set[str]
    our_loaded_params = set()  # type: set[str]

    def import_var(our_var, blocks_param):
        """
        :param tf.Variable our_var:
        :param str|numpy.ndarray blocks_param:
        """
        assert isinstance(our_var, tf.Variable)
        if isinstance(blocks_param, str):
            blocks_param = load_blocks_var(blocks_param)
        assert isinstance(blocks_param, numpy.ndarray)
        assert tuple(our_var.shape.as_list()) == blocks_param.shape
        our_loaded_params.add(our_var.name[:-2])
        our_var.load(blocks_param, session=rnn.engine.tf_session)

    def load_blocks_var(blocks_param_name):
        """
        :param str blocks_param_name:
        :rtype: numpy.ndarray
        """
        assert isinstance(blocks_param_name, str)
        assert blocks_param_name in blocks_params
        blocks_used_params.add(blocks_param_name)
        return blocks_params[blocks_param_name]

    enc_name = "bidirectionalencoder"
    enc_embed_name = "EncoderLookUp0.W"
    assert enc_name in blocks_params_hierarchy
    assert enc_embed_name in blocks_params_hierarchy[enc_name]  # input embedding
    num_encoder_layers = max(
        [
            int(re.match(".*([0-9]+)", s).group(1))
            for s in blocks_params_hierarchy[enc_name]
            if s.startswith("EncoderBidirectionalLSTM")
        ]
    )
    blocks_input_dim, blocks_input_embed_dim = blocks_params["%s/%s" % (enc_name, enc_embed_name)].shape
    print("Blocks input dim: %i, embed dim: %i" % (blocks_input_dim, blocks_input_embed_dim))
    print("Blocks num encoder layers: %i" % num_encoder_layers)
    expected_enc_entries = ["EncoderLookUp0.W"] + [
        "EncoderBidirectionalLSTM%i" % i for i in range(1, num_encoder_layers + 1)
    ]
    assert set(expected_enc_entries) == set(blocks_params_hierarchy[enc_name].keys())

    our_input_layer = find_our_input_embed_layer()
    assert our_input_layer.input_data.dim == blocks_input_dim
    assert our_input_layer.output.dim == blocks_input_embed_dim
    assert not our_input_layer.with_bias
    import_var(our_input_layer.params["W"], "%s/%s" % (enc_name, enc_embed_name))

    dec_name = "decoder/sequencegenerator"
    dec_hierarchy_base = get_in_hierarchy(dec_name, blocks_params_hierarchy)
    assert set(dec_hierarchy_base.keys()) == {"att_trans", "readout"}
    dec_embed_name = "readout/lookupfeedbackwmt15/lookuptable.W"
    get_in_hierarchy(dec_embed_name, dec_hierarchy_base)  # check

    for i in range(num_encoder_layers):
        # Assume standard LSTMCell.
        # i = input_gate, j = new_input, f = forget_gate, o = output_gate
        # lstm_matrix = self._linear1([inputs, m_prev])
        # i, j, f, o = array_ops.split(value=lstm_matrix, num_or_size_splits=4, axis=1)
        # bias (4*in), kernel (in+out,4*out), w_(f|i|o)_diag (out)
        # prefix: rec/rnn/lstm_cell
        # Blocks: gate-in, gate-forget, next-in, gate-out
        for direction in ("fwd", "bwd"):
            our_layer = get_network().layers["lstm%i_%s" % (i, direction[:2])]
            blocks_prefix = "bidirectionalencoder/EncoderBidirectionalLSTM%i" % (i + 1,)
            # (in,out*4), (out*4,)
            W_in, b = [
                load_blocks_var(
                    "%s/%s_fork/fork_inputs.%s" % (blocks_prefix, {"bwd": "back", "fwd": "fwd"}[direction], p)
                )
                for p in ("W", "b")
            ]
            W_re = load_blocks_var(
                "%s/bidirectionalseparateparameters/%s.W_state"
                % (blocks_prefix, {"fwd": "forward", "bwd": "backward"}[direction])
            )
            W = numpy.concatenate([W_in, W_re], axis=0)
            b = lstm_vec_blocks_to_tf(b)
            W = lstm_vec_blocks_to_tf(W)
            import_var(our_layer.params["rnn/lstm_cell/bias"], b)
            import_var(our_layer.params["rnn/lstm_cell/kernel"], W)
            import_var(
                our_layer.params["initial_c"],
                "%s/bidirectionalseparateparameters/%s.initial_cells"
                % (blocks_prefix, {"fwd": "forward", "bwd": "backward"}[direction]),
            )
            import_var(
                our_layer.params["initial_h"],
                "%s/bidirectionalseparateparameters/%s.initial_state"
                % (blocks_prefix, {"fwd": "forward", "bwd": "backward"}[direction]),
            )
            for s1, s2 in [
                ("W_cell_to_in", "w_i_diag"),
                ("W_cell_to_forget", "w_f_diag"),
                ("W_cell_to_out", "w_o_diag"),
            ]:
                import_var(
                    our_layer.params["rnn/lstm_cell/%s" % s2],
                    "%s/bidirectionalseparateparameters/%s.%s"
                    % (blocks_prefix, {"fwd": "forward", "bwd": "backward"}[direction], s1),
                )
    import_var(
        get_network().layers["enc_ctx"].params["W"],
        "decoder/sequencegenerator/att_trans/attention/encoder_state_transformer.W",
    )
    import_var(
        get_network().layers["enc_ctx"].params["b"],
        "decoder/sequencegenerator/att_trans/attention/encoder_state_transformer.b",
    )
    import_var(our_params["output/rec/s/initial_c"], "decoder/sequencegenerator/att_trans/lstm_decoder.initial_cells")
    import_var(our_params["output/rec/s/initial_h"], "decoder/sequencegenerator/att_trans/lstm_decoder.initial_state")
    import_var(
        our_params["output/rec/weight_feedback/W"],
        "decoder/sequencegenerator/att_trans/attention/sum_alignment_transformer.W",
    )
    import_var(
        our_params["output/rec/target_embed/W"], "decoder/sequencegenerator/readout/lookupfeedbackwmt15/lookuptable.W"
    )
    import_var(our_params["fertility/W"], "decoder/sequencegenerator/att_trans/attention/fertility_transformer.W")
    import_var(our_params["output/rec/energy/W"], "decoder/sequencegenerator/att_trans/attention/energy_comp/linear.W")
    prev_s_trans_W_states = load_blocks_var(
        "decoder/sequencegenerator/att_trans/attention/state_trans/transform_states.W"
    )
    prev_s_trans_W_cells = load_blocks_var(
        "decoder/sequencegenerator/att_trans/attention/state_trans/transform_cells.W"
    )
    prev_s_trans_W = numpy.concatenate([prev_s_trans_W_cells, prev_s_trans_W_states], axis=0)
    import_var(our_params["output/rec/prev_s_transformed/W"], prev_s_trans_W)
    import_var(
        our_params["output/rec/s/rec/lstm_cell/bias"], numpy.zeros(our_params["output/rec/s/rec/lstm_cell/bias"].shape)
    )
    dec_lstm_kernel_in_feedback = load_blocks_var(
        "decoder/sequencegenerator/att_trans/feedback_to_decoder/fork_inputs.W"
    )
    dec_lstm_kernel_in_ctx = load_blocks_var("decoder/sequencegenerator/att_trans/context_to_decoder/fork_inputs.W")
    dec_lstm_kernel_re = load_blocks_var("decoder/sequencegenerator/att_trans/lstm_decoder.W_state")
    dec_lstm_kernel = numpy.concatenate(
        [dec_lstm_kernel_in_feedback, dec_lstm_kernel_in_ctx, dec_lstm_kernel_re], axis=0
    )
    dec_lstm_kernel = lstm_vec_blocks_to_tf(dec_lstm_kernel)
    import_var(our_params["output/rec/s/rec/lstm_cell/kernel"], dec_lstm_kernel)
    for s1, s2 in [("W_cell_to_in", "w_i_diag"), ("W_cell_to_forget", "w_f_diag"), ("W_cell_to_out", "w_o_diag")]:
        import_var(
            our_params["output/rec/s/rec/lstm_cell/%s" % s2], "decoder/sequencegenerator/att_trans/lstm_decoder.%s" % s1
        )
    readout_in_W_states = load_blocks_var("decoder/sequencegenerator/readout/merge/transform_states.W")
    readout_in_W_feedback = load_blocks_var("decoder/sequencegenerator/readout/merge/transform_feedback.W")
    readout_in_W_att = load_blocks_var("decoder/sequencegenerator/readout/merge/transform_weighted_averages.W")
    readout_in_W = numpy.concatenate([readout_in_W_states, readout_in_W_feedback, readout_in_W_att], axis=0)
    import_var(our_params["output/rec/readout_in/W"], readout_in_W)
    import_var(
        our_params["output/rec/readout_in/b"],
        "decoder/sequencegenerator/readout/initializablefeedforwardsequence/maxout_bias.b",
    )
    import_var(
        our_params["output/rec/output_prob/W"],
        "decoder/sequencegenerator/readout/initializablefeedforwardsequence/softmax1.W",
    )
    import_var(
        our_params["output/rec/output_prob/b"],
        "decoder/sequencegenerator/readout/initializablefeedforwardsequence/softmax1.b",
    )

    print("Not initialized own params:")
    count = 0
    for key, v in sorted(our_params.items()):
        if key in our_loaded_params:
            continue
        print("  %s: %s, %s" % (key, v.shape, v.dtype.base_dtype.name))
        count += 1
    if not count:
        print("  None.")
    print("Not used Blocks params:")
    count = 0
    for key, value in sorted(blocks_params.items()):
        if key in blocks_used_params:
            continue
        print("  %s: %s, %s" % (key, value.shape, value.dtype))
        count += 1
    if not count:
        print("  None.")
    print("Done.")

    blocks_debug_dump_output = config.value("blocks_debug_dump_output", None)
    if blocks_debug_dump_output:
        print("Will read Blocks debug dump output from %r and compare with Returnn outputs." % blocks_debug_dump_output)
        blocks_initial_outputs = numpy.load("%s/initial_states_data.0.npz" % blocks_debug_dump_output)
        blocks_search_log = pickle.load(open("%s/search.log.pkl" % blocks_debug_dump_output, "rb"), encoding="bytes")
        blocks_search_log = {d[b"step"]: d for d in blocks_search_log}
        input_seq = blocks_initial_outputs["input"]
        beam_size, seq_len = input_seq.shape
        input_seq = input_seq[0]  # all the same, select beam 0
        assert isinstance(input_seq, numpy.ndarray)
        print("Debug input seq: %s" % input_seq.tolist())
        from returnn.datasets.generating import StaticDataset

        dataset = StaticDataset(
            data=[{"data": input_seq}],
            output_dim={"data": get_network().extern_data.get_default_input_data().get_kwargs()},
        )
        dataset.init_seq_order(epoch=0)
        extract_output_dict = {
            "enc_src_emb": get_network().layers["source_embed"].output.get_placeholder_as_batch_major(),
            "encoder": get_network().layers["encoder"].output.get_placeholder_as_batch_major(),
            "enc_ctx": get_network().layers["enc_ctx"].output.get_placeholder_as_batch_major(),
            "output": get_network().layers["output"].output.get_placeholder_as_batch_major(),
        }
        from returnn.tf.layers.basic import concat_sources

        for i in range(num_encoder_layers):
            extract_output_dict["enc_layer_%i" % i] = concat_sources(
                [get_network().layers["lstm%i_fw" % i], get_network().layers["lstm%i_bw" % i]]
            ).get_placeholder_as_batch_major()
        extract_output_dict["enc_layer_0_fwd"] = (
            get_network().layers["lstm0_fw"].output.get_placeholder_as_batch_major()
        )
        our_output = rnn.engine.run_single(dataset=dataset, seq_idx=0, output_dict=extract_output_dict)
        blocks_out = blocks_initial_outputs["bidirectionalencoder_EncoderLookUp0__EncoderLookUp0_apply_output"]
        our_out = our_output["enc_src_emb"]
        print("our enc emb shape:", our_out.shape)
        print("Blocks enc emb shape:", blocks_out.shape)
        assert our_out.shape[:2] == (1, seq_len)
        assert blocks_out.shape[:2] == (seq_len, beam_size)
        assert our_out.shape[2] == blocks_out.shape[2]
        assert_almost_equal(our_out[0], blocks_out[:, 0], decimal=5)
        blocks_lstm0_out_ref = calc_lstm(blocks_out[:, 0], blocks_params)
        blocks_lstm0_out = blocks_initial_outputs[
            "bidirectionalencoder_EncoderBidirectionalLSTM1_bidirectionalseparateparameters_forward__forward_apply_states"
        ]
        our_lstm0_out = our_output["enc_layer_0_fwd"]
        assert blocks_lstm0_out.shape == (seq_len, beam_size) + blocks_lstm0_out_ref.shape
        assert our_lstm0_out.shape == (1, seq_len) + blocks_lstm0_out_ref.shape
        assert_almost_equal(blocks_lstm0_out[0, 0], blocks_lstm0_out_ref, decimal=6)
        print("Blocks LSTM0 frame 0 matched to ref calc.")
        assert_almost_equal(our_lstm0_out[0, 0], blocks_lstm0_out_ref, decimal=6)
        print("Our LSTM0 frame 0 matched to ref calc.")
        for i in range(num_encoder_layers):
            blocks_out = blocks_initial_outputs[
                "bidirectionalencoder_EncoderBidirectionalLSTM%i_bidirectionalseparateparameters__bidirectionalseparateparameters_apply_output_0"
                % (i + 1,)
            ]
            our_out = our_output["enc_layer_%i" % i]
            print("our enc layer %i shape:" % i, our_out.shape)
            print("Blocks enc layer %i shape:" % i, blocks_out.shape)
            assert our_out.shape[:2] == (1, seq_len)
            assert blocks_out.shape[:2] == (seq_len, beam_size)
            assert our_out.shape[2] == blocks_out.shape[2]
            assert_almost_equal(our_out[0], blocks_out[:, 0], decimal=6)
        print("our encoder shape:", our_output["encoder"].shape)
        blocks_encoder_out = blocks_initial_outputs["bidirectionalencoder__bidirectionalencoder_apply_representation"]
        print("Blocks encoder shape:", blocks_encoder_out.shape)
        assert our_output["encoder"].shape[:2] == (1, seq_len)
        assert blocks_encoder_out.shape[:2] == (seq_len, beam_size)
        assert our_output["encoder"].shape[2] == blocks_encoder_out.shape[2]
        assert_almost_equal(our_output["encoder"][0], blocks_encoder_out[:, 0], decimal=6)
        blocks_first_frame_outputs = numpy.load("%s/next_states.0.npz" % blocks_debug_dump_output)
        blocks_enc_ctx_out = blocks_first_frame_outputs[
            "decoder_sequencegenerator_att_trans_attention__attention_preprocess_preprocessed_attended"
        ]
        our_enc_ctx_out = our_output["enc_ctx"]
        print("Blocks enc ctx shape:", blocks_enc_ctx_out.shape)
        assert blocks_enc_ctx_out.shape[:2] == (seq_len, beam_size)
        assert our_enc_ctx_out.shape[:2] == (1, seq_len)
        assert blocks_enc_ctx_out.shape[2:] == our_enc_ctx_out.shape[2:]
        assert_almost_equal(blocks_enc_ctx_out[:, 0], our_enc_ctx_out[0], decimal=5)
        fertility = numpy.dot(
            blocks_encoder_out[:, 0],
            blocks_params["decoder/sequencegenerator/att_trans/attention/fertility_transformer.W"],
        )
        fertility = sigmoid(fertility)
        assert fertility.shape == (seq_len, 1)
        fertility = fertility[:, 0]
        assert fertility.shape == (seq_len,)
        our_dec_outputs = {v["step"]: v for v in _SubnetworkRecCell._debug_out}
        assert our_dec_outputs
        print("our dec frame keys:", sorted(our_dec_outputs[0].keys()))
        our_dec_search_outputs = {v["step"]: v for v in ChoiceLayer._debug_out}
        assert our_dec_search_outputs
        print("our dec search frame keys:", sorted(our_dec_search_outputs[0].keys()))
        print("Blocks search frame keys:", sorted(blocks_search_log[0].keys()))
        dec_lookup = blocks_params["decoder/sequencegenerator/readout/lookupfeedbackwmt15/lookuptable.W"]
        last_lstm_state = blocks_params["decoder/sequencegenerator/att_trans/lstm_decoder.initial_state"]
        last_lstm_cells = blocks_params["decoder/sequencegenerator/att_trans/lstm_decoder.initial_cells"]
        last_accumulated_weights = numpy.zeros((seq_len,), dtype="float32")
        last_output = 0
        dec_seq_len = 0
        for dec_step in range(100):
            blocks_frame_state_outputs_fn = "%s/next_states.%i.npz" % (blocks_debug_dump_output, dec_step)
            blocks_frame_probs_outputs_fn = "%s/logprobs.%i.npz" % (blocks_debug_dump_output, dec_step)
            if dec_step > 3:
                if not os.path.exists(blocks_frame_state_outputs_fn) or not os.path.exists(
                    blocks_frame_probs_outputs_fn
                ):
                    print("Seq not ended yet but frame not found for step %i." % dec_step)
                    break
            blocks_frame_state_outputs = numpy.load(blocks_frame_state_outputs_fn)
            blocks_frame_probs_outputs = numpy.load(blocks_frame_probs_outputs_fn)
            blocks_search_frame = blocks_search_log[dec_step]
            our_dec_frame_outputs = our_dec_outputs[dec_step]
            assert our_dec_frame_outputs["step"] == dec_step
            assert our_dec_frame_outputs[":i.output"].tolist() == [dec_step]
            our_dec_search_frame_outputs = our_dec_search_outputs[dec_step]

            blocks_last_lstm_state = blocks_frame_probs_outputs[
                "decoder_sequencegenerator__sequencegenerator_generate_states"
            ]
            blocks_last_lstm_cells = blocks_frame_probs_outputs[
                "decoder_sequencegenerator__sequencegenerator_generate_cells"
            ]
            assert blocks_last_lstm_state.shape == (beam_size, last_lstm_state.shape[0])
            assert_almost_equal(blocks_last_lstm_state[0], last_lstm_state, decimal=5)
            assert_almost_equal(blocks_last_lstm_cells[0], last_lstm_cells, decimal=5)
            our_last_lstm_cells = our_dec_frame_outputs["prev:s.extra.state"][0]
            our_last_lstm_state = our_dec_frame_outputs["prev:s.extra.state"][1]
            assert our_last_lstm_state.shape == our_last_lstm_cells.shape == (beam_size, last_lstm_state.shape[0])
            assert_almost_equal(our_last_lstm_state[0], last_lstm_state, decimal=5)
            assert_almost_equal(our_last_lstm_cells[0], last_lstm_cells, decimal=5)
            our_last_s = our_dec_frame_outputs["prev:s.output"]
            assert our_last_s.shape == (beam_size, last_lstm_state.shape[0])
            assert_almost_equal(our_last_s[0], last_lstm_state, decimal=5)

            blocks_last_accum_weights = blocks_frame_probs_outputs[
                "decoder_sequencegenerator__sequencegenerator_generate_accumulated_weights"
            ]
            assert blocks_last_accum_weights.shape == (beam_size, seq_len)
            assert_almost_equal(blocks_last_accum_weights[0], last_accumulated_weights, decimal=5)
            our_last_accum_weights = our_dec_frame_outputs["prev:accum_att_weights.output"]
            assert our_last_accum_weights.shape == (beam_size, seq_len if dec_step > 0 else 1, 1)
            if dec_step > 0:
                assert_almost_equal(our_last_accum_weights[0, :, 0], last_accumulated_weights, decimal=4)
            else:
                assert_almost_equal(our_last_accum_weights[0, 0, 0], last_accumulated_weights.sum(), decimal=4)

            energy_sum = numpy.copy(blocks_enc_ctx_out[:, 0])  # (T,enc-ctx-dim)
            weight_feedback = numpy.dot(
                last_accumulated_weights[:, None],
                blocks_params["decoder/sequencegenerator/att_trans/attention/sum_alignment_transformer.W"],
            )
            energy_sum += weight_feedback
            transformed_states = numpy.dot(
                last_lstm_state[None, :],
                blocks_params["decoder/sequencegenerator/att_trans/attention/state_trans/transform_states.W"],
            )
            transformed_cells = numpy.dot(
                last_lstm_cells[None, :],
                blocks_params["decoder/sequencegenerator/att_trans/attention/state_trans/transform_cells.W"],
            )
            energy_sum += transformed_states + transformed_cells
            assert energy_sum.shape == (seq_len, blocks_enc_ctx_out.shape[-1])
            blocks_energy_sum_tanh = blocks_frame_probs_outputs[
                "decoder_sequencegenerator_att_trans_attention_energy_comp_tanh__tanh_apply_output"
            ]
            assert blocks_energy_sum_tanh.shape == (seq_len, beam_size, energy_sum.shape[-1])
            assert_almost_equal(blocks_energy_sum_tanh[:, 0], numpy.tanh(energy_sum), decimal=5)
            assert our_dec_frame_outputs["weight_feedback.output"].shape == (
                beam_size,
                seq_len if dec_step > 0 else 1,
                blocks_enc_ctx_out.shape[-1],
            )
            assert our_dec_frame_outputs["prev_s_transformed.output"].shape == (beam_size, blocks_enc_ctx_out.shape[-1])
            our_energy_sum = our_dec_frame_outputs["energy_in.output"]
            assert our_energy_sum.shape == (beam_size, seq_len, blocks_enc_ctx_out.shape[-1])
            assert_almost_equal(our_energy_sum[0], energy_sum, decimal=4)
            blocks_energy = blocks_frame_probs_outputs[
                "decoder_sequencegenerator_att_trans_attention_energy_comp__energy_comp_apply_output"
            ]
            assert blocks_energy.shape == (seq_len, beam_size, 1)
            energy = numpy.dot(
                numpy.tanh(energy_sum),
                blocks_params["decoder/sequencegenerator/att_trans/attention/energy_comp/linear.W"],
            )
            assert energy.shape == (seq_len, 1)
            assert_almost_equal(blocks_energy[:, 0], energy, decimal=4)
            our_energy = our_dec_frame_outputs["energy.output"]
            assert our_energy.shape == (beam_size, seq_len, 1)
            assert_almost_equal(our_energy[0], energy, decimal=4)
            weights = softmax(energy[:, 0])
            assert weights.shape == (seq_len,)
            our_weights = our_dec_frame_outputs["att_weights.output"]
            assert our_weights.shape == (beam_size, seq_len, 1)
            assert_almost_equal(our_weights[0, :, 0], weights, decimal=4)
            accumulated_weights = last_accumulated_weights + weights / (2.0 * fertility)
            assert accumulated_weights.shape == (seq_len,)
            # blocks_accumulated_weights = blocks_frame_probs_outputs["decoder_sequencegenerator_att_trans_attention__attention_take_glimpses_accumulated_weights"]
            # assert blocks_accumulated_weights.shape == (beam_size, seq_len)
            # assert_almost_equal(blocks_accumulated_weights[0], accumulated_weights, decimal=5)
            blocks_weights = blocks_frame_probs_outputs[
                "decoder_sequencegenerator_att_trans_attention__attention_compute_weights_output_0"
            ]
            assert blocks_weights.shape == (seq_len, beam_size)
            assert_almost_equal(weights, blocks_weights[:, 0], decimal=4)
            our_accum_weights = our_dec_frame_outputs["accum_att_weights.output"]
            assert our_accum_weights.shape == (beam_size, seq_len, 1)
            weighted_avg = (weights[:, None] * blocks_encoder_out[:, 0]).sum(axis=0)  # att in our
            assert weighted_avg.shape == (blocks_encoder_out.shape[-1],)
            blocks_weighted_avg = blocks_frame_probs_outputs[
                "decoder_sequencegenerator_att_trans_attention__attention_compute_weighted_averages_output_0"
            ]
            assert blocks_weighted_avg.shape == (beam_size, blocks_encoder_out.shape[-1])
            assert_almost_equal(blocks_weighted_avg[0], weighted_avg, decimal=4)
            our_att = our_dec_frame_outputs["att.output"]
            assert our_att.shape == (beam_size, blocks_encoder_out.shape[-1])
            assert_almost_equal(our_att[0], weighted_avg, decimal=4)

            blocks_last_output = blocks_frame_probs_outputs[
                "decoder_sequencegenerator__sequencegenerator_generate_outputs"
            ]
            assert blocks_last_output.shape == (beam_size,)
            assert max(blocks_last_output[0], 0) == last_output
            last_target_embed = dec_lookup[last_output]
            if dec_step == 0:
                last_target_embed = numpy.zeros_like(last_target_embed)
            our_last_target_embed = our_dec_frame_outputs["prev:target_embed.output"]
            assert our_last_target_embed.shape == (beam_size, dec_lookup.shape[-1])
            assert_almost_equal(our_last_target_embed[0], last_target_embed, decimal=4)

            readout_in_state = numpy.dot(
                last_lstm_state, blocks_params["decoder/sequencegenerator/readout/merge/transform_states.W"]
            )
            blocks_trans_state = blocks_frame_probs_outputs[
                "decoder_sequencegenerator_readout_merge__merge_apply_states"
            ]
            assert blocks_trans_state.shape == (beam_size, last_lstm_state.shape[0])
            assert_almost_equal(blocks_trans_state[0], readout_in_state, decimal=4)
            readout_in_feedback = numpy.dot(
                last_target_embed, blocks_params["decoder/sequencegenerator/readout/merge/transform_feedback.W"]
            )
            blocks_trans_feedback = blocks_frame_probs_outputs[
                "decoder_sequencegenerator_readout_merge__merge_apply_feedback"
            ]
            assert blocks_trans_feedback.shape == (beam_size, readout_in_feedback.shape[0])
            assert_almost_equal(blocks_trans_feedback[0], readout_in_feedback, decimal=4)
            readout_in_weighted_avg = numpy.dot(
                weighted_avg, blocks_params["decoder/sequencegenerator/readout/merge/transform_weighted_averages.W"]
            )
            blocks_trans_weighted_avg = blocks_frame_probs_outputs[
                "decoder_sequencegenerator_readout_merge__merge_apply_weighted_averages"
            ]
            assert blocks_trans_weighted_avg.shape == (beam_size, readout_in_weighted_avg.shape[0])
            assert_almost_equal(blocks_trans_weighted_avg[0], readout_in_weighted_avg, decimal=4)
            readout_in = readout_in_state + readout_in_feedback + readout_in_weighted_avg
            blocks_readout_in = blocks_frame_probs_outputs[
                "decoder_sequencegenerator_readout_merge__merge_apply_output"
            ]
            assert blocks_readout_in.shape == (beam_size, readout_in.shape[0])
            assert_almost_equal(blocks_readout_in[0], readout_in, decimal=4)
            readout_in += blocks_params[
                "decoder/sequencegenerator/readout/initializablefeedforwardsequence/maxout_bias.b"
            ]
            assert readout_in.shape == (
                blocks_params["decoder/sequencegenerator/readout/initializablefeedforwardsequence/maxout_bias.b"].shape[
                    0
                ],
            )
            our_readout_in = our_dec_frame_outputs["readout_in.output"]
            assert our_readout_in.shape == (beam_size, readout_in.shape[0])
            assert_almost_equal(our_readout_in[0], readout_in, decimal=4)
            readout = readout_in.reshape((readout_in.shape[0] // 2, 2)).max(axis=1)
            our_readout = our_dec_frame_outputs["readout.output"]
            assert our_readout.shape == (beam_size, readout.shape[0])
            assert_almost_equal(our_readout[0], readout, decimal=4)
            prob_logits = (
                numpy.dot(
                    readout,
                    blocks_params["decoder/sequencegenerator/readout/initializablefeedforwardsequence/softmax1.W"],
                )
                + blocks_params["decoder/sequencegenerator/readout/initializablefeedforwardsequence/softmax1.b"]
            )
            assert prob_logits.ndim == 1
            blocks_prob_logits = blocks_frame_probs_outputs[
                "decoder_sequencegenerator_readout__readout_readout_output_0"
            ]
            assert blocks_prob_logits.shape == (beam_size, prob_logits.shape[0])
            assert_almost_equal(blocks_prob_logits[0], prob_logits, decimal=4)
            output_prob = softmax(prob_logits)
            log_output_prob = log_softmax(prob_logits)
            assert_almost_equal(numpy.log(output_prob), log_output_prob, decimal=4)
            our_output_prob = our_dec_frame_outputs["output_prob.output"]
            assert our_output_prob.shape == (beam_size, output_prob.shape[0])
            assert_almost_equal(our_output_prob[0], output_prob, decimal=4)
            blocks_nlog_prob = blocks_frame_probs_outputs["logprobs"]
            assert blocks_nlog_prob.shape == (beam_size, output_prob.shape[0])
            assert_almost_equal(blocks_nlog_prob[0], -log_output_prob, decimal=4)
            assert_almost_equal(our_dec_search_frame_outputs["scores_in_orig"][0], output_prob, decimal=4)
            assert_almost_equal(blocks_search_frame[b"logprobs"][0], -log_output_prob, decimal=4)
            # for b in range(beam_size):
            #  assert_almost_equal(-numpy.log(our_output_prob[b]), blocks_frame_probs_outputs["logprobs"][b], decimal=4)
            ref_output = numpy.argmax(output_prob)
            # Note: Don't take the readout.emit outputs. They are randomly sampled.
            blocks_dec_output = blocks_search_frame[b"outputs"]
            assert blocks_dec_output.shape == (beam_size,)
            our_dec_output = our_dec_frame_outputs["output.output"]
            assert our_dec_output.shape == (beam_size,)
            print("Frame %i: Ref best greedy output symbol: %i" % (dec_step, int(ref_output)))
            print("Blocks labels:", blocks_dec_output.tolist())
            print("Our labels:", our_dec_output.tolist())
            # Well, the following two could be not true if all the other beams have much better scores,
            # but this is unlikely.
            assert ref_output in blocks_dec_output
            assert ref_output in our_dec_output
            if dec_step == 0:
                # This assumes that the results are ordered by score which might not be true (see tf.nn.top_k).
                assert blocks_dec_output[0] == our_dec_output[0] == ref_output
            # We assume that the best is the same. Note that this also might not be true if there are two equally best scores.
            # It also assumes that it's ordered by the score which also might not be true (see tf.nn.top_k).
            # For the same reason, the remaining list and entries might also not perfectly match.
            assert our_dec_output[0] == blocks_dec_output[0]
            # Just follow the first beam.
            ref_output = blocks_dec_output[0]
            assert our_dec_search_frame_outputs["src_beam_idxs"].shape == (1, beam_size)
            assert our_dec_search_frame_outputs["scores"].shape == (1, beam_size)
            print("Blocks src_beam_idxs:", blocks_search_frame[b"indexes"].tolist())
            print("Our src_beam_idxs:", our_dec_search_frame_outputs["src_beam_idxs"][0].tolist())
            print("Blocks scores:", blocks_search_frame[b"chosen_costs"].tolist())
            print("Our scores:", our_dec_search_frame_outputs["scores"][0].tolist())
            if list(our_dec_search_frame_outputs["src_beam_idxs"][0]) != list(blocks_search_frame[b"indexes"]):
                print("Warning, beams do not match.")
                print("Blocks scores base:", blocks_search_frame[b"scores_base"].flatten().tolist())
                print("Our scores base:", our_dec_search_frame_outputs["scores_base"].flatten().tolist())
                # print("Blocks score in orig top k:", sorted(blocks_search_frame[b'logprobs'].flatten())[:beam_size])
                # print("Our score in orig top k:", sorted(-numpy.log(our_dec_search_frame_outputs["scores_in_orig"].flatten()))[:beam_size])
                print(
                    "Blocks score in top k:",
                    sorted(
                        (blocks_search_frame[b"logprobs"] * blocks_search_log[dec_step - 1][b"mask"][:, None]).flatten()
                    )[:beam_size],
                )
                print("Our score in top k:", sorted(-our_dec_search_frame_outputs["scores_in"].flatten())[:beam_size])
                blocks_scores_combined = blocks_search_frame[b"next_costs"]
                our_scores_combined = our_dec_search_frame_outputs["scores_combined"]
                print("Blocks scores combined top k:", sorted(blocks_scores_combined.flatten())[:beam_size])
                print("Our neg scores combined top k:", sorted(-our_scores_combined.flatten())[:beam_size])
                # raise Exception("beams mismatch")
            assert our_dec_search_frame_outputs["src_beam_idxs"][0][0] == blocks_search_frame[b"indexes"][0]
            beam_idx = our_dec_search_frame_outputs["src_beam_idxs"][0][0]
            if beam_idx != 0:
                print("Selecting different beam: %i." % beam_idx)
                # Just overwrite the needed states by Blocks outputs.
                accumulated_weights = blocks_frame_state_outputs[
                    "decoder_sequencegenerator_att_trans_attention__attention_take_glimpses_accumulated_weights"
                ][0]
                weighted_avg = blocks_frame_state_outputs[
                    "decoder_sequencegenerator__sequencegenerator_generate_weighted_averages"
                ][0]
                last_lstm_state = blocks_frame_state_outputs[
                    "decoder_sequencegenerator__sequencegenerator_generate_states"
                ][0]
                last_lstm_cells = blocks_frame_state_outputs[
                    "decoder_sequencegenerator__sequencegenerator_generate_cells"
                ][0]

            # From now on, use blocks_frame_state_outputs instead of blocks_frame_probs_outputs because
            # it will have the beam reordered.
            blocks_target_emb = blocks_frame_state_outputs[
                "decoder_sequencegenerator_fork__fork_apply_feedback_decoder_input"
            ]
            assert blocks_target_emb.shape == (beam_size, dec_lookup.shape[1])
            target_embed = dec_lookup[ref_output]
            assert target_embed.shape == (dec_lookup.shape[1],)
            assert_almost_equal(blocks_target_emb[0], target_embed)

            feedback_to_decoder = numpy.dot(
                target_embed, blocks_params["decoder/sequencegenerator/att_trans/feedback_to_decoder/fork_inputs.W"]
            )
            context_to_decoder = numpy.dot(
                weighted_avg, blocks_params["decoder/sequencegenerator/att_trans/context_to_decoder/fork_inputs.W"]
            )
            lstm_z = feedback_to_decoder + context_to_decoder
            assert (
                lstm_z.shape
                == feedback_to_decoder.shape
                == context_to_decoder.shape
                == (last_lstm_state.shape[-1] * 4,)
            )
            blocks_feedback_to_decoder = blocks_frame_state_outputs[
                "decoder_sequencegenerator_att_trans_feedback_to_decoder__feedback_to_decoder_apply_inputs"
            ]
            blocks_context_to_decoder = blocks_frame_state_outputs[
                "decoder_sequencegenerator_att_trans_context_to_decoder__context_to_decoder_apply_inputs"
            ]
            assert (
                blocks_feedback_to_decoder.shape
                == blocks_context_to_decoder.shape
                == (beam_size, last_lstm_state.shape[-1] * 4)
            )
            assert_almost_equal(blocks_feedback_to_decoder[0], feedback_to_decoder, decimal=4)
            assert_almost_equal(blocks_context_to_decoder[0], context_to_decoder, decimal=4)
            lstm_state, lstm_cells = calc_raw_lstm(
                lstm_z,
                blocks_params=blocks_params,
                prefix="decoder/sequencegenerator/att_trans/lstm_decoder.",
                last_state=last_lstm_state,
                last_cell=last_lstm_cells,
            )
            assert lstm_state.shape == last_lstm_state.shape == lstm_cells.shape == last_lstm_cells.shape
            blocks_lstm_state = blocks_frame_state_outputs[
                "decoder_sequencegenerator_att_trans_lstm_decoder__lstm_decoder_apply_states"
            ]
            blocks_lstm_cells = blocks_frame_state_outputs[
                "decoder_sequencegenerator_att_trans_lstm_decoder__lstm_decoder_apply_cells"
            ]
            assert blocks_lstm_state.shape == blocks_lstm_cells.shape == (beam_size, last_lstm_state.shape[-1])
            assert_almost_equal(blocks_lstm_state[0], lstm_state, decimal=4)
            assert_almost_equal(blocks_lstm_cells[0], lstm_cells, decimal=4)
            our_lstm_cells = our_dec_frame_outputs["s.extra.state"][0]
            our_lstm_state = our_dec_frame_outputs["s.extra.state"][1]
            assert our_lstm_state.shape == our_lstm_cells.shape == (beam_size, lstm_state.shape[0])
            assert_almost_equal(our_lstm_state[0], lstm_state, decimal=4)
            assert_almost_equal(our_lstm_cells[0], lstm_cells, decimal=4)
            our_s = our_dec_frame_outputs["s.output"]
            assert our_s.shape == (beam_size, lstm_state.shape[0])
            assert_almost_equal(our_s[0], lstm_state, decimal=4)

            last_accumulated_weights = accumulated_weights
            last_lstm_state = lstm_state
            last_lstm_cells = lstm_cells
            last_output = ref_output
            if last_output == 0:
                print("Sequence finished, seq len %i." % dec_step)
                dec_seq_len = dec_step
                break
        assert dec_seq_len > 0
        print("All outputs seem to match.")
    else:
        print(
            "blocks_debug_dump_output not specified. It will not compare the model outputs." % blocks_debug_dump_output
        )

    if dry_run:
        print("Dry-run, not saving model.")
    else:
        rnn.engine.save_model(our_model_fn)
    print("Finished importing.")


def lstm_vec_blocks_to_tf(x):
    """
    Blocks order: gate-in, gate-forget, next-in, gate-out
    TF order: i = input_gate, j = new_input, f = forget_gate, o = output_gate
    :param numpy.ndarray x: (..., dim*4)
    :rtype: numpy.ndarray
    """
    axis = x.ndim - 1
    i, f, j, o = numpy.split(x, 4, axis=axis)
    return numpy.concatenate([i, j, f, o], axis=axis)


def calc_lstm(x, blocks_params, t=0):
    """
    :param numpy.ndarray x: (seq_len, in_dim)
    :param dict[str,numpy.ndarray] blocks_params:
    :param int t:
    :rtype: numpy.ndarray
    """
    prefix = "bidirectionalencoder/EncoderBidirectionalLSTM1"
    prefix1 = "%s/bidirectionalseparateparameters/forward." % prefix
    prefix2 = "%s/fwd_fork/fork_inputs." % prefix
    W_in, b = blocks_params[prefix2 + "W"], blocks_params[prefix2 + "b"]
    assert b.ndim == 1
    assert b.shape[0] % 4 == 0
    out_dim = b.shape[0] // 4
    seq_len, in_dim = x.shape
    assert W_in.shape == (in_dim, out_dim * 4)
    z = numpy.dot(x[t], W_in) + b
    assert z.shape == (out_dim * 4,)
    cur_state, cur_cell = calc_raw_lstm(z, blocks_params=blocks_params, prefix=prefix1)
    return cur_state


def calc_raw_lstm(z, blocks_params, prefix, last_state=None, last_cell=None):
    """
    :param numpy.ndarray z: shape (out_dim * 4,)
    :param dict[str,numpy.ndarray] blocks_params:
    :param str prefix: e.g. "bidirectionalencoder/EncoderBidirectionalLSTM1/bidirectionalseparateparameters/forward."
    :param numpy.ndarray|None last_state: (out_dim,)
    :param numpy.ndarray|None last_cell: (out_dim,)
    :rtype: numpy.ndarray
    """
    assert z.ndim == 1
    assert z.shape[-1] % 4 == 0
    out_dim = z.shape[-1] // 4
    W_re = blocks_params[prefix + "W_state"]
    assert W_re.shape == (out_dim, out_dim * 4)
    W_cell_to_in = blocks_params[prefix + "W_cell_to_in"]
    W_cell_to_forget = blocks_params[prefix + "W_cell_to_forget"]
    W_cell_to_out = blocks_params[prefix + "W_cell_to_out"]
    assert W_cell_to_in.shape == W_cell_to_forget.shape == W_cell_to_out.shape == (out_dim,)
    if last_cell is None and last_state is None:
        initial_state, initial_cell = blocks_params[prefix + "initial_state"], blocks_params[prefix + "initial_cells"]
        assert initial_state.shape == initial_cell.shape == (out_dim,)
        last_state = initial_state
        last_cell = initial_cell
    z = z + numpy.dot(last_state, W_re)
    assert z.shape == (out_dim * 4,)
    # Blocks order: gate-in, gate-forget, next-in, gate-out
    gate_in, gate_forget, next_in, gate_out = numpy.split(z, 4)
    gate_in = gate_in + W_cell_to_in * last_cell
    gate_forget = gate_forget + W_cell_to_forget * last_cell
    gate_in = sigmoid(gate_in)
    gate_forget = sigmoid(gate_forget)
    next_in = numpy.tanh(next_in)
    cur_cell = last_cell * gate_forget + next_in * gate_in
    gate_out = gate_out + W_cell_to_out * cur_cell
    gate_out = sigmoid(gate_out)
    cur_state = numpy.tanh(cur_cell) * gate_out
    return cur_state, cur_cell


def softmax(x, axis=-1):
    assert isinstance(x, numpy.ndarray)
    e_x = numpy.exp(x - numpy.max(x, axis=axis, keepdims=True))
    assert isinstance(e_x, numpy.ndarray)
    return e_x / e_x.sum(axis=axis, keepdims=True)


def log_softmax(x, axis=-1):
    assert isinstance(x, numpy.ndarray)
    xdev = x - x.max(axis=axis, keepdims=True)
    lsm = xdev - numpy.log(numpy.sum(numpy.exp(xdev), axis=axis, keepdims=True))
    return lsm


def sigmoid(x):
    return 1.0 / (1.0 + numpy.exp(-x))


if __name__ == "__main__":
    better_exchook.install()
    main()
