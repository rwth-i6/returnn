#!/usr/bin/env python3

"""
This is intended to test a compiled TF graph which was compiled via `compile_tf_graph.py`
with the option `--rec_step_by_step`.
This is just for demonstration, testing and debugging purpose. The search itself is performed in pure Python.
"""

# No RETURNN dependency needed for the basic search. Just TF itself.

import typing
import os
import json
import argparse
import tensorflow as tf
import numpy


class Hyp:
    """
    Represents a hypothesis in a given decoder step, including the label sequence so far.
    """

    def __init__(self, idx):
        """
        :param int idx: hyp idx (to identify it in a beam)
        """
        self.idx = idx
        self.source_idx = None  # type: typing.Optional[int]  # source hyp idx
        self.score = 0.0
        self.seq = []  # label seq

    def expand(self, idx, label, score):
        """
        :param int idx:
        :param int label:
        :param float score:
        :rtype: Hyp
        """
        new_hyp = Hyp(idx=idx)
        new_hyp.source_idx = self.idx
        new_hyp.seq = list(self.seq) + [label]
        new_hyp.score = score
        return new_hyp


def main():
    """
    Main entry.
    """
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--graph", help="compiled TF graph", required=True)
    arg_parser.add_argument("--chkpt", help="TF checkpoint (model params)", required=True)
    arg_parser.add_argument("--beam_size", type=int, default=12)
    arg_parser.add_argument("--rec_step_by_step_json", required=True)
    args = arg_parser.parse_args()

    # We operate only on a single seq (i.e. initially batch dim == 1, then batch dim == beam size).

    def make_initial_feed_dict():
        """
        :return: whatever placeholders we have for input features...
        :rtype: dict
        """
        # TODO...
        return {}

    # Load rec-step-by-step info. See compile_tf_graph.py for details.
    info = json.load(open(args.rec_step_by_step_json))
    assert isinstance(info, dict)

    # Load the graph
    if os.path.splitext(args.graph)[1] in [".meta", ".metatxt"]:  # meta graph
        saver = tf.compat.v1.train.import_meta_graph(args.graph)
    else:  # normal graph
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(open(args.graph, "rb").read())
        tf.import_graph_def(graph_def)
        saver = tf.compat.v1.train.Saver()

    with tf.compat.v1.Session() as session:
        # Load the params.
        saver.restore(session, args.chkpt)

        # Calculate encoder states and initial decoder states.
        initial_feed_dict = make_initial_feed_dict()
        session.run(info["init_op"], feed_dict=initial_feed_dict)

        hyps = [Hyp(idx=0)]

        # Now loop over decoder steps.
        max_dec_len = 100  # TODO better default... depending on input len. or configurable...
        for i in range(max_dec_len):

            # Loop over all stochastic variables.
            for stochastic_var in info["stochastic_var_order"]:
                assert isinstance(stochastic_var, str)

                # Calculate the scores.
                session.run(info["stochastic_vars"][stochastic_var]["calc_scores_op"])
                # Get the scores (probabilities in +log space).
                scores = session.run(info["state_vars"]["stochastic_var_scores_%s" % stochastic_var])
                assert isinstance(scores, numpy.ndarray) and scores.ndim == 2 and scores.shape[0] == len(hyps)
                all_possibilities = [
                    (hyp.score + scores[i, j], j, hyp) for i, hyp in enumerate(hyps) for j in range(scores.shape[1])
                ]

                # TODO: length norm here?

                # Select new hypotheses.
                best_possibilities = sorted(all_possibilities)[
                    : args.beam_size
                ]  # type: typing.List[typing.Tuple[float,int,Hyp]]
                assert len(best_possibilities) == args.beam_size
                hyps = [
                    hyp.expand(idx=i, label=label, score=score)
                    for i, (score, label, hyp) in enumerate(best_possibilities)
                ]

                # Set the choices.
                session.run(
                    info["state_vars"]["stochastic_var_scores_%s" % stochastic_var] + "/Assign...?",  # TODO...
                    feed_dict={
                        info["state_vars"]["stochastic_var_scores_%s" % stochastic_var]
                        + "/Initial...?": [[hyp.seq[-1] for hyp in hyps]]  # TODO...
                    },
                )

                # Select source beams.
                session.run(
                    info["select_src_beams"]["op"],
                    feed_dict={info["select_src_beams"]["src_beams_placeholder"]: [[hyp.source_idx] for hyp in hyps]},
                )

            # Update state.
            session.run(info["next_step_op"])

            # TODO: stopping criterion?

    print("Best hypotheses:")
    for hyp in hyps:
        print("score %.2f: %r" % (hyp.score, hyp.seq))


if __name__ == "__main__":
    from returnn.util import better_exchook

    better_exchook.install()
    main()
