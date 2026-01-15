"""
Generic framework independent reference implementations
"""

from typing import Any, Optional, Union, Dict, List, Tuple
from collections import defaultdict
import numpy


NEG_INF = -float("inf")


def logsumexp(x: Union[numpy.ndarray, Any]) -> float:
    """
    Stable log sum exp.
    """
    x = numpy.array(x)
    if (x == NEG_INF).all():
        return NEG_INF
    a_max = x.max()
    lsp = numpy.log(numpy.sum(numpy.exp(x - a_max)))
    return a_max + lsp


def py_baum_welch(am_scores, float_idx, edges, weights, start_end_states):
    """
    Pure Python Forward-backward (Baum Welch) algorithm.
    The parameters are in the same format as our native fast_baum_welch op.

    :param numpy.ndarray am_scores: (time, batch, dim), in -log space
    :param numpy.ndarray float_idx: (time, batch) -> 0 or 1 (index mask, via seq lens)
    :param numpy.ndarray edges: (4,num_edges), edges of the graph (from,to,emission_idx,sequence_idx)
    :param numpy.ndarray weights: (num_edges,), weights of the edges
    :param numpy.ndarray start_end_states: (2, batch), (start,end) state idx in FSA. there is only one single FSA.
    :return: (fwdbwd, obs_scores), fwdbwd is (time, batch, dim), obs_scores is (time, batch), in -log space
    :rtype: (numpy.ndarray, numpy.ndarray)
    """
    # We get it in -log space, but we calculate in +log space.
    am_scores = -am_scores
    weights = -weights
    n_time, n_batch, dim = am_scores.shape
    assert float_idx.shape == (n_time, n_batch)
    assert edges.ndim == 2 and weights.ndim == 1
    (n_edges,) = weights.shape
    assert edges.shape == (4, n_edges)
    assert start_end_states.shape == (2, n_batch)

    zero_score = float("-inf")
    fwdbwd = numpy.zeros((n_time, n_batch, dim), dtype=am_scores.dtype) + zero_score
    obs_scores = numpy.zeros((n_time, n_batch), dtype=am_scores.dtype) + zero_score

    def collect_scores(forward: bool) -> List[Dict[int, float]]:
        """
        :param forward:
        """
        start_idx, end_idx = start_end_states[:, sequence_idx]
        states: Dict[int, float] = defaultdict(lambda: zero_score)  # state-idx -> score.
        states[start_idx if forward else end_idx] = 0.0
        scores_over_t: List[Optional[Dict[int, float]]] = [None] * (n_time + 1)
        scores_over_t[0 if forward else -1] = dict(states)
        for t in range(n_time) if forward else reversed(range(n_time)):
            if float_idx[t, sequence_idx] == 1:
                scores: Dict[int, List[float]] = defaultdict(list)  # state-idx -> list[score]
                for edge_idx in range(n_edges):
                    from_idx, to_idx, emission_idx, sequence_idx_ = edges[:, edge_idx]
                    if not forward:
                        from_idx, to_idx = to_idx, from_idx
                    if sequence_idx_ != sequence_idx:
                        continue
                    if from_idx not in states or states[from_idx] == zero_score:
                        continue
                    assert 0 <= emission_idx < dim
                    score = states[from_idx] + weights[edge_idx] + am_scores[t, sequence_idx, emission_idx]
                    scores[to_idx].append(score)
                states.clear()
                for state_idx in scores.keys():
                    states[state_idx] = float(logsumexp(scores[state_idx]))
            scores_over_t[(t + 1) if forward else t] = dict(states)
        return scores_over_t

    def gamma():
        """
        :return: nothing, fill fwdbwd and obs_scores
        """
        for t in range(n_time):
            if float_idx[t, sequence_idx] == 1:
                scores: Dict[int, List[float]] = defaultdict(list)  # emission-idx -> list[score]
                all_scores: List[float] = []
                for edge_idx in range(n_edges):
                    from_idx, to_idx, emission_idx, sequence_idx_ = edges[:, edge_idx]
                    if sequence_idx_ != sequence_idx:
                        continue
                    assert 0 <= emission_idx < dim
                    if from_idx not in fwd_scores[t]:
                        continue
                    if to_idx not in bwd_scores[t + 1]:
                        continue
                    score = (
                        fwd_scores[t][from_idx]
                        + weights[edge_idx]
                        + am_scores[t, sequence_idx, emission_idx]
                        + bwd_scores[t + 1][to_idx]
                    )
                    scores[emission_idx].append(score)
                    all_scores.append(score)
                obs_scores[t, sequence_idx] = logsumexp(all_scores) if all_scores else zero_score
                for emission_idx, values in scores.items():
                    if not values:
                        fwdbwd[t, sequence_idx, emission_idx] = zero_score
                    else:
                        fwdbwd[t, sequence_idx, emission_idx] = float(logsumexp(values)) - obs_scores[t, sequence_idx]

    for sequence_idx in range(n_batch):
        fwd_scores = collect_scores(forward=True)
        bwd_scores = collect_scores(forward=False)
        gamma()

    # -log space
    return -fwdbwd, -obs_scores


def py_viterbi(am_scores, am_seq_len, edges, weights, start_end_states):
    """
    Pure Python Viterbi algorithm, to find the best path/alignment.
    The parameters are in the same format as our native fast_baum_welch op.

    :param numpy.ndarray am_scores: (time, batch, dim), in +log space
    :param numpy.ndarray am_seq_len: (batch,) -> int32
    :param numpy.ndarray edges: (4,num_edges), edges of the graph (from,to,emission_idx,sequence_idx)
    :param numpy.ndarray weights: (num_edges,), weights of the edges
    :param numpy.ndarray start_end_states: (2, batch), (start,end) state idx in FSA. there is only one single FSA.
    :return: (alignment, obs_scores), alignment is (time, batch), obs_scores is (batch,), in +log space
    :rtype: (numpy.ndarray, numpy.ndarray)
    """
    n_time, n_batch, dim = am_scores.shape
    assert am_seq_len.shape == (n_batch,)
    assert edges.ndim == 2 and weights.ndim == 1
    (n_edges,) = weights.shape
    assert edges.shape == (4, n_edges)
    assert start_end_states.shape == (2, n_batch)

    zero_score = float("-inf")
    alignment = numpy.zeros((n_time, n_batch), dtype="int32")
    obs_scores = numpy.zeros((n_batch,), dtype=am_scores.dtype) + zero_score

    def search():
        """
        :rtype: list[dict[int,(float,int)]]
        """
        start_idx, _ = start_end_states[:, sequence_idx]
        states: Dict[int, Tuple[float, int]] = defaultdict(lambda: (zero_score, -1))  # state-idx -> score/edge
        states[start_idx] = (0.0, -1)
        res: List[Dict[int, Tuple[float, int]]] = []
        for t in range(n_time):
            if t >= am_seq_len[sequence_idx]:
                break
            scores: Dict[int, List[Tuple[float, int]]] = defaultdict(list)  # state-idx -> list[score/edge]
            for edge_idx in range(n_edges):
                from_idx, to_idx, emission_idx, sequence_idx_ = edges[:, edge_idx]
                if sequence_idx_ != sequence_idx:
                    continue
                if from_idx not in states or states[from_idx] == zero_score:
                    continue
                assert 0 <= emission_idx < dim
                score = states[from_idx][0] + weights[edge_idx] + am_scores[t, sequence_idx, emission_idx]
                scores[to_idx].append((score, edge_idx))
            states.clear()
            for state_idx in scores.keys():
                states[state_idx] = max(scores[state_idx], key=lambda _item: (_item[0], -_item[1]))
            res.append(dict(states))
        assert len(res) == am_seq_len[sequence_idx]
        return res

    def select_best():
        """
        :return: nothing, fill alignment and obs_scores
        """
        _, end_idx = start_end_states[:, sequence_idx]
        state_idx = end_idx
        for t in reversed(range(am_seq_len[sequence_idx])):
            if state_idx not in fwd_search_res[t]:  # no path?
                alignment[t, sequence_idx] = 0
                continue
            score, edge_idx = fwd_search_res[t][state_idx]
            if t == am_seq_len[sequence_idx] - 1:
                obs_scores[sequence_idx] = score
            from_idx, to_idx, emission_idx, sequence_idx_ = edges[:, edge_idx]
            assert sequence_idx_ == sequence_idx
            alignment[t, sequence_idx] = emission_idx
            state_idx = from_idx

    for sequence_idx in range(n_batch):
        fwd_search_res = search()
        select_best()

    return alignment, obs_scores
