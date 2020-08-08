#!/usr/bin/env python2.7

from __future__ import print_function
from __future__ import division

import logging
logging.getLogger('tensorflow').disabled = True

import os
import sys
import datetime
import unittest
import time
import numpy
import tensorflow as tf

print("TF version:", tf.__version__)

print("__file__:", __file__)
base_path = os.path.realpath(os.path.dirname(os.path.abspath(__file__)) + "/..")
print("base path:", base_path)
import _setup_test_env  # noqa
# Returnn imports
import returnn.util.fsa as fsa_util
import returnn.tf.compat as tf_compat
from returnn.tf.util.basic import is_gpu_available


class Lexicon:

  def __init__(self):
    self.lemmas = {}


class StateTying:

  def __init__(self):
    # TODO load state tying file as py dic
    self.allo_map = {}


def main_custom():
  start_time = time.time()
  date_str = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")

  lemmas = [
    'Halloween is a fantastic event',
    'Halloween is a fantastic event Halloween is a fantastic event Halloween is a fantastic event '
    'Halloween is a fantastic event Halloween is a fantastic event',
    'This is a great day',
    'hallucinations',
    'not',
    'great',
    'for',
    'driving',
    'To be or not to be That is the question'
  ]

  lexicon_start_time = time.time()

  lexicon = Lexicon()

  lexicon.lemmas["halloween"] = {"orth": "halloween", "phons": [
    {"phon": "hh ae l ow w iy n", "score": 0.530628251062},
    {"phon": "hh ax l ow w iy n", "score": 0.887303195001}
  ]}

  lexicon.lemmas["is"] = {"orth": "is", "phons": [
    {"phon": "ih z", "score": 0.0}
  ]}

  lexicon.lemmas["a"] = {"orth": "a", "phons": [
    {"phon": "ah", "score": 2.30421991506},
    {"phon": "ax", "score": 0.1584136243},
    {"phon": "ey", "score": 3.06472514504}
  ]}

  lexicon.lemmas["fantastic"] = {"orth": "fantastic", "phons": [
    {"phon": "f ae n t ae s t ih k", "score": 0.0}
  ]}

  lexicon.lemmas["event"] = {"orth": "event", "phons": [
    {"phon": "ih v eh n t", "score": 0.124454174473},
    {"phon": "iy v eh n t", "score": 2.14539950947}
  ]}

  lexicon.lemmas["this"] = {"orth": "this", "phons": [
    {"phon": "dh ih s", "score": 0.0}
  ]}

  lexicon.lemmas["great"] = {"orth": "great", "phons": [
    {"phon": "g r ey t", "score": 0.0}
  ]}

  lexicon.lemmas["day"] = {"orth": "day", "phons": [
    {"phon": "d ey", "score": 0.0}
  ]}

  lexicon.lemmas["hallucinations"] = {"orth": "hallucinations", "phons": [
    {"phon": "hh ax l uw s ih n ey sh n z", "score": 0.0}
  ]}

  lexicon.lemmas["aren't"] = {"orth": "aren't", "phons": [
    {"phon": "aa n t", "score": 0.0}
  ]}

  lexicon.lemmas["driving"] = {"orth": "driving", "phons": [
    {"phon": "d r ay v ih ng", "score": 0.0}
  ]}

  lexicon.lemmas["to"] = {"orth": "to", "phons": [
    {"phon": "t ax", "score": 0.0866560671538},
    {"phon": "t uw", "score": 2.48882341292}
  ]}

  lexicon.lemmas["or"] = {"orth": "or", "phons": [
    {"phon": "ao", "score": 2.17712006721},
    {"phon": "ao r", "score": 0.120324758972}
  ]}

  lexicon.lemmas["for"] = {"orth": "for", "phons": [
    {"phon": "f ao", "score": 3.90224166756},
    {"phon": "f ao r", "score": 2.28076991556},
    {"phon": "f ax", "score": 1.76517101304},
    {"phon": "f ax r", "score": 3.75190315662},
    {"phon": "f er", "score": 0.381308177114}
  ]}

  lexicon.lemmas["be"] = {"orth": "be", "phons": [
    {"phon": "b iy", "score": 0.0}
  ]}

  lexicon.lemmas["not"] = {"orth": "not", "phons": [
    {"phon": "n oh t", "score": 0.0}
  ]}

  lexicon.lemmas["that"] = {"orth": "that", "phons": [
    {"phon": "dh ae t", "score": 0.0838468092005},
    {"phon": "dh ax t", "score": 2.52039433822}
  ]}

  lexicon.lemmas["the"] = {"orth": "the", "phons": [
    {"phon": "dh ax", "score": 0.0645240603743},
    {"phon": "dh iy", "score": 2.77280565879}
  ]}

  lexicon.lemmas["question"] = {"orth": "question", "phons": [
    {"phon": "k w eh s ch ax n", "score": 0.0}
  ]}

  lexicon_end_time = time.time()

  with open("./tmp/timings_{}.txt".format(date_str), 'wb') as timings:
    timings.write("Date: {}\n\n".format(date_str))
    timings.write("Lexicon load time: {}\n\n".format(lexicon_end_time - lexicon_start_time))

    for lemma in lemmas:
      timings.write("Lemma: {}\n\n".format(lemma))

      fsa = fsa_util.Graph(lemma)
      fsa.filename = lemma.lower().replace(' ', '-')

      word_start_time = time.time()
      word = fsa_util.AllPossibleWordsFsa(fsa)
      word.lexicon = lexicon
      word_run_start_time = time.time()
      word.run()
      word_run_end_time = time.time()
      sav_word = fsa_util.Store(fsa.num_states_word, fsa.edges_word)
      sav_word.filename = "{}_word_{}".format(fsa.filename, date_str)
      sav_word.fsa_to_dot_format()
      sav_word.save_to_file()
      word_end_time = time.time()

      timings.write("FSA over all possible words\n")
      timings.write("Total time: {}\n".format(word_end_time - word_start_time))
      timings.write("Init time:  {}\n".format(word_run_start_time - word_start_time))
      timings.write("Run time:   {}\n".format(word_run_end_time - word_run_start_time))
      timings.write("Save time:  {}\n".format(word_end_time - word_run_end_time))

      asg_start_time = time.time()
      asg = fsa_util.Asg(fsa)
      asg.label_conversion = False
      asg.asg_repetition = 2
      asg_run_start_time = time.time()
      asg.run()
      asg_run_end_time = time.time()
      sav_asg = fsa_util.Store(fsa.num_states_asg, fsa.edges_asg)
      sav_asg.filename = "{}_asg_{}".format(fsa.filename, date_str)
      sav_asg.fsa_to_dot_format()
      sav_asg.save_to_file()
      asg_end_time = time.time()

      timings.write("ASG FSA\n")
      timings.write("Total time: {}\n".format(asg_end_time - asg_start_time))
      timings.write("Init time:  {}\n".format(asg_run_start_time - asg_start_time))
      timings.write("Run time:   {}\n".format(asg_run_end_time - asg_run_start_time))
      timings.write("Save time:  {}\n".format(asg_end_time - asg_run_end_time))

      ctc_start_time = time.time()
      ctc = fsa_util.Ctc(fsa)
      ctc.label_conversion = False
      ctc_run_start_time = time.time()
      ctc.run()
      ctc_run_end_time = time.time()
      sav_ctc = fsa_util.Store(fsa.num_states_ctc, fsa.edges_ctc)
      sav_ctc.filename = "_+_ctc_{}".format(fsa.filename, date_str)
      sav_ctc.fsa_to_dot_format()
      sav_ctc.save_to_file()
      ctc_end_time = time.time()

      timings.write("CTC FSA\n")
      timings.write("Total time: {}\n".format(ctc_end_time - ctc_start_time))
      timings.write("Init time:  {}\n".format(ctc_run_start_time - ctc_start_time))
      timings.write("Run time:   {}\n".format(ctc_run_end_time - ctc_run_start_time))
      timings.write("Save time:  {}\n".format(ctc_end_time - ctc_run_end_time))

      hmm_start_time = time.time()
      hmm = fsa_util.Hmm(fsa)
      hmm.lexicon = lexicon
      hmm.allo_num_states = 3
      hmm.state_tying_conversion = False
      hmm_run_start_time = time.time()
      hmm.run()
      hmm_run_end_time = time.time()
      sav_hmm = fsa_util.Store(fsa.num_states_hmm, fsa.edges_hmm)
      sav_hmm.filename = "{}_hmm_{}".format(fsa.filename, date_str)
      sav_hmm.fsa_to_dot_format()
      sav_hmm.save_to_file()
      hmm_end_time = time.time()

      timings.write("HMM FSA\n")
      timings.write("Total time: {}\n".format(hmm_end_time - hmm_start_time))
      timings.write("Init time:  {}\n".format(hmm_run_start_time - hmm_start_time))
      timings.write("Run time:   {}\n".format(hmm_run_end_time - hmm_run_start_time))
      timings.write("Save time:  {}\n\n\n".format(hmm_end_time - hmm_run_end_time))

    end_time = time.time()

    timings.write("Total program time: {}\n\n".format(end_time - start_time))


def slow_full_sum_staircase_uniform(num_classes, out_seq_len, with_loop=False):
  """
  :param int num_classes:
  :param int out_seq_len:
  :param bool with_loop:
  :return: array shape (out_seq_len, num_classes)
  :rtype: numpy.ndarray
  """
  def iter_seqs(t, c):
    if c >= num_classes:
      return
    for c_ in range(c, num_classes):
      if t == out_seq_len - 1:
        yield [c_]
      else:
        for seq in iter_seqs(t + 1, c_ if with_loop else (c_ + 1)):
          yield [c_] + seq
  seqs = list(iter_seqs(0, 0))
  print("slow_full_sum_staircase_uniform(%i, %i, with_loop=%r)" % (num_classes, out_seq_len, with_loop))
  print("  num seqs:", len(seqs))
  assert seqs, "did not found any"
  seqs = numpy.array(seqs)
  assert seqs.shape[1] == out_seq_len
  print("  seqs:")
  print(seqs)
  res = numpy.zeros((out_seq_len, num_classes))
  for t in range(out_seq_len):
    for c in range(num_classes):
      count = numpy.count_nonzero(seqs[:, t] == c)
      res[t, c] = float(count) / seqs.shape[0]
  print("  res sums (should be close to 1):", res.sum(axis=1))
  return res


def tf_baum_welch(fsa, am_scores=None, num_classes=None, out_seq_len=None):
  """
  :param Fsa.FastBaumWelchBatchFsa fsa:
  :param numpy.ndarray am_scores:
  :param int num_classes:
  :param int out_seq_len:
  :return: shape (out_seq_len, num_classes)
  :rtype: numpy.ndarray
  """
  print("tf_baum_welch...")
  n_batch = fsa.num_batch
  assert n_batch == 1
  edges = tf.constant(fsa.edges, dtype=tf.int32)
  weights = tf.constant(fsa.weights, dtype=tf.float32)
  start_end_states = tf.constant(fsa.start_end_states, dtype=tf.int32)
  if am_scores is None:
    am_scores = numpy.ones((out_seq_len, n_batch, num_classes), dtype="float32") * numpy.float32(1.0 / num_classes)
  else:
    if am_scores.shape == (out_seq_len, num_classes):
        am_scores = am_scores[:, None, :]
    assert am_scores.shape == (out_seq_len, n_batch, num_classes)
  am_scores = -numpy.log(am_scores)  # in -log space
  am_scores = tf.constant(am_scores, dtype=tf.float32)
  float_idx = tf.ones((out_seq_len, n_batch), dtype=tf.float32)
  # from returnn.tf.util.basic import sequence_mask_time_major
  # float_idx = tf.cast(sequence_mask_time_major(tf.convert_to_tensor(list(range(seq_len - n_batch + 1, seq_len + 1)))), dtype=tf.float32)
  print("Construct call...")
  from returnn.tf.native_op import fast_baum_welch
  fwdbwd, obs_scores = fast_baum_welch(
    am_scores=am_scores, float_idx=float_idx,
    edges=edges, weights=weights, start_end_states=start_end_states)
  print("Done.")
  print("Eval:")
  session = tf_compat.get_default_session()
  fwdbwd, score = session.run([fwdbwd, obs_scores])
  print("BW score:")
  print(repr(score))
  assert score.shape == (out_seq_len, n_batch)
  #bw = numpy.maximum(-fwdbwd, -100.)
  bw = numpy.exp(-fwdbwd)
  #print("Baum-Welch soft alignment:")
  #print(repr(bw))
  assert bw.shape == (out_seq_len, n_batch, num_classes)
  return bw[:, 0]  # expect single batch...


def check_fast_bw_fsa_staircase(num_classes, out_seq_len, with_loop):
  print("check_fast_bw_fsa_staircase(%i, %i, with_loop=%r)" % (num_classes, out_seq_len, with_loop))
  expected = slow_full_sum_staircase_uniform(num_classes=num_classes, out_seq_len=out_seq_len, with_loop=with_loop)
  print("expected full sum:")
  print(expected)
  fsa = fsa_util.fast_bw_fsa_staircase(seq_lens=[num_classes], with_loop=with_loop)
  with tf_compat.v1.Session().as_default():
    res = tf_baum_welch(fsa, num_classes=num_classes, out_seq_len=out_seq_len)
  print("baum-welch:")
  print(res)
  is_close = numpy.isclose(expected, res).all()
  print("close:", is_close)
  assert is_close


# Note: we could replace tf_baum_welch by some CPU/Python code...
@unittest.skipIf(not is_gpu_available(), "no gpu on this system; needed for tf_baum_welch")
def test_fast_bw_fsa_staircase():
  check_fast_bw_fsa_staircase(2, 2, with_loop=False)
  check_fast_bw_fsa_staircase(2, 2, with_loop=True)
  check_fast_bw_fsa_staircase(3, 2, with_loop=False)
  check_fast_bw_fsa_staircase(3, 2, with_loop=True)
  check_fast_bw_fsa_staircase(3, 3, with_loop=False)
  check_fast_bw_fsa_staircase(3, 3, with_loop=True)


if __name__ == "__main__":
  from returnn.util import better_exchook
  better_exchook.install()
  if len(sys.argv) <= 1:
    for k, v in sorted(globals().items()):
      if k.startswith("test_"):
        print("-" * 40)
        print("Executing: %s" % k)
        try:
          v()
        except unittest.SkipTest as exc:
          print("SkipTest:", exc)
        print("-" * 40)
    print("Finished all tests.")
  else:
    assert len(sys.argv) >= 2
    for arg in sys.argv[1:]:
      print("Executing: %s" % arg)
      if arg in globals():
        globals()[arg]()  # assume function and execute
      else:
        eval(arg)  # assume Python code and execute
