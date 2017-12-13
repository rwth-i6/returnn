#!/usr/bin/env python2.7

from __future__ import print_function
from __future__ import division


import sys
import datetime
import time
sys.path += ["./.."]
import Fsa


class Lexicon:

  def __init__(self):
    self.lemmas = {}


class StateTying:

  def __init__(self):
    # TODO load state tying file as py dic
    self.allo_map = {}


def main():
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

      fsa = Fsa.Graph(lemma)
      fsa.filename = lemma.lower().replace(' ', '-')

      word_start_time = time.time()
      word = Fsa.AllPossibleWordsFsa(fsa)
      word.lexicon = lexicon
      word_run_start_time = time.time()
      word.run()
      word_run_end_time = time.time()
      sav_word = Fsa.Store(fsa.num_states_word, fsa.edges_word)
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
      asg = Fsa.Asg(fsa)
      asg.label_conversion = False
      asg.asg_repetition = 2
      asg_run_start_time = time.time()
      asg.run()
      asg_run_end_time = time.time()
      sav_asg = Fsa.Store(fsa.num_states_asg, fsa.edges_asg)
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
      ctc = Fsa.Ctc(fsa)
      ctc.label_conversion = False
      ctc_run_start_time = time.time()
      ctc.run()
      ctc_run_end_time = time.time()
      sav_ctc = Fsa.Store(fsa.num_states_ctc, fsa.edges_ctc)
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
      hmm = Fsa.Hmm(fsa)
      hmm.lexicon = lexicon
      hmm.allo_num_states = 3
      hmm.state_tying_conversion = False
      hmm_run_start_time = time.time()
      hmm.run()
      hmm_run_end_time = time.time()
      sav_hmm = Fsa.Store(fsa.num_states_hmm, fsa.edges_hmm)
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


if __name__ == "__main__":
  main()
