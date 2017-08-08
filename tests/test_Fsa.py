#!/usr/bin/env python2.7

from __future__ import print_function
from __future__ import division


import sys
sys.path += ["./.."]
import Fsa


class Lexicon:

  def __init__(self):
    self.lemmas = {}


def main():
  import time

  start_time = time.time()

  # load lexicon (longest single op) and state tying file
  lexicon_start_time = time.time()

  lexicon = Lexicon()

  lexicon.lemmas = {}

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

  lemmas = [
    'Halloween is a fantastic event',
    'This is a great day',
    'hallucinations',
    'not',
    'great',
    'for',
    'driving',
    'To be or not to be That is the question'
  ]

  for lemma in lemmas:
    fsa = Fsa.Graph(lemma)
    fsa.filename = lemma.lower().replace(' ', '-')

    asg = Fsa.Asg(fsa)
    asg.run()
    sav_asg = Fsa.Store(fsa.num_states_asg, fsa.edges_asg)
    sav_asg.filename = fsa.filename + '_asg'
    sav_asg.fsa_to_dot_format()
    sav_asg.save_to_file()

    ctc = Fsa.Ctc(fsa)
    ctc.run()
    sav_ctc = Fsa.Store(fsa.num_states_ctc, fsa.edges_ctc)
    sav_ctc.filename = fsa.filename + '_ctc'
    sav_ctc.fsa_to_dot_format()
    sav_ctc.save_to_file()

    hmm = Fsa.Hmm(fsa)
    hmm.lexicon = lexicon
    hmm.run()
    sav_hmm = Fsa.Store(fsa.num_states_hmm, fsa.edges_hmm)
    sav_hmm.filename = fsa.filename + '_hmm'
    sav_hmm.fsa_to_dot_format()
    sav_hmm.save_to_file()

  lexicon_end_time = time.time()

  run_start_time = time.time()

  run_end_time = time.time()

  print("Run time:", run_end_time - run_start_time, "seconds")

  print("Lexicon load time:", lexicon_end_time - lexicon_start_time, "seconds")

  print("Total time:", time.time() - start_time, "seconds")


if __name__ == "__main__":
  main()
