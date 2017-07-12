#!/usr/bin/env python2.7

from __future__ import print_function
from __future__ import division

try:
  from Fsa import Fsa
except ImportError:
  import os
  import sys
  sys.path.append(os.path.relpath("./../"))
  from Fsa import Fsa


def test_01():
  pass

def main():
  import time

  start_time = time.time()

  lex = './../recog.150k.final.lex.gz'
  st_ty = './../state-tying.txt'

  automaton = Fsa()

  automaton.set_lemma(lemma='Halloween is a fantastic event')

  automaton.set_fsa_type(fsa_type='hmm')

  lexicon_start_time = time.time()

  automaton.set_lexicon(lexicon_name=lex)

  automaton.set_state_tying(st_ty)

  lexicon_end_time = time.time()

  run_start_time = time.time()

  automaton.run()

  run_end_time = time.time()

  print("Run time:", run_end_time - run_start_time, "seconds")

  print("Lexicon load time:", lexicon_end_time - lexicon_start_time, "seconds")

  print("Total time:", time.time() - start_time, "seconds")


if __name__ == "__main__":
  main()
