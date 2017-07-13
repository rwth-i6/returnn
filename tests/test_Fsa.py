#!/usr/bin/env python2.7

from __future__ import print_function
from __future__ import division


import os
import sys
sys.path.append(os.path.relpath("./../"))
from Fsa import Fsa
from Fsa import fsa_to_dot_format


def main():
  import time

  start_time = time.time()

  lex = './../recog.150k.final.lex.gz'
  st_ty = './../state-tying.txt'
  lem = 'test_Fsa_lemma.txt'
  fsatype = ['asg', 'ctc', 'hmm']

  # load file
  lemmas = open(lem, "r")

  # init FSA class
  automaton = Fsa()

  # load lexicon (longest single op) and state tying file
  lexicon_start_time = time.time()

  automaton.set_lexicon(lexicon_name=lex)

  automaton.set_state_tying(st_ty)

  automaton.set_hmm_depth(6)

  lexicon_end_time = time.time()

  run_start_time = time.time()

  for lemma in lemmas:
    automaton.set_lemma(lemma=lemma)

    for ft in fsatype:
      automaton.set_fsa_type(fsa_type=ft)

      automaton.run()

      filename = lemma.lower().strip().replace(" ", "-") + '-' + automaton.fsa_type
      fsa_to_dot_format(file=filename, num_states=automaton.num_states, edges=automaton.edges)

  run_end_time = time.time()

  lemmas.close()

  print("Run time:", run_end_time - run_start_time, "seconds")

  print("Lexicon load time:", lexicon_end_time - lexicon_start_time, "seconds")

  print("Total time:", time.time() - start_time, "seconds")


if __name__ == "__main__":
  main()
