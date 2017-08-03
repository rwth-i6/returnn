#!/usr/bin/env python2.7

from __future__ import print_function
from __future__ import division


import sys
sys.path += ["./.."]
from Fsa import Fsa
from Fsa import fsa_to_dot_format


def main():
  import time

  start_time = time.time()

  lex_name = './tmp/recog.150k.final.lex.gz'

  lex = []

  lex_w = file(lex_name, "w")
  lex_w.writelines(lex)

  st_ty_name = './tmp/state-tying.txt'

  st_ty = []

  st_ty_w = file(st_ty, "w")
  st_ty_w.writelines(st_ty)

  fsatype = ['asg', 'ctc', 'hmm']

  lemmas = [
    'Halloween is a fantastic event',
    'This is a great day',
    "hallucinations aren't great for driving",
    'To be or not to be That is the question'
  ]

  # init FSA class
  automaton = Fsa()

  # load lexicon (longest single op) and state tying file
  lexicon_start_time = time.time()

  automaton.set_lexicon(lexicon_name=lex_name)

  automaton.set_state_tying(st_ty_name)

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

  print("Run time:", run_end_time - run_start_time, "seconds")

  print("Lexicon load time:", lexicon_end_time - lexicon_start_time, "seconds")

  print("Total time:", time.time() - start_time, "seconds")


if __name__ == "__main__":
  main()
