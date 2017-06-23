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

  lex = './../recog.150k.final.lex.gz'

  automaton = Fsa(lemma='Halloween is a fantastic event', fsa_type='ctc')

  lexicon_time = time.time()

  automaton.set_lexicon(lexicon_name=lex)

  print(time.time() - lexicon_time, "seconds to load the lexicon")


if __name__ == "__main__":
  main()
