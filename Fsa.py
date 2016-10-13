
def ctc_fsa_for_label_seq(num_labels, label_seq):
  """
  :param int num_labels: number of labels
  :param list[int] label_seq: sequences of label indices, i.e. numbers >= 0 and < num_labels
  :returns (num_states, edges)
  where:
    num_states: int, number of states.
      per convention, state 0 is start state, state (num_states - 1) is single final state
    edges: list[(from,to,label_idx,weight)]
      from and to are state_idx >= 0 and < num_states,
      label_idx >= 0 and label_idx < num_labels  --or-- label_idx == num_labels for blank symbol
      weight is a float, in -log space
  """
  # TODO @Chris ...


def hmm_fsa_for_word_seq(word_seq, lexicon_file,
                         allo_num_states=3, allo_context_len=1,
                         state_tying_file=None,
                         tdps=None  # ...
                         ):
  """
  :param list[str] word_seq: sequences of words
  ... (like in LmDataset.PhoneSeqGenerator)
  :returns (num_states, edges) like above
  """
  # TODO @Chris ...


def fsa_to_dot_format(num_states, edges):
  f = open("/tmp/dummy-fsa.dot", "w")
  # evtl mit graphviz
  # dot -T svg /tmp/dummy-fsa.dot > dummy-fsa.svg
