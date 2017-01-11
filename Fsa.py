#!/usr/bin/env python2.7

from __future__ import print_function

"""
TODO
- tuple -> sets
- allophone acceptor
- state tying acceptor
- class num_states, edges, first and last char
"""


def convert_label_seq_to_indices(num_labels, label_seq):
  """
  takes label sequence of chars and converts to indices (a->0, b->1, ...)
  :param int num_labels: total number of labels
  :param str label_seq: sequence of labels
  :return list[int] label_indices: labels converted into indices
  """
  label_indices = []

  for label in label_seq:
    label_index = ord(label) - 97
    assert label_index < num_labels, "Index of label exceeds number of labels"
    label_indices.append(label_index)

  return label_indices


def ctc_fsa_for_label_seq(num_labels, label_seq):
  """
  :param int num_labels: number of labels without blank
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
  edges = []
  # calculate number of states
  num_states = 2 * (len(label_seq) + 1) - 1

  # create edges from the label sequence without loops and no empty labels
  num_states, edges = __create_states_from_label_seq_for_ctc(label_seq, num_states, num_labels, edges)

  # adds blank labels to fsa
  num_states, edges = __adds_blank_states_for_ctc(label_seq, num_states, num_labels, edges)

  # adds loops to fsa
  num_states, edges = __adds_loop_edges(num_states, edges)

  # creates end state
  num_states, edges = __adds_last_state_for_ctc(label_seq, num_states, num_labels, edges)

  return num_states, edges


def __create_states_from_label_seq_for_ctc(label_seq, num_states, num_labels, edges):
  """
  :param list[int] label_seq: sequence of labels (normally some kind of word)
  :param int num_states: number of states
  :param int num_labels: number of labels
  :param list edges: list of edges
  :returns (num_states, edges)
  where:
    num_states: int, number of states.
      per convention, state 0 is start state, state (num_states - 1) is single final state
    edges: list[(from,to,label_idx,weight)]
      from and to are state_idx >= 0 and < num_states,
      label_idx >= 0 and label_idx < num_labels  --or-- label_idx == num_labels for blank symbol
      weight is a float, in -log space
  """

  # go through the whole label sequence and create the state for each label
  for label_index in range(0, len(label_seq)):
    # if to remove skips if two equal labels follow each other
    if label_seq[label_index] != label_seq[label_index - 1]:
      n = 2 * label_index
      edges.append((n, n + 2, label_seq[label_index], 1.))

  return num_states, edges


def __adds_blank_states_for_ctc(label_seq, num_states, num_labels, edges):
  """
  :param list[int] label_seq: sequence of labels (normally some kind of word)
  :param int label: label number
  :param int num_states: number of states
  :param int num_labels: number of labels
  :param list edges: list of edges
  :returns (num_states, edges)
  where:
    num_states: int, number of states.
      per convention, state 0 is start state, state (num_states - 1) is single final state
    edges: list[(from,to,label_idx,weight)]
      from and to are state_idx >= 0 and < num_states,
      label_idx >= 0 and label_idx < num_labels  --or-- label_idx == num_labels for blank symbol
      weight is a float, in -log space
  """

  # adds blank labels to fsa
  for label_index in range(0, len(label_seq)):
    label_blank = 2 * label_index + 1
    edges.append((label_blank - 1, label_blank, num_labels + 1, 1.))
    edges.append((label_blank, label_blank + 1, label_seq[label_index], 1.))

  return num_states, edges


def __adds_loop_edges(num_states, edges):
  """
  :param int num_states: number of states
  :param list edges: list of edges
  :returns (num_states, edges)
  where:
    num_states: int, number of states.
      per convention, state 0 is start state, state (num_states - 1) is single final state
    edges: list[(from,to,label_idx,weight)]
      from and to are state_idx >= 0 and < num_states,
      label_idx >= 0 and label_idx < num_labels  --or-- label_idx == num_labels for blank symbol
      weight is a float, in -log space
  """

  # adds loops to fsa
  for state in range(1, num_states):
    edges_included = [edge_index for edge_index, edge in enumerate(edges) if (edge[1] == state)]
    edges.append((state, state, edges[edges_included[0]][2], 1.))

  return num_states, edges


def __adds_last_state_for_ctc(label_seq, num_states, num_labels, edges):
  """
  :param list[int] label_seq: sequence of labels (normally some kind of word)
  :param int label: label number
  :param int num_states: number of states
  :param int num_labels: number of labels
  :param list edges: list of edges
  :returns (num_states, edges)
  where:
    num_states: int, number of states.
      per convention, state 0 is start state, state (num_states - 1) is single final state
    edges: list[(from,to,label_idx,weight)]
      from and to are state_idx >= 0 and < num_states,
      label_idx >= 0 and label_idx < num_labels  --or-- label_idx == num_labels for blank symbol
      weight is a float, in -log space
  """

  i = num_states
  edges.append((i, i, num_labels + 1, 1.))
  edges.append((i - 1, i, num_labels + 1, 1.))
  edges.append((i - 2, i, label_seq[-1], 1.))
  edges.append((i - 3, i, label_seq[-1], 1.))
  num_states += 1

  return num_states, edges


def __make_single_final_state(final_states, num_states, edges):
  """
  takes the graph and merges all final nodes into one single final node
  idea:
  - add new single final node
  - for all edge which ended in a former final node:
    - create new edge from stating node to new single final node with the same label
  :param list[int] final_states: list of index numbers of the final states
  :param int num_states: number of states
  :param list[tuples(start[int], end[int], label, weight)] edges: list of edges
  :return num_states, edges:
  """
  if len(final_states) == 1 and final_states[0] == num_states - 1:  # nothing to change
    return num_states, edges

  num_states += 1
  for fstate in final_states:
    edges_fstate = [edge_index for edge_index, edge in enumerate(edges) if (edge[1] == fstate)]
    for fstate_edge in edges_fstate:
      edges.append((fstate, num_states - 1, fstate_edge[2], 1.))

  return num_states, edges


def __determine_edges(num_states, edges):
  """
  transforms the graph from non-deterministic to deterministic
  specifically removing epsilon edges
  :param int num_states: number of states
  :param list[tuples(start[int], end[int], label, weight)] edges: list of edges
  :return num_states, edges:
  """
  new_states = []  # type: list[set[int]]
  start_states = __discover_eps([0], num_states, edges)
  todo = [start_states]


  return num_states, edges


def __discover_eps(node, num_states, edges):
  """
  starting at a specific node, the nodes connected via epsilon are found
  :param list[int] node:
    list of nodes in the graph which are starting points for epsilon edge search
  :param int num_states: number of states
  :param list[tuples(start[int], end[int], label, weight)] edges: list of edges
  :return list[tuples(start[int], end[int], label, weight)] eps_edges:
    all edges from specific node with label epsilon
  """
  eps_edges = []

  return eps_edges



def asg_fsa_for_label_seq(num_labels, label_seq, repetitions):
  """
  :param int num_labels: number of labels
  :param list[int] label_seq: sequences of label indices, i.e. numbers >= 0 and < num_labels
  :param int repetitions: number of label repetitions
  :returns (num_states, edges)
  where:
    num_states: int, number of states.
      per convention, state 0 is start state, state (num_states - 1) is single final state
    edges: list[(from,to,label_idx,weight)]
      from and to are state_idx >= 0 and < num_states,
      label_idx >= 0 and label_idx < num_labels  --or-- label_idx == num_labels for blank symbol
      weight is a float, in -log space
  """

  edges = []

  rep_seq = __check_for_repetitions(num_labels, label_seq, repetitions)

  num_states, edges = __create_states_from_label_for_asg(rep_seq, edges)
  num_states, edges = __adds_loop_edges(num_states, edges)

  return num_states, edges


def __check_for_repetitions(num_labels, label_indices, repetitions):
  """
  checks the label indices for repetitions, if the n-1 label index is a repetition n in reps gets set to 1 otherwise 0
  :param list[int] label_indices: sequence of label indices
  :return: list[int] reps: list of indices of label repetitions
  """

  reps = []
  rep_count = 0
  index_old = None

  if repetitions == 0:
    reps = label_indices
  else:
    for index in label_indices:
      index_t = index
      if index_t == index_old:
        if rep_count < repetitions:
          rep_count += 1
        elif rep_count != 0:
          reps.append(num_labels + rep_count)
          rep_count = 1
        else:
          print("Something went wrong")
      elif index_t != index_old:
        if rep_count != 0:
          reps.append(num_labels + rep_count)
          rep_count = 0
        reps.append(index)
      else:
        print("Something went wrong")
      index_old = index

  return reps


def __create_states_from_label_for_asg(rep_seq, edges):
  """
  :param int rep_index: label number
  :param int num_labels: number of labels
  :param list edges: list of edges
  :returns (num_states, edges)
  where:
    num_states: int, number of states.
      per convention, state 0 is start state, state (num_states - 1) is single final state
    edges: list[(from,to,label_idx,weight)]
      from and to are state_idx >= 0 and < num_states,
      label_idx >= 0 and label_idx < num_labels  --or-- label_idx == num_labels for blank symbol
      weight is a float, in -log space
  """
  for rep_index, rep_label in enumerate(rep_seq):
    edges.append((rep_index, rep_index+1, rep_label, 1.))

  num_states = len(rep_seq) + 1

  return num_states, edges


def hmm_fsa_for_word_seq(word_seq, lexicon_file, depth=5,
                         allo_num_states=3, allo_context_len=1,
                         state_tying_file=None,
                         tdps=None  # ...
                         ):
  """
  :param list[str] word_seq: sequences of words
  :param str lexicon_file: lexicon XML file
  :param int allo_num_states: hom much HMM states per allophone
  :param int allo_context_len: how much context to store left and tight. 1 -> triphone
  :param str | None state_tying_file: for state-tying, if you want that
  ... (like in LmDataset.PhoneSeqGenerator)
  :returns (num_states, edges) like above
  """
  print("Word sequence:", word_seq)
  sil = 'sil'
  print("Silence: sil")
  print("Place holder: epsilon")
  depth = int(depth)
  if depth == 1:
    print("Lemma acceptor chosen.")
    num_states, edges = __lemma_acceptor_for_hmm_fsa(sil, word_seq)
  elif depth == 2:
    print("Phoneme acceptor chosen.")
    num_states, edges = __lemma_acceptor_for_hmm_fsa(sil, word_seq)
    allo_seq, allo_seq_score, phon = __find_allo_seq_in_lex(word_seq, lexicon_file)
    num_states, edges = __phoneme_acceptor_for_hmm_fsa(sil, word_seq, allo_seq, num_states, edges)
  elif depth == 3:
    print("Triphone acceptor chosen.")
    num_states, edges = __lemma_acceptor_for_hmm_fsa(sil, word_seq)
    allo_seq, allo_seq_score, phon = __find_allo_seq_in_lex(word_seq, lexicon_file)
    num_states, edges = __triphone_acceptor_for_hmm_fsa(sil, word_seq, allo_seq, num_states, edges)
  elif depth == 4:
    print("Allophone state acceptor chosen.")
    allo_seq, allo_seq_score, phon = __find_allo_seq_in_lex(word_seq, lexicon_file)
    num_states, edges = __allophone_state_acceptor_for_hmm_fsa(allo_seq)
  elif depth == 5:
    print("HMM acceptor chosen.")
    num_states, edges = __lemma_acceptor_for_hmm_fsa(sil, word_seq)
    allo_seq, allo_seq_score, phon = __find_allo_seq_in_lex(word_seq, lexicon_file)
    num_states, edges = __triphone_acceptor_for_hmm_fsa(sil, word_seq, allo_seq, num_states, edges)
    num_states, edges = __hmm_acceptor_for_hmm_fsa(num_states, edges)
  elif depth == 6:
    print("State tying chosen.")
    num_states, edges = __state_tying_for_hmm_fsa()
  else:
    print("No acceptor chosen! Try again!")
    num_states = 0
    edges = []

  return num_states, edges


def __lemma_acceptor_for_hmm_fsa(sil, word_seq):
  """
  :param word_seq:
  :return: num_states, edges
  """
  edges = []
  if type(word_seq) is str:
    word_seq_len = 1
    num_states = 4
    num_states_start = 0
    num_states_end = num_states - 1
  else:
    word_seq_len = len(word_seq)
    num_states = 2 + 2 + 4 * (word_seq_len)  # start/end + 2 for sil + number of states for word * 4
    num_states_start = 0
    num_states_end = num_states - 1

  edges.append((num_states_start, num_states_start + 1, sil, 1.))
  edges.append((num_states_end - 1, num_states_end, sil, 1.))
  if type(word_seq) is str:
    for i in range(num_states_start, num_states):
      for j in range(i, num_states):
        edges_included = [m for m, n in enumerate(edges) if
                          (n[0] == i and n[1] == j and n[2] == sil)]
        if len(edges_included) == 0 and not (i == j):
          edges.append((i, j, word_seq, 1.))
  else:
    for i in range(num_states_start, num_states_end):
      for char in word_seq:
        print(char)

  return num_states, edges


def __phoneme_acceptor_for_hmm_fsa(sil, word_seq, allo_seq, num_states, edges):
  allo_len = len(allo_seq)
  num_states_new = num_states + 4 * (allo_len - 1)
  edges_new = []
  state_idx = 2

  for edge in edges:
    if edge[2] == sil and edge[1] == num_states - 1:
      lst = list(edge)
      lst[0] = num_states_new - 2
      lst[1] = num_states_new - 1
      edge = tuple(lst)
      edges_new.append(edge)
    elif edge[2] == word_seq:
      for allo_idx in range(allo_len):
        if allo_idx == 0:
          idx1 = edge[0]
          idx2 = state_idx
        elif allo_idx == allo_len - 1:
          idx1 = state_idx
          if edge[1] == 3:
            edge_idx_t = 1
          elif edge[1] == 2:
            edge_idx_t = 2
          idx2 = num_states_new - edge_idx_t
          state_idx += 1
        else:
          idx1 = state_idx
          state_idx += 1
          idx2 = state_idx
        edge_t = (idx1, idx2, allo_seq[allo_idx], 1.)
        edges_new.append(edge_t)
    else:
      edges_new.append(edge)

  return num_states_new, edges_new


def __triphone_acceptor_for_hmm_fsa(sil, word_seq, allo_seq, num_states, edges):
  allo_len = len(allo_seq)
  num_states_new = num_states + 4 * (allo_len - 1)
  edges_new = []
  state_idx = 2

  tri_seq = __triphone_from_phon(allo_seq)

  for edge in edges:
    if edge[2] == sil and edge[1] == num_states - 1:
      lst = list(edge)
      lst[0] = num_states_new - 2
      lst[1] = num_states_new - 1
      edge = tuple(lst)
      edges_new.append(edge)
    elif edge[2] == word_seq:
      for allo_idx in range(allo_len):
        if allo_idx == 0:
          idx1 = edge[0]
          idx2 = state_idx
        elif allo_idx == allo_len - 1:
          idx1 = state_idx
          if edge[1] == 3:
            edge_idx_t = 1
          elif edge[1] == 2:
            edge_idx_t = 2
          idx2 = num_states_new - edge_idx_t
          state_idx += 1
        else:
          idx1 = state_idx
          state_idx += 1
          idx2 = state_idx
        edge_t = (idx1, idx2, tri_seq[allo_idx], 1.)
        edges_new.append(edge_t)
    else:
      edges_new.append(edge)

  return num_states_new, edges_new


def __allophone_state_acceptor_for_hmm_fsa(allo_seq):
  num_states = 0
  edges = []
  return num_states, edges


def __hmm_acceptor_for_hmm_fsa(num_states, edges):
  for state in range(1, num_states):
    edges_included = [edge_index for edge_index, edge in enumerate(edges) if (edge[1] == state)]
    edges.append((state, state, edges[edges_included[0]][2], 1.
                  ))
  return num_states, edges


def __state_tying_for_hmm_fsa():
  num_states = 0
  edges = []
  return num_states, edges


def __load_lexicon(lexFile):
  '''
  loads a lexicon from a file, loads the xml and returns its conent
  :param lexFile: lexicon file with xml structure
  :return lex: variable with xml structure
  where:
    lex.lemmas and lex.phonemes important
  '''
  from os.path import isfile
  from Log import log
  from LmDataset import Lexicon

  assert isfile(lexFile), "Lexicon does not exists"

  log.initialize(verbosity=[5])
  lex = Lexicon(lexFile)

  return lex


def __find_allo_seq_in_lex(lemma, lexi):
  '''
  searches a lexicon xml structure for a watching word and
  returns the matching allophone sequence as a list
  :param lemma: the word to search for in the lexicon
  :param lexi: the lexicon
  :return allo_seq: allophone sequence with the highest score as a list
  :return phons: phonemes matching the lemma as a list of dictionaries with score and phon
  '''
  lex = __load_lexicon(lexi)

  assert lex.lemmas[lemma], "Lemma not in lexicon"

  phons = lex.lemmas[lemma]['phons']

  phons_sorted = sorted(phons, key=lambda phon: phon['score'], reverse=True)

  allo_seq = phons_sorted[0]['phon'].split(' ')
  allo_seq_score = phons_sorted[0]['score']

  return allo_seq, allo_seq_score, phons


def __triphone_from_phon(word_seq):
  '''
  :param word_seq: sequence of allophones
  :return tri_seq: list of three phonemes
  uses the sequence of allophones and splits into a list of triphones.
  triphones are composed of the current phon and the left and right phons
  '''
  tri_seq = []

  for allo_index in range(0, len(word_seq)):
    if allo_index <= 0:
      tri_l = ''
    else:
      tri_l = word_seq[allo_index - 1]
    if allo_index >= len(word_seq) - 1:
      tri_r = ''
    else:
      tri_r = word_seq[allo_index + 1]
    tri_c = word_seq[allo_index]
    tri = (tri_l, tri_c, tri_r)
    tri_seq.append(tri)

  return tri_seq


def fsa_to_dot_format(file, num_states, edges):
  '''
  :param num_states:
  :param edges:
  :return:

  converts num_states and edges to dot file to svg file via graphviz
  '''
  import graphviz
  G = graphviz.Digraph(format='svg')

  nodes = []
  for i in range(0, num_states):
    nodes.append(str(i))

  __add_nodes(G, nodes)
  __add_edges(G, edges)

  # print(G.source)
  filepath = "./tmp/" + file
  filename = G.render(filename=filepath)
  print("File saved in:", filename)


def __add_nodes(graph, nodes):
  for n in nodes:
    if isinstance(n, tuple):
      graph.node(n[0], **n[1])
    else:
      graph.node(n)
  return graph


def __add_edges(graph, edges):
  for e in edges:
    e = ((str(e[0]), str(e[1])), {'label': str(e[2])})
    if isinstance(e[0], tuple):
      graph.edge(*e[0], **e[1])
    else:
      graph.edge(*e)
  return graph


def main():
  from argparse import ArgumentParser
  arg_parser = ArgumentParser()
  arg_parser.add_argument("--file", required=True)
  arg_parser.add_argument("--num_labels", type=int, required=True)
  arg_parser.add_argument("--label_seq", required=True)
  arg_parser.add_argument("--fsa", required=True)
  arg_parser.add_argument("--lexicon")
  arg_parser.add_argument("--depth")
  arg_parser.add_argument("--asg_repetition")
  args = arg_parser.parse_args()

  label_indices = convert_label_seq_to_indices(num_labels=int(args.num_labels),
                                               label_seq=args.label_seq)

  if (args.fsa.lower() == 'ctc'):
    num_states, edges = ctc_fsa_for_label_seq(num_labels=int(args.num_labels),
                                              label_seq=label_indices)
  elif (args.fsa.lower() == 'asg'):
    assert args.asg_repetition, "Specify number of asg repetition labels in argument options: --asg_repetition [int]"
    num_states, edges = asg_fsa_for_label_seq(num_labels=int(args.num_labels),
                                              label_seq=label_indices,
                                              repetitions=int(args.asg_repetition))
    print("Number of labels (a-z == 27 labels):", args.num_labels)
    print("Number of repetition symbols:", args.asg_repetition)
    for rep in range(1, int(args.asg_repetition) + 1):
      print("Repetition label:", int(args.num_labels) + rep, "meaning", rep, "repetitions")
  elif (args.fsa.lower() == 'hmm'):
    assert args.lexicon, "Specify lexicon in argument options: --lexicon [path]"
    assert args.depth, "Specify the depth in argument options: --depth [int]"
    num_states, edges = hmm_fsa_for_word_seq(word_seq=args.label_seq,
                                             lexicon_file=args.lexicon,
                                             depth=int(args.depth))

  fsa_to_dot_format(file=args.file, num_states=num_states, edges=edges)


if __name__ == "__main__":
  main()
