#!/usr/bin/env python2.7

from __future__ import print_function


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
  edges = []
  # calculate number of states
  num_states = 2 * (len(label_seq) + 1) - 1

  # create edges from the label sequence without loops and no empty labels
  num_states, edges = __create_states_from_label_seq_for_ctc(label_seq, num_states, num_labels, edges)

  # adds blank labels to fsa
  num_states, edges = __adds_empty_states_for_ctc(label_seq, num_states, num_labels, edges)

  # adds loops to fsa
  num_states, edges = __adds_loop_edges_for_ctc(label_seq, num_states, num_labels, edges)

  # creates end state
  num_states, edges = __adds_last_state_for_ctc(label_seq, num_states, num_labels, edges)

  # removes states to create skips
  num_states, edges = __removes_edges_nodes_for_skip_for_ctc(label_seq, num_states, num_labels, edges)

  return num_states, edges


def __create_states_from_label_seq_for_ctc(label_seq, num_states, num_labels, edges):
  """
  :param str label_seq: sequence of labels (normally some kind of word)
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
  for m in range(0, len(label_seq)):
    n = 2 * m
    edges.append((str(n), str(n+2), label_seq[m], 1.))

  return num_states, edges


def __adds_empty_states_for_ctc(label_seq, num_states, num_labels, edges):
  """
  :param str label_seq: sequence of labels (normally some kind of word)
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
  for m in range(0, len(label_seq)):
    n = 2 * m + 1
    edges.append((str(n-1), str(n), 'blank', 1.))
    edges.append((str(n), str(n+1), label_seq[m], 1.))

  return num_states, edges


def __adds_loop_edges_for_ctc(label_seq, num_states, num_labels, edges):
  """
  :param str label_seq: sequence of labels (normally some kind of word)
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

  # adds loops to fsa
  for state in range(1, num_states):
    edges_included = [edge_index for edge_index, edge in enumerate(edges) if (edge[1] == str(state))]
    edges.append((str(state), str(state), edges[edges_included[0]][2], 1.))

  return num_states, edges


def __adds_last_state_for_ctc(label_seq, num_states, num_labels, edges):
  """
  :param str label_seq: sequence of labels (normally some kind of word)
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
  edges.append((str(i - 1), str(i), 'blank', 1.))
  edges.append((str(i), str(i), 'blank', 1.))
  edges.append((str(i - 2), str(i), label_seq[-1], 1.))
  edges.append((str(i - 3), str(i), label_seq[-1], 1.))
  num_states += 1

  return num_states, edges


def __removes_edges_nodes_for_skip_for_ctc(label_seq, num_states, num_labels, edges):
  """
  :param str label_seq: sequence of labels (normally some kind of word)
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
  # compare label_seq positions to each other
  # calculate position of state to delete
  # remove from graph

  return num_states, edges


def asg_fsa_for_label_seq(num_labels, label_seq):
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
  num_states = 0
  edges = []

  for m in range(0, len(label_seq)):
    num_states, edges = __create_states_from_label_for_asg(m, num_labels, edges)
    print("label:", label_seq[m], "=", m)

  return num_states, edges


def __create_states_from_label_for_asg(label, num_labels, edges):
  """
  :param int label: label number
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
  i = label
  edges.append((str(i), str(i + 1), label, 1.))
  edges.append((str(i + 1), str(i + 1), label, 1.))
  num_states = num_labels

  return num_states, edges


def hmm_fsa_for_word_seq(word_seq, lexicon_file,
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
  print("Silence: sil")
  print("Place holder: epsilon")
  num_states, edges = __lemma_acceptor_for_hmm_fsa(word_seq)
  allo_seq, allo_seq_score, phon = __find_allo_seq_in_lex(word_seq, lexicon_file)

  return num_states, edges


def __lemma_acceptor_for_hmm_fsa(word_seq):
  """
  :param word_seq:
  :return: num_states, edges
  """
  sil = 'sil'
  edges = []
  if type(word_seq) is str:
    word_seq_len = 1
    num_states = 4
    num_states_start = 0
    num_states_end = num_states - 1
  else:
    word_seq_len = len(word_seq)
    num_states = 2 + 2 + 4 * (word_seq_len) # start/end + 2 for sil + number of states for word * 4
    num_states_start = 0
    num_states_end = num_states - 1

  edges.append((str(num_states_start), str(num_states_start+1), sil, 1.))
  edges.append((str(num_states_end-1), str(num_states_end), sil, 1.))
  if type(word_seq) is str:
    for i in range(num_states_start, num_states):
      for j in range(i, num_states):
        edges_included = [m for m, n in enumerate(edges) if (n[0] == str(i) and n[1] == str(j) and n[2] == sil)]
        if len(edges_included) == 0 and not (i == j):
          edges.append((str(i), str(j), word_seq, 1.))
  else:
    for i in range(num_states_start, num_states_end):
      for char in word_seq:
        print(char)

  return num_states, edges

def __phoneme_acceptor_for_hmm_fsa(allo_seq):
  pass


def __triphone_acceptor_for_hmm_fsa():
  pass


def __allophone_state_acceptor_for_hmm_fsa():
  pass


def __hmm_loops_for_hmm_fsa():
  pass


def __state_tying_for_hmm_fsa():
  pass


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

  #print(G.source)
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
    e = ((e[0], e[1]), {'label': str(e[2])})
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
  arg_parser.add_argument("--lexicon", required=True)
  args = arg_parser.parse_args()

  if (args.fsa.lower() == 'ctc'):
    num_states, edges = ctc_fsa_for_label_seq(num_labels=args.num_labels, label_seq=args.label_seq)
  elif (args.fsa.lower() == 'asg'):
    num_states, edges = asg_fsa_for_label_seq(num_labels=args.num_labels, label_seq=args.label_seq)
  elif (args.fsa.lower() == 'hmm'):
    num_states, edges = hmm_fsa_for_word_seq(word_seq=args.label_seq, lexicon_file=args.lexicon)

  fsa_to_dot_format(file=args.file, num_states=num_states, edges=edges)


if __name__ == "__main__":
  main()
