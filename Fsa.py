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

  num_states = 0
  edges = []

  for m in range(0, label_seq.__len__()):
    num_states, edges = __create_states_from_label_for_ctc(label_seq, m, num_states, num_labels, edges)
    print("label:", label_seq[m], "=", m)

  num_states, edges = __create_last_state_for_ctc(label_seq, m, num_states, num_labels, edges)
  print("label: blank =", m+1)

  return num_states, edges


def __create_states_from_label_for_ctc(label_seq, label, num_states, num_labels, edges):
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
  i = 2 * (label + 1) - 2
  edges.append((str(i), str(i+1), num_labels, 1.))
  edges.append((str(i + 1), str(i + 1), num_labels, 1.))
  edges.append((str(i + 1), str(i + 2), label, 1.))
  edges.append((str(i + 2), str(i + 2), label, 1.))
  if (label_seq[label] != label_seq[label-1]):
    edges.append((str(i), str(i + 2), label, 1.))
  num_states = 2 * (label + 2) - 1

  return num_states, edges


def __create_last_state_for_ctc(label_seq, label, num_states, num_labels, edges):
  """
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
  edges.append((str(i - 1), str(i), num_labels, 1.))
  edges.append((str(i), str(i), num_labels, 1.))
  edges.append((str(i - 2), str(i), label, 1.))
  edges.append((str(i - 3), str(i), label, 1.))
  num_states += 1

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

  for m in range(0, label_seq.__len__()):
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
  ... (like in LmDataset.PhoneSeqGenerator)
  :returns (num_states, edges) like above
  """
  # TODO @Chris ...


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
  args = arg_parser.parse_args()

  if (args.fsa.lower() == 'ctc'):
    num_states, edges = ctc_fsa_for_label_seq(num_labels=args.num_labels, label_seq=args.label_seq)
  elif (args.fsa.lower() == 'asg'):
    num_states, edges = asg_fsa_for_label_seq(num_labels=args.num_labels, label_seq=args.label_seq)
  elif (args.fsa.lower() == 'hmm'):
    num_states, edges = hmm_fsa_for_word_seq()

  fsa_to_dot_format(file=args.file, num_states=num_states, edges=edges)


if __name__ == "__main__":
  main()
