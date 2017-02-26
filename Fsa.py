#!/usr/bin/env python2.7

from __future__ import print_function
from __future__ import division


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
  final_states = []
  # calculate number of states
  num_states = 2 * (len(label_seq) + 1) - 1

  # create edges from the label sequence without loops and no empty labels
  final_states, num_states, edges = __create_states_from_label_seq_for_ctc(label_seq, num_labels, final_states, num_states, edges)

  # adds blank labels to fsa
  final_states, num_states, edges = __adds_blank_states_for_ctc(label_seq, num_labels, final_states, num_states, edges)

  # creates end state
  final_states, num_states, edges = __adds_last_state_for_ctc(label_seq, num_labels, final_states, num_states, edges)

  # adds loops to fsa
  num_states, edges = __adds_loop_edges(num_states, edges)

  # makes one single final state
  num_states, edges = __make_single_final_state(final_states, num_states, edges)

  return num_states, edges


def __create_states_from_label_seq_for_ctc(label_seq, num_labels, final_states, num_states, edges):
  """
  creates states from label sequence, skips repetitions
  :param list[int] label_seq: sequence of labels (normally some kind of word)
  :param int num_labels: number of labels
  :param list[int] final_states: list of final states
  :param int num_states: number of states
  :param list[tuple] edges: list of edges
  :returns (num_states, edges)
  where:
    num_states: int, number of states.
      per convention, state 0 is start state, state (num_states - 1) is single final state
    edges: list[(from,to,label_idx,weight)]
      from and to are state_idx >= 0 and < num_states,
      label_idx >= 0 and label_idx < num_labels  --or-- label_idx == num_labels for blank symbol
      weight is a float, in -log space
  """
  print("Creating nodes and edges from label sequence...")
  # go through the whole label sequence and create the state for each label
  for label_index in range(0, len(label_seq)):
    # if to remove skips if two equal labels follow each other
    if label_seq[label_index] != label_seq[label_index - 1]:
      n = 2 * label_index
      edges.append((n, n + 2, label_seq[label_index], 1.))

  return final_states, num_states, edges


def __adds_blank_states_for_ctc(label_seq, num_labels, final_states, num_states, edges):
  """
  adds blank edges and repetitions to ctc
  :param list[int] label_seq: sequence of labels (normally some kind of word)
  :param int num_labels: number of labels
  :param list[int] final_states: list of final states
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
  print("Adding blank states and edges...")
  # adds blank labels to fsa
  for label_index in range(0, len(label_seq)):
    label_blank_idx = 2 * label_index + 1
    label_blank = 'blank' #  num_labels + 1
    edges.append((label_blank_idx - 1, label_blank_idx, label_blank, 1.))
    edges.append((label_blank_idx, label_blank_idx + 1, label_seq[label_index], 1.))
  final_states.append(label_blank_idx + 1)

  return final_states, num_states, edges


def __adds_last_state_for_ctc(label_seq, num_labels, final_states, num_states, edges):
  """
  adds last states for ctc
  :param list[int] label_seq: sequence of labels (normally some kind of word)
  :param int num_labels: number of labels
  :param list[int] final_states: list of final states
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
  print("Adds final states and edges...")
  i = num_states
  label_blank = 'blank' #  num_labels + 1
  edges.append((i - 3, i, label_blank, 1.))
  edges.append((i, i + 1, label_seq[-1], 1.))
  edges.append((i + 1, i + 2, label_blank, 1.))
  num_states += 3
  final_states.append(num_states - 1)

  return final_states, num_states, edges


def __adds_loop_edges(num_states, edges):
  """
  for every node loops with edge label pointing to node
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
  print("Adding loops...")
  # adds loops to fsa (loops on first and last node excluded)
  for state in range(1, num_states - 1):
    edges_included = [edge_index for edge_index, edge in enumerate(edges) if (edge[1] == state)]
    edges.append((state, state, edges[edges_included[0]][2], 1.))

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
  print("Creates single final state...")
  if len(final_states) == 1 and final_states[0] == num_states - 1:  # nothing to change
    return num_states, edges

  num_states += 1
  for fstate in final_states:
    edges_fstate = [edge_index for edge_index, edge in enumerate(edges) if (edge[1] == fstate)]
    for fstate_edge in edges_fstate:
      edges.append((edges[fstate_edge][0], num_states - 1, edges[fstate_edge][2], 1.))

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

  rep_seq = __check_for_repetitions_for_asg(num_labels, label_seq, repetitions)

  num_states, edges = __create_states_from_label_for_asg(rep_seq, edges)
  num_states, edges = __adds_loop_edges(num_states, edges)

  return num_states, edges


def __check_for_repetitions_for_asg(num_labels, label_indices, repetitions):
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


def hmm_fsa_for_word_seq(word_seq, lexicon_file, state_tying_file, depth=6,
                         allo_num_states=3, allo_context_len=1,
                         tdps=None  # ...
                         ):
  """
  :param list[str] or str word_seq: sequences of words
  :param str lexicon_file: lexicon XML file
  :param int allo_num_states: hom much HMM states per allophone
  :param int allo_context_len: how much context to store left and tight. 1 -> triphone
  :param str | None state_tying_file: for state-tying, if you want that
  :param int depth: depth / level of the algorithm
  ... (like in LmDataset.PhoneSeqGenerator)
  :returns (num_states, edges) like above
  """
  print("Word sequence:", word_seq)
  global sil
  sil = '_'
  print("Place holder silence:", sil)
  global eps
  eps = '*'
  print("Place holder epsilon:", eps)
  if depth is None:
    depth = 6
  print("Depth level is", depth)
  if depth >= 1:
    print("Lemma acceptor...")
    word_list, num_states, edges = __lemma_acceptor_for_hmm_fsa(word_seq)
  else:
    print("No acceptor chosen! Try again!")
    num_states = 0
    edges = []
  if depth >= 2:
    lexicon = __load_lexicon(lexicon_file)
    print("Getting allophone sequence...")
    phon_dict = __find_allo_seq_in_lex(word_list, lexicon)
    print("Phoneme acceptor...")
    word_pos, phon_pos, num_states, edges = __phoneme_acceptor_for_hmm_fsa(word_list, phon_dict, num_states, edges)
  if depth >= 3:
    print("Triphone acceptor...")
    num_states, edges = __triphone_acceptor_for_hmm_fsa(num_states, edges)
  if depth >= 4:
    print("Allophone state acceptor...")
    num_states, edges = __allophone_state_acceptor_for_hmm_fsa(allo_seq, sil, allo_num_states, num_states, edges)
  if depth >= 5:
    print("HMM acceptor...")
    num_states, edges = __adds_loop_edges(num_states, edges)
  if depth >= 6:
    print("State tying...")
    num_states, edges = __state_tying_for_hmm_fsa(state_tying_file, num_states, edges)

  return num_states, edges


def __lemma_acceptor_for_hmm_fsa(word_seq):
  """
  :param str word_seq:
  :return list word_list:
  :return int num_states:
  :return list edges:
  """
  global sil, eps
  epsil = [sil, eps]

  edges = []
  num_states = 0

  if isinstance(word_seq, str):
    word_list = word_seq.split(" ")
  elif isinstance(word_seq, list):
    word_list = word_seq
  else:
    print("word sequence is not a str or a list. i will try...")
    word_list = word_seq

  assert isinstance(word_list, list), "word list is not a list"

  for word_idx in range(len(word_list)):
    assert isinstance(word_list[word_idx], str), "word is not a str"
    start_node = 2 * (word_idx + 1) - 1
    end_node = start_node + 1
    edges.append([start_node, end_node, word_list[word_idx], 0.])
    for i in epsil:
      if word_idx == 0:
        edges.append([start_node - 1, end_node - 1, i, 0.])
        num_states += 1
      edges.append([start_node + 1, end_node + 1, i, 0.])
      num_states += 1

  return word_list, num_states, edges


def __phoneme_acceptor_for_hmm_fsa(word_list, phon_dict, num_states, edges):
  """
  phoneme acceptor
  :param list word_list:
  :param dict phon_dict:
  :param int num_states:
  :param list edges:
  :return list of dict word_pos: letter positions in word
  :return list of list phon_pos: phoneme positions in lemma
        0: phoneme sequence
        1, 2: start end point
        len = 1: no start end point
  :return int num_states:
  :return list edges_phon:
  """
  global sil, eps

  edges_phon_t = []

  """
  replaces chars with phonemes
  """
  while (edges):
    edge = edges.pop(0)
    if edge[2] != sil and edge[2] != eps:
      phon_current = phon_dict[edge[2]]
      for phons in phon_current:
        phon_score = phons['score']  # calculate phon score correctly log space
        edges_phon_t.append([edge[0], edge[1], phons['phon'], phon_score])
    elif edge[2] == sil or edge[2] == eps:
      edges_phon_t.append(edge)  # adds eps and sil edges unchanged
    else:
      assert 1 == 0, "unrecognized phoneme"  # all edges should be handled

  """
  splits word and marks the letters next to a silence
  """
  word_pos = []
  while (word_list):
    word = word_list.pop(0)
    for idx, letter in enumerate(word):
      if idx == 0 and idx == len(word) - 1:
        word_pos.append({letter: ['i', 'f']})
      elif idx == 0:
        word_pos.append({letter: ['i']})
      elif idx == len(word) - 1:
        word_pos.append({letter: ['f']})
      else:
        word_pos.append({letter: ['']})

  """
  splits phoneme sequence and marks the phoneme next to a silence
  """
  edges_t = []
  edges_t.extend(edges_phon_t)
  phon_pos = []

  edges_t.sort(key=lambda x: x[0])

  while (edges_t):
    edge = edges_t.pop(0)  # edge is tuple start node, end node, label, score
    if edge[2] != sil and edge[2] != eps:  # sil and eps ignored
      phon_list = edge[2].split(" ")
      letter_pos = []
      for idx, letter in enumerate(phon_list):
        if idx == 0 and idx == len(phon_list) - 1:
          letter_pos.append([letter, 'i', 'f'])
        elif idx == 0:
          letter_pos.append([letter, 'i'])
        elif idx == len(phon_list) - 1:
          letter_pos.append([letter, 'f'])
        else:
          letter_pos.append([letter])
      phon_pos.append(letter_pos)

  """
  splits phoneme edge into several edges
  """
  edges_phon = []
  edges_phon_t.sort(key=lambda x: x[0])

  while (edges_phon_t):
    edge = edges_phon_t.pop(0)
    if edge[2] != sil and edge[2] != eps:
      phon_seq = edge[2].split(" ")
      for phon_idx, phon_label in enumerate(phon_seq):
        phon_seq_len = len(phon_seq)
        if phon_seq_len == 1:
          edges_phon.append(edge)
        elif phon_seq_len > 1:
          if phon_idx == 0:
            start_node = edge[0]
            end_node = num_states
            phon_score = edge[3]
            edges_phon.append([start_node, end_node, phon_label, phon_score])
            num_states += 1
          elif phon_idx == phon_seq_len - 1:
            start_node = num_states - 1
            end_node = edge[1]
            phon_score = 0.
            edges_phon.append([start_node, end_node, phon_label, phon_score])
          else:
            start_node = num_states - 1
            end_node = num_states
            phon_score = 0.
            edges_phon.append([start_node, end_node, phon_label, phon_score])
            num_states += 1
        else:
          assert 1 == 0, "Something went wrong while expanding phoneme sequence"
    else:
      edges_phon.append(edge)
    edges_phon.sort(key=lambda x: x[0])

  edges_phon = __sort_node_num(edges_phon)

  return word_pos, phon_pos, num_states, edges_phon


def __check_node_existance(node_num, edges):
  """
  checks if the node numbers already exist in edges list
  :param float node_num: node number to be checked
  :return bool: true if node in edges
  """
  node_list = [edge_index for edge_index, edge in enumerate(edges)
                      if (edge[0] == node_num or edge[1] == node_num)]

  if len(node_list) > 0:
    return True
  else:
    return False


def __sort_node_num(edges):
  """
  reorders the node numbers: always rising numbers. never 40 -> 11
  uses some kind of sorting algorithm (quicksort, ...)
  :param int num_states: number od states / nodes
  :param list edges: list with unordered nodes
  :return list edges: list with ordered nodes
  """
  idx = 0

  while (idx < len(edges)):  # traverse all edges from 0 to num_states
    cur_edge = edges[idx]         # gets the current edge
    cur_edge_start = cur_edge[0]  # with current start
    cur_edge_end = cur_edge[1]    # and end node

    if cur_edge_start > cur_edge_end:  # only something to do if start node number > end node number
      edges_cur_start = __find_node_edges(cur_edge_start, edges)  # find start node in all edges
      edges_cur_end = __find_node_edges(cur_edge_end, edges)  # find end node in all edges

      for edge_key in edges_cur_start.keys():  # loop over edge which have the specific node
        edges[edge_key][edges_cur_start[edge_key]] = cur_edge_end  # replaces the start node number

      for edge_key in edges_cur_end.keys():  # edge_key: idx from edge in edges
        edges[edge_key][edges_cur_end[edge_key]] = cur_edge_start  # replaces the end node number

      # reset idx: restarts traversing at the beginning of graph
      # swapping may introduce new disorders
      idx = 0

    idx += 1

  return edges


def __find_node_edges(node, edges):
  """
  find a specific node in all edges
  :param int node: node number
  :param list edges: all edges
  :return dict node_dict: dict of nodes where
        key: edge index
        value: 0 = node at edge start position
        value: 1 = node at edge end position
        value: 2 = node at edge start and edge postion
  """
  node_dict = {}

  pos_start = [edge_index for edge_index, edge in enumerate(edges) if (edge[0] == node)]
  pos_end = [edge_index for edge_index, edge in enumerate(edges) if (edge[1] == node)]
  pos_start_end = [edge_index for edge_index, edge in enumerate(edges) if
                   (edge[0] == node and edge[1] == node)]

  for pos in pos_start:
    node_dict[pos] = 0

  for pos in pos_end:
    node_dict[pos] = 1

  for pos in pos_start_end:
    node_dict[pos] = 2

  return node_dict


def __triphone_acceptor_for_hmm_fsa(num_states, edges):
  """
  changes the labels of the edges from phonemes to triphones
  :param list[str] or str word_seq: sequences of words
  :param dict phon_dict:
        key: lemma from the list
        value: list of dictionaries with phon and score (keys)
  :param list of dict word_pos: letter positions in word
  :param list of list phon_pos: phoneme positions in lemma
        0: phoneme sequence
        1, 2: start end point
        len = 1: no start end point
  :param int num_states: number of states
  :param list edges: list of edges
  :return int num_states: number of states
  :return list edges_tri: list of edges
  """
  global sil
  global eps

  edges_tri = []
  edges_t = []
  edges_t.extend(edges)

  while(edges_t):
    edge_t = edges_t.pop(0)
    if edge_t[2] == sil or edge_t[2] == eps:
      edges_tri.append(edge_t)
    else:
      prev_edge_t = __find_prev_next_edge(edge_t, 0, edges)
      next_edge_t = __find_prev_next_edge(edge_t, 1, edges)

      label_tri = [prev_edge_t[2], edge_t[2], next_edge_t[2]]

      edge_n = [edge_t[0], edge_t[1], label_tri, edge_t[3]]
      edges_tri.append(edge_n)

  return num_states, edges_tri


def __find_prev_next_edge(cur_edge, pn_switch, edges):
  """
  find the next/previous edge within the edges list
  :param list cur_edge: current edge
  :param int pn_switch: toggles between previous (0) and next (1) edge
  :param list edges: list of edges
  :return list pn_edge: previous/next edge
  """
  global sil
  global eps

  assert pn_switch == 0 or pn_switch == 1, ("Previous/Next switch has wrong value:", pn_switch)

  # finds indexes of previous edges
  prev_edge_cand_idx = [edge_index for edge_index, edge in enumerate(edges)
                        if (cur_edge[pn_switch] == edge[1 - pn_switch])]

  # remove eps and sil edges
  prev_edge_cand_idx_len = len(prev_edge_cand_idx)
  if prev_edge_cand_idx_len > 1:
    for idx in prev_edge_cand_idx:
      assert edges[idx][2] == sil or edges[idx][2] == eps, "Edge found which is not sil or eps"
  else:
    assert prev_edge_cand_idx_len <= 1, ("Too many previous edges found:", prev_edge_cand_idx)

  assert prev_edge_cand_idx_len >= 0, ("Negative edges found. Something went wrong..")

  # sets pn_edge to the previous edge or if sil/eps then empty edge
  if prev_edge_cand_idx_len == 1:
    pn_edge = edges[prev_edge_cand_idx[0]]
  else:
    pn_edge = [None, None, '', None]

  return pn_edge


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


def __allophone_state_acceptor_for_hmm_fsa(allo_seq, sil, allo_num_states, num_states_input, edges_input):
  """
  the edges which are not sil or eps are split into three allophone states / components
    marked with 0, 1, 2
  :param allo_seq: sequence of allophones
  :param str sil: placeholder for silence
  :param int num_states_input: number of states
  :param list[tuples(int, int, tuple(str, str, str), float)] edges_input: edges with label and weight
  :return int num_states_output, list[tuples(int, int, tuple(str, str, str, int), float)] edges_output:
  """
  allo_len = len(allo_seq)
  allo_count = 4 * allo_len
  edges_count = __count_all_edges_non_sil_or_eps(edges_input, sil)

  assert edges_count == allo_count, "the count for the non-sil, non-eps edges varies: %i != %i"\
                                    % (edges_count, allo_count)

  global num_states_check
  num_states_check = num_states_input + (allo_num_states - 1) * edges_count
  num_states_output = num_states_input

  current_node = 0

  edges_traverse = []
  edges_updated = []
  edges_output = []

  edges_updated.extend(edges_input)
  edges_updated.sort(key=lambda x: x[1])

  assert id(edges_updated) != id(edges_input), "same id for edges is wrong"

  num_states_output, edges_output = \
    __walk_graph_add_allo_states_for_hmm_fsa(current_node,
                                             sil,
                                             allo_num_states,
                                             num_states_input,
                                             edges_input,
                                             edges_traverse,
                                             edges_updated,
                                             num_states_output,
                                             edges_output)

  assert num_states_output == num_states_check,\
    "the number of states does not match: %i != %i" % (num_states_output, num_states_check)

  return num_states_output, edges_output


def __count_all_edges_non_sil_or_eps(edges, sil='sil', eps='eps'):
  """
  count all edges in a graph which are NOT silence or placeholders (epsilon)
  :param list[tuples(int, int, tuple(str, str, str), float)] edges: edges with label and weight
  :param str sil: silence
  :param str eps: epsilon placeholder / skip edge
  :return int edges_count: number of edges where NOT silence or skips
  """
  edges_count = 0

  for edge in edges:
    if edge[2] != sil and edge[2] != eps:
      edges_count += 1

  return edges_count


def __walk_graph_add_allo_states_for_hmm_fsa(current_node,
                                             sil,
                                             allo_num_states,
                                             num_states_input,
                                             edges_input,
                                             edges_traverse,
                                             edges_updated,
                                             num_states_output,
                                             edges_output):
  """
  idea: go to edge. do not change start node. take end node. search in edges at position start
  node (only if ![sil], no change propagates from [sil]). add 2 to start and end node for all
  following nodes (add nodes with index 1, 2 while traversing)

  algorithm idea:
  - take current_edge
  - search for all edges with a start and end node >= current_edge[end node] and add to edges_traverse
  - expand current_edge and add three edges to edges_expand
  - take all edges from edges_traverse and add =+2 to start and end node in edges

  :param int current_node:
  :param list [tuples(int, int, tuple(str, str, str), float)] edges_traverse:
    edges to traverse and expand from one triphone into three allophone states,
    double entries are allowed, with the last entry the edge should be expanded (triphone
    to allophone states
  :param list [tuples(int, int, tuple(str, str, str), float)] edges_expanded:
    list of edges with triphones expanded into three allophone states
  :param str sil: placeholder for silence
  :param int num_states_input: expanded number of states
  :param int num_states_input: number of states
  :param list[tuples(int, int, tuple(str, str, str), float)] edges_input: edges with label and weight
  :return int current_node:
  :return list [tuples(int, int, tuple(str, str, str), float)] edges_to_traverse:
    edges to traverse and expand from one triphone into three allophone states,
    double entries are allowed, with the last entry the edge should be expanded (triphone
    to allophone states
  :return list [tuples(int, int, tuple(str, str, str), float)] edges_expanded:
    list of edges with triphones expanded into three allophone states
  :return int num_states_input: expanded number of states
  :return int num_states: number of states
  :return list[tuples(int, int, tuple(str, str, str), float)] edges: edges with label and weight
  """
  edges_input.sort(key=lambda x: x[1])
  edges_traverse.sort(key=lambda x: x[1])
  edges_output.sort(key=lambda x: x[1])

  if len(edges_updated) > 0:
    current_edge = edges_updated.pop(0)

    edges_traverse = __find_edges_after_current_for_hmm_fsa(current_edge, edges_updated)

    edges_updated, edges_output = __change_edge_to_higher_node_num_for_hmm_fsa(current_edge,
                                                                  sil,
                                                                  allo_num_states,
                                                                  edges_traverse,
                                                                  edges_updated,
                                                                  edges_output)

    edges_updated, num_states_output, edges_output = __expand_tri_edge_for_hmm_fsa(current_edge,
                                                                      sil,
                                                                      allo_num_states,
                                                                      num_states_output,
                                                                      edges_updated,
                                                                      edges_output)

    num_states_output, edges_output = \
      __walk_graph_add_allo_states_for_hmm_fsa(current_node,
                                               sil,
                                               allo_num_states,
                                               num_states_input,
                                               edges_input,
                                               edges_traverse,
                                               edges_updated,
                                               num_states_output,
                                               edges_output)

  return num_states_output, edges_output


def __find_edges_after_current_for_hmm_fsa(current_edge, edges):
  """
  search for all edges with a start node >= current_edge[end node] and add to edges_traverse
  :param tuple(int, int, tuple(str, str, str), float) current_edge: the currently selected edge
  :param list[tuples(int, int, tuple(str, str, str), float)] edges: list of edges
  :return list[tuples(int, int, tuple(str, str, str), float)] edges_traverse: list of edges where
    start node >= current_edge[end node]
  """
  edges_gequal_cur = [edge_index for edge_index, edge in enumerate(edges)
                      if (edge[0] >= current_edge[1] or edge[1] >= current_edge[1])]

  edges_traverse = []
  for edge_idx in edges_gequal_cur:
    edges_traverse.append(edges[edge_idx])

  edges_traverse.sort(key=lambda x: x[1])

  return edges_traverse


def __change_edge_to_higher_node_num_for_hmm_fsa(current_edge,
                                                 sil,
                                                 allo_num_states,
                                                 edges_traverse,
                                                 edges_updated,
                                                 edges_output):
  """
  idea: change start / end node id number += 2 for edges
  :param tuples(int, int, tuple(str, str, str), float) current_edge: current edge
  :param str sil: placeholder for silence
  :param list[tuples(int, int, tuple(str, str, str), float)] edges_updated:
    list of edges with expanded allo states
  :param list[tuples(int, int, tuple(str, str, str), float)] edges_traverse:
    list of edges after current edge
  :param list[tuples(int, int, tuple(str, str, str), float)] edges: list of edges
  :return list[tuples(int, int, tuple(str, str, str), float)] edges_expanded:
   list of edges where the start and end node have been raised by two
  """
  if current_edge[2] == sil and current_edge[0] == 0:
    edges_output.append(current_edge)
  elif current_edge[2] == sil and current_edge[0] != 0:
    edge_t = (current_edge[0] + (allo_num_states - 1) * len(edges_updated),
              current_edge[1] + (allo_num_states - 1) * len(edges_updated),
              current_edge[2],
              current_edge[3])
    edges_output.append(edge_t)
    edges_t = []
    for edge in edges_updated:  # necessary because sil edge is moved backwards
      edges_t.append((edge[0], edge[1] - 1, edge[2], edge[3]))
    edges_updated = edges_t
  else:
    # construct list of current edges
    current_edge_list = [current_edge for n in range(len(edges_traverse))]
    # take all edges which have to be traversed and move them to higher nodes
    edges_high = map(__map_higher_node, edges_traverse, current_edge_list)
    # create new list of edges from edges_updated which are in edges_traverse
    edges_sub = filter(lambda x: x in edges_traverse, edges_updated)

    for edge in edges_sub:
      if edge in edges_updated:
        edges_updated.remove(edge)

    edges_updated.extend(edges_high)

  edges_updated.sort(key=lambda x: x[1])

  return edges_updated, edges_output


def __map_higher_node(x, y):
  assert isinstance(x, tuple), "x has to be a tuple(int, int, tuple(str, str, str), float)"
  assert isinstance(y, tuple), "y should be a tuple(int, int, tuple(str, str, str), float)"
  assert len(x) == len(y), "x and y have different lengths"
  if (x[0] >= y[1]):
    return (x[0] + 2, x[1] + 2, x[2], x[3])
  elif (x[1] >= y[1]):
    return (x[0], x[1] + 2, x[2], x[3])


def __expand_tri_edge_for_hmm_fsa(current_edge,
                                  sil,
                                  allo_num_states,
                                  num_states_t,
                                  edges_updated,
                                  edges_output):
  """

  :param tuple(int, int, tuple(str, str, str), float) current_edge: the current edge
  :param str sil: placeholder for silence
  :param int num_states_t: new calculation of number of states
    where the node count has been raised by two
  :param list[tuples(int, int, tuple(str, str, str), float)] edges_expanded: list of edges
    where the node count has been raised by two
  :return int num_states_output:
  :return list[tuples(int, int, tuple(str, str, str), float)] edges_expanded:
  """
  global num_states_check
  edges_expanded = []
  start_node = current_edge[0]
  if len(edges_updated) > 4 and current_edge[2] != sil:
    end_node = current_edge[1]
  else:
    end_node = current_edge[1]
  if current_edge[2] == sil:
    num_states_output = num_states_t
  else:
    for state_t in range(0, allo_num_states):
      tuple_t = (current_edge[2][0], current_edge[2][1], current_edge[2][2], state_t)

      if len(edges_updated) < 5 and state_t == allo_num_states - 1:
        end_node = num_states_check - len(edges_updated) % 2 - 1

      edge_t = (start_node, end_node, tuple_t, current_edge[3])

      edges_expanded.append(edge_t)

      start_node = end_node
      end_node += 1

    num_states_output = num_states_t + (allo_num_states - 1)

  edges_output.extend(edges_expanded)
  edges_output.sort(key=lambda x: x[1])

  return edges_updated, num_states_output, edges_output


def __state_tying_for_hmm_fsa(state_tying_file, num_states, edges):
  """
  idea: take file with mapping char to number and apply to edge labels
  :param int num_states:
  :param list[tuples(start[int], end[int], label, weight)] edges:
  :return: num_states, edges
  """
  global sil
  edges_ts = []
  edges_st = []
  statetying = __load_state_tying_file(state_tying_file)

  for edge in edges:
    allo_state_tying = edge[2]

    allo_syntax = __build_allo_syntax_for_mapping(allo_state_tying)

    allo_id_num = statetying.allo_map[allo_syntax]

    edges_ts.append((edge[0], edge[1], allo_syntax, edge[3]))
    edges_st.append((edge[0], edge[1], allo_id_num, edge[3]))

  edges_ts.sort()
  edges_st.sort()

  return num_states, edges_st


def __load_state_tying_file(stFile):
  '''
  loads a state tying map from a file, loads the file and returns its content
  :param stFile: state tying map file (allo_syntax int)
  :return state_tying: variable with state tying mapping
  where:
    statetying.allo_map important
  '''
  from os.path import isfile
  from LmDataset import StateTying

  print("Loading state tying file:", stFile)

  assert isfile(stFile), "State tying file does not exists"

  statetying = StateTying(stFile)

  print("Finished state tying mapping:", len(statetying.allo_map), "allos to int")

  return statetying


def __build_allo_syntax_for_mapping(label):
  """
  builds a conforming allo syntax for mapping
  :param str pos_seq:
  :param int pos_allo:
  :param str or tuple(str, str, str) label: a allo either string or tuple
  :return str allo_map: a allo syntax ready for mapping
  """
  global sil
  assert isinstance(label, str) or isinstance(label, tuple), "Something went wrong while building allo syntax for mapping"

  if isinstance(label, str) and label == sil:
    allo_start = "%s{#+#}" % ('[SILENCE]')
  else:
    if label[0] == '' and label[2] == '':
      allo_start = "%s{#+#}" % (label[0])
    elif label[0] == '':
      allo_start = "%s{#+%s}" % (label[1], label[2])
    elif label[2] == '':
      allo_start = "%s{%s+#}" % (label[1], label[0])
    else:
      allo_start = "%s{%s+%s}" % (label[1], label[0], label[2])

  allo_middle = ''
  if label[0] == '':
    allo_middle = "@%s" % ('i')
  if label[2] == '':
    allo_middle = "@%s" % ('f')

  if label == sil:
    allo_end = ".0"
  else:
    allo_end = ".%i" % (label[3])

  allo_map = "%s%s%s" % (allo_start, allo_middle, allo_end)

  return allo_map


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


def __find_allo_seq_in_lex(lemma_list, lex):
  '''
  searches a lexicon xml structure for a watching word and
  returns the matching allophone sequence as a list
  :param lemma_list: the word / lemma sequence to search for in the lexicon
  :param lex: the lexicon
  :return dict phon_dict:
        key: lemma from the list
        value: list of dictionaries with phon and score (keys)
  '''
  if isinstance(lemma_list, str):
    lemma_list = lemma_list.split(" ")

  assert isinstance(lemma_list, list), " word list is not list"

  phon_dict = {}

  for lemma in lemma_list:
    assert isinstance(lemma, str), "word is not str"
    phon_dict[lemma] = lex.lemmas[lemma]['phons']

  return phon_dict


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
  arg_parser.add_argument("--state_tying")
  arg_parser.add_argument("--lexicon")
  arg_parser.add_argument("--depth", type=int)
  arg_parser.add_argument("--asg_repetition", type=int)
  arg_parser.add_argument("--label_conversion", type=bool)
  args = arg_parser.parse_args()

  if (args.fsa.lower() == 'ctc'):
    if args.label_conversion:
      label_seq = convert_label_seq_to_indices(args.num_labels, args.label_seq)
    else:
      label_seq = args.label_seq
    num_states, edges = ctc_fsa_for_label_seq(num_labels=args.num_labels,
                                              label_seq=label_seq)
  elif (args.fsa.lower() == 'asg'):
    assert args.asg_repetition, "Specify number of asg repetition labels in argument options: --asg_repetition [int]"
    if args.label_conversion:
      label_seq = convert_label_seq_to_indices(args.num_labels, args.label_seq)
    else:
      label_seq = args.label_seq
    num_states, edges = asg_fsa_for_label_seq(num_labels=args.num_labels,
                                              label_seq=label_seq,
                                              repetitions=args.asg_repetition)
    print("Number of labels (ex.: a-z == 27 labels):", args.num_labels)
    print("Number of repetition symbols:", args.asg_repetition)
    for rep in range(1, args.asg_repetition + 1):
      print("Repetition label:", args.num_labels + rep, "meaning", rep, "repetitions")
  elif (args.fsa.lower() == 'hmm'):
    assert args.lexicon, "Specify lexicon in argument options: --lexicon [path]"
    assert args.state_tying, "Specify state tying file in argument options: --state_tying [path]"
    num_states, edges = hmm_fsa_for_word_seq(word_seq=args.label_seq,
                                             lexicon_file=args.lexicon,
                                             state_tying_file=args.state_tying,
                                             depth=args.depth)

  fsa_to_dot_format(file=args.file, num_states=num_states, edges=edges)


if __name__ == "__main__":
  main()
