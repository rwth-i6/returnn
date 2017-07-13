#!/usr/bin/env python2.7

from __future__ import print_function
from __future__ import division


class Fsa:
  """class to create a finite state automaton"""
  _SIL = '_'
  _EPS = '*'
  _BLANK = '%'

  def __init__(self):
    """
    :param str or list lemma: word or sentence
    :param str fsa_type: determines finite state automaton type: asg, ctc, hmm
    :param int num_states: number of states
    :param list edges: list of edges
    where:
      num_states: int, number of states.
        per convention, state 0 is start state, state (num_states - 1) is single final state
      edges: list[(from,to,label_idx,weight)]
        from and to are state_idx >= 0 and < num_states,
        label_idx >= 0 and label_idx < num_labels  --or-- label_idx == num_labels for blank symbol
        weight is a float, in -log space
    :param str filename: name of file to store graph
    :param int asg_repetition: repetition symbols for asg
    :param int num_labels: number of labels
    :param bool label_conversion: use chars or indexes
    :param list[int] final_states: list of final states
    :param int depth: depth / level of hmm
    :param int allo_num_states: number of allophone states
    :param str lexicon: lexicon file name
    :param str state_tying: state tying file name
    :param dict phon_dict: dictionary of phonemes, loaded from lexicon file
    """
    # needed by ASG, CTC and HMM
    self.num_states = 0

    # 0: starting node
    # 1: ending node
    # 2: label
    # 3: weight
    # 4: label position
    self.edges = []
    self.edges_single_state = []

    self.fsa_type = None

    self.lemma_orig = None
    self.lemma = None

    self.filename = 'fsa'

    self.single_state = False

    # needed by ASG
    self.asg_repetition = 2

    # needed by ASG and CTC
    self.num_labels = 27
    self.label_conversion = None

    # needed by CTC
    self.final_states = []

    # needed by HMM
    self.depth = 6
    self.allo_num_states = 3
    self.lexicon_name = ''
    self.lexicon = None
    self.state_tying_name = ''
    self.state_tying = None
    self.phon_dict = {}

  def set_params(self,
                 asg_repetition=2,
                 num_labels=256,  # ascii number of labels
                 label_conversion=False,
                 depth=6,
                 allo_num_states=3,
                 lexicon_name='',
                 state_tying_name='',
                 single_state=False):
    """
    sets the parameters for FSA generator
    checks if needed params for fsa type available otherwise erquests user input
    :param str filename: sets the output file name
    :param int asg_repetition:
      if a label is repeated within the lemma how many repetitions will be substituted
      with a specific repetition symbol
    :param int num_labels: total number of labels
    :param bool label_conversion:
      true: each label converted to index of its label
      false: no conversion
    :param int depth: depth of the hmm acceptor
    :param int allo_num_states: umber of allophone states
    :param str lexicon: lexicon file name
    :param str state_tying: state tyting file name
    :param bool single_state: produce additional fsa: single node
    :return:
    """
    self.single_state = single_state
    print("Single state set to:", self.single_state)

    if not isinstance(label_conversion, bool):
      print("Set label conversion option:")
      print("1 (On) or 0 (Off)")
      label_conversion = raw_input("--> ")
    self.label_conversion = bool(int(label_conversion))
    assert isinstance(self.label_conversion, bool), "Label conversion not set"

    if self.fsa_type == 'asg' or self.fsa_type == 'ctc':
      if self.fsa_type == 'asg' and asg_repetition < 0:
        print("Enter length of repetition symbols:")
        print("Example: 3 -> 2 repetition symbols for 2 and 3 repetitions")
        asg_repetition = raw_input("--> ")
      self.asg_repetition = int(asg_repetition)
      assert isinstance(self.asg_repetition, int), "ASG repetition wrong type"
      assert self.asg_repetition >= 0, "ASG repetition not set"

      if num_labels <= 0:
        print("Enter number of labels:")
        num_labels = raw_input("--> ")
      self.num_labels = int(num_labels)
      assert self.num_labels > 0, "Number of labels not set"

    elif self.fsa_type == 'hmm':
      self.lemma_orig = self.lemma_orig.lower()
      if depth < 0:
        print("Set the depth level of HMM:")
        depth = raw_input("--> ")
      self.depth = int(depth)
      assert isinstance(self.depth, int) and self.depth > 0, "Depth for HMM not set"

      if allo_num_states < 1:
        print("Set the number of allophone states:")
        allo_num_states = raw_input("--> ")
      self.allo_num_states = int(allo_num_states)
      assert isinstance(self.allo_num_states, int) and self.allo_num_states > 0,\
        "Number of allophone states not set"
      self.lexicon_name = lexicon_name
      self.state_tying_name = state_tying_name

    else:
      print("No finite state automaton matches to chosen type")
      sys.exit(-1)

  def set_lemma(self, lemma):
    """
    :param str or list lemma: word or sentence
    """
    assert isinstance(lemma, str) or isinstance(lemma, list), "Lemma type not correct"
    self.lemma_orig = lemma.lower().rstrip()
    assert isinstance(self.lemma_orig, str) or isinstance(self.lemma_orig, list),\
      "Lemma type not correct"
    self.lemma = None

  def set_fsa_type(self, fsa_type):
    """
    :param str fsa_type: determines finite state automaton type: asg, ctc, hmm
    """
    assert isinstance(fsa_type, str), "FSA type input not a string"
    self.fsa_type = fsa_type.lower()
    assert isinstance(self.fsa_type, str), "FSA type not a string"
    print("Setting parameters for", self.fsa_type)

  def set_filename(self, filename):
    """
    :param str filename: name of file to store graph
    """
    assert isinstance(filename, str), "filename is not a string"
    self.filename = filename

  def set_hmm_depth(self, depth):
    assert isinstance(depth, int), "depth is not a int"
    self.depth = depth

  def set_lexicon(self, lexicon_name=None):
    """
    sets a new lexicon
    :param str lexicon_name: lexicon path
    """
    if isinstance(lexicon_name, str):
      self.lexicon_name = lexicon_name
      self._load_lexicon()

  def set_state_tying(self, state_tying=None):
    """
    sets a new state tying file
    :param str state_tying: state tying file/path
    """
    assert isinstance(state_tying, str), "state tying is not a string"
    self.state_tying_name = state_tying
    self._load_state_tying()

  def _load_lexicon(self, reload = False):
    '''
    loads a lexicon from a file, loads the xml and returns its content
    where:
      lex.lemmas and lex.phonemes important
    :param bool reload: should lexicon be reloaded
    '''
    from LmDataset import Lexicon
    if not isinstance(self.lexicon, Lexicon):
      reload = True

    if reload:
      from os.path import isfile
      from Log import log

      assert isfile(self.lexicon_name), "Lexicon does not exists"

      log.initialize(verbosity=[5])

      self.lexicon = Lexicon(self.lexicon_name)

  def run(self):
    """
    runs the FSA
    """
    if self.fsa_type == 'asg':
      if self.label_conversion == True:
        self.convert_label_seq_to_indices()
      else:
        self.lemma = self.lemma_orig

      assert isinstance(self.lemma, str) or isinstance(self.lemma, list), "Lemma not str or list"

      print("Number of labels (ex.: ascii: 265 labels):", self.num_labels)
      print("Number of repetition symbols:", self.asg_repetition)
      for rep in range(1, self.asg_repetition + 1):
        print("Repetition label:", self.num_labels + rep, "meaning", rep, "repetitions")

      self.edges = []

      self._check_for_repetitions_for_asg()
      self._create_states_from_label_for_asg()
      self._adds_loop_edges()
    elif self.fsa_type == 'ctc':
      print("Place holder blank:", self._BLANK)
      if self.label_conversion == True:
        self.convert_label_seq_to_indices()
      else:
        self.lemma = self.lemma_orig

      assert isinstance(self.lemma, str) or isinstance(self.lemma, list), "Lemma not str or list"

      self.edges = []
      self.final_states = []

      # calculate number of states
      self.num_states = 2 * (len(self.lemma) + 1) - 1

      # create edges from the label sequence without loops and no empty labels
      self._create_states_from_label_seq_for_ctc()

      # adds blank labels to fsa
      self._adds_blank_states_for_ctc()

      # creates end state
      self._adds_last_state_for_ctc()

      # adds loops to fsa
      self._adds_loop_edges()

      # makes one single final state
      self._make_single_final_state()
    elif self.fsa_type == 'hmm':
      print("Word sequence:", self.lemma_orig)
      print("Place holder silence:", self._SIL)
      print("Place holder epsilon:", self._EPS)
      print("Depth level is", self.depth)
      if self.depth >= 1:
        print("Lemma acceptor...")
        self._lemma_acceptor_for_hmm_fsa()
      else:
        print("No acceptor chosen! Try again!")
        self.num_states = 0
        self.edges = []
      if self.depth >= 2:
        self._load_lexicon()
        print("Getting allophone sequence...")
        self._find_allo_seq_in_lex()
        print("Phoneme acceptor...")
        self._phoneme_acceptor_for_hmm_fsa()
      if self.depth >= 3:
        print("Triphone acceptor...")
        self._triphone_acceptor_for_hmm_fsa()
      if self.depth >= 4:
        print("Allophone state acceptor...")
        print("Number of allophone states:", self.allo_num_states)
        self._allophone_state_acceptor_for_hmm_fsa()
      if self.depth >= 5:
        print("HMM acceptor...")
        self._adds_loop_edges()
      if self.depth >= 6:
        print("State tying...")
        self._state_tying_for_hmm_fsa()
      if self.depth >= 7:
        print("No depth level higher than 6!")
    else:
      print("No finite state automaton matches to chosen type")
      sys.exit(-1)

  def convert_label_seq_to_indices(self):
    """
    takes label sequence of chars and converts to indices (ascii numbering)
    """
    label_indices = []
    label_seq = self.lemma_orig

    for label in label_seq:
      label_index = ord(label)
      assert label_index < self.num_labels, "Index of label exceeds number of labels"
      label_indices.append(label_index)

    self.lemma = label_indices

  def reduce_node_num(self):
    """
    takes the edges and nodes, then reduces all to one node
    """
    if (self.num_states > 1 and self.single_state == True):
      self.edges_single_state = [(0, 0, edge[2], edge[3])for edge in self.edges]

  def _adds_loop_edges(self):
    """
    for every node loops with edge label pointing to node
    """
    print("Adding loops...")
    if self.fsa_type == 'asg' or self.fsa_type == 'ctc':  # loops on first node excluded
      countloops = self.num_states
    elif self.fsa_type == 'hmm':  # loops on first and last node excluded
      countloops = self.num_states - 1
    else:
      assert 0 == 1, "No finite state automaton matches to chosen type"

    # adds loops to fsa
    for state in range(1, countloops):
      edges_included = [edge_index for edge_index, edge in enumerate(self.edges) if
                        (edge[1] == state and edge[2] != self._EPS)]
      for edge_inc in edges_included:
        try:
          label_pos = self.edges[edge_inc][4]
        except Exception:
          label_pos = None
        if self.fsa_type == 'hmm':
          edge_n = [state, state, self.edges[edge_inc][2], 0., self.edges[edge_inc][4]]
          assert len(edge_n) == 5,  "length of edge wrong"
        else:
          edge_n = [state, state, self.edges[edge_inc][2], 0.]
          assert len(edge_n) == 4, "length of edge wrong"
        self.edges.append(edge_n)

  def _check_for_repetitions_for_asg(self):
    """
    checks the label indices for repetitions,
    if the n-1 label index is a repetition n in reps gets set to 1 otherwise 0
    """
    reps = []
    rep_count = 0
    index_old = None

    if self.asg_repetition == 0:
      reps = self.lemma
    else:
      for index in self.lemma:
        index_t = index
        if index_t == index_old:
          if rep_count < self.asg_repetition:
            rep_count += 1
          elif rep_count != 0:
            reps.append(self.num_labels + rep_count)
            rep_count = 1
          else:
            print("Something went wrong")
        elif index_t != index_old:
          if rep_count != 0:
            reps.append(self.num_labels + rep_count)
            rep_count = 0
          reps.append(index)
        else:
          print("Something went wrong")
        index_old = index

    self.lemma = reps

  def _create_states_from_label_for_asg(self):
    """
    create states from lemma
    """
    for rep_index, rep_label in enumerate(self.lemma):
      self.edges.append((rep_index, rep_index+1, rep_label, 1.))

    self.num_states = len(self.lemma) + 1

  def _create_states_from_label_seq_for_ctc(self):
    """
    creates states from label sequence, skips repetitions
    """
    print("Create nodes and edges from label sequence...")
    # go through the whole label sequence and create the state for each label
    for label_index in range(0, len(self.lemma)):
      # if to remove skips if two equal labels follow each other
      if self.lemma[label_index] != self.lemma[label_index - 1]:
        n = 2 * label_index
        self.edges.append([n, n + 2, self.lemma[label_index], 1.])

  def _adds_blank_states_for_ctc(self):
    """
    adds blank edges and repetitions to ctc
    """
    print("Add blank states and edges...")
    label_blank_idx = 0
    # adds blank labels to fsa
    for label_index in range(0, len(self.lemma)):
      label_blank_idx = 2 * label_index + 1
      self.edges.append([label_blank_idx - 1, label_blank_idx, self._BLANK, 1.])
      self.edges.append([label_blank_idx, label_blank_idx + 1, self.lemma[label_index], 1.])
    self.final_states.append(label_blank_idx + 1)

  def _adds_last_state_for_ctc(self):
    """
    adds last states for ctc
    """
    print("Add final states and edges...")
    i = self.num_states
    self.edges.append([i - 3, i, self._BLANK, 1.])
    self.edges.append([i, i + 1, self.lemma[-1], 1.])
    self.edges.append([i + 1, i + 2, self._BLANK, 1.])
    self.num_states += 3
    self.final_states.append(self.num_states - 1)

  def _make_single_final_state(self):
    """
    takes the graph and merges all final nodes into one single final node
    idea:
      - add new single final node
      - for all edge which ended in a former final node:
      - create new edge from stating node to new single final node with the same label
    """
    print("Create single final state...")
    if len(self.final_states) == 1 and self.final_states[0] == self.num_states - 1:  # nothing to change
      pass
    else:
      self.num_states += 1
      for fstate in self.final_states:
        edges_fstate = [edge_index for edge_index, edge in enumerate(self.edges) if (edge[1] == fstate)]
        for fstate_edge in edges_fstate:
          self.edges.append([self.edges[fstate_edge][0], self.num_states - 1, self.edges[fstate_edge][2], 1.])

  def _lemma_acceptor_for_hmm_fsa(self):
    """
    takes lemma, turns into graph with epsilon and silence
    """
    epsil = [self._SIL, self._EPS]

    self.edges = []
    self.num_states = 0

    if isinstance(self.lemma_orig, str):
      self.lemma = self.lemma_orig.split(" ")
    elif isinstance(self.lemma_orig, list):
      self.lemma = self.lemma_orig
    else:
      print("word sequence is not a str or a list. i will try...")
      self.lemma = self.lemma_orig

    assert isinstance(self.lemma, list), "lemma is not a list"

    for word_idx in range(len(self.lemma)):
      assert isinstance(self.lemma[word_idx], str), "lemma is not a str"
      start_node = 2 * (word_idx + 1) - 1
      end_node = start_node + 1
      self.edges.append([start_node, end_node, self.lemma[word_idx], 0.])
      for i in epsil:
        if word_idx == 0:
          self.edges.append([start_node - 1, end_node - 1, i, 0.])
          self.num_states += 1
        self.edges.append([start_node + 1, end_node + 1, i, 0.])
        self.num_states += 1

  def _find_allo_seq_in_lex(self):
    '''
    searches a lexicon xml structure for a watching word and
    returns the matching allophone sequence as a list
    :return dict phon_dict:
          key: lemma from the list
          value: list of dictionaries with phon and score (keys)
    '''
    if isinstance(self.lemma, str):
      self.lemma = self.lemma.split(" ")

    assert isinstance(self.lemma, list), "lemma not list"

    self.phon_dict = {}

    for lemma in self.lemma:
      assert isinstance(lemma, str), "word not str"
      self.phon_dict[lemma] = self.lexicon.lemmas[lemma]['phons']

  def _phoneme_acceptor_for_hmm_fsa(self):
    """
    phoneme acceptor
    :return list of dict word_pos: letter positions in word
    :return list of list phon_pos: phoneme positions in lemma
          0: phoneme sequence
          1, 2: start end point
          len = 1: no start end point
    :return int num_states:
    :return list edges_phon:
    """
    edges_phon_t = []

    """
    replaces chars with phonemes
    """
    while self.edges:
      edge = self.edges.pop(0)
      if edge[2] != self._SIL and edge[2] != self._EPS:
        phon_current = self.phon_dict[edge[2]]
        for phons in phon_current:
          phon_score = phons['score']  # calculate phon score correctly log space
          edges_phon_t.append([edge[0], edge[1], phons['phon'], phon_score])
      elif edge[2] == self._SIL or edge[2] == self._EPS:
        edges_phon_t.append(edge)  # adds eps and sil edges unchanged
      else:
        assert 1 == 0, "unrecognized phoneme"  # all edges should be handled
    assert len(self.edges) == 0, "Edges left"
    self.edges.extend(edges_phon_t)

    """
    splits word and marks the letters next to a silence
    """
    word_pos = []
    assert isinstance(self.lemma, list), "Lemma not list"
    word_list = []
    word_list.extend(self.lemma)
    while word_list:
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
    edges_t.extend(self.edges)
    phon_pos = []

    edges_t.sort(key=lambda x: x[0])

    while edges_t:
      edge = edges_t.pop(0)  # edge is tuple start node, end node, label, score
      if edge[2] != self._SIL and edge[2] != self._EPS:  # sil and eps ignored
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
    edges_tt = []
    edges_tt.extend(self.edges)
    edges_tt.sort(key=lambda x: x[0])
    self.edges = []

    while edges_tt:
      edge = edges_tt.pop(0)
      if edge[2] != self._SIL and edge[2] != self._EPS:
        phon_seq = edge[2].split(" ")
        for phon_idx, phon_label in enumerate(phon_seq):
          phon_seq_len = len(phon_seq)
          if phon_seq_len == 1:
            start_node = edge[0]
            end_node = edge[1]
            phon_score = edge[3]
            self.edges.append([start_node, end_node, phon_label, phon_score, 'if'])
          elif phon_seq_len > 1:
            if phon_idx == 0:
              start_node = edge[0]
              end_node = self.num_states
              phon_score = edge[3]
              self.edges.append([start_node, end_node, phon_label, phon_score, 'i'])
              self.num_states += 1
            elif phon_idx == phon_seq_len - 1:
              start_node = self.num_states - 1
              end_node = edge[1]
              phon_score = 0.
              self.edges.append([start_node, end_node, phon_label, phon_score, 'f'])
            else:
              start_node = self.num_states - 1
              end_node = self.num_states
              phon_score = 0.
              self.edges.append([start_node, end_node, phon_label, phon_score, ''])
              self.num_states += 1
          else:
            assert 1 == 0, "Something went wrong while expanding phoneme sequence"
      else:
        start_node = edge[0]
        end_node = edge[1]
        phon_label = edge[2]
        phon_score = edge[3]
        self.edges.append([start_node, end_node, phon_label, phon_score, ''])
      self.edges.sort(key=lambda x: x[0])

    self.edges = self._sort_node_num(self.edges)

  def _sort_node_num(self, edges):
    """
    reorders the node numbers: always rising numbers. never 40 -> 11
    uses some kind of sorting algorithm (binarysort, quicksort, ...)
    :param int num_states: number od states / nodes
    :param list edges: list with unordered nodes
    :return list edges: list with ordered nodes
    """
    idx = 0

    while (idx < len(edges)):  # traverse all edges from 0 to num_states
      cur_edge = edges[idx]  # gets the current edge
      cur_edge_start = cur_edge[0]  # with current start
      cur_edge_end = cur_edge[1]  # and end node

      if cur_edge_start > cur_edge_end:  # only something to do if start node number > end node number
        edges_cur_start = self._find_node_edges(cur_edge_start, edges)  # find start node in all edges
        edges_cur_end = self._find_node_edges(cur_edge_end, edges)  # find end node in all edges

        for edge_key in edges_cur_start.keys():  # loop over edge which have the specific node
          edges[edge_key][
            edges_cur_start[edge_key]] = cur_edge_end  # replaces the start node number

        for edge_key in edges_cur_end.keys():  # edge_key: idx from edge in edges
          edges[edge_key][edges_cur_end[edge_key]] = cur_edge_start  # replaces the end node number

        # reset idx: restarts traversing at the beginning of graph
        # swapping may introduce new disorders
        idx = 0

      idx += 1

    return edges

  def _find_node_edges(self, node, edges):
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

  def _triphone_acceptor_for_hmm_fsa(self):
    """
    changes the labels of the edges from phonemes to triphones
    """
    edges_tri = []
    edges_t = []
    edges_t.extend(self.edges)

    while edges_t:
      edge_t = edges_t.pop(0)
      if edge_t[2] == self._SIL or edge_t[2] == self._EPS:
        edges_tri.append(edge_t)
      else:
        prev_edge_t = self._find_prev_next_edge(edge_t, 0, self.edges)
        next_edge_t = self._find_prev_next_edge(edge_t, 1, self.edges)

        label_tri = [prev_edge_t[2], edge_t[2], next_edge_t[2]]

        edge_n = [edge_t[0], edge_t[1], label_tri, edge_t[3], edge_t[4]]
        edges_tri.append(edge_n)

    self.edges = edges_tri

  def _find_prev_next_edge(self, cur_edge, pn_switch, edges):
    """
    find the next/previous edge within the edges list
    :param list cur_edge: current edge
    :param int pn_switch: either previous (0) and next (1) edge
    :param list edges: list of edges
    :return list pn_edge: previous/next edge
    """
    assert pn_switch == 0 or pn_switch == 1, ("Previous/Next switch has wrong value:", pn_switch)

    # finds indexes of previous edges
    prev_edge_cand_idx = [edge_index for edge_index, edge in enumerate(edges)
                          if (cur_edge[pn_switch] == edge[1 - pn_switch])]

    # remove eps and sil edges
    prev_edge_cand_idx_len = len(prev_edge_cand_idx)
    if prev_edge_cand_idx_len > 1:
      for idx in prev_edge_cand_idx:
        assert edges[idx][2] == self._SIL or edges[idx][2] == self._EPS, "Edge found which is not sil or eps"
    else:
      assert prev_edge_cand_idx_len <= 1, ("Too many previous edges found:", prev_edge_cand_idx)

    assert prev_edge_cand_idx_len >= 0, ("Negative edges found. Something went wrong..")

    # sets pn_edge to the previous edge or if sil/eps then empty edge
    if prev_edge_cand_idx_len == 1:
      pn_edge = edges[prev_edge_cand_idx[0]]
    else:
      pn_edge = [None, None, '', None]

    return pn_edge

  def _allophone_state_acceptor_for_hmm_fsa(self):
    """
    the edges which are not sil or eps are split into three allophone states / components
      marked with 0, 1, 2
    """
    num_states_output = self.num_states
    edges_t = []
    edges_t.extend(self.edges)
    edges_output = []

    while edges_t:
      edge_t = edges_t.pop(0)
      if edge_t[2] == self._SIL or edge_t[2] == self._EPS:
        edges_output.append(edge_t)  # adds sil/eps edge unchanged
      else:
        if self.allo_num_states > 1:  # requirement for edges to change
          for state in range(self.allo_num_states):  # loop through all required states
            edge_label = []
            edge_label.extend(edge_t[2])
            edge_label.append(state)
            edge_score = edge_t[3]
            edge_if = edge_t[4]
            if state == 0:  # first state
              edge_start = edge_t[0]
              edge_end = num_states_output
              num_states_output += 1
            elif state == self.allo_num_states - 1:  # last state
              edge_start = num_states_output
              edge_end = edge_t[1]
              num_states_output += 1
            else:  # states in between
              edge_start = num_states_output - 1
              edge_end = num_states_output
            edge_n = [edge_start, edge_end, edge_label, edge_score, edge_if]
            edges_output.append(edge_n)

    edges_output = self._sort_node_num(edges_output)

    self.num_states = num_states_output
    self.edges = edges_output

  def _state_tying_for_hmm_fsa(self):
    """
    idea: take file with mapping char to number and apply to edge labels
    """
    edges_t = []
    edges_t.extend(self.edges)
    edges_orig = self.edges
    error_status = False
    self.edges = []
    self._load_state_tying()

    while (edges_t):
      edge_t = edges_t.pop(0)
      assert len(edge_t) == 5, ("edge length != 5", edge_t)
      label = edge_t[2]
      pos = edge_t[4]

      allo_syntax = self._build_allo_syntax_for_mapping(label, pos)

      if label == self._EPS:
        allo_id_num = '*'
      else:
        try:
          allo_id_num = self.state_tying.allo_map[allo_syntax]
        except:
          print("Error converting label:", label, pos, allo_syntax)
          error_status = True

      if self.label_conversion:
        self.edges.append((edge_t[0], edge_t[1], allo_id_num, edge_t[3]))
      else:
        self.edges.append((edge_t[0], edge_t[1], allo_syntax, edge_t[3]))

      if error_status:
        self.edges = edges_orig

  def _load_state_tying(self, reload = False):
    '''
    loads a state tying map from a file, loads the file and returns its content
    :param stFile: state tying map file (allo_syntax int)
    :return state_tying: variable with state tying mapping
    where:
      statetying.allo_map important
    '''
    from os.path import isfile
    from Log import log
    from LmDataset import StateTying

    if not isinstance(self.state_tying, StateTying):
      reload = True

    if reload:
      print("Loading state tying file:", self.state_tying_name)

      assert isfile(self.state_tying_name), "State tying file does not exists"

      log.initialize(verbosity=[5])
      self.state_tying = StateTying(self.state_tying_name)

      print("Finished state tying mapping:", len(self.state_tying.allo_map), "allos to int")

  def _build_allo_syntax_for_mapping(self, label, pos=''):
    """
    builds a conforming allo syntax for mapping
    :param str or list label: a allo either string or list
    :param str pos: position of allophone within the word
    :return str allo_map: a allo syntax ready for mapping
    """
    assert isinstance(label, str) or isinstance(label,
                                                list), "Something went wrong while building allo syntax for mapping"

    if isinstance(label, str) and label == self._SIL:
      allo_start = "%s{#+#}" % ('[SILENCE]')
    elif isinstance(label, str) and label == self._EPS:
      allo_start = "*"
    else:
      if label[0] == '' and label[2] == '':
        allo_start = "%s{#+#}" % (label[1])
      elif label[0] == '':
        allo_start = "%s{#+%s}" % (label[1], label[2])
      elif label[2] == '':
        allo_start = "%s{%s+#}" % (label[1], label[0])
      else:
        allo_start = "%s{%s+%s}" % (label[1], label[0], label[2])

    allo_middle = ''
    if pos == 'if':
      allo_middle = "@%s@%s" % ('i', 'f')
    elif pos == 'i':
      allo_middle = "@%s" % ('i')
    elif pos == 'f':
      allo_middle = "@%s" % ('f')

    if label == self._SIL:
      allo_end = ".0"
    elif label == self._EPS:
      allo_end = ""
    else:
      allo_end = ".%i" % (label[3])

    allo_map = "%s%s%s" % (allo_start, allo_middle, allo_end)

    return allo_map


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

  _add_nodes(G, nodes)
  _add_edges(G, edges)

  # print(G.source)
  filepath = "./tmp/" + file
  filename = G.render(filename=filepath)
  print("File saved in:", filename)


def _add_nodes(graph, nodes):
  for n in nodes:
    if isinstance(n, tuple):
      graph.node(n[0], **n[1])
    else:
      graph.node(n)
  return graph


def _add_edges(graph, edges):
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
  arg_parser.add_argument("--fsa", type=str, required=True)
  arg_parser.add_argument("--label_seq", type=str, required=True)
  arg_parser.add_argument("--file", type=str)
  arg_parser.set_defaults(file='fsa')
  arg_parser.add_argument("--asg_repetition", type=int)
  arg_parser.set_defaults(asg_repetition=3)
  arg_parser.add_argument("--num_labels", type=int)
  arg_parser.set_defaults(num_labels=265)  # ascii number of labels
  arg_parser.add_argument("--label_conversion_on", dest="label_conversion", action="store_true")
  arg_parser.add_argument("--label_conversion_off", dest="label_conversion", action="store_false")
  arg_parser.set_defaults(label_conversion=None)
  arg_parser.add_argument("--depth", type=int)
  arg_parser.set_defaults(depth=6)
  arg_parser.add_argument("--allo_num_states", type=int)
  arg_parser.set_defaults(allo_num_states=3)
  arg_parser.add_argument("--lexicon", type=str)
  arg_parser.set_defaults(lexicon='recog.150k.final.lex.gz')
  arg_parser.add_argument("--state_tying", type=str)
  arg_parser.set_defaults(state_tying='state-tying.txt')
  arg_parser.add_argument("--single_state_on", dest="single_state", action="store_true")
  arg_parser.add_argument("--single_state_off", dest="single_state", action="store_false")
  arg_parser.set_defaults(single_state=False)
  args = arg_parser.parse_args()

  fsa_gen = Fsa()

  fsa_gen.set_lemma(args.label_seq)

  fsa_gen.set_fsa_type(args.fsa)

  fsa_gen.set_filename(args.file)

  fsa_gen.set_params(asg_repetition=args.asg_repetition,
                     num_labels=args.num_labels,
                     label_conversion=args.label_conversion,
                     depth=args.depth,
                     allo_num_states=args.allo_num_states,
                     lexicon_name=args.lexicon,
                     state_tying_name=args.state_tying,
                     single_state=args.single_state)

  fsa_gen.set_lexicon(args.lexicon)

  fsa_gen.run()

  fsa_to_dot_format(file=fsa_gen.filename, num_states=fsa_gen.num_states, edges=fsa_gen.edges)

  if (fsa_gen.single_state == True):
    fsa_gen.reduce_node_num()
    fsa_to_dot_format(file=fsa_gen.filename + "_single_state", num_states=1, edges=fsa_gen.edges_single_state)


if __name__ == "__main__":
  import time

  start_time = time.time()

  main()

  print("Total time:", time.time() - start_time, "seconds")
