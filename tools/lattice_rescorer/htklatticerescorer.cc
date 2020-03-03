/* Implement main rescoring algorithm.

Code adapted from rwthlm:
  http://www-i6.informatik.rwth-aachen.de/~sundermeyer/rwthlm.html
==================================================================*/

#include <cmath>
#include <algorithm>
#include <functional>
#include <sstream>
#include <boost/algorithm/string/trim.hpp>
#include <boost/range/adaptor/reversed.hpp>
#include "file.h"
#include "htklatticerescorer.h"
#include <fstream>
#include <sstream>

std::pair<size_t, size_t> HtkLatticeRescorer::ParseField(
    const std::string &line,
    const std::string &field,
    int *const result) const {
  std::string s;
  const auto match = ParseField(line, field, &s);
  std::istringstream(s) >> *result;
  return match;
}

std::pair<size_t, size_t> HtkLatticeRescorer::ParseField(
    const std::string &line,
    const std::string &field,
    float *const result) const {
  std::string s;
  const auto match = ParseField(line, field, &s);
  std::istringstream(s) >> *result;
  return match;
}

std::pair<size_t, size_t> HtkLatticeRescorer::ParseField(
    const std::string &line,
    const std::string &field,
    std::string *const result) const {
  size_t start = line.find(field + '='), end;
  assert(start != std::string::npos);
  assert(line.find(field + '=', start + 1) == std::string::npos);
  const size_t start0 = start;
  start += field.length() + 1;
  if (line[start] == '"') {
      end = line.find('"', ++start + 1);
  } else {
      end = std::min(line.find(' ', start + 1), line.find('\t', start + 1));
      if (end == std::string::npos)
          end = line.size();
  }
  if (result != nullptr)
    *result = line.substr(start, end - start);
  return std::make_pair(start0, end - start0 + 1);
}

void HtkLatticeRescorer::ParseLine(const std::string &line, int *num_links) {
  // log base e is default in HTK SLF
  assert(!boost::algorithm::starts_with(line, "base") ||
         boost::algorithm::starts_with(line, "base=2.718"));
  if (line[0] == 'N' || line[0] == 'L') {
    int n;
    std::istringstream tokenizer;
    tokenizer.str(line);
    while (!tokenizer.eof()) {
      std::string token;
      tokenizer >> token;
      if (token[0] == 'N') {
        ParseField(token, token[1] == '=' ? "N" : "NODES", &n);
        successor_links_.resize(n);
        hypotheses_.resize(n);
      } else {
        ParseField(line, line[1] == '=' ? "L" : "LINKS", &n);
        links_.resize(n);
      }
    }
  } else if (line[0] == 'I') {
    Node node;
    ParseField(line, "I", &node.id);
    float time;
    ParseField(line, "t", &time);
    node.time = static_cast<int>(floor(100. * time + .5));
    node.look_ahead_score = std::numeric_limits<float>::infinity();
    nodes_.push_back(node);
    nodes_by_time_[node.time].push_back(node.id);
  }
  else if (line[0] == 'J') {
    int id;
    ++*num_links;
    Link link;
    ParseField(line, "J", &id);
    ParseField(line, "S", &link.from);
    ParseField(line, "E", &link.to);
    ParseField(line, "l", &link.lm_score);
    ParseField(line, "a", &link.am_score);
    ParseField(line, "v", &link.pronunciation);

    if (clear_initial_links_ && link.from == 0) {
      link.lm_score = 0.;
      link.am_score = 0.;
    }
    // We always have progress in time, except for sentence-end arcs or
    // initial arcs (whose scores may have just been set to zero explicitly).
    assert(nodes_[link.from].time < nodes_[link.to].time ||
      nodes_[link.from].time == nodes_[link.to].time &&
      link.am_score == 0. && (link.lm_score != 0. ||
      nodes_[link.from].time == 0));
    link.lm_score = -link.lm_score;  // our scores are positive!
    link.am_score = -link.am_score;
    std::string word;
    ParseField(line, "W", &word);
    link.word = vocabulary_->GetIndex(word);
    if (link.word == unk_index_)
      oov_by_link_[id] = word;
    links_[id] = link;
    successor_links_[link.from].push_back(id);
  }
}

void HtkLatticeRescorer::Reset() {
  Hypothesis hypothesis;
  hypothesis.score = 0.;
  hypothesis.traceback_id = 0;
  if (is_dependent_)
    nodes_.clear();
  if (nodes_.empty()) {
    InitStateVars(session_, state_vars_assign_ops_, state_vars_assign_inputs_, state_vars_size_);
    TF_ExtractState(session_, state_vars_, &hypothesis.state);
  } else {
    // use best ending state of previous lattice
    const int final_node_id = sorted_nodes_.back()->id;
    assert(successor_links_[final_node_id].empty());
    auto &node_hypotheses = hypotheses_[final_node_id];
    assert(!node_hypotheses.empty());
    while (node_hypotheses.size() > 1)
      node_hypotheses.pop();
    hypothesis.state = node_hypotheses.top().state;
  }

  nodes_.clear();
  single_best_.clear();
  nodes_by_time_.clear();
  sorted_nodes_.clear();
  links_.clear();
  successor_links_.clear();
  hypotheses_.clear();
  hypotheses_.resize(1);
  hypotheses_[0].push(hypothesis);  // assumption: node 0 is start node
  best_score_by_time_.clear();
  traceback_.clear();
  oov_by_link_.clear();
  topological_order_.clear();
  AddTraceback(-1,  // illegal link ID
               vocabulary_->sb_index(),
               0,  // predecessor traceback ID
               0,  // current node id
               0.);  // initial cost
}

void HtkLatticeRescorer::SortTopologically() {
  sorted_nodes_.clear();
  std::unordered_set<int> visited;
  SortTopologicallyHelper(0, &visited);
  assert(visited.size() == nodes_.size());
  std::reverse(sorted_nodes_.begin(), sorted_nodes_.end());
}

void HtkLatticeRescorer::SortTopologicallyHelper(
    const int node_id,
    std::unordered_set<int> *visited) {
  if (visited->count(node_id))
    return;
  visited->insert(node_id);

  for (const int link_id : successor_links_[node_id])
    SortTopologicallyHelper(links_[link_id].to, visited);
  sorted_nodes_.push_back(&nodes_[node_id]);
}

void HtkLatticeRescorer::ReadLattice(const std::string &file_name) {
  // read lattice from htk file
  ReadableFile file(file_name);
  std::string line, token;
  std::istringstream tokenizer, parser;

  int num_links = 0;
  while (file.GetLine(&line)) {
    boost::trim(line);
    ParseLine(line, &num_links);
  }
  assert(nodes_.size() == successor_links_.size());
  assert(links_.size() == num_links);

  // Terminal links with non-zero LM score should be end-of-sentence links.
  for (Link &link : links_) {
    const auto &successors = successor_links_[link.to];
    if (set_sb_last_links_ && successors.empty() ||
        set_sb_next_to_last_links_ && !successors.empty() &&
        successor_links_[links_[successors[0]].to].empty()) {
      assert(link.word == unk_index_ || link.word == vocabulary_->sb_index());
      assert(successors.empty() || successors.size() == 1);
      link.word = vocabulary_->sb_index();
    }
  }
  // sort nodes by time stamp
  SortTopologically();
  std::stable_sort(sorted_nodes_.begin(), sorted_nodes_.end(),
      [](const Node *const node1, const Node *const node2) {
        return *node1 < *node2;
      });
  topological_order_.resize(nodes_.size(), -1);
  int i = 0;
  for (const Node *const node : sorted_nodes_)
    topological_order_[node->id] = i++;

  ComputeLookAheadScores();
}

void HtkLatticeRescorer::ComputeLookAheadScores() {
  for (Node *node : boost::adaptors::reverse(sorted_nodes_)) {
    float look_ahead_score = 0.;
    // if there are no successors the look ahead score is zero
    for (const int link_id : successor_links_[node->id]) {
      const Link &link = links_[link_id];
      const float score = look_ahead_lm_scale_ * link.lm_score +
                         link.am_score + nodes_[link.to].look_ahead_score;
      if (semiring_ == kTropical) {
        if (look_ahead_score == 0. || score < look_ahead_score)
          look_ahead_score = score;
      } else if (semiring_ == kLog) {
        assert(semiring_ == kLog);
        if (look_ahead_score == 0.)
          look_ahead_score = score;
        else
          look_ahead_score = ScaledLogAdd(look_ahead_lm_scale_,
                                          look_ahead_score,
                                          score);
      } else {
        assert(semiring_ == kNone);
      }
    }
    node->look_ahead_score = look_ahead_score;
  }
  // update initial hypothesis
  traceback_[0].score = nodes_[0].look_ahead_score;
  Hypothesis hypothesis = hypotheses_[0].top();
  hypotheses_[0].pop();
  hypothesis.score = nodes_[0].look_ahead_score;
  hypotheses_[0].push(hypothesis);
}

void HtkLatticeRescorer::Prune(const int time) {
  size_t num_before = 0,
         num_after = 0,
         min_num_before = std::numeric_limits<size_t>::max(),
         max_num_before = std::numeric_limits<size_t>::min(),
         min_num_after = std::numeric_limits<size_t>::max(),
         max_num_after = std::numeric_limits<size_t>::min();
  const float threshold = best_score_by_time_[time] + pruning_threshold_;
  for (const int node_id : nodes_by_time_[time]) {
    std::priority_queue<Hypothesis> &node_hypotheses = hypotheses_[node_id];
    num_before += node_hypotheses.size();
    min_num_before = std::min(min_num_before, node_hypotheses.size());
    max_num_before = std::max(max_num_before, node_hypotheses.size());

    // beam pruning
    std::unordered_map<size_t, Hypothesis> recombined_hypotheses;
    assert(!node_hypotheses.empty());
    while (!node_hypotheses.empty()) {
      const Hypothesis &hypothesis = node_hypotheses.top();
      // ensure at least one hypothesis per node survives
      if (hypothesis.score <= threshold || node_hypotheses.size() == 1) {
        // dynamic programming recombination
        Hypothesis &best = recombined_hypotheses[Hash(hypothesis.traceback_id)];
        if (best.score == 0. || best.score > hypothesis.score)
          best = hypothesis;
      }
      node_hypotheses.pop();
    }
    // write back surviving hypotheses
    for (const std::pair<size_t, Hypothesis> &pair : recombined_hypotheses)
      node_hypotheses.push(pair.second);
    // cardinality pruning
    while (node_hypotheses.size() > pruning_limit_)
      node_hypotheses.pop();
    min_num_after = std::min(min_num_after, node_hypotheses.size());
    max_num_after = std::max(max_num_after, node_hypotheses.size());
    num_after += node_hypotheses.size();
  }
  std::cout << "t=" << std::setw(5) << std::right << time <<
               " #hyps/#nodes: " << std::setw(5) << num_before << "/" <<
               std::setw(3) << nodes_by_time_[time].size() << " (min: " <<
               std::setw(3) << min_num_before << ", max: " << std::setw(4) <<
               max_num_before << ") -> " << std::setw(4) << num_after <<
               " (min: " << std::setw(2) << min_num_after << ", max: " <<
               std::setw(2) << max_num_after << ")" << std::endl;
}

void HtkLatticeRescorer::RescoreLattice() {
  int last_time = sorted_nodes_[0]->time;

  for (const Node *const node : sorted_nodes_) {
    if (node->id == sorted_nodes_.back()->id)
      break;
    if (node->time > last_time) {
      Prune(node->time);
      last_time = node->time;
    }
    std::priority_queue<Hypothesis> &node_hypotheses = hypotheses_[node->id];
    while (!node_hypotheses.empty()) {
      const Hypothesis &hypothesis = node_hypotheses.top();
      for (const int link_id : successor_links_[node->id]) {
        // Sets the current value of state variables in LSTM cell
        TF_SetState(session_, state_vars_assign_ops_, state_vars_assign_inputs_, state_vars_size_, hypothesis.state);
        const Link &link = links_[link_id];

        Hypothesis new_hypothesis;
        new_hypothesis.score = hypothesis.score + link.am_score -
        node->look_ahead_score + nodes_[link.to].look_ahead_score;
        const int history_word = GetHistoryWord(hypothesis.traceback_id);
        // links with lm_score == 0. will not be evaluated
        if (link.lm_score == 0.) {
          new_hypothesis.state = hypothesis.state;
        } else {
          const int target_word = link.word;
          const tensorflow::int64 word_index = hypothesis.history_word_index;
          new_hypothesis.score -= log((1. - nn_lambda_) * exp(-link.lm_score) +
            nn_lambda_ * exp(TF_ComputeLogProbability(session_, tensor_names_, history_word, target_word, word_index)) /
            (link.word == unk_index_ ? num_oov_words_ + 1. : 1.)) * lm_scale_;
          new_hypothesis.history_word_index = hypothesis.history_word_index+1;
          // Extracts the new value of state variables in LSTM cell and store them in the corresponding hypothesis
          TF_ExtractState(session_, state_vars_, &new_hypothesis.state);
        }
        // This must be done, independent of whether we have an LM score or not!
        float &best_score = best_score_by_time_[nodes_[link.to].time];
        if (best_score == 0. || new_hypothesis.score < best_score)
          best_score = new_hypothesis.score;
        new_hypothesis.traceback_id = AddTraceback(
          link_id,
          link.lm_score == 0. ? history_word : link.word,
          hypothesis.traceback_id,
          link.to,
          new_hypothesis.score);
        hypotheses_[link.to].push(new_hypothesis);
      }
      node_hypotheses.pop();
    }
  }
  TraceBack();
}

void HtkLatticeRescorer::TraceBack() {
  std::vector<int> ordered_traceback(traceback_.size());
  ordered_traceback.resize(traceback_.size());
  for (size_t i = 0; i < ordered_traceback.size(); ++i)
    ordered_traceback[i] = i;
  std::sort(ordered_traceback.begin(),
            ordered_traceback.end(),
            [this](const int id1, const int id2) {
              return topological_order_[GetToNodeID(id1)] >
                     topological_order_[GetToNodeID(id2)] ||
                     topological_order_[GetToNodeID(id1)] ==
                     topological_order_[GetToNodeID(id2)] &&
                     GetScore(id1) < GetScore(id2);
            });
  assert(single_best_.empty());
  bool fill_single_best = true;
  std::unordered_set<int> visited;
  for (Link &link : links_)
    link.lm_score = -std::numeric_limits<float>::infinity();
  for (int traceback_id : ordered_traceback) {
    while (traceback_id > 0 && !visited.count(traceback_id)) {
      Link &link = links_[GetLinkID(traceback_id)];
      float &lm_score = link.lm_score;
      if (lm_score < 0.)
        lm_score = GetLinkLmScore(traceback_id);
      if (fill_single_best) {
        const int link_id = GetLinkID(traceback_id);
        single_best_.push_back(link_id);
      }
      visited.insert(traceback_id);
      traceback_id = GetPredecessorID(traceback_id);
    }
    fill_single_best = false;
  }
  std::cout << "single best: " << std::setw(10) << std::fixed <<
               std::setprecision(5) <<
               traceback_[ordered_traceback[0]].score << '\n';
  for (const int link_id : boost::adaptors::reverse(single_best_)) {
    const int word = links_[link_id].word;
    std::cout << "\tW=" << std::left << std::setw(20) <<
                 (word == unk_index_ ? oov_by_link_[link_id] :
                 vocabulary_->GetWord(word)) << "\tJ=" <<
                 std::setw(6) << link_id << "\tl=" << std::fixed <<
                 std::setw(8) << std::setprecision(5) <<
                 -links_[link_id].lm_score << std::endl;
  }
  assert(links_[single_best_[0]].word == vocabulary_->sb_index());
}

void HtkLatticeRescorer::WriteLattice(const std::string &file_name) {
  switch (output_format_) {
  case kCtm:
    WriteCtm(file_name);
    break;
  case kLattice:
    WriteHtkLattice(file_name);
    break;
  case kExpandedLattice:
    WriteExpandedHtkLattice(file_name);
    break;
  }
}

void HtkLatticeRescorer::WriteCtm(const std::string &file_name) {
  const std::string file_name_prefix(ExtendedFileName(file_name, ""));
  WritableFile out_file(ExtendedFileName(file_name, ".ctm"));

  // leave out final token (which is <sb> and should not occur in a ctm file)
  for (auto it = single_best_.rbegin(); it + 1 != single_best_.rend(); ++it) {
    const int link_id = *it;
    if (links_[link_id].lm_score < epsilon_)  // ignore non-speech events
      continue;
    const int start_time = nodes_[links_[link_id].from].time,
              end_time = nodes_[links_[link_id].to].time,
              word = links_[link_id].word;
    const double duration = (end_time - start_time) / 100.,
                 start = start_time / 100.;
    out_file << file_name_prefix << " 1 " << std::fixed << std::setprecision(3) <<
                start << ' ' << duration << ' ' << (word == unk_index_ ?
                oov_by_link_[link_id] : vocabulary_->GetWord(word)) << '\n';
  }
}

void HtkLatticeRescorer::WriteHtkLattice(const std::string &file_name) {
  // write back file
  ReadableFile in_file(file_name);
  WritableFile out_file(ExtendedFileName(file_name, ".rescored"));

  std::string line, token;
  while (in_file.GetLine(&line)) {
    boost::trim(line);
    if (line[0] == 'J') {
      int link_id;
      ParseField(line, "J", &link_id);

      const Link &link = links_[link_id];
      if (clear_initial_links_ && link.from == 0) {
        const auto match = ParseField(line, "a");
        line.replace(match.first, match.second, "a=0.0 ");
      }

      const auto match = ParseField(line, "l");
      out_file << line.replace(
          match.first,  // position
          match.second,  // length
          "l=" + std::to_string(-link.lm_score)) << '\n';
    } else {
      out_file << line << '\n';
    }
  }
}

void HtkLatticeRescorer::WriteExpandedHtkLattice(const std::string &file_name) {
  // Build some helper data structures.
  std::unordered_set<int> pruned_traceback_ids;
  for (size_t i = 0; i < traceback_.size(); ++i)
    pruned_traceback_ids.insert(i);
  // A pruned traceback node has no successors and is not a final node.
  for (size_t i = 0; i < traceback_.size(); ++i)
    pruned_traceback_ids.erase(GetPredecessorID(i));
  const int final_node_id = sorted_nodes_.back()->id,
            final_time = sorted_nodes_.back()->time;
  assert(successor_links_[final_node_id].empty());
  std::unordered_set<int> final_traceback_ids;
  for (const int traceback_id : pruned_traceback_ids) {
    const int to_node_id = GetToNodeID(traceback_id);
    if (to_node_id == final_node_id) {
      final_traceback_ids.insert(traceback_id);
    } else {
      // verify: original lattice only has a single final node
      assert(nodes_[to_node_id].time < final_time);
    }
  }
  for (const int traceback_id : final_traceback_ids)
    pruned_traceback_ids.erase(traceback_id);
  // recombine pruned hypothesis with worst surviving hypothesis
  std::function<bool (const int, const int)> score_comparator =
      [this] (const int traceback_id1, const int traceback_id2) {
        // disable look ahead for recombination
        return GetScore(traceback_id1) -
               nodes_[GetToNodeID(traceback_id1)].look_ahead_score >
               GetScore(traceback_id2) -
               nodes_[GetToNodeID(traceback_id2)].look_ahead_score;
      };
  std::vector<std::vector<int>> traceback_by_node(nodes_.size());
  for (size_t i = 0; i < traceback_.size(); ++i)
    if (!pruned_traceback_ids.count(i))
      traceback_by_node[GetToNodeID(i)].push_back(i);
  for (auto &traceback_ids : traceback_by_node)
    std::sort(traceback_ids.begin(), traceback_ids.end(), score_comparator);

  // HTK SLF header
  WritableFile out_file(ExtendedFileName(file_name, ".rescored"));
  out_file << "VERSION=1.0\nUTTERANCE=" <<
              FileNameWithoutExtension(file_name) << "\nlmscale=" <<
              lm_scale_ << '\n';
  out_file << "NODES=" << traceback_.size() - pruned_traceback_ids.size() + 1 <<
              "\nLINKS=" << traceback_.size() - 1 +
              final_traceback_ids.size() << '\n';

  // node definitions
  int index = 0;
  std::vector<int> new_node_id(traceback_.size(), -1);
  for (size_t i = 0; i < traceback_.size(); ++i) {
    if (!pruned_traceback_ids.count(i)) {
      new_node_id[i] = index++;
      const int time = GetTime(i);
      out_file << "I=" << new_node_id[i] << " t=" <<
                          (time / 100) << '.' << std::setfill('0') <<
                          std::setw(2) << (time % 100) << '\n';
    }
  }
  // pruned nodes: use worst possible surviving traceback ID for recombination
  for (int traceback_id : pruned_traceback_ids) {
    const auto &traceback_ids = traceback_by_node[GetToNodeID(traceback_id)];
    const auto it = std::lower_bound(traceback_ids.begin(),
                                     traceback_ids.end(),
                                     traceback_id,
                                     score_comparator);
    assert(it != traceback_ids.end());
    new_node_id[traceback_id] = new_node_id[*it];
  }

  // new final node
  const int new_final_node = index;
  out_file << "I=" << new_final_node << " t=" << (final_time / 100) << '.' <<
              std::setfill('0') << std::setw(2) << (final_time % 100) << '\n';

  // link definitions
  index = 0;
  for (size_t i = 1; i < traceback_.size(); ++i) {
    const int link_id = GetLinkID(i);
    const Link &link = links_[link_id];
    out_file << "J=" << index++ << " S=" <<
                new_node_id[GetPredecessorID(i)] << " E=" <<
                new_node_id[i] << " W=\"" <<
                (oov_by_link_.count(link_id) ? oov_by_link_[link_id] :
                vocabulary_->GetWord(link.word)) << "\" v=" <<
                link.pronunciation << " a=" << -link.am_score << " l=" <<
                -GetLinkLmScore(i) << '\n';
  }
  // epsilon links to new final node
  for (int traceback_id : final_traceback_ids) {
    out_file << "J=" << index++ << " S=" << new_node_id[traceback_id] <<
                " E=" << new_final_node << " W=\"!NULL\" v=-1 a=-0 l=-0\n";
  }
}

// Initialize the state variables with zeros in the LSTM cell
void HtkLatticeRescorer::InitStateVars(tensorflow::Session* session,
                                       const std::vector<std::string> state_vars_assign_ops,
                                       const std::vector<std::string> state_vars_assign_inputs,
                                       const std::vector<int> state_vars_size) {
  assert(state_vars_assign_ops.size()==state_vars_assign_inputs.size());
  std::vector<tensorflow::Tensor> outputs;
  std::string var_assign_op_name, var_assign_op_input_name;
  for (int i=0; i < state_vars_assign_ops.size(); i++) {
    tensorflow::Input::Initializer zeros(0.0f, tensorflow::TensorShape({1, state_vars_size[i]}));
    var_assign_op_name = state_vars_assign_ops[i];
    var_assign_op_input_name = state_vars_assign_inputs[i];
    // Please check the following link if you don't know how to use session->Run(),
    // https://haosdent.gitbooks.io/tensorflow-document/content/api_docs/cc/ClassSession.html
    TF_CHECK_OK(session->Run({{var_assign_op_input_name, zeros.tensor}}, {var_assign_op_name}, {}, &outputs));
  }
}

// Set the states variables in LSTM cell
void HtkLatticeRescorer::TF_SetState(tensorflow::Session *session, const std::vector<std::string> state_vars_assign_ops,
                                     const std::vector<std::string> state_vars_assign_inputs,
                                     const std::vector<int> state_vars_size, const State &state) {
  assert(state_vars_assign_ops.size() == state_vars_assign_inputs.size());
  std::vector<tensorflow::Tensor> outputs;
  std::string var_assign_op_name, var_assign_op_input_name;
  for (int i=0; i < state_vars_assign_ops.size(); i++) {
    tensorflow::Input::Initializer keep_state_var(0.0f, tensorflow::TensorShape({1, state_vars_size[i]}));
    var_assign_op_name = state_vars_assign_ops[i];
    var_assign_op_input_name = state_vars_assign_inputs[i];
    std::copy(state.states[i].begin(), state.states[i].begin() + state.states[i].size(),
              keep_state_var.tensor.flat<float>().data());
    TF_CHECK_OK(session->Run({{var_assign_op_input_name, keep_state_var.tensor}}, {var_assign_op_name}, {}, &outputs));
  }
}

// Extract the current states of LSTM and store them in the hypothesis
void HtkLatticeRescorer::TF_ExtractState(tensorflow::Session* session,
                                         const std::vector<std::string> state_vars, State *state) const {
  std::vector<float> lstm_state;
  std::vector<tensorflow::Tensor> outputs;
  for (int i=0; i < state_vars.size(); i++) {
    TF_CHECK_OK(session->Run({}, {state_vars[i]}, {}, &outputs));
    float *keep_state_var_ptr = outputs[0].flat<float>().data(); // access the value of the Tensor "score"
    lstm_state.resize(outputs[0].dim_size(1));
    std::copy_n(keep_state_var_ptr, lstm_state.size(), lstm_state.begin());
    state->states.push_back(lstm_state);
  }
}

// Compute log p(w|h)
float HtkLatticeRescorer::TF_ComputeLogProbability(tensorflow::Session* session,
                                                   const std::vector<std::string> tensor_names,
                                                   const int history_word, const int target_word,
                                                   const tensorflow::int64 word_index) {
  tensorflow::Tensor in_word = tensorflow::Tensor(tensorflow::DT_INT32, tensorflow::TensorShape({1, 1}));
  in_word.scalar<int>()() = history_word;
  tensorflow::Tensor delayed_dim0_size = tensorflow::Tensor(tensorflow::DT_INT32, tensorflow::TensorShape({1}));
  delayed_dim0_size.scalar<int>()() = 1;
  tensorflow::Tensor epoch_step = tensorflow::Tensor(tensorflow::DT_INT64, tensorflow::TensorShape({}));
  epoch_step.scalar<tensorflow::int64>()() = word_index;
  std::vector<tensorflow::Tensor> outputs;
  std::vector<std::pair<std::string, tensorflow::Tensor>> inputs;
  inputs = {{tensor_names[0], delayed_dim0_size}, {tensor_names[1], in_word}, {tensor_names[2], epoch_step}};
  TF_CHECK_OK(session->Run(inputs,{tensor_names[3]}, {tensor_names[4]}, &outputs));
  tensorflow::Tensor probs = outputs[0];
  float *probs_ptr = probs.flat<float>().data();  // access the value of the Tensor "score"
  std::vector<float> probs_vec;
  probs_vec.resize(probs.dim_size(2));
  std::copy(probs_ptr, probs_ptr + probs_vec.size(), probs_vec.begin());
  return log(probs_vec[target_word]);
}
