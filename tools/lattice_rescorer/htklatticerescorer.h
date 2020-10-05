/*  Code adapted from rwthlm:

  http://www-i6.informatik.rwth-aachen.de/~sundermeyer/rwthlm.html
====================================================================*/

#pragma once
#include <algorithm>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <boost/functional/hash.hpp>
#include "rescorer.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/c/c_api.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "state_ops.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/cc/framework/ops.h"

class HtkLatticeRescorer : public Rescorer {
public:
  enum LookAheadSemiring {
    kTropical, kLog, kNone
  };

  enum OutputFormat {
    kCtm, kLattice, kExpandedLattice
  };

  HtkLatticeRescorer(const ConstVocabularyPointer &vocabulary,
                     tensorflow::Session* session,
                     const std::vector<std::string> state_vars,
                     const std::vector<std::string> state_vars_assign_ops,
                     const std::vector<std::string> state_vars_assign_inputs,
                     const std::vector<int> state_vars_size,
                     const std::vector<std::string> tensor_names,
                     const OutputFormat output_format,
                     const int num_oov_words,
                     const float nn_lambda,
                     const LookAheadSemiring semiring,
                     const float look_ahead_lm_scale,
                     const float lm_scale,
                     const float pruning_threshold,
                     const size_t pruning_limit,
                     const int dp_order,
                     const bool is_dependent,
                     const bool clear_initial_links,
                     const bool set_sb_next_to_last_links,
                     const bool set_sb_last_links)
      : Rescorer(vocabulary, num_oov_words, nn_lambda),
        unk_index_(vocabulary_->HasUnk() ?
                   vocabulary_->GetIndex(vocabulary_->unk()) : -1),
        session_(session),
        state_vars_(state_vars),
        state_vars_assign_ops_(state_vars_assign_ops),
        state_vars_assign_inputs_(state_vars_assign_inputs),
        state_vars_size_(state_vars_size),
        tensor_names_(tensor_names),
        output_format_(output_format),
        pruning_limit_(pruning_limit),
        semiring_(semiring),
        look_ahead_lm_scale_(look_ahead_lm_scale),
        lm_scale_(lm_scale),
        pruning_threshold_(pruning_threshold),
        epsilon_(1e-8),
        dp_order_(dp_order),
        is_dependent_(is_dependent),
        clear_initial_links_(clear_initial_links),
        set_sb_next_to_last_links_(set_sb_next_to_last_links),
        set_sb_last_links_(set_sb_last_links) {
  }

  ~HtkLatticeRescorer() {
  }

  virtual void ReadLattice(const std::string &file_name);
  virtual void RescoreLattice();
  virtual void WriteLattice(const std::string &file_name);

private:
  struct Node {
    int id, time;
    float look_ahead_score;
    bool operator<(const Node &other) const {
      return time < other.time;
    };
  };

  struct Link {
    int from, to, word, pronunciation;
    float lm_score, am_score;
  };

  struct Hypothesis {
    Hypothesis() {
      traceback_id = 0;
      score = 0.;
      history_word_index = 0;
    }
    bool operator<(const Hypothesis &other) const {
      return score < other.score;
    }
    State state;
    size_t traceback_id;
    float score;
    tensorflow::int64 history_word_index;
  };

  struct Trace {
    Trace(const int link_id,
          const int history_word,
          const int predecessor_traceback_id,
          const int to_node_id,
          const float score)
        : link_id(link_id),
          history_word(history_word),
          to_node_id(to_node_id),
          predecessor_traceback_id(predecessor_traceback_id),
          score(score) {
    }
    int link_id, to_node_id, predecessor_traceback_id, history_word;
    float score;
  };

  std::pair<size_t, size_t> ParseField(
      const std::string &line,
      const std::string &field,
      int *const result) const;
  std::pair<size_t, size_t> ParseField(
      const std::string &line,
      const std::string &field,
      float *const result) const;
  std::pair<size_t, size_t> ParseField(
      const std::string &line,
      const std::string &field,
      std::string *const result = nullptr) const;
  void ParseLine(const std::string &line, int *num_links);

  int AddTraceback(const int link_id,
                   const int history_word,
                   const int predecessor_traceback_id,
                   const int to_node_id,
                   const float score) {
    traceback_.push_back(Trace(link_id,
                               history_word,
                               predecessor_traceback_id,
                               to_node_id,
                               score));
    return traceback_.size() - 1;
  }

  int GetHistoryWord(const int traceback_id) const {
    return traceback_[traceback_id].history_word;
  }

  int GetTime(const int traceback_id) const {
    return nodes_[GetToNodeID(traceback_id)].time;
  }

  int GetToNodeID(const int traceback_id) const {
    return traceback_[traceback_id].to_node_id;
  }

  int GetLinkID(const int traceback_id) const {
    const int result = traceback_[traceback_id].link_id;
    return result;
  }

  float GetScore(const int traceback_id) const {
    return traceback_[traceback_id].score;
  }

  float GetLinkLmScore(const int traceback_id) const {
    const Link &link = links_[traceback_[traceback_id].link_id];
    const float from_look_ahead_score = nodes_[link.from].look_ahead_score,
               to_look_ahead_score = nodes_[link.to].look_ahead_score;
    return (GetScore(traceback_id) - GetScore(GetPredecessorID(traceback_id)) -
            link.am_score - to_look_ahead_score + from_look_ahead_score) / lm_scale_;
  }

  int GetPredecessorID(const int traceback_id) const {
    return traceback_[traceback_id].predecessor_traceback_id;
  }

  size_t Hash(int traceback_id) const {
    size_t hash = 0;
    for (int i = 0; i < dp_order_; ++i) {
      // stop at beginning-of-sentence or "real" word
      while (GetLinkID(traceback_id) >= 0 &&
             links_[GetLinkID(traceback_id)].lm_score == 0.)
        traceback_id = GetPredecessorID(traceback_id);
      boost::hash_combine(hash,
                          static_cast<size_t>(GetHistoryWord(traceback_id)));
      traceback_id = GetPredecessorID(traceback_id);
    }
    return hash;
  }

  float ScaledLogAdd(const float scale, float x, float y) {
    if (y >= std::numeric_limits<float>::max())
      return x;
    if (x >= std::numeric_limits<float>::max())
      return y;
    const float inverted_scale = 1. / scale;
    x *= inverted_scale;
    y *= inverted_scale;
    const float min = std::min(x, y);
    return scale * (min - LogOnePlusX(exp(min - std::max(x, y))));
  }

  float LogOnePlusX(double x) {
    if (x <= -1.) {
      assert(false);
      return -1.;
    }
    if (fabs(x) > 1e-4)
      return log(1.0 + x);
    return (1. - 0.5 * x) * x;
  }

  void SortTopologically();
  void SortTopologicallyHelper(const int node_id,
                               std::unordered_set<int> *visited);
  void ComputeLookAheadScores();
  void Reset();
  void Prune(const int time);
  void TraceBack();
  void TraceBackCtm();
  void TraceBackLattice();
  void TraceBackExpandedLattice();
  void WriteCtm(const std::string &file_name);
  void WriteHtkLattice(const std::string &file_name);
  void WriteExpandedHtkLattice(const std::string &file_name);
  // Initialize the state variables in LSTM cell with zeros
  // state_vars_assign_ops: the names of assignment operation nodes of state variables in LSTM cell.
  //   These names can be found is .metatxt
  // state_vars_assign_inputs: the inputs of state variables assignment,
  //   each includes the name of a state variable and the name of
  void InitStateVars(tensorflow::Session* session, const std::vector<std::string> state_vars_assign_ops,
                     const std::vector<std::string> state_vars_assign_inputs, const std::vector<int> state_vars_size);
  // Set the state variables in LSTM cell
  // state_vars: the names of state variables in LSTM cell
  // state: the values of state variables of a hypothesis
  void TF_SetState(tensorflow::Session* session, const std::vector<std::string> state_vars_assign_ops,
                   const std::vector<std::string> state_vars_assign_inputs, const std::vector<int> state_vars_size,
                   const State &state);
  // Extract the current value of state variables and store them in the corresponding hypothesis.state
  // state_vars: the names of state variables in LSTM cell
  // state: a pointer to new_hypothesis.state, to store the current values of the new hypothesis
  void TF_ExtractState(tensorflow::Session* session, const std::vector<std::string> state_vars, State *state) const;
  // Compute P(w|h)
  // tensor_names:
  //   the names of tensors needed for feeding and fetching, please check the text file example/tensor_names_list
  // history_word: the index of the word fed into the graph when doing inference
  // target_word: the index of the word to be predicted in the next time frame
  // word_index: the position of the history_word in the sentence,
  //            e.g a sentence <sb> a b c <sb>
  //                word_index of <sb> = 0
  //                word index of 'a' = 1
  //                work index of 'b' = 2
  float TF_ComputeLogProbability(tensorflow::Session* session, const std::vector<std::string> tensor_names,
                                 const int history_word, const int target_word, const tensorflow::int64 word_index);
  const int unk_index_, dp_order_;
  const size_t pruning_limit_;
  const LookAheadSemiring semiring_;
  const OutputFormat output_format_;
  const float look_ahead_lm_scale_, lm_scale_, pruning_threshold_, epsilon_;
  const bool is_dependent_,
             clear_initial_links_,
             set_sb_next_to_last_links_,
             set_sb_last_links_;
  std::vector<int> single_best_, topological_order_, state_vars_size_;
  std::unordered_map<int, std::string> oov_by_link_;
  std::vector<Node> nodes_;
  std::vector<Node *> sorted_nodes_;
  std::vector<Link> links_;
  std::vector<std::vector<int>> successor_links_;
  std::vector<std::priority_queue<Hypothesis>> hypotheses_;
  std::unordered_map<int, std::vector<int>> nodes_by_time_;
  std::vector<Trace> traceback_;
  std::unordered_map<int, float> best_score_by_time_;
  tensorflow::Session* session_;
  const std::vector<std::string> state_vars_, state_vars_assign_ops_,
    state_vars_assign_inputs_, tensor_names_;
};
