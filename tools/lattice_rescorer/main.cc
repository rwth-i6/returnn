/* Code adapted from rwthlm:

  http://www-i6.informatik.rwth-aachen.de/~sundermeyer/rwthlm.html
=====================================================================*/

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <boost/program_options.hpp>
#include <boost/filesystem/operations.hpp>
#include "htklatticerescorer.h"
#include "vocabulary.h"
#include <algorithm>
#include <numeric>
#include <file.h>
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"

namespace po = boost::program_options;

// Loads the lstm op libraries, README.md for more details
void LoadLibsReturnn(const std::string& path_to_returnn_libs) {
  std::ifstream infile(path_to_returnn_libs);
  std::string lib;
  TF_Status* status = TF_NewStatus();
  while(!infile.eof()){
    getline(infile, lib);
    TF_Library* lib_ptr = TF_LoadLibrary(lib.c_str(), status);
    tensorflow::string status_msg(TF_Message(status));
    std::cerr << status_msg << std::endl;
    if(lib_ptr == nullptr)
      std::cout << lib << "load failed!\n";
    else
      std::cout << lib << " successfully loaded!\n";
   }
   TF_DeleteStatus(status);
}

// Loads the parameters of the graph
// path_to_ckpt: path to the checkpoint file, we include the graph for inference in checkpoint,
// README.md for more details
void LoadGraphParams(tensorflow::MetaGraphDef& graph_def,
                     const std::string& path_to_ckpt, tensorflow::Session* session ){
  // Load the parameters of models using checkpoint files
  tensorflow::Tensor checkpointPathTensor(tensorflow::DT_STRING, tensorflow::TensorShape());
  checkpointPathTensor.scalar<std::string>()() = path_to_ckpt;
  TF_CHECK_OK(session->Run({{graph_def.saver_def().filename_tensor_name(), checkpointPathTensor}},
    {}, {graph_def.saver_def().restore_op_name()}, nullptr));
  std::cout << "Graph parameters successfully loaded!\n";
}

// Read the information about the state variables in LSTM cell from a text file
// example/README.md for more details
void StateVars(const std::string path_to_state_vars_list, std::vector<std::string>& state_vars,
               std::vector<std::string>& state_vars_assign_ops,
               std::vector<std::string>& state_vars_assign_inputs, std::vector<int>& state_vars_size) {
  std::ifstream infile(path_to_state_vars_list);
  std::string line;
  std::string state_var_item;
  if(!infile.eof()){
    getline(infile, line);
    while(line.length() != 0) {
      std::stringstream stringin(line);
      stringin >> state_var_item;
      state_vars.push_back(state_var_item);
      stringin >> state_var_item;
      state_vars_assign_ops.push_back(state_var_item);
      stringin >> state_var_item;
      state_vars_assign_inputs.push_back(state_var_item);
      stringin >> state_var_item;
      state_vars_size.push_back(atoi(state_var_item.c_str()));
      getline(infile, line);
    }
  }
}

// Read the names of tensors needed for feeding and fetching from a text file
// example/README.md for more details
void TensorNames(const std::string path_to_run_tensor_names_list, std::vector<std::string>& tensor_names) {
  std::ifstream infile(path_to_run_tensor_names_list);
  std::string line;
  int i = 0;
  while(!infile.eof()){
    std::cout << i << std::endl;
    getline(infile, line);
    tensor_names.push_back(line);
    i++;
  }
}

void ParseCommandLine(const int argc,
                      const char *const argv[],
                      po::variables_map *options) {
  // define command line options
  po::options_description desc("Options"), hidden, all;
  desc.add_options()
      ("help", "produce this help message")
      ("config", po::value<std::string>()->default_value(""), "config file")
      ("vocab", po::value<std::string>(), "vocabulary file")
      ("map-unk", po::value<std::string>()->default_value("<unk>"),
       "name of unknown token")
      ("map-sb", po::value<std::string>()->default_value("<sb>"),
       "name of sentence boundary token")
      ("num-oovs", po::value<size_t>()->default_value(0),
        "difference in recognition and neural network LM vocabulary size")

      ("lambda", po::value<float>(),
       "interpolation weight of neural network LM")
      ("look-ahead-semiring",
       po::value<std::string>()->default_value("none"),
       "none, tropical or log semiring")
      ("dependent", po::value<bool>()->default_value(1),
       "rescore lattices independently, do not use previous best state for rescoring current lattice")
      ("look-ahead-lm-scale", po::value<float>(),
       "Look ahead LM scale for lattice decoding (default: lm-scale)")
      ("lm-scale", po::value<float>()->default_value(1.0),
       "LM scale for lattice decoding")
      ("pruning-threshold", po::value<float>(),
       "beam pruning threshold for lattice rescoring, zero means unlimited")
      ("pruning-limit", po::value<size_t>()->default_value(0),
       "maximum number of hypotheses per lattice node, zero means unlimited")
      ("dp-order", po::value<int>()->default_value(3),
       "dynamic programming order for lattice rescoring")
      ("output", po::value<std::string>()->default_value("lattice"),
       "ctm, lattice, or expanded-lattice")
      ("clear-initial-links",  // some options for compatibility with RWTH ASR software
       po::value<bool>()->default_value(0),
       "set scores of initial links in a lattice to zero")
      ("set-sb-next-to-last", po::value<bool>()->default_value(0),
       "set link label of next to last links in a lattice to <sb>")
      ("set-sb-last", po::value<bool>()->default_value(1),
       "set link label of last links in a lattice to <sb>")
      ("ops-Returnn", po::value<std::string>(),
       "libraries of the custom ops defined in Returnn")
      ("checkpoint-files", po::value<std::string>(),
       "checkpoint of tensorflow model")
      ("state-vars-list", po::value<std::string>(),
       "list of state variables and their assignment operation names")
      ("tensor-names-list", po::value<std::string>(),
       "list of tensor names for feeding and fetching");

  hidden.add_options()
      ("lattices", po::value<std::vector<std::string>>(),
       "lattices to be rescored");
  all.add(desc).add(hidden);

  // define positional options
  po::positional_options_description positional_description;
  positional_description.add("lattices", -1);
  try {
  // parse command line options
  // po::store(po::parse_command_line(argc, argv, visible), *options);
    po::store(po::command_line_parser(argc, argv).
              options(all).
                positional(positional_description).
        run(), *options);
    po::notify(*options);
    // help option?
    if (options->count("help")||!options->count("lattices")) {
      std::cout << "Usage: lattice_rescorer [OPTION]...[LATTICE]\n";
      std::cout << desc;
      exit(0);
    }
    // config file?
    const std::string config_file = (*options)["config"].as<std::string>();
    if (config_file != "") {
      assert(boost::filesystem::exists(config_file));
      std::ifstream file(config_file, std::ifstream::in);
      assert(file.good());
      std::cout << "Reading options from config file '" << config_file <<
                   "' ..." << std::endl;
      po::store(po::parse_config_file(file, desc), *options);
    }
  } catch (std::exception &e) {
    // unable to parse: print error message
    std::cerr << e.what() << '\n';
    exit(1);
  }
}

void EvaluateCommandLine(const po::variables_map &options) {
  try {
    std::vector<std::string> lattices = options["lattices"].as<std::vector<std::string>>();
    // parse arguments for which default values have been defined
    const std::string unk = options["map-unk"].as<std::string>(),
                      sb = options["map-sb"].as<std::string>();
    const size_t num_oovs = options["num-oovs"].as<size_t>();

    // Parse arguments for lattice rescoring using tf language model
    std::string path_to_returnn_libs,
                path_to_ckpt,
                path_to_lattice,
                path_to_state_vars_list,
                path_to_tensor_names_list;
    if (options.count("ops-Returnn"))
      path_to_returnn_libs = options["ops-Returnn"].as<std::string>();
    if (options.count("checkpoint-files"))
      path_to_ckpt = options["checkpoint-files"].as<std::string>();
    if (options.count("state-vars-list"))
      path_to_state_vars_list = options["state-vars-list"].as<std::string>();
    if (options.count("tensor-names-list"))
      path_to_tensor_names_list = options["tensor-names-list"].as<std::string>();
    // set up vocabulary
    ConstVocabularyPointer vocabulary;
    const std::string vocab_file = options["vocab"].as<std::string>();
    assert(boost::filesystem::exists(vocab_file));
    std::cout << "Reading vocabulary from file '" << vocab_file << "' ..." << std::endl;
    vocabulary = Vocabulary::ConstructFromVocabFile(vocab_file, unk, sb);

    std::cout << "Parsing arguments for lattice rescoring\n";
    // remaining positional arguments: lattices for rescoring
    const float lambda = options["lambda"].as<float>();
    assert(lambda >= 0. && lambda <= 1.);

    HtkLatticeRescorer::LookAheadSemiring semiring;
    const std::string name = options["look-ahead-semiring"].as<std::string>();
    if (name == "tropical")
      semiring = HtkLatticeRescorer::kTropical;
    else if (name == "log")
      semiring = HtkLatticeRescorer::kLog;
    else if (name == "none")
      semiring = HtkLatticeRescorer::kNone;
    else
      assert(false);

    HtkLatticeRescorer::OutputFormat output_format;
    const std::string format = options["output"].as<std::string>();
    if (format == "ctm")
      output_format = HtkLatticeRescorer::kCtm;
    else if (format == "lattice")
      output_format = HtkLatticeRescorer::kLattice;
    else if (format == "expanded-lattice")
      output_format = HtkLatticeRescorer::kExpandedLattice;
    else
      assert(false);

    const float lm_scale = options["lm-scale"].as<float>(),
               look_ahead_lm_scale = options.count("look-ahead-lm-scale") ==
                   0 ? lm_scale : options["look-ahead-lm-scale"].as<float>(),
               beam = options["pruning-threshold"].as<float>();
    const size_t limit = options["pruning-limit"].as<size_t>();

    std::vector<std::string> state_vars, state_vars_assign_ops, state_vars_assign_inputs, tensor_names;
    std::vector<int> state_vars_size;
    StateVars(path_to_state_vars_list, state_vars, state_vars_assign_ops, state_vars_assign_inputs, state_vars_size);
    TensorNames(path_to_tensor_names_list, tensor_names);
    LoadLibsReturnn(path_to_returnn_libs);
    tensorflow::Session* session;
    TF_CHECK_OK(tensorflow::NewSession(tensorflow::SessionOptions(), &session));
    std::cout << "Session created!\n";
    tensorflow::MetaGraphDef graph_def;
    // Loads the meta graph
    TF_CHECK_OK(tensorflow::ReadBinaryProto(tensorflow::Env::Default(), path_to_ckpt+".meta", &graph_def));
    TF_CHECK_OK(session->Create(graph_def.graph_def()));
    std::cout << "Graph successfully loaded!\n";
    LoadGraphParams(graph_def, path_to_ckpt, session);
    RescorerPointer rescorer(new HtkLatticeRescorer(
          vocabulary,
          session,
          state_vars,
          state_vars_assign_ops,
          state_vars_assign_inputs,
          state_vars_size,
          tensor_names,
          output_format,
          num_oovs,
          lambda,
          semiring,
          look_ahead_lm_scale,
          lm_scale,
          beam == 0. ? std::numeric_limits<float>::infinity() : beam,
          limit == 0 ? std::numeric_limits<size_t>::max() : limit,
          options["dp-order"].as<int>(),
          options["dependent"].as<bool>(),
          options["clear-initial-links"].as<bool>(),
          options["set-sb-next-to-last"].as<bool>(),
          options["set-sb-last"].as<bool>()));
    rescorer->Rescore(lattices);
    session->Close();
    exit(0);
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    exit(1);
  }
}

int main(int argc, char* argv[]) {
  po::variables_map options;
  ParseCommandLine(argc, argv, &options);
  EvaluateCommandLine(options);
  return 0;
}
