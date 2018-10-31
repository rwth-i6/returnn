#pragma once
#include <cassert>
#include <memory>
#include <regex>
#include <set>
#include <string>
#include <boost/algorithm/string/predicate.hpp>
#include <boost/filesystem.hpp>
#include <iostream>
#include <vocabulary.h>
#include <iomanip>

struct State {
	std::vector<std::vector<float>> states;
};

class Rescorer {
public:
  Rescorer(const ConstVocabularyPointer &vocabulary,
          const int num_oov_words,
           const float nn_lambda)
      : vocabulary_(vocabulary),
        nn_lambda_(nn_lambda),
        num_oov_words_(num_oov_words){
  }

  virtual ~Rescorer() {
  }

  void Rescore(const std::vector<std::string> &file_names) {
    std::cout << "Rescoring ..." << std::endl;
    for (auto &file_name : file_names) {
      std::cout << "lattice '" << file_name << "' ..." << std::endl;
      Reset();
      ReadLattice(file_name);
      RescoreLattice();
      WriteLattice(file_name);
    }
  }

protected:
  virtual void Reset() = 0;
  virtual void ReadLattice(const std::string &file_name) = 0;
  virtual void RescoreLattice() = 0;
  virtual void WriteLattice(const std::string &file_name) = 0;


  static std::string ExtendedFileName(const std::string &file_name,
                                      const std::string &extension) {
    const bool ends_with_gz = boost::algorithm::ends_with(file_name, ".gz");
    return (ends_with_gz ? file_name.substr(0, file_name.size() - 3) :
            file_name) + extension + (ends_with_gz ? ".gz" : "");
  }

  static std::string FileNameWithoutExtension(const std::string &file_name) {
    const bool ends_with_gz = boost::algorithm::ends_with(file_name, ".gz");
    return file_name.substr(
        0,
        file_name.substr(0,
                         file_name.size() - (ends_with_gz ? 3 : 0)).rfind('.'));
  }

  const float nn_lambda_;
  const int num_oov_words_;
  const ConstVocabularyPointer &vocabulary_;
};

typedef std::unique_ptr<Rescorer> RescorerPointer;
