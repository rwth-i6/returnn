#pragma once
#include <cassert>
#include <algorithm>
#include <memory>
#include <string>
#include <unordered_map>
#include <iostream>

class Vocabulary;

typedef std::shared_ptr<Vocabulary> VocabularyPointer;
typedef std::shared_ptr<const Vocabulary> ConstVocabularyPointer;

class Vocabulary {
public:
  static ConstVocabularyPointer ConstructFromVocabFile(
      const std::string &vocab_file,
      const std::string &unk,
      const std::string &sb);

  static ConstVocabularyPointer ConstructFromTrainFile(
      const std::string &train_file,
      const std::string &unk,
      const std::string &sb);

  void Save(const std::string &file_name) const;

  bool Contains(const std::string &word) const {
    return index_by_word_.find(word) != index_by_word_.end();
  }

  int ComputeShortlistSize() const {
    int result = 0;
    for (int size : class_size_) {
      if (size > 1)
        break;
      ++result;
    }
    return result;
  }

  int GetIndex(const std::string &word) const {
    const StringToInt::const_iterator it = index_by_word_.find(word);
    if (it == index_by_word_.end()) {
      assert(unk_ != "");
      return index_by_word_.find(unk_)->second;
    }
    return it->second;
  }

  std::string GetWord(const int index) const {
    const IntToString::const_iterator it = word_by_index_.find(index);
    assert (it != word_by_index_.end());
    return it->second;
  }

  int GetClass(const int index) const {
    assert(index >= 0 && index < GetVocabularySize());
    return class_by_index_[index];
  }

  int GetClassSize(const int clazz) const {
    return class_size_[clazz];
  }

  int GetMaxClassSize() const {
    return *std::max_element(class_size_.begin(), class_size_.end());
  }

  // includes <sb> and <unk>
  int GetVocabularySize() const {
    return index_by_word_.size();
  }

  int GetNumClasses() const {
    return class_size_.size();
  }

  bool HasUnk() const {
    const StringToInt::const_iterator it = index_by_word_.find(unk());
    return it != index_by_word_.end();
  }

  bool IsSentenceBoundary(const std::string &word) const {
    return word == sb_;
  }

  int sb_index() const {
    return sb_index_;
  }

  std::string sb() const {
    return sb_;
  }

  std::string unk() const {
    return unk_;
  }

private:
  typedef std::unordered_map<std::string, int> StringToInt;
  typedef std::unordered_map<int, std::string> IntToString;
  typedef std::unordered_map<int, int> IntToInt;

  Vocabulary(const std::string &unk, const std::string &sb) :
      unk_(unk), sb_(sb) {
  };

  static void Remap(const IntToInt &class_by_index, VocabularyPointer v);

  // vocabulary has to contain </b> and may contain <unk>
  const std::string unk_, sb_;
  int sb_index_;
  StringToInt index_by_word_;
  IntToString word_by_index_;
  std::vector<int> class_by_index_;
  std::vector<int> class_size_;
};
