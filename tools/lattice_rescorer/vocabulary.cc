/*  Code adapted from rwthlm:

  http://www-i6.informatik.rwth-aachen.de/~sundermeyer/rwthlm.html
====================================================================*/

#include <cassert>
#include <cstdint>
#include <set>
#include <vector>
#include <sstream>
#include <boost/tuple/tuple.hpp>
#include <boost/tuple/tuple_comparison.hpp>
#include <boost/algorithm/string/trim.hpp>
#include "file.h"
#include "vocabulary.h"

ConstVocabularyPointer Vocabulary::ConstructFromVocabFile(
    const std::string &vocab_file,
    const std::string &unk,
    const std::string &sb) {
  assert(sb != "");
  VocabularyPointer v = VocabularyPointer(new Vocabulary(unk, sb));

  // we do not know the number of classes yet
  IntToInt class_by_index_map;

  // read words (and, if available, word classes) from vocab file
  int index = 0, max_class = -1;
  std::string line;
  ReadableFile file(vocab_file);
  while (file.GetLine(&line)) {
    boost::trim(line);
    std::istringstream iss(line);
    std::string word;
    iss >> word;
    assert(v->index_by_word_.find(word) == v->index_by_word_.end());
    v->index_by_word_[word] = index;
    // class information available?
    if (!iss.eof()) {
      assert(class_by_index_map.find(index) == class_by_index_map.end());
      int clazz;
      iss >> clazz;
      max_class = std::max(clazz, max_class);
      class_by_index_map[index] = clazz;
    } else {
      class_by_index_map[index] = index;
      max_class = index;
    }
    ++index;
  }

  // add <sb> automatically if not present (mkcls compatibility)
  if (v->index_by_word_.find(sb) == v->index_by_word_.end()) {
    v->index_by_word_[sb] = index;
    class_by_index_map[index] = max_class + 1;
  }

  assert(v->index_by_word_.find(sb) != v->index_by_word_.end());
  assert(unk == "" || v->index_by_word_.find(unk) != v->index_by_word_.end());
  Remap(class_by_index_map, v);
  v->sb_index_ = v->index_by_word_.find(sb)->second;

  // initialize index to word map
  for (const auto &tuple : v->index_by_word_)
    v->word_by_index_[tuple.second] = tuple.first;
  return v;
}

void Vocabulary::Remap(const IntToInt &class_by_index, VocabularyPointer v) {
  // count number of words belonging to each class
  IntToInt size_by_class;
  for (auto ic : class_by_index)
    ++size_by_class[ic.second];

  // sort classes by size in ascending order
  typedef boost::tuple<int, int, int> Triple;
  std::set<Triple> sorted;
  for (auto ic : class_by_index) {
    // (class size, class, word index)
    sorted.insert(boost::make_tuple(size_by_class[ic.second], ic.second,
                  ic.first));
  }

  // create mapping old word index -> new word index, old class -> new class
  int new_class = 0, new_index = 0, last_class = -1;
  IntToInt new_class_by_old_class;
  const int num_words = class_by_index.size();
  std::vector<int> new_index_by_old_index(num_words);
  for (auto sci : sorted) {
    new_index_by_old_index[sci.get<2>()] = new_index++;
    if (last_class != sci.get<1>()) {
      new_class_by_old_class[sci.get<1>()] = new_class++;
      last_class = sci.get<1>();
    }
  }

  // write back to vocabulary object
  const int num_classes = size_by_class.size();
  v->class_size_.resize(num_classes);

  for (const auto cs : size_by_class)
    v->class_size_[new_class_by_old_class[cs.first]] = cs.second;
  v->class_by_index_.resize(num_words);
  for (auto &wi : v->index_by_word_) {
    const int new_index = new_index_by_old_index[wi.second];
    v->class_by_index_[new_index] = new_class_by_old_class[
        class_by_index.find(wi.second)->second];
    wi.second = new_index;
  }
}

ConstVocabularyPointer Vocabulary::ConstructFromTrainFile(
    const std::string &train_file,
    const std::string &unk,
    const std::string &sb) {
  // read text from file, add words to vocabulary
  int index = 0;
  VocabularyPointer v = VocabularyPointer(new Vocabulary(unk, sb));
  std::string line, word;
  ReadableFile file(train_file);
  int64_t num_sentences = 0;
  while (file.GetLine(&line)) {
    boost::trim(line);
    std::istringstream iss(line);
    bool first_word = true;
    while (!iss.eof()) {
      if (first_word) {
        ++num_sentences;
        first_word = false;
      }
      iss >> word;
      auto it = v->index_by_word_.find(word);
      if (it == v->index_by_word_.end()) {
        it = v->index_by_word_.insert(std::make_pair(word, index++)).first;
      }
    }
  }

  // training data contain <sb> token?
  if (!v->Contains(sb)) {
    v->index_by_word_[sb] = index;
    v->sb_index_ = index;
  }

  // put each word into a single class
  const size_t vocabulary_size = v->index_by_word_.size();
  v->class_size_.resize(vocabulary_size, 1);
  v->class_by_index_.resize(vocabulary_size);
  for (size_t i = 0; i < vocabulary_size; ++i)
    v->class_by_index_[i] = i;

  // initialize index to word map
  for (const auto &tuple : v->index_by_word_)
    v->word_by_index_[tuple.second] = tuple.first;
  return v;
}

void Vocabulary::Save(const std::string &file_name) const {
  std::vector<const std::string *> sorted_words(GetVocabularySize());
  for (auto it = index_by_word_.begin(); it != index_by_word_.end(); ++it)
    sorted_words[it->second] = &it->first;
  WritableFile file(file_name);
  for (int i = 0; i < GetVocabularySize(); ++i) {
    file << *sorted_words[i] << "\t" << GetClass(i);
    if (i != sorted_words.size() - 1)
      file << '\n';
  }
}
