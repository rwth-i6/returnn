/* Code adapted from rwthlm:

  http://www-i6.informatik.rwth-aachen.de/~sundermeyer/rwthlm.html
==================================================================*/

#pragma once
#include <cassert>
#include <fstream>
#include <string>
#include <boost/algorithm/string/predicate.hpp>
#include <boost/filesystem/operations.hpp>
#pragma warning(push)
#pragma warning(disable: 4244)
#include <boost/iostreams/filter/gzip.hpp>
#pragma warning(pop)
#include <boost/iostreams/filtering_stream.hpp>

class ReadableFile {
public:
  ReadableFile(const std::string &file_name)
    : file_(file_name.c_str(), std::ios::in | std::ios::binary) {
    assert(boost::filesystem::exists(file_name));
    assert(file_.good());
    if (boost::algorithm::ends_with(file_name, ".gz"))
      stream_.push(boost::iostreams::gzip_decompressor());
    stream_.push(file_);
  }

  ~ReadableFile() {
    file_.close();
  }

  std::istream &GetLine(std::string *line) {
    return getline(stream_, *line);
  }

private:
  std::ifstream file_;
  boost::iostreams::filtering_stream<boost::iostreams::input> stream_;
};


class WritableFile {
public:
  WritableFile(const std::string &file_name)
    : file_(file_name.c_str(), std::ios::out | std::ios::binary) {
    assert(file_.good());
    if (boost::algorithm::ends_with(file_name, ".gz"))
      stream_.push(boost::iostreams::gzip_compressor());
    stream_.push(file_);
  }

  ~WritableFile() {
    stream_.reset();  // reset() before close() to avoid corrupted files
    file_.close();
  }

  template <typename T>
  std::ostream &operator<<(T t) {
    return stream_ << t;
  }

private:
  std::ofstream file_;
  boost::iostreams::filtering_ostream stream_;
};
