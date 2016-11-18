#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Pavel Golik (golik@cs.rwth-aachen.de)

"""
This module is about reading (maybe later also writing) the Sprint archive format.
"""

import sys
import os
import array
from struct import pack, unpack
import numpy
import zlib
import mmap


class FileInfo:
  def __init__(self, name, pos, size, compressed, index):
    self.name       = name
    self.pos        = pos
    self.size       = size
    self.compressed = compressed
    self.index      = index

  def __repr__(self):
    return "FileInfo(%s)" % " ".join(str(s) for s in self.__dict__.values())


class FileArchive:

  # read routines
  def read_u32(self):
    return int(unpack("i", self.f.read(4))[0])

  def read_U32(self):
    return int(unpack("I", self.f.read(4))[0])

  def read_u64(self):
    return int(unpack("q", self.f.read(8))[0])

  def read_char(self):
    return unpack("b", self.f.read(1))[0]

  def read_bytes(self, l):
    return unpack('%ds' % l, self.f.read(l))[0]

  def read_str(self, l, enc='ascii'):
    return self.read_bytes(l).decode(enc)

  def read_f32(self):
    return float(unpack("f", self.f.read(4))[0])

  def read_f64(self):
    return float(unpack("d", self.f.read(8))[0])

  def read_v(self, typ, size):
    if   typ == 'f':
        b = 4
        t = numpy.float32
    elif typ == 'd':
        b = 8
        t = numpy.float64
    else:
        raise NotImplementedError("typ: %r" % typ)
    if isinstance(self.f, mmap.mmap):
        res = numpy.frombuffer(self.f, t, size, self.f.tell())
        self.f.seek(b * size, os.SEEK_CUR)
    else:
        res = numpy.fromfile(self.f, t, size, '')
    return res

  # write routines
  def write_str(self, s):
    return self.f.write(pack("%ds" % len(s), s))

  def write_char(self, i):
    return self.f.write(pack("b", i))

  def write_u32(self, i):
    return self.f.write(pack("i", i))

  def write_U32(self, i):
    return self.f.write(pack("I", i))

  def write_u64(self, i):
    return self.f.write(pack("q", i))

  def write_f32(self, i):
    return self.f.write(pack("f", i))

  def write_f64(self, i):
    return self.f.write(pack("d", i))

  def __init__(self, filename, must_exists=True):
    self.header = "SP_ARC1\0"
    self.start_recovery_tag = 0xaa55aa55
    self.end_recovery_tag   = 0x55aa55aa

    self.ft = {}
    if os.path.exists(filename):
      self.allophones = []
      self.f = open(filename, 'rb')
      header = self.read_str(8)
      assert header == self.header

      ft = bool(self.read_char())
      if ft:
        self.readFileInfoTable()
      else:
        self.scanArchive()

    else:
      assert not must_exists, "File does not exist: %r" % filename
      self.f = open(filename, 'wb')
      self.write_str(self.header)
      self.write_char(1)

      #print(self.ft)

  def __del__(self):
    self.f.close()

  def file_list(self):
    return self.ft.keys()

  def finalize(self):
    self.writeFileInfoTable()

  def readFileInfoTable(self):
    self.f.seek(-8, 2)
    pos_count = self.read_u64()
    self.f.seek(pos_count)
    count = self.read_u32()
    if not count > 0: return
    for i in range(count):
      l    = self.read_u32()
      name = self.read_str(l)
      pos  = self.read_u64()
      size = self.read_u32()
      comp = self.read_u32()
      self.ft[name] = FileInfo(name, pos, size, comp, i)
      # TODO: read empty files

  def writeFileInfoTable(self):
    pos = self.f.tell()
    self.write_u32(len(self.ft))

    for fi in self.ft.values():
      self.write_u32(len(fi.name))
      self.write_str(fi.name)
      self.write_u64(fi.pos)
      self.write_u32(fi.size)
      self.write_u32(fi.compressed)

    self.write_u64(0)
    self.write_u64(pos)

  def scanArchive(self):
    i = 0
    self.f.seek(0, 2)
    end = self.f.tell()
    self.f.seek(0)
    while self.f.tell() < end:
      if self.read_U32() != self.start_recovery_tag: continue

      fn_len = self.read_u32()
      name = self.read_str(fn_len)
      pos  = self.f.tell()
      size = self.read_u32()
      comp = self.read_u32()
      chk  = self.read_u32()

      self.f.seek(size, 1)
      self.ft[name] = FileInfo(name, pos, size, comp, i)
      i += 1

      self.read_U32() # end tag

      #raise NotImplementedError("Need to scan archive if no "
      #                          "file info table found.")

  def _raw_read(self, size, typ):
    if typ == "str":
      return self.read_str(size)

    elif typ == "feat":
      type_len = self.read_U32()
      typ = self.read_str(type_len)
      # print(typ)
      assert typ == "vector-f32"
      count = self.read_U32()
      data = [None] * count
      time = [None] * count
      for i in range(count):
        size = self.read_U32()
        data[i] = self.read_v("f", size)  # size x f32
        time[i] = self.read_v("d", 2)  # 2    x f64
      return time, data

    elif typ == "align":
      type_len = self.read_U32()
      typ = self.read_str(type_len)
      assert typ == "flow-alignment"
      flag = self.read_u32()  # ?
      typ = self.read_str(8)
      if typ == "ALIGNRLE":
        size = self.read_U32()
        if size < (1 << 31):
          # RLE scheme
          time = 0
          alignment = []
          while len(alignment) < size:
            n = self.read_char()
            # print(n)
            if n > 0:
              while n > 0:
                mix, state = self.getState(self.read_u32())
                # print(mix, state)
                # print(time, self.allophones[mix])
                alignment.append((time, mix, state))
                time += 1
                n -= 1
            elif n < 0:
              mix, state = self.getState(self.read_u32())
              while n < 0:
                # print(mix, state)
                # print(time, self.allophones[mix])
                alignment.append((time, mix, state))
                time += 1
                n += 1
            else:
              time = self.read_u32()
              # print("time", time)
          return alignment
        else:
          raise NotImplementedError("No support for weighted "
                                    "alignments yet.")
      else:
        raise Exception("No valid alignment header found. Wrong cache?")

  def read(self, filename, typ):
    fi = self.ft[filename]
    self.f.seek(fi.pos)
    size = self.read_U32()
    comp = self.read_U32()
    chk  = self.read_U32()
    if size == 0:
      return None

    if comp > 0:
      # read compressed bytes into memory as 'bytearray'
      a = array.array('b')
      a.fromfile(self.f, comp)
      # unpack
      b = zlib.decompress(a.tostring(), 15+32)
      # substitute self.f by an anonymous memmap file object
      # restore original file handle after we're done
      backup_f = self.f
      self.f = mmap.mmap(-1, len(b))
      self.f.write(b)
      self.f.seek(0)
      try:
        return self._raw_read(size=fi.size, typ=typ)
      finally:
        self.f = backup_f

    return self._raw_read(size=fi.size, typ=typ)

  def getState(self, mix):
    max_states = 6
    #print("Was:", mix)
    for state in range(max_states):
      if mix >= len(self.allophones):
        mix -= (1<<26)
    #print("Now:", mix)
    return (mix, state)

  def setAllophones(self, f):
    for l in open(f):
      l = l.strip()
      if l.startswith("#"): continue
      self.allophones.append(l)

  def addFeatureCache(self, filename, features, times):
    self.write_U32(self.start_recovery_tag)
    self.write_u32(len(filename))
    self.write_str(filename)
    pos = self.f.tell()
    size = 4 + 10 + 4 + len(features) * (4 + len(features[0]) * 4 + 2 * 8)
    self.write_u32(size)
    self.write_u32(0)
    self.write_u32(0)

    self.write_u32(10)
    self.write_str("vector-f32")
    assert len(features) == len(times)
    self.write_u32(len(features))
    for f, t in zip(features, times):
      self.write_u32(len(f)) #f.shape[0])
      #print filename,len(f),f,t[0],t[1]
      #f.tofile(self.f,"f")
      for x in f.flat:
        self.write_f32(x)
      self.write_f64(t[0])
      self.write_f64(t[1])

    self.ft[filename] = FileInfo(filename, pos, size, 0, len(self.ft))
    self.write_U32(self.end_recovery_tag)

    self.addAttributes(filename, len(features[0]), times[-1][1])

  def addAttributes(self, filename, dim, duration):
    data = '<flow-attributes><flow-attribute name="datatype" value="vector-f32"/><flow-attribute name="sample-size" value="%d"/><flow-attribute name="total-duration" value="%.5f"/></flow-attributes>' % (dim, duration)
    self.write_U32(self.start_recovery_tag)
    filename = "%s.attribs" % filename
    self.write_u32(len(filename))
    self.write_str(filename)
    pos  = self.f.tell()
    size = len(data)
    self.write_u32(size)
    self.write_u32(0)
    self.write_u32(0)
    self.write_str(data)
    self.write_U32(self.end_recovery_tag)
    self.ft[filename] = FileInfo(filename, pos, size, 0, len(self.ft))


class FileArchiveBundle():

  def __init__(self, filename):
    self.archives = {}  # :type: dict[str,FileArchive]  # filename -> FileArchive
    self.files = {}  # :type: dict[str,FileArchive]  # archive content file -> FileArchive
    for l in open(filename).read().splitlines():
      self.archives[l] = a = FileArchive(l, must_exists=True)
      for f in a.ft.keys():
        self.files[f] = a

  def file_list(self):
    return self.files.keys()

  def read(self, filename, typ):
    return self.files[filename].read(filename, typ)

  def setAllophones(self, filename):
    for a in self.archives.values():
      a.setAllophones(filename)


def open_file_archive(archive_filename, must_exists=True):
  if archive_filename.endswith(".bundle"):
    assert must_exists
    return FileArchiveBundle(archive_filename)
  else:
    return FileArchive(archive_filename, must_exists=must_exists)


###############################################################################

class MixtureSet:
  # read routines
  def read_u32(self):
    return int(unpack('i', self.f.read(4))[0])

  def read_U32(self):
    return int(unpack('I', self.f.read(4))[0])

  def read_u64(self):
    return int(unpack('q', self.f.read(8))[0])

  def read_char(self):
    return unpack('b', self.f.read(1))[0]

  def read_str(self, l, enc='ascii'):
    a = array.array('b')
    a.fromfile(self.f, l)
    return a.tostring().decode(enc)

  def read_f32(self):
    return float(unpack('f', self.f.read(4))[0])

  def read_f64(self):
    return float(unpack('d', self.f.read(8))[0])

  def read_v(self, size, a):
    a.fromfile(self.f, size)
    return a

  def __init__(self, filename):
    self.header = 'MIXSET\0'
    self.f = open(filename, 'rb')
    header = self.read_str(8)
    assert header[:7] == self.header
    self.version = self.read_u32()
    self.dim = self.read_u32()

    self.nMeans = self.read_u32()
    self.means = np.zeros([self.nMeans, self.dim], dtype=np.float64)
    self.meanWeights = np.zeros(self.nMeans, dtype=np.float64)

    for n in range(self.nMeans):
      size = self.read_u32()
      assert size == self.dim
      arr_f = array.array('d')
      arr_f.fromfile(self.f, self.dim)
      self.means[n, :] = np.array(arr_f)
      self.meanWeights[n] = self.read_f64()

    self.nCovs = self.read_u32()
    self.covs = np.zeros([self.nCovs, self.dim], dtype=np.float64)
    self.covWeights = np.zeros(self.nCovs, dtype=np.float64)

    for n in range(self.nCovs):
      size = self.read_u32()
      assert size == self.dim
      arr_f = array.array('d')
      arr_f.fromfile(self.f, self.dim)
      self.covs[n, :] = np.array(arr_f)
      self.covWeights[n] = self.read_f64()

    self.nDensities = self.read_u32()
    self.densities = np.zeros((self.nDensities, 2), dtype=np.int32)
    for n in range(self.nDensities):
      mean_idx = self.read_u32()
      cov_idx = self.read_u32()
      self.densities[n, 0] = mean_idx
      self.densities[n, 1] = cov_idx

    self.nMixtures = self.read_u32()
    self.mixtures = [None] * self.nMixtures
    for n in range(self.nMixtures):
      nDensities = self.read_u32()
      dns_idx = []
      dns_weight = []
      for i in range(nDensities):
        dns_idx.append(self.read_u32())
        dns_weight.append(self.read_f64())
      self.mixtures[n] = (dns_idx, dns_weight)

  def write(self, filename):
    self.f = open(filename, 'wb')
    self.write_str(self.header + 't')
    self.write_u32(self.version)
    self.write_u32(self.dim)

    self.write_u32(self.nMeans)

    for n in range(self.nMeans):
      self.write_u32(self.dim)
      arr_f = array.array('d')
      arr_f.fromlist(list(self.means[n, :]))
      arr_f.tofile(self.f)
      self.write_f64(self.meanWeights[n])

    self.write_u32(self.nCovs)

    for n in range(self.nCovs):
      self.write_u32(self.dim)
      arr_f = array.array('d')
      arr_f.fromlist(list(self.covs[n, :]))
      arr_f.tofile(self.f)
      self.write_f64(self.covWeights[n])

    self.write_u32(self.nDensities)
    for n in range(self.nDensities):
      self.write_u32(int(self.densities[n, 0]))
      self.write_u32(int(self.densities[n, 1]))

    self.write_u32(self.nMixtures)
    for n in range(self.nMixtures):
      nDensities = len(self.mixtures[n][0])
      self.write_u32(nDensities)
      for i in range(nDensities):
        self.write_u32(self.mixtures[n][0][i])
        self.write_f64(self.mixtures[n][1][i])

  def __del__(self):
    self.f.close()

  def getMeanByIdx(self, idx):
    return np.float32(self.means[idx, :] / self.meanWeights[idx])

  def getCovByIdx(self, idx):
    return np.float32(self.covs[idx, :] / self.covWeights[idx])

  def getNumberMixtures(self):
    return self.nMixtures


###############################################################################

def main():
  from argparse import ArgumentParser
  arg_parser = ArgumentParser()
  arg_parser.add_argument("archive")
  arg_parser.add_argument("file", nargs="?")
  arg_parser.add_argument("--mode", default="list", help="list, show")
  arg_parser.add_argument("--type", default="feat", help="ascii, feat, align, bin-matrix, flow-cache")
  arg_parser.add_argument("--allophone-file")
  args = arg_parser.parse_args()

  a = open_file_archive(args.archive, must_exists=True)

  if args.mode == "list":
    print("\n".join(sorted(s for s in a.file_list())))
    sys.exit(0)

  elif args.mode == "show":
    assert args.file, "need to provide 'file' for --mode show. see --help"
    if args.type == "align":
      if args.allophone_file:
        a.setAllophones(args.allophone_file)

      f = a.read(args.file, "align")
      for row in f:
        print(" ".join("%.6f " % x for x in row))

    elif args.type == "feat":
      t, f = a.read(args.file, "feat")
      for row, time in zip(f, t):
        print(str(time) + "--------" + " ".join("%.6f " % x for x in row))

    else:
      raise NotImplementedError("invalid --type %r" % args.type)

  else:
    raise NotImplementedError("invalid --mode %r" % args.mode)


if __name__ == "__main__":
  import better_exchook
  better_exchook.install()
  main()
