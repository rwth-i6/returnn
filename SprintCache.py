#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Pavel Golik (golik@cs.rwth-aachen.de)
import sys, array
from struct import pack, unpack
from os.path import exists

class FileInfo:
    def __init__(self, name, pos, size, compressed, index):
        self.name       = name
        self.pos        = pos
        self.size       = size
        self.compressed = compressed
        self.index      = index

    def __repr__(self):
        return " ".join(str(s) for s in self.__dict__.values())

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
    
    def read_str(self, l, enc='ascii'):
        a = array.array('b')
        a.fromfile(self.f, l)
        return a.tostring().decode(enc)
        #return unpack("%ds" % l, self.f.read(l))[0].decode(enc)

    def read_f32(self):
        return float(unpack("f", self.f.read(4))[0])

    def read_f64(self):
        return float(unpack("d", self.f.read(8))[0])

    def read_v(self, typ, size):
        a = array.array(typ)
        a.fromfile(self.f, size)
        return a.tolist()

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

    def __init__(self, filename):
        self.header = "SP_ARC1\0"
        self.start_recovery_tag = 0xaa55aa55
        self.end_recovery_tag   = 0x55aa55aa

        self.ft = {}
        if exists(filename):
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
            self.f = open(filename, 'wb')
            self.write_str(self.header)
            self.write_char(1)

        #print(self.ft)

    def __del__(self):
        self.f.close()

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

    def read(self, filename, typ):
        fi = self.ft[filename]
        self.f.seek(fi.pos)
        size = self.read_u32()
        comp = self.read_u32()
        chk  = self.read_u32()
        if typ == "str":
            if fi.compressed > 0:
                return self.read_str(fi.compressed)
            else:
                return self.read_str(fi.size)

        elif typ == "feat":
            type_len = self.read_u32()
            typ = self.read_str(type_len)
            #print(typ)
            assert typ == "vector-f32"
            count = self.read_u32()
            data  = [None] * count
            time  = [None] * count
            for i in range(count):
                size    = self.read_u32()
                data[i] = self.read_v("f", size) # size x f32
                time[i] = self.read_v("d", 2)    # 2    x f64
            return time, data

        elif typ == "align":
            type_len = self.read_u32()
            typ = self.read_str(type_len)
            assert typ == "flow-alignment"
            flag = self.read_u32() #?
            typ  = self.read_str(8)
            if typ == "ALIGNRLE":
                size = self.read_u32()
                if size < (1<<31):
                    # RLE scheme
                    time = 0
                    alignment = []
                    while len(alignment) < size:
                        n = self.read_char()
                        #print(n)
                        if n > 0:
                            while n>0:
                                mix, state = self.getState(self.read_u32())
                                #print(mix, state)
                                #print(time, self.allophones[mix])
                                alignment.append((time, mix, state))
                                time += 1
                                n    -= 1
                        elif n < 0:
                            mix, state = self.getState(self.read_u32())
                            while n<0:
                                #print(mix, state)
                                #print(time, self.allophones[mix])
                                alignment.append((time, mix, state))
                                time += 1
                                n    += 1
                        else:
                            time = self.read_u32()
                            #print("time", time)
                    return alignment
                else:
                    raise NotImplementedError("No support for weighted "
                                              "alignments yet.")
            else:
                raise Exception("No valid alignment header found. Wrong cache?")
                

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

###############################################################################

def main(argv):
    a = FileArchive(argv[1])

    if len(argv) < 3:
        print("\n".join(sorted(s for s in a.ft)))
        sys.exit(0)

    if len(argv) == 4:
      a.setAllophones(argv[2])
      f = a.read(argv[3], "align")
      t=f
    else:
      t,f = a.read(argv[2], "feat")
    for row, time in zip(f, t):
      print(str(time) + "--------" + " ".join("%.6f " % x for x in row))

def usage(prog):
	print("USAGE: %s <alignment.cache> [<allophones.txt>]" % prog)
	sys.exit(-1)

if __name__ == "__main__":
	if len(sys.argv) < 2: usage(sys.argv[0])
	main(sys.argv)
