#!/usr/bin/env python

from __future__ import annotations
import h5py
import numpy

with open("chars.txt") as f:
    chars = [l.strip() for l in f.readlines()] + ["_blank"]

with h5py.File("mdlstm_real_valid.h5", "r") as f:
    x = f["inputs"][...]
    x = numpy.argmax(x, axis=1)
    x = [chars[idx] for idx in x]
    lens = f["seqLengths"][...]
    tags = f["seqTags"][...]
    start = 0
    for tag, len_ in zip(tags, lens):
        y = []
        last_char = None
        for c in x[start : start + len_]:
            if last_char != c:
                y.append(c)
                last_char = c
        y = [" " if c == "|" else c for c in y if c != "_blank"]
        output = "".join(y).strip()
        print(tag, output)
        start += len_
