#!/usr/bin/env python
# coding=utf-8
import h5py
import numpy

def hdf5_strings(handle, name, data):
  try:
    S=max([len(d) for d in data])
    dset = handle.create_dataset(name, (len(data),), dtype="S"+str(S))
    dset[...] = data
  except Exception:
    dt = h5py.special_dtype(vlen=unicode)
    del handle[name]
    dset = handle.create_dataset(name, (len(data),), dtype=dt)
    dset[...] = data

def write_to_hdf(img_list, transcription_list, charlist, out_file_name, dataset_prefix="train"):
  with h5py.File(out_file_name, "w") as f:
    f.attrs["inputPattSize"] = 1
    f.attrs["numDims"] = 1
    f.attrs["numSeqs"] = len(img_list)
    classes = charlist

    inputs = []
    sizes = []
    seq_lengths = []
    targets = []
    for img, transcription in zip(img_list, transcription_list):
      targets += transcription
      sizes.append(img.shape)
      img = img.reshape(img.size, 1)
      inputs.append(img)
      seq_lengths.append([[img.size, len(transcription), 2]])

    inputs = numpy.concatenate(inputs, axis=0)
    sizes = numpy.concatenate(numpy.array(sizes, dtype="int32"), axis=0)
    seq_lengths = numpy.concatenate(numpy.array(seq_lengths, dtype="int32"), axis=0)
    targets = numpy.array(targets, dtype="int32")

    f.attrs["numTimesteps"] = inputs.shape[0]

    f["inputs"] = inputs.astype("float32") / 255.0

    hdf5_strings(f, "labels", classes)
    f["seqLengths"] = seq_lengths
    seq_tags = [dataset_prefix + "/" + str(idx) for idx in range(len(img_list))]
    hdf5_strings(f, "seqTags", seq_tags)

    f["targets/data/classes"] = targets
    f["targets/data/sizes"] = sizes
    hdf5_strings(f, "targets/labels/classes", classes)
    hdf5_strings(f, "targets/labels/sizes", ["foo"]) #TODO, can we just omit it?
    g = f.create_group("targets/size")
    g.attrs["classes"] = len(classes)
    g.attrs["sizes"] = 2

def main():
  #TODO: replace this by the list of your chars (do not include the blank from CTC here, but you can include whitespace)
  # attention: at the moment, the number of chars needs to be hardcoded in the config: "classes": [4,1], there the 4 must be len(char_list)
  char_list = ['a', 'b', 'c', 'd']
  #TODO: replace this by some read images to use your own data
  img_list = [numpy.zeros((14,14), dtype="float32"), numpy.zeros((12,12), dtype="float32")]
  #TODO: replace this by some useful transcriptions (as indices into the char_list) to use your own data
  transcription_list = [[0,1,2], [2,0,1]]
  out_file_name = "test.h5"
  write_to_hdf(img_list, transcription_list, char_list, out_file_name)

if __name__ == "__main__":
  main()
