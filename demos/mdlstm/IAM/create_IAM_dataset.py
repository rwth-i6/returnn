#!/usr/bin/env python
# coding=utf-8
import h5py
import numpy
from scipy.ndimage import imread
from scipy.misc import imsave, imresize
import os
import glob
import sys
import errno
import re

def mkdir_p(path):
  try:
    os.makedirs(path)
  except OSError as exc:
    if exc.errno == errno.EEXIST and os.path.isdir(path):
      pass
    else:
      raise

def hdf5_strings(handle, name, data):
  try:
    S=max([len(d) for d in data])
    dset = handle.create_dataset(name, (len(data),), dtype="S"+str(S))
    dset[...] = data
  except Exception:
    dt = h5py.special_dtype(vlen=str)
    del handle[name]
    dset = handle.create_dataset(name, (len(data),), dtype=dt)
    dset[...] = data


def load_char_list(char_list_path):
  charlist = []
  with open(char_list_path) as f:
    for l in f:
      charlist.append(l.strip())
  return charlist


def load_file_list_and_transcriptions_and_sizes_and_n_labels(file_list_path, char_list_path, pad_whitespace, base_path):
  charlist = load_char_list(char_list_path)
  file_list = []
  transcription_list = []
  size_list = []
  with open(file_list_path) as f:
    for l in f:
      #IAM format
      if l.startswith("#"):
        continue
      sp = l.split()
      status = sp[1]
      assert status in ("ok", "err"), status
      assert len(sp) >= 9, l
      name = sp[0]
      text = "".join(sp[8:])
      #add space before '
      text = re.sub("([^|])'" , "\g<1>|'", text)
      width = int(sp[6])
      height = int(sp[7])
      if height < 1 or width < 1:
        continue
      size_list.append((height, width))
      s = name.split('-')
      name = base_path + name + '.png'
      text = [charlist.index(c) for c in text]
      if pad_whitespace:
        text = [charlist.index("|")] + text + [charlist.index("|")]
      file_list.append(name)
      transcription_list.append(text)
  return file_list, transcription_list, size_list, len(charlist)


def write_to_hdf(file_list, transcription_list, charlist, n_labels, out_file_name, dataset_prefix, pad_y=15, pad_x=15, compress=True):
  with h5py.File(out_file_name, "w") as f:
    f.attrs["inputPattSize"] = 1
    f.attrs["numDims"] = 1
    f.attrs["numSeqs"] = len(file_list)
    classes = charlist

    inputs = []
    sizes = []
    seq_lengths = []
    targets = []
    for i, (img_name, transcription) in enumerate(zip(file_list, transcription_list)):
      targets += transcription
      img = imread(img_name)
      img = 255 - img
      img = numpy.pad(img, ((pad_y, pad_y), (pad_x, pad_x)), 'constant')
      sizes.append(img.shape)
      img = img.reshape(img.size, 1)
      inputs.append(img)
      seq_lengths.append([[img.size, len(transcription), 2]])
      if i % 100 == 0:
        print(i, "/", len(file_list))

    inputs = numpy.concatenate(inputs, axis=0)
    sizes = numpy.concatenate(numpy.array(sizes, dtype="int32"), axis=0)
    seq_lengths = numpy.concatenate(numpy.array(seq_lengths, dtype="int32"), axis=0)
    targets = numpy.array(targets, dtype="int32")
    
    f.attrs["numTimesteps"] = inputs.shape[0]

    if compress:
      f.create_dataset("inputs", compression="gzip", data=inputs.astype("float32") / 255.0)
    else:
      f["inputs"] = inputs.astype("float32") / 255.0
    hdf5_strings(f, "labels", classes)
    f["seqLengths"] = seq_lengths
    seq_tags = [dataset_prefix + "/" + tag.split("/")[-1].split(".png")[0] for tag in file_list]
    hdf5_strings(f, "seqTags", seq_tags)

    f["targets/data/classes"] = targets
    f["targets/data/sizes"] = sizes
    hdf5_strings(f, "targets/labels/classes", classes)
    hdf5_strings(f, "targets/labels/sizes", ["foo"]) #TODO, can we just omit it?
    g = f.create_group("targets/size")
    g.attrs["classes"] = len(classes)
    g.attrs["sizes"] = 2


def sort_by_size(file_list, transcription_list, size_list):
  zipped = list(zip(file_list, transcription_list, size_list))
  #sort by (width, height)
  sorted_lists = sorted(zipped, key=lambda x: (x[2][1], x[2][0]))
  return [x[0] for x in sorted_lists], [x[1] for x in sorted_lists], [x[2] for x in sorted_lists]


def convert(file_list_path, char_list_path, selections, out_file_names, pad_whitespace, dataset_prefix, base_path, compress):
  charlist = load_char_list(char_list_path)
  file_list, transcription_list, size_list, n_labels = load_file_list_and_transcriptions_and_sizes_and_n_labels(
      file_list_path, char_list_path, pad_whitespace, base_path)
  file_list, transcription_list, size_list = sort_by_size(file_list, transcription_list, size_list)

  for selection, out_file_name in zip(selections, out_file_names):
    print(out_file_name)
    selection_set = set(selection)
    assert selection_set.issubset(set(x.split("/")[-1].split(".png")[0] for x in file_list))
    selected_file_list = []
    selected_transcription_list = []
    for f, t in zip(file_list, transcription_list):
      if f.split("/")[-1].split(".png")[0] in selection_set:
        selected_file_list.append(f)
        selected_transcription_list.append(t)
    write_to_hdf(selected_file_list, selected_transcription_list, charlist, n_labels, out_file_name, dataset_prefix, compress=compress)


def get_image_list(train_list_path):
  with open(train_list_path) as f:
    imgs = f.readlines()
  imgs = [img.replace("\n", "") for img in imgs]
  return imgs

def get_train_and_train_valid_lists(train_list_path, blacklist, train_fraction=0.9):
  with open(train_list_path) as f:
    imgs = f.readlines()
  imgs = [img.replace("\n", "") for img in imgs]
  n_train = int(round(train_fraction * len(imgs)))

  train_imgs = imgs[:n_train]
  train_valid_imgs = imgs[n_train:]

  n_before = len(train_imgs)
  train_imgs = [s for s in train_imgs if s not in blacklist]
  n_after = len(train_imgs)
  if n_before != n_after:
    print("removed", n_before - n_after, "blacklisted images from train")
  
  n_before = len(train_valid_imgs)
  train_valid_imgs = [s for s in train_valid_imgs if s not in blacklist]
  n_after = len(train_valid_imgs)
  if n_before != n_after:
    print("removed", n_before - n_after, "blacklisted images from train_valid")

  return train_imgs, train_valid_imgs

def convert_IAM_lines_demo(base_path_imgs, tag, blacklist=[]):
  base_path_out = "features/" + tag + "/"
  mkdir_p(base_path_out)
  file_list_path = "lines.txt"
  char_list_path = "chars.txt"
  selection_list_path = "split/demo.txt"
  out_file_name_demo = base_path_out + "demo.h5"

  print("converting IAM_lines to", out_file_name_demo)
  demo_list = ["a01-000u-00", "a01-007-04", "a01-007-06"]
  selections = [demo_list]
  out_file_names = [out_file_name_demo]
  
  convert(file_list_path, char_list_path, selections, out_file_names, pad_whitespace=True, dataset_prefix="trainset", base_path=base_path_imgs, compress=True)

def convert_IAM_lines_train(base_path_imgs, tag, blacklist=[]):
  base_path_out = "features/" + tag + "/"
  mkdir_p(base_path_out)
  file_list_path = "lines.txt"
  char_list_path = "chars.txt"
  selection_list_path = "split/train.txt"
  out_file_name_train1 = base_path_out + "train.1.h5"
  out_file_name_train2 = base_path_out + "train.2.h5"
  out_file_name_train_valid = base_path_out + "train_valid.h5"

  print("converting IAM_lines to", out_file_name_train1, "and", out_file_name_train2)
  train_list, train_valid_list = get_train_and_train_valid_lists(selection_list_path, blacklist, 0.9)
  len1 = len(train_list) / 2
  train_list1 = train_list[:len1]
  train_list2 = train_list[len1:]
  selections = [train_list1, train_list2, train_valid_list]
  out_file_names = [out_file_name_train1, out_file_name_train2, out_file_name_train_valid]

  convert(file_list_path, char_list_path, selections, out_file_names, pad_whitespace=True, dataset_prefix="trainset", base_path=base_path_imgs, compress=True)


def convert_IAM_lines_valid_test(base_path_imgs, tag):
  base_path_out = "features/" + tag + "/"
  mkdir_p(base_path_out)
  char_list_path = "chars.txt"
  selection_list_path_valid = "split/valid.txt"
  selection_list_path_test = "split/eval.txt"
  out_file_name_valid = base_path_out + "valid.h5"
  out_file_name_test = base_path_out + "test.h5"
  prefix_valid = "validationset"
  prefix_test = "testset"

  charlist = load_char_list(char_list_path)
  n_labels = len(charlist)

  for selection_list_path, out_file_name, prefix in zip([selection_list_path_valid, selection_list_path_test],
                                 [out_file_name_valid, out_file_name_test], [prefix_valid, prefix_test]):
    selection = [x.strip() for x in open(selection_list_path).readlines()]
    imgs = []
    for sel in selection:
      pattern = base_path_imgs + sel + '-[0-9][0-9].png'
      new_imgs = glob.glob(pattern)
      assert len(new_imgs) > 0, (sel, pattern)
      imgs.extend(new_imgs)
    imgs = sorted(imgs)
    transcriptions = [[]] * len(imgs)

    print("converting IAM_lines to", out_file_name)
    write_to_hdf(imgs, transcriptions, charlist, n_labels, out_file_name, dataset_prefix=prefix, compress=True)


def main():
  base_path_imgs = "IAM_lines"
  tag = "raw"
  
  if base_path_imgs[-1] != "/":
    base_path_imgs += "/"
  convert_IAM_lines_demo(base_path_imgs, tag)
  #convert_IAM_lines_train(base_path_imgs, tag)
  #convert_IAM_lines_valid_test(base_path_imgs, tag)

if __name__ == "__main__":
  main()

