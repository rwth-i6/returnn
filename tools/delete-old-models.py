#!/usr/bin/env python3

import sys
import os
import argparse
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import better_exchook
better_exchook.install()

from LearningRateControl import LearningRateControl
from Util import human_bytes_size


def confirm(s):
	while True:
		try:
			r = input("%s Confirm? [yes/no]" % s)
		except KeyboardInterrupt:
			print("KeyboardInterrupt")
			sys.exit(1)
		if not r:
			continue
		if r in ["y", "yes"]:
			return
		if r in ["n", "no"]:
			sys.exit(1)
		print("Invalid response %r." % r)


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--max_epoch", type=int, default=float("inf"))
arg_parser.add_argument("--keep_last_n", type=int, default=10)
arg_parser.add_argument("--base_dir", default=".")
arg_parser.add_argument("--model", default="net-model/network")
arg_parser.add_argument("--scores", default="newbob.data")
args = arg_parser.parse_args()

assert os.path.exists("%s/%s" % (args.base_dir, args.scores))
scores_s = open("%s/%s" % (args.base_dir, args.scores)).read()
scores_epoch_data = eval(scores_s, {"nan": float("nan"), "EpochData": LearningRateControl.EpochData})

model_dir = os.path.dirname("%s/%s" % (args.base_dir, args.model))
assert os.path.isdir(model_dir)
model_name = os.path.basename(args.model)

epoch_files = {}  # epoch -> set of files

for fn in os.listdir(model_dir):
	if not fn.startswith(model_name + "."):
		continue
	fn_postfix = fn[len(model_name + "."):]
	m = re.match(r"^([0-9][0-9][0-9])((\.index)|(\.meta)|(\.data(-.*)?))?$", fn_postfix)
	if not m:
		continue
	epoch = int(m.group(1))
	if epoch not in epoch_files:
		epoch_files[epoch] = set()
	epoch_files[epoch].add(fn)

if not epoch_files:
	print("No models found.")
	sys.exit(1)

epochs = sorted(epoch_files.keys())
last_epoch = epochs[-1]

print("Found epochs:", epochs)

epoch_files = {epoch: fs for (epoch, fs) in epoch_files.items() if epoch < args.max_epoch}
if args.keep_last_n:
	epoch_files = {epoch: fs for (epoch, fs) in epoch_files.items() if epoch not in epochs[-args.keep_last_n:]}

if not epoch_files:
	print("No epochs to delete.")
	sys.exit(0)

print("Epochs to delete:", sorted(epoch_files.keys()), "keeping:", [epoch for epoch in epochs if epoch not in epoch_files])
if last_epoch in epoch_files:
	print("WARNING: Last epoch is part to be deleted.")
confirm("Remove the model files.")

removed_file_size = 0

for epoch in sorted(epoch_files.keys()):
	for fn in sorted(epoch_files[epoch]):
		fn = "%s/%s" % (model_dir, fn)
		assert os.path.exists(fn)
		removed_file_size += os.stat(fn).st_size
		os.remove(fn)

print("Finished. Removed %s." % human_bytes_size(removed_file_size))
