#!/usr/bin/env python3

import sys
import os
from argparse import ArgumentParser
from subprocess import check_call
import better_exchook

better_exchook.install()
argparser = ArgumentParser()
argparser.add_argument("--input_type", default="audio", help="audio or text. default is audio")
argparser.add_argument("--input_file", help="existing file")
argparser.add_argument("--input_text", help="some text")
argparser.add_argument("--ssh_host")
argparser.add_argument("--http_host", default="http://localhost:12380")
args = argparser.parse_args()

if args.input_file:
  assert os.path.exists(args.input_file)
  fn = args.input_file

elif args.input_type == "text" or args.input_text:
  if args.input_text:
    text = args.input_text
  else:
    text = input("Please insert some text: ")
  print("Text:", text)
  fn = "/tmp/input.txt"
  with open(fn, "w") as f:
    f.write("%s\n" % text)

elif args.input_type == "audio":
  #print("Noise profile")
  #check_call(["sox", "-n", "-d", "trim", "0", "1.5", "noiseprof", "/tmp/speech.noise-profile"])

  print("Using Sox for recording. Speak something, then pause 3 sec.")
  fn = "/tmp/rec.wav"
  if os.path.exists(fn):
    os.remove(fn)
  check_call([
    "sox", "-d", "-c", "1", "-r", "16k", "-b", "16", fn,
    # "noisered", "/tmp/speech.noise-profile", "0.3",
    "silence", "1", "0.1", "0.1%", "1", "2.0", "1%",
    "norm"],
    stdin=sys.stdin)
  assert os.path.exists(fn)

def get_curl_cmd():
  return ["curl", "-F", "file=@%s" % fn, args.http_host]

if args.ssh_host:
  print("copy to remote machine", args.ssh_host)
  check_call(["rsync", fn, "%s:/tmp" % args.ssh_host])
  fn = "/tmp/%s" % os.path.basename(fn)

  check_call(["ssh", args.ssh_host] + get_curl_cmd())

else:
  check_call(get_curl_cmd())
