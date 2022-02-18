#!/usr/bin/env python3

"""
This explicitly compiles some of the native ops, and will tell you the so-filenames.
Normally all native ops (e.g. NativeLstm2 etc) are compiled on-the-fly within RETURNN.
When you export the computation graph (e.g. via ``compile_tf_graph.py``),
you explicitly must load these native ops.
"""

from __future__ import print_function

import os
import sys
import typing

import _setup_returnn_env  # noqa
from returnn import __main__ as rnn
from returnn.log import log
import argparse
import returnn.util.basic as util


config = None  # type: typing.Optional["returnn.config.Config"]


def init(config_filename, log_verbosity):
  """
  :param str config_filename: filename to config-file
  :param int log_verbosity:
  """
  rnn.init_better_exchook()
  rnn.init_thread_join_hack()
  if config_filename:
    print("Using config file %r." % config_filename)
    assert os.path.exists(config_filename)
  rnn.init_config(config_filename=config_filename, command_line_options=[])
  global config
  config = rnn.config
  config.set("log", None)
  config.set("log_verbosity", log_verbosity)
  config.set("use_tensorflow", True)
  rnn.init_log()
  print("Returnn compile-native-op starting up.", file=log.v1)
  rnn.returnn_greeting()
  rnn.init_backend_engine()
  assert util.BackendEngine.is_tensorflow_selected(), "this is only for TensorFlow"
  rnn.init_faulthandler()
  rnn.init_config_json_network()
  if 'network' in config.typed_dict:
    print("Loading network")
    from returnn.tf.network import TFNetwork
    network = TFNetwork(
      name="",
      config=config,
      rnd_seed=1,
      train_flag=False,
      eval_flag=True,
      search_flag=False)
    network.construct_from_dict(config.typed_dict["network"])


def main(argv):
  """
  Main entry.
  """
  from returnn.tf.util.basic import CudaEnv, NativeCodeCompiler
  CudaEnv.verbose_find_cuda = True
  NativeCodeCompiler.CollectedCompilers = []

  argparser = argparse.ArgumentParser(description='Compile some op')
  argparser.add_argument('--config', help="filename to config-file")
  argparser.add_argument('--native_op', help="op name. e.g. 'LstmGenericBase'")
  argparser.add_argument('--blas_lib', default=None,
                         help="specify which blas lib to use (path to .so or file name to search for)")
  argparser.add_argument('--search_for_numpy_blas', dest='search_for_numpy_blas', action='store_true',
                         help="search for blas inside numpys .libs folder")
  argparser.add_argument('--no_search_for_numpy_blas', dest='search_for_numpy_blas', action='store_false',
                         help="do not search for blas inside numpys .libs folder")
  argparser.add_argument("--verbosity", default=4, type=int, help="5 for all seqs (default: 4)")
  argparser.add_argument("--output_file", help='if given, will write the list of libs to this file')
  args = argparser.parse_args(argv[1:])
  init(config_filename=args.config, log_verbosity=args.verbosity)

  import returnn.native_op as native_op
  from returnn.tf.native_op import make_op, OpMaker
  if args.native_op:
    print("Loading native op %r" % args.native_op)
    op_gen = getattr(native_op, args.native_op)
    assert issubclass(op_gen, native_op.NativeOpGenBase)
    make_op(op_gen, compiler_opts={"verbose": True},
            search_for_numpy_blas=args.search_for_numpy_blas, blas_lib=args.blas_lib)

  libs = []
  if OpMaker.with_cuda and OpMaker.tf_blas_gemm_workaround:
    print('CUDA BLAS lib:', OpMaker.cuda_blas_gemm_so_filename())
    libs.append(OpMaker.cuda_blas_gemm_so_filename())
  elif OpMaker.with_cuda is False:
    print('No CUDA.')

  for compiler in NativeCodeCompiler.CollectedCompilers:
    assert isinstance(compiler, NativeCodeCompiler)
    print(compiler)
    # noinspection PyProtectedMember
    libs.append(compiler._so_filename)

  if libs:
    print("libs:")
    for fn in libs:
      print(fn)
  else:
    print("no libs compiled. use --native_op or --config")

  if args.output_file:
    with open(args.output_file, "w") as f:
      for fn in libs:
        f.write(fn + '\n')
    print("Wrote lib list to file:", args.output_file)


if __name__ == '__main__':
  main(sys.argv)
