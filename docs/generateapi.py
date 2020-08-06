

"""
Generate stuff like:

.. automodule:: rnn
    :members:
    :undoc-members:

.. automodule:: Engine
    :members:
    :undoc-members:

This will automatically get called by conf.py.

"""

import os

exclude = {"autonet", "mod"}


def generate():
  if not os.path.exists("api"):
    os.mkdir("api")

  def makeapi(modname):
    fn = "api/%s.rst" % modname[len("returnn."):]
    if os.path.exists(fn):
      return
    f = open(fn, "w")
    title = ":mod:`%s`" % modname
    f.write("\n%s\n%s\n\n" % (title, "-" * len(title)))
    f.write(".. automodule:: %s\n\t:members:\n\t:undoc-members:\n\n" % modname)
    f.close()

  def scan_modules(modpath):
    """

    :param list modpath:
    :return:
    """
    path = "/".join(modpath)
    for fn in sorted(os.listdir(path)):
      if fn.startswith("_"):
        continue
      if os.path.isdir(os.path.join(path, fn)):
        scan_modules(modpath + [fn])
      if not fn.endswith(".py"):
        continue
      modname, _ = os.path.splitext(os.path.basename(fn))
      if modname in exclude:
        continue
      makeapi(".".join(modpath + [modname]))

  scan_modules(["returnn"])
