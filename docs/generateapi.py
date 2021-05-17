

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
    """
    :param str modname:
    """
    fn = "api/%s.rst" % (modname[len("returnn."):] or "___base")
    if os.path.exists(fn):
      return
    f = open(fn, "w")
    target_python_file_path = modname.replace(".", "/")
    if os.path.isfile(target_python_file_path + ".py"):
      f.write(":github_url: https://github.com/rwth-i6/returnn/blob/master/%s.py\n\n" % modname.replace(".", "/"))
    else:
      f.write(":github_url: https://github.com/rwth-i6/returnn/blob/master/%s.py\n\n" %
              os.path.join(modname.replace(".", "/"), "__init__"))
    title = ":mod:`%s`" % modname
    f.write("\n%s\n%s\n\n" % (title, "-" * len(title)))
    f.write(".. automodule:: %s\n\t:members:\n\t:undoc-members:\n\n" % modname)
    f.close()

  def scan_modules(modpath):
    """
    :param list[str] modpath:
    """
    makeapi(".".join(modpath))
    path = "/".join(modpath)

    # First all sub packages.
    for fn in sorted(os.listdir(path)):
      if not os.path.isdir(os.path.join(path, fn)):
        continue
      if os.path.exists(os.path.join(path, fn, ".git")):
        continue
      if fn == "__pycache__":
        continue
      if not os.path.exists(os.path.join(path, fn, "__init__.py")):
        continue
      scan_modules(modpath + [fn])

    # Now all sub modules in this package.
    for fn in sorted(os.listdir(path)):
      if os.path.isdir(os.path.join(path, fn)):
        continue
      if not fn.endswith(".py"):
        continue
      if fn == "__init__.py":
        continue
      modname, _ = os.path.splitext(os.path.basename(fn))
      if modname in exclude:
        continue
      makeapi(".".join(modpath + [modname]))

  scan_modules(["returnn"])
