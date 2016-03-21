

"""
Generate stuff like:

.. automodule:: rnn
    :members:
    :undoc-members:

.. automodule:: Engine
    :members:
    :undoc-members:

"""

import os

exclude = {"autonet", "mod"}

def generate():
  def makeapi(modname):
    fn = "api/%s.rst" % modname
    if os.path.exists(fn): return
    f = open(fn, "w")
    title = ":mod:`%s`" % modname
    f.write("\n%s\n%s\n\n" % (title, "-" * len(title)))
    f.write(".. automodule:: %s\n\t:members:\n\t:undoc-members:\n\n" % modname)
    f.close()

  for fn in sorted(os.listdir("crnn")):
    if not fn.endswith(".py"): continue
    if fn.startswith("_"): continue
    modname, _ = os.path.splitext(fn)
    if modname in exclude: continue
    makeapi(modname)
