import os
import TFUpdater

header_text = """
.. _optimizer:

=========
Optimizer
=========

"""


def generate():
  if not os.path.exists("returnn"):
    os.symlink("..", "returnn")

  TFUpdater._init_optimizer_classes_dict()
  optimizer_dict = TFUpdater._OptimizerClassesDict


  rst_file = open("optimizer.rst", "w")
  rst_file.write(header_text)

  for optimizer_name, optimizer_class in sorted(optimizer_dict.items()):

    if not optimizer_name.endswith("optimizer"):
      module = optimizer_class.__module__
      class_name = optimizer_class.__name__
      name = class_name[:-len("Optimizer")]

      rst_file.write("\n")
      rst_file.write("%s\n" % name)
      rst_file.write("%s\n" % ("-" * len(name)))
      rst_file.write("\n")
      rst_file.write(".. autoclass:: %s.%s\n" % (module, class_name))
      rst_file.write("    :members:\n")
      rst_file.write("    :undoc-members:\n")
      rst_file.write("\n")

  rst_file.close()

