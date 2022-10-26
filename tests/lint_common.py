
"""
Common settings for linters, e.g. pycharm-inspect.py or pylint.py.

Some resources:
https://github.com/google/styleguide
https://google.github.io/styleguide/pyguide.html
https://chromium.googlesource.com/chromium/src/+/master/docs/code_reviews.md#OWNERS-files

Our Python style:
2 space indentation, 120 char line limit, otherwise mostly PEP8,
and follow further common Python conventions (closely follow PyCharm warnings).

We want to require the CI/Travis PyCharm code style checks to be mandatory for passing.

PyCharm is by far the best tool we know for this.
Pylint or Flake8 do not come close.
They report many more false positives, and fail to detect many problems.
Maybe this changes at some point, but that is why we should use PyCharm for now.
This is the script ```pycharm-inspect.py``.

PyCharm is not perfect either, though.
There are some false positives.
Maybe not always due to PyCharms fault but some lib (TensorFlow or NumPy) does strange things.
We should filter these out.

There are multiple ways how we could filter out false positives:

* ``pycharm-inspect``:

  * `inspect_class_blacklist`.
    Will completely remove/skip some type from the report, even not mention it (as ignored).
    Currently not used at all.
    Maybe could have `PyTypeCheckerInspection`, although the spell checking is useful in general.

  * `inspect_class_not_counted`.
    This will globally not count any such warnings as critical.
    It will still report them.
    This should also only rarely be used, and only for warnings which we generally do not care about,
    or which are false positives in >=50% of the cases or so.
    `PyTypeCheckerInspection` is currently here, and should be removed later,
    once there are less false positives on PyCharm side.
    Actually there are also often true positives, so this is a helpful inspection.
    Otherwise there are things which really are not so critical.

  * In `report_inspect_xml`, there is some extra code to filter out some false positives by common patterns.
    E.g. we normally check for `PyArgumentListInspection`,
    but there are some cases for NumPy which produces false warnings.

* `tests/PyCharm.idea/inspectionProfiles/Project_Default.xml`:
  Includes general inspection settings. And e.g. list of valid imports which would not lead to a warning.
  This file rarely needs to be touched.

* `lint_common.ignore_count_for_files`:
  These are mostly old files, which are not fixed yet, but which are also only rarely used, so not so important.
  New code files should normally not be added here.
  (Unless maybe it's stuff for `extern` which we copied from somewhere.)

* In the code itself, at the place of the false positive:

  * You could add some ``# noinspection PyProtectedMember``, ``# noinspection PyBroadException`` or so
    in front of the statement.
    PyCharm can do that automatically. You click on "suppress warning for this statement" or sth like that.
    Try to only use this rarely.
    This is in most cases a valid warning, no false warning, but you are ok to write the code like this.

  * You could do the same for a false warning.
    But really try to avoid that.
    We should not clutter our code with stuff which are just to workaround a false warning
    in some specific PyCharm version (which hopefully will get fixed in some later PyCharm version).
    Better filter it out by some pattern in `pycharm-inspect`.

  * Maybe your Python code is just somewhat unusual,
    and you can potentially just rewrite it slightly in a cleaner way,
    which also will get rid of the warning.
    If that is the case, definitely do it!
    Maybe some explicit ``assert isinstance(...)`` sometimes can help as well.

  * Your line got too long only because of added type information.
    Just add `` # nopep8`` at the very end of the line.
    This should still be rare.
    If you can (almost always), rewrite it in a way that it spans multiple lines.

  * You have some rare case which needs unusual code and PyCharm will show some warning.
    Then it is valid to add ``# noinspection ...``. Or even ``# noqa`` or ``# nopep8``.

* Spelling dict ``spelling.dic``.
  Do not just everything in there. Only things which you consider so common that we should accept them.
  I.e. words (variable names) like ``tmpdir`` should better just be written as ``tmp_dir``,
  or ``argparser`` as ``arg_parser``, instead of adding all such cases to the spelling dict.

"""

import os

_my_dir = os.path.dirname(os.path.abspath(__file__))
_root_dir = os.path.dirname(_my_dir)
assert os.path.exists("%s/rnn.py" % _root_dir)


# Proceed like this: Fix all warnings for some file, then remove it from this list.
# I removed already all files which really should not have warnings (mostly the TF backend + shared files).
ignore_count_for_files = {
  'returnn/util/fsa.py',
  'returnn/util/task_system.py',
  'returnn/datasets/bundle_file.py',
  'returnn/datasets/cached.py',
  'returnn/datasets/normalization_data.py',
  'returnn/datasets/raw_wav.py',
  'returnn/datasets/stereo.py',
  'returnn/tf/layers/segmental_model.py',

  # Copied to RETURNN, still work-in-progress to clean up.
  "returnn/extern/graph_editor/reroute.py",
  "returnn/extern/graph_editor/subgraph.py",
  "returnn/extern/graph_editor/transform.py",
  "returnn/extern/graph_editor/util.py",
  "returnn/extern/official_tf_resnet/resnet_model.py",

  # Ignore some outdated or rarely used tools/demos.
  'tools/collect-orth-symbols.py',
  'tools/debug-plot-search-scores.py',
  'tools/import-blocks-mt-model.py',
  'tools/import-sprint-nn.py',
  'tools/import-t2t-mt-model.py',
  'demos/mdlstm/IAM/create_IAM_dataset.py',
  'demos/mdlstm/IAM/decode.py',
  'demos/mdlstm/artificial/create_test_h5.py',
  'demos/mdlstm/artificial_rgb/create_test_h5.py',
}


def find_all_py_source_files():
  """
  :rtype: list[str]
  """
  # Earlier this was a `glob("%s/*.py" % _root_dir)`. But not anymore, since we have the new package structure.
  src_files = []
  for root, dirs, files in os.walk(_root_dir):
    if root == _root_dir:
      root = ""
    else:
      assert root.startswith(_root_dir + "/")
      root = root[len(_root_dir) + 1:]  # relative to the root
      root += "/"
    # Ignore tests, or other irrelevant directories.
    if root == "":
      dirs[:] = ["returnn", "demos", "tools"]
    else:
      dirs[:] = sorted(dirs)
      # Ignore extern git submodules.
      dirs[:] = [d for d in dirs if not os.path.exists("%s/%s%s/.git" % (_root_dir, root, d))]
    for file in sorted(files):
      if not file.endswith(".py"):
        continue
      if file == "_setup_info_generated.py":
        continue
      src_files.append(root + file)
  return src_files
