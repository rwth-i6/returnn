#!/usr/bin/env python3

"""
This is mostly intended to be run as a test.
This also demonstrates how the SprintInterface is being used by Sprint (RASR).
"""

import os
import tempfile
import sys


_my_dir = os.path.dirname(os.path.abspath(__file__))
_base_dir = os.path.dirname(_my_dir)
assert os.path.exists("%s/rnn.py" % _base_dir)


def main():
    """
    Main entry.
    """
    tmp_dir = tempfile.mkdtemp()
    os.symlink("%s/returnn" % _base_dir, "%s/returnn" % tmp_dir)
    config_fn = "%s/returnn.config" % tmp_dir
    with open(config_fn, "w") as f:
        f.write("#!rnn.py\n")  # Python format
        f.write("use_tensorflow = True\n")
        f.write("num_inputs, num_outputs = 3, 5\n")
        f.write("network = {'output': {'class': 'softmax', 'target': 'classes'}}\n")
        f.write("model = %r + '/model'\n" % tmp_dir)
    open("%s/model.001.meta" % tmp_dir, "w").close()
    sys.path.insert(0, tmp_dir)
    print("Import SprintInterface (relative import).")
    import returnn.sprint.interface

    print("SprintInterface.init")
    returnn.sprint.interface.init(
        inputDim=3,
        outputDim=5,
        cudaEnabled=0,
        targetMode="forward-only",
        config="epoch:1,action:nop,configfile:%s" % config_fn,
    )  # normally action:forward
    print("Ok.")


if __name__ == "__main__":
    main()
    print("demo-sprint-interface exit.")
