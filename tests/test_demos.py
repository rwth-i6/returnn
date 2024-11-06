from __future__ import annotations

import _setup_test_env  # noqa
from subprocess import Popen, PIPE, STDOUT, CalledProcessError
import re
import os
import sys
from glob import glob
import shutil
import unittest
from nose.tools import assert_less, assert_in
from returnn.util import better_exchook
from returnn.util.basic import which_pip


try:
    import torch
except ImportError:
    torch = None
else:
    print("Torch:", torch.__version__)


if "RETURNN_DISABLE_TF" in os.environ and int(os.environ["RETURNN_DISABLE_TF"]) == 1:
    tf = None
else:
    try:
        import tensorflow as tf
    except ImportError:
        tf = None
    else:
        print("TensorFlow:", tf.__version__)


py = sys.executable
print("Python:", py)

os.chdir((os.path.dirname(__file__) or ".") + "/..")
assert os.path.exists("rnn.py")


def build_env(env_update=None):
    """
    :param dict[str,str]|None env_update:
    :return: env dict for Popen
    :rtype: dict[str,str]
    """
    env_update_ = os.environ.copy()
    if env_update:
        env_update_.update(env_update)
    return env_update_


def run(*args, env_update=None, print_stdout=False):
    args = list(args)
    print("run:", args)
    # RETURNN by default outputs on stderr, so just merge both together
    p = Popen(args, stdout=PIPE, stderr=STDOUT, env=build_env(env_update=env_update))
    out, _ = p.communicate()
    out = out.decode("utf8")
    if p.returncode != 0:
        print("Return code is %i" % p.returncode)
        print("std out/err:\n---\n%s\n---\n" % out)
        raise CalledProcessError(cmd=args, returncode=p.returncode, output=out)
    if print_stdout:
        print("std out/err:\n---\n%s\n---\n" % out)
    return out


def parse_last_fer(out: str) -> float:
    """
    :param out:
    :return: FER
    """
    parsed_fer = None
    for line in out.splitlines():
        # example: epoch 5 score: 0.0231807245472 elapsed: 0:00:04 dev: score 0.0137521058997 error 0.00268961807423
        m = re.match("epoch [0-9]+ score: .* dev: .* error ([0-9.]+)\\s?", line)
        if not m:
            # example: dev: score 0.03350000149202181 error 0.009919877954075871
            m = re.match("dev: score .* error ([0-9.]+)\\s?", line)
        if not m:
            # example: Epoch 2 evaluation: dev: ce 0.034 fer 0.019
            m = re.match("Epoch [0-9]+ evaluation: dev: .* fer ([0-9.]+)\\s?", line)
        if not m:
            continue
        parsed_fer = float(m.group(1))
    err_msg = "ERROR: No epoch dev errors found in output"
    assert parsed_fer is not None, "%s.\nOutput:\n\n%s\n\n%s." % (err_msg, out, err_msg)
    return parsed_fer


def run_and_parse_last_fer(*args, **kwargs):
    out = run(*args, **kwargs)
    return parse_last_fer(out)


def run_config_get_fer(
    config_filename, *args, env_update=None, log_verbosity=5, print_stdout=False, pre_cleanup=True, post_cleanup=True
):
    if pre_cleanup:
        cleanup_tmp_models(config_filename)
    fer = run_and_parse_last_fer(
        py,
        "rnn.py",
        config_filename,
        "++log_verbosity",
        str(log_verbosity),
        *args,
        env_update=env_update,
        print_stdout=print_stdout,
    )
    print("FER:", fer)
    if post_cleanup:
        cleanup_tmp_models(config_filename)
    return fer


def get_model_filename(config_filename: str) -> str:
    assert os.path.exists(config_filename)
    from returnn.config import Config

    config = Config()
    config.load_file(config_filename)
    model_filename = config.value("model", "")
    assert model_filename
    # Remove existing models
    assert model_filename.startswith("/tmp/")
    return model_filename


def cleanup_tmp_models(config_filename: str):
    model_filename = get_model_filename(config_filename)
    for f in glob(model_filename + ".*"):
        os.remove(f)


@unittest.skipIf(not tf, "no TF")
def test_demo_tf_task12ax():
    fer = run_config_get_fer("demos/demo-tf-native-lstm.12ax.config", print_stdout=True)
    # The FER limit here is somewhat arbitrary.
    # It's (more or less) deterministic for some given hardware, some given TF version.
    # Earlier we had limit 0.01, but now that the random order in Task12AXDataset changed,
    # this seems not to be correct anymore, at least in the GitHub CI env.
    # On my local machine (Mac M1), I actually get it quite a bit lower, like 0.00127.
    # I'm not 100% sure that there is maybe sth wrong or not quite optimal...
    assert_less(fer, 0.015)


@unittest.skipIf(not tf, "no TF")
def test_demo_tf_task12ax_eval():
    # The test is specifically for task="eval". We had the problem that the dataset init was broken for this case.
    # Task=eval requires a model though, so first train one epoch.
    cfg_filename = "demos/demo-tf-native-lstm.12ax.config"
    train_dataset_repr = """{"class": "Task12AXDataset", "num_seqs": 10}"""
    dev_dataset_repr = """{"class": "Task12AXDataset", "num_seqs": 10}"""
    fer1 = run_config_get_fer(
        cfg_filename,
        "++num_epochs",
        "2",
        "++train",
        train_dataset_repr,
        "++dev",
        dev_dataset_repr,
        print_stdout=True,
        post_cleanup=False,
    )
    fer2 = run_config_get_fer(
        cfg_filename,
        "++task",
        "eval",
        "++load_epoch",
        "2",
        "++train",
        "None",
        "++dev",
        dev_dataset_repr,
        print_stdout=True,
        pre_cleanup=False,
        post_cleanup=False,
    )
    assert fer1 == fer2
    model_filename = get_model_filename(cfg_filename)
    ep2_files = glob(model_filename + ".002.*")
    assert ep2_files, f"No model files found for epoch 2, {model_filename}"
    for fn in ep2_files:
        shutil.copy(fn, fn.replace(".002.", ".003."))
    fer3 = run_config_get_fer(
        cfg_filename,
        "++task",
        "eval",
        "++load_epoch",
        "3",
        "++train",
        "None",
        "++dev",
        dev_dataset_repr,
        print_stdout=True,
        pre_cleanup=False,
    )
    assert fer3 != fer2  # because random seed, we intentionally did not use fixed_random_seed


@unittest.skipIf(not torch, "no PyTorch")
def test_demo_torch_task12ax():
    cleanup_tmp_models("demos/demo-torch.config")
    out = run(py, "rnn.py", "demos/demo-torch.config", print_stdout=True)
    # Also see test_demo_tf_task12ax above.
    fer = parse_last_fer(out)
    assert_less(fer, 0.02)


def _test_torch_export_to_onnx(cfg_filename: str) -> str:
    """
    Executes the demo passed as a parameter and returns the ONNX exported model as a filename.

    :param cfg_filename: Demo filename, either "demos/demo-rf.config" or "demos/demo-torch.config".
    :return: Filename representing the ONNX model location.
    """
    cleanup_tmp_models(cfg_filename)
    out = run(py, "rnn.py", cfg_filename, "--num_epochs", "1", "--device", "cpu")
    out_pt_model = re.search(r"\nSave model under (.*)\n", out, re.MULTILINE)
    assert out_pt_model, "Could not find the model filename in the output:\n%s" % out
    out_pt_model = out_pt_model.group(1)
    print("*** PT model:", out_pt_model)
    out_onnx_model = out_pt_model.replace(".pt", ".onnx")
    run(py, "tools/torch_export_to_onnx.py", cfg_filename, out_pt_model, out_onnx_model, print_stdout=True)

    return out_onnx_model


def _test_torch_onnx_inference_no_seq_lens(out_onnx_model: str):
    """
    Tests the inference of the torch demo with an ONNX model passed as parameter.
    """
    import onnxruntime as ort

    torch.manual_seed(42)
    # Set arbitrary values to pass to the model, respecting the input size (9).
    dummy_data = torch.randn([3, 50, 9])

    session = ort.InferenceSession(out_onnx_model)
    outputs_onnx = session.run(
        None,
        {
            "data": dummy_data.numpy(),
        },
    )
    print("*** Result:", torch.FloatTensor(outputs_onnx[0]))
    print("*** Batch size:", torch.IntTensor(outputs_onnx[1]))
    print("*** Sequence lengths:", torch.IntTensor(outputs_onnx[2]))


def _test_torch_onnx_inference_seq_lens_in_out(out_onnx_model: str):
    """
    Tests the inference of the torch demo with an ONNX model passed as parameter.
    """
    print(out_onnx_model)
    import onnxruntime as ort

    torch.manual_seed(42)
    # Set arbitrary values to pass to the model, respecting the input size (9).
    dummy_data = torch.randn([3, 50, 9])
    dummy_seq_lens = torch.tensor([27, 50, 43], dtype=torch.int32)

    session = ort.InferenceSession(out_onnx_model)
    outputs_onnx = session.run(
        None,
        {
            "data": dummy_data.numpy(),
            "data:size1": dummy_seq_lens.numpy(),
        },
    )
    out_result = torch.FloatTensor(outputs_onnx[0])
    out_seq_lens = torch.IntTensor(outputs_onnx[1])
    print("*** Result:", out_result)
    print("*** Sequence lengths:", out_seq_lens)

    assert out_result.shape == torch.Size([3, 50, 2])


@unittest.skipIf(not torch, "no PyTorch")
def test_demo_torch_export_to_onnx():
    out_onnx_model = _test_torch_export_to_onnx("demos/demo-torch.config")
    _test_torch_onnx_inference_seq_lens_in_out(out_onnx_model)


@unittest.skipIf(not torch, "no PyTorch")
def test_demo_rf_export_to_onnx():
    out_onnx_model = _test_torch_export_to_onnx("demos/demo-rf.config")
    _test_torch_onnx_inference_seq_lens_in_out(out_onnx_model)


@unittest.skipIf(not torch, "no PyTorch")
def test_demo_rf_torch_task12ax():
    cleanup_tmp_models("demos/demo-rf.config")
    out = run(py, "rnn.py", "demos/demo-rf.config", print_stdout=True)
    # Also see test_demo_tf_task12ax above.
    fer = parse_last_fer(out)
    assert_less(fer, 0.02)


@unittest.skipIf(not tf, "no TF")
def test_demo_rf_tf_task12ax():
    cleanup_tmp_models("demos/demo-rf.config")
    out = run(py, "rnn.py", "demos/demo-rf.config", "++backend", "tensorflow-net-dict", print_stdout=True)
    # Also see test_demo_tf_task12ax above.
    fer = parse_last_fer(out)
    assert_less(fer, 0.02)


def test_demo_iter_dataset_task12ax():
    # there should be no actual TF dependency, we just iterate the dataset
    cleanup_tmp_models("demos/demo-tf-vanilla-lstm.12ax.config")
    # pick any 12ax config for the dataset test
    out = run(py, "demos/demo-iter-dataset.py", "demos/demo-tf-vanilla-lstm.12ax.config")
    assert_in("Epoch 5.", out.splitlines())


@unittest.skipIf(not tf, "no TF")
def test_demo_returnn_as_framework():
    print("Prepare.")
    import subprocess
    import shutil
    from glob import glob
    from returnn.util.basic import get_login_username

    # echo via subprocess, because this stdout as well as the other will always be visible.
    subprocess.check_call(["echo", "travis_fold:start:test_demo_returnn_as_framework"])
    assert os.path.exists("setup.py")
    if glob("dist/*.tar.gz"):
        # we want it unique below
        for fn in glob("dist/*.tar.gz"):
            os.remove(fn)
    if os.path.exists("MANIFEST"):
        os.remove("MANIFEST")  # auto-generated. will be recreated
    if os.path.exists("docs/crnn"):
        os.remove("docs/crnn")  # this is auto-generated, and confuses setup.py sdist
    if os.path.exists("docs/returnn"):
        os.remove("docs/returnn")  # this is auto-generated, and confuses setup.py sdist
    tmp_model_dir = "/tmp/%s/returnn-demo-as-framework" % get_login_username()
    if os.path.exists(tmp_model_dir):
        shutil.rmtree(tmp_model_dir, ignore_errors=True)
    print("setup.py sdist, to create package.")
    subprocess.check_call([py, "setup.py", "sdist"])
    dist_fns = glob("dist/*.tar.gz")
    assert len(dist_fns) == 1
    dist_fn = os.path.abspath(dist_fns[0])
    pip_path = which_pip()
    print("Pip install Returnn.")
    in_virtual_env = hasattr(sys, "real_prefix")  # https://stackoverflow.com/questions/1871549/
    cmd = [py, pip_path, "install"]
    if not in_virtual_env:
        cmd += ["--user"]
    cmd += ["-v", dist_fn]
    print("$ %s" % " ".join(cmd))
    subprocess.check_call(cmd, cwd="/")
    print("Running demo now.")
    subprocess.check_call([py, "demo-returnn-as-framework.py"], cwd="demos")
    print("Success.")
    subprocess.check_call(["echo", "travis_fold:end:test_demo_returnn_as_framework"])


@unittest.skipIf(not tf, "no TF")
def test_demo_sprint_interface():
    import subprocess

    # echo via subprocess, because this stdout as well as the other will always be visible.
    subprocess.check_call(["echo", "travis_fold:start:test_demo_sprint_interface"])
    subprocess.check_call([py, os.path.abspath("demos/demo-sprint-interface.py")], cwd="/")
    subprocess.check_call(["echo", "travis_fold:end:test_demo_sprint_interface"])


def test_returnn_as_framework_TaskSystem():
    import subprocess

    # echo via subprocess, because this stdout as well as the other will always be visible.
    subprocess.check_call(["echo", "travis_fold:start:test_returnn_as_framework_TaskSystem"])
    subprocess.check_call([py, os.path.abspath("tests/returnn-as-framework.py"), "test_TaskSystem_Pickler()"], cwd="/")
    subprocess.check_call(["echo", "travis_fold:end:test_returnn_as_framework_TaskSystem"])


@unittest.skipIf(not tf, "no TF")
def test_returnn_as_framework_old_style_crnn_TFUtil():
    """
    Check that old-style `import crnn.TFUtil` works.

    It's not so much about TFUtil, it also could be some other module.
    It's about the old-style module names.
    This is the logic in __old_mod_loader__.
    """
    import subprocess

    # echo via subprocess, because this stdout as well as the other will always be visible.
    subprocess.check_call(["echo", "travis_fold:start:test_returnn_as_framework_old_style_crnn_TFUtil"])
    subprocess.check_call(
        [
            py,
            os.path.abspath("tests/returnn-as-framework.py"),
            "--old-style",
            "--returnn-package-name",
            "crnn",
            "test_old_style_import_crnn_TFUtil()",
        ],
        cwd="/",
    )
    subprocess.check_call(["echo", "travis_fold:end:test_returnn_as_framework_old_style_crnn_TFUtil"])


@unittest.skipIf(not tf, "no TF")
def test_returnn_as_framework_old_style_TFUtil():
    """
    Check that old-style `import TFUtil` works.
    See also :func:`test_returnn_as_framework_old_style_crnn_TFUtil`.
    """
    import subprocess

    # echo via subprocess, because this stdout as well as the other will always be visible.
    subprocess.check_call(["echo", "travis_fold:start:test_returnn_as_framework_old_style_TFUtil"])
    subprocess.check_call(
        [py, os.path.abspath("tests/returnn-as-framework.py"), "--old-style", "test_old_style_import_TFUtil()"], cwd="/"
    )
    subprocess.check_call(["echo", "travis_fold:end:test_returnn_as_framework_old_style_TFUtil"])


if __name__ == "__main__":
    better_exchook.install()
    import sys

    fns = sys.argv[1:]
    if not fns:
        fns = [arg for arg in globals() if arg.startswith("test_")]
    for arg in fns:
        f = globals()[arg]
        print("-" * 20)
        print("Run:", f)
        print("-" * 20)
        try:
            f()
        except unittest.SkipTest as exc:
            print("(SkipTest: %s)" % exc)
        print("-" * 20)
        print("Ok.")
        print("-" * 20)
