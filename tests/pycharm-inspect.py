#!/usr/bin/env python3

"""
Will use the PyCharm code inspection.

See here:
  https://github.com/albertz/pycharm-inspect
  https://stackoverflow.com/questions/55323910/pycharm-code-style-check-via-command-line
  https://youtrack.jetbrains.com/issue/PY-34863
  https://youtrack.jetbrains.com/issue/PY-34864
"""

import os
import sys
import re
import time
import shutil
import subprocess
import tempfile
from glob import glob
import argparse
from xml.dom import minidom
from xml.etree import ElementTree

my_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(my_dir)
sys.path.insert(0, root_dir)
os.chdir(root_dir)

from returnn.util import better_exchook  # noqa
from returnn.util.basic import pip_install, which, which_pip, pip_check_is_installed, hms  # noqa

travis_env = os.environ.get("TRAVIS") == "true"
github_env = os.environ.get("GITHUB_ACTIONS") == "true"

gray_color = "black"  # black is usually gray
if github_env:
    gray_color = "white"  # black is black, on black background. so just use white


class _StdoutTextFold:
    def __init__(self, name):
        """
        :param str name:
        """
        self.name = name
        self.start_time = time.time()

        if github_env:
            # https://github.community/t/has-github-action-somthing-like-travis-fold/16841
            if not folds:  # nested folds not supported, https://github.com/actions/toolkit/issues/112
                print("::group::%s" % name)

        if travis_env:
            # travis_fold: https://github.com/travis-ci/travis-ci/issues/1065
            print("travis_fold:start:%s" % name)

        sys.stdout.flush()

    def finish(self):
        """
        End fold.
        """
        elapsed_time = time.time() - self.start_time
        print("%s: Elapsed time: %s" % (self.name, hms(elapsed_time)))

        if travis_env:
            print("travis_fold:end:%s" % folds[-1])

        if github_env:
            if len(folds) == 1:
                print("::endgroup::")

        sys.stdout.flush()


folds = []  # type: typing.List[_StdoutTextFold]


def fold_start(name):
    """
    :param str name:
    """
    folds.append(_StdoutTextFold(name))


def fold_end():
    """
    Ends the fold.
    """
    assert folds
    folds[-1].finish()
    folds.pop(-1)


def check_pycharm_dir(pycharm_dir):
    """
    :param str pycharm_dir:
    """
    assert os.path.isdir(pycharm_dir)
    assert os.path.exists("%s/bin/inspect.sh" % pycharm_dir)


def install_pycharm():
    """
    :return: pycharm dir
    :rtype: str
    """
    fold_start("script.install")
    print("travis_fold:start:script.install")
    install_dir = tempfile.mkdtemp()
    pycharm_dir = "%s/pycharm" % install_dir
    print("Install PyCharm into:", pycharm_dir)
    sys.stdout.flush()

    # Keep the pin recent enough for the CI python of the pycharm-inspect job:
    # 2020.2 topped out at language level ~3.10, and with a 3.12 interpreter it fell back to "3.1"
    # (PyInterpreterInspection EOL warning in every file + wrong-syntax inspections).
    # Since 2025.3 there is no separate Community tarball anymore, only the unified PyCharm
    # (free mode included) -- pycharm-<ver>.tar.gz, python plugin dir "python" instead of "python-ce".
    name = "pycharm-2026.2"
    fn = "%s.tar.gz" % name

    subprocess.check_call(
        ["wget", "--progress=dot:mega", "-c", "https://download.jetbrains.com/python/%s" % fn],
        cwd=install_dir,
        stderr=subprocess.STDOUT,
    )
    tar_out = subprocess.check_output(["tar", "-xzvf", fn], cwd=install_dir, stderr=subprocess.STDOUT)
    print((b"\n".join(tar_out.splitlines()[-10:])).decode("utf8"))
    # the tarball's top-level dir does not always match `name` exactly (patch-versioned): take it from the listing
    top_dir = tar_out.splitlines()[0].decode("utf8").split("/")[0].strip()
    assert os.path.isdir("%s/%s" % (install_dir, top_dir))
    os.remove("%s/%s" % (install_dir, fn))
    os.rename("%s/%s" % (install_dir, top_dir), pycharm_dir)
    check_pycharm_dir(pycharm_dir)

    fold_end()
    return pycharm_dir


def get_version_str_from_pycharm(pycharm_dir):
    """
    :param str pycharm_dir:
    :return: e.g. "CE2018.3"
    :rtype: str
    """
    import re
    import json

    if os.path.exists("%s/product-info.json" % pycharm_dir):
        d = json.load(open("%s/product-info.json" % pycharm_dir))
        name = d["dataDirectoryName"]
        assert isinstance(name, str)
        assert name.startswith("PyCharm")
        return name[len("PyCharm") :]
    # This works on PyCharm 2019.
    code = open("%s/bin/pycharm.sh" % pycharm_dir).read()
    m = re.search("-Didea\\.paths\\.selector=PyCharm(\\S+) ", code)
    assert m, "pycharm %r not as expected" % pycharm_dir
    return m.group(1)


def parse_pycharm_version(version_str):
    """
    :param str version_str: e.g. "CE2018.3"
    :rtype: ((int,int),str)
    :return: e.g. (2018,3),"CE"
    """
    name = ""
    if version_str.startswith("CE"):
        name = "CE"
        version_str = version_str[2:]
    assert version_str.startswith("2")
    version_str_parts = version_str.split(".")
    assert len(version_str_parts) == 2, "version %r" % version_str
    return tuple([int(p) for p in version_str_parts]), name


def create_stub_dir(pycharm_dir, stub_dir, pycharm_major_version):
    """
    :param str pycharm_dir:
    :param str stub_dir:
    :param int pycharm_major_version:
    """
    fold_start("script.create_python_stubs")
    print("Generating Python stubs via helpers/generator3.py...")
    if pycharm_major_version >= 2020:
        # "python-ce" in the Community edition, "python" in the unified PyCharm (2025.3+)
        for plugin_name in ["python", "python-ce"]:
            helpers_dir = "%s/plugins/%s/helpers" % (pycharm_dir, plugin_name)
            if os.path.exists("%s/generator3/__main__.py" % helpers_dir):
                break
        assert helpers_dir and os.path.exists("%s/generator3/__main__.py" % helpers_dir)
        # -m with cwd=helpers: running __main__.py by path puts the package dir itself on sys.path,
        # and `import generator3` then fails (generator3 became a package at some point after 2020.2)
        cmd = [sys.executable, "-m", "generator3", "-d", stub_dir]
        # The stdout can sometimes be very long. Thus we pipe and filter it a bit.
        proc = subprocess.Popen(cmd, cwd=helpers_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        stdout, _ = proc.communicate()
        if proc.returncode != 0:
            raise subprocess.CalledProcessError(returncode=proc.returncode, cmd=cmd, output=stdout)
        for line in stdout.splitlines():
            line = line.decode("utf8")
            if len(line) < 240:
                print(line)
            else:
                print(line[:240] + "...")
    elif pycharm_major_version <= 2019:
        generator_path = "%s/helpers/generator3.py" % pycharm_dir
        assert os.path.exists(generator_path)
        subprocess.check_call([sys.executable, generator_path, "-d", stub_dir, "-b"])
        print("Collecting further native modules...")
        sys.stdout.flush()
        mod_names = []
        for line in subprocess.check_output([sys.executable, generator_path, "-L"]).decode("utf8").splitlines()[1:]:
            # First line is version, so we skipped those.
            # Then we get sth like "<module name> <other things>...".
            assert isinstance(line, str)
            mod_name = line.split()[0]
            # There are duplicates. Ignore.
            if mod_name not in mod_names:
                mod_names.append(mod_name)
        for mod_name in mod_names:
            print("Generate for %r." % mod_name)
            sys.stdout.flush()
            # Ignore errors here.
            subprocess.call([sys.executable, generator_path, "-d", stub_dir, mod_name])
    fold_end()


_use_stub_zip = False


def setup_pycharm_python_interpreter(pycharm_dir, install_py_deps=False):
    """
    Unfortunately, the headless PyCharm bin/inspect will use the global PyCharm settings,
    and requires that we have a Python interpreter set up,
    with the same name as we use in our `.idea` settings, which we will link in :func:`prepare_src_dir`.
    See here: https://youtrack.jetbrains.com/issue/PY-34864

    Our current way to work around this: We create (or extend) the file
    ``~/.PyCharm<VERSION>/config/options/jdk.table.xml`` such that it has the right Python interpreter.

    :param str pycharm_dir:
    :param bool install_py_deps: pip-install TF and further packages the inspection expects (for CI)
    """
    if install_py_deps:
        # only with --install_py_deps (CI): never modify a user's local env implicitly
        fold_start("script.opt_install_further_py_deps")
        if not pip_check_is_installed("tensorflow") and not pip_check_is_installed("tensorflow-gpu"):
            pip_install("tensorflow")
        # Note: Horovod will usually fail to install in this env.
        for pkg in ["typing", "librosa==0.8.1", "PySoundFile", "nltk", "matplotlib", "mpi4py", "pycodestyle"]:
            if not pip_check_is_installed(pkg):
                try:
                    pip_install(pkg)
                except subprocess.CalledProcessError as exc:
                    print("Pip install failed:", exc)
                    print("Ignore...")
        fold_end()

    fold_start("script.setup_pycharm_python_interpreter")
    print("Setup PyCharm Python interpreter... (jdk.table.xml)")
    print("Current Python:", sys.executable, sys.version, sys.version_info)
    name = "Python 3 (.../bin/python3)"  # used in our PyCharm.idea. this should match.
    pycharm_version_str = get_version_str_from_pycharm(pycharm_dir)
    pycharm_version, pycharm_version_name = parse_pycharm_version(pycharm_version_str)
    if sys.platform == "darwin":
        pycharm_config_dir = os.path.expanduser("~/Library/Preferences/PyCharm%s" % pycharm_version_str)
        pycharm_system_dir = os.path.expanduser("~/Library/Caches/PyCharm%s" % pycharm_version_str)
    else:  # assume Linux/Unix
        if pycharm_version[0] >= 2020:
            pycharm_config_dir = os.path.expanduser("~/.config/JetBrains/PyCharm%s" % pycharm_version_str)
            pycharm_system_dir = os.path.expanduser("~/.cache/JetBrains/PyCharm%s" % pycharm_version_str)
        else:  # <= 2020
            pycharm_config_dir = os.path.expanduser("~/.PyCharm%s/config" % pycharm_version_str)
            pycharm_system_dir = os.path.expanduser("~/.PyCharm%s/system" % pycharm_version_str)
    # Env overrides, for running this OUTSIDE a disposable CI runner:
    # the default dirs are the user's REAL IDE config (a running IDE both holds the
    # single-instance lock and would have its jdk.table.xml edited underneath it).
    pycharm_config_dir = os.environ.get("PYCHARM_INSPECT_CONFIG_DIR", pycharm_config_dir)
    pycharm_system_dir = os.environ.get("PYCHARM_INSPECT_SYSTEM_DIR", pycharm_system_dir)

    # I just zipped the stubs from my current installation on Linux.
    # Maybe we can also reuse these stubs for other PyCharm versions, or even other Python versions.
    if _use_stub_zip:
        stub_base_name = "pycharm2018.3-python3.6-stubs"
        stub_fn = "%s/python_stubs/%s.zip" % (pycharm_system_dir, stub_base_name)
        stub_dir = "%s/python_stubs/%s" % (pycharm_system_dir, stub_base_name)
        os.makedirs(os.path.dirname(stub_fn), exist_ok=True)
        if os.path.exists(stub_dir):
            print("Python stubs dir exists already:", stub_dir)
        else:
            if not os.path.exists(stub_fn):
                subprocess.check_call(
                    ["wget", "https://www-i6.informatik.rwth-aachen.de/web/Software/returnn/%s.zip" % stub_base_name],
                    cwd=os.path.dirname(stub_fn),
                )
            assert os.path.exists(stub_fn)
            subprocess.check_call(
                ["unzip", "%s.zip" % stub_base_name, "-d", stub_base_name], cwd=os.path.dirname(stub_fn)
            )
            assert os.path.isdir(stub_dir)
    else:
        stub_dir = "%s/python_stubs/python%s-generated" % (pycharm_system_dir, "%i.%i.%i" % sys.version_info[:3])
        if os.path.exists(stub_dir):
            print("Python stubs already exists, not recreating (%s)" % stub_dir)
        else:
            print("Generate stub dir:", stub_dir)
            os.makedirs(stub_dir)
            create_stub_dir(pycharm_dir=pycharm_dir, stub_dir=stub_dir, pycharm_major_version=pycharm_version[0])

    jdk_table_fn = "%s/options/jdk.table.xml" % pycharm_config_dir
    print("Filename:", jdk_table_fn)
    os.makedirs(os.path.dirname(jdk_table_fn), exist_ok=True)

    if os.path.exists(jdk_table_fn):
        print("Loading existing jdk.table.xml.")
        et = ElementTree.parse(jdk_table_fn)
        root = et.getroot()
        assert isinstance(root, ElementTree.Element)
        jdk_collection = root.find("./component")
        assert isinstance(jdk_collection, ElementTree.Element)
        assert jdk_collection.tag == "component" and jdk_collection.attrib["name"] == "ProjectJdkTable"
    else:
        print("Creating new jdk.table.xml.")
        root = ElementTree.Element("application")
        et = ElementTree.ElementTree(root)
        jdk_collection = ElementTree.SubElement(root, "component", name="ProjectJdkTable")
        assert isinstance(jdk_collection, ElementTree.Element)

    existing_jdk = jdk_collection.find("./jdk/name[@value='%s']/.." % name)
    # `is not None`, NOT truthiness: an Element is falsy when it has no children
    # (and Python warns about it), so the old `if existing_jdk:` also missed empty entries.
    if existing_jdk is not None:
        assert isinstance(existing_jdk, ElementTree.Element)
        assert existing_jdk.find("./name").attrib["value"] == name
        home_el = existing_jdk.find("./homePath")
        existing_home = home_el.attrib["value"] if home_el is not None else ""
        # the IDE writes the $USER_HOME$ macro back into this file
        existing_home = existing_home.replace("$USER_HOME$", os.path.expanduser("~"))
        if existing_home and os.path.realpath(existing_home) == os.path.realpath(sys.executable):
            # Same interpreter: KEEP the entry as the IDE last left it, do not rewrite.
            # Rewriting it every run is what made the inspection unreliable: we point the SDK
            # back at OUR stub dir, so on the next start the IDE finds its own (hashed) skeleton
            # root missing, runs PySkeletonRefresher ~50 s INTO the run and then updates the SDK
            # -- and every file analyzed after that point stops resolving `torch` (measured:
            # nothing wrong in the first ~30% of the file order, ~1200 bogus
            # "Cannot find reference ... in 'torch'" after it).
            # Recreating is still right when the interpreter DIFFERS -- that is the cross-env
            # stale-SDK case this remove/recreate was originally added for.
            print("Existing Python interpreter %r already points at %s. Keeping it." % (name, existing_home))
            fold_end()
            return
        print(
            "Found existing Python interpreter %r for a DIFFERENT interpreter (%s != %s)."
            " Remove and recreate." % (name, existing_home, sys.executable)
        )
        jdk_collection.remove(existing_jdk)

    # Example content:
    """
  <application>
  <component name="ProjectJdkTable">
    <jdk version="2">
      <name value="Python 2.7.3 (/usr/bin/python2.7)" />
      <type value="Python SDK" />
      <version value="Python 2.7.12" />
      <homePath value="/usr/bin/python2.7" />
      <roots>
        <classPath>
          <root type="composite">
            <root url="file:///usr/bin" type="simple" />
            ...
          </root>
        </classPath>
        <sourcePath>
          <root type="composite" />
        </sourcePath>
      </roots>
      <additional />
    </jdk>
  </component>
  </application>
  """

    jdk_entry = ElementTree.SubElement(jdk_collection, "jdk", version="2")
    ElementTree.SubElement(jdk_entry, "name", value=name)
    ElementTree.SubElement(jdk_entry, "type", value="Python SDK")
    ElementTree.SubElement(jdk_entry, "version", value="Python %i.%i.%i" % sys.version_info[:3])
    ElementTree.SubElement(jdk_entry, "homePath", value=sys.executable)
    paths_root = ElementTree.SubElement(jdk_entry, "roots")
    classes_paths = ElementTree.SubElement(ElementTree.SubElement(paths_root, "classPath"), "root", type="composite")
    relevant_paths = list(sys.path)
    if root_dir in relevant_paths:
        relevant_paths.remove(root_dir)
    if my_dir in relevant_paths:
        relevant_paths.remove(my_dir)
    relevant_paths.extend(
        [
            stub_dir,
            "$APPLICATION_HOME_DIR$/helpers/python-skeletons",
            "$APPLICATION_HOME_DIR$/helpers/typeshed/stdlib/3",
            "$APPLICATION_HOME_DIR$/helpers/typeshed/stdlib/2and3",
            "$APPLICATION_HOME_DIR$/helpers/typeshed/third_party/3",
            "$APPLICATION_HOME_DIR$/helpers/typeshed/third_party/2and3",
        ]
    )
    # Maybe also add Python stubs path? How to generate them?
    for path in relevant_paths:
        ElementTree.SubElement(classes_paths, "root", url="file://%s" % path, type="simple")
    ElementTree.SubElement(ElementTree.SubElement(paths_root, "sourcePath"), "root", type="composite")
    ElementTree.SubElement(jdk_entry, "additional")

    print("Save XML.")
    et.write(jdk_table_fn, encoding="UTF-8")

    fold_start("script.jdk_table")
    print("XML content:")
    rough_string = ElementTree.tostring(root, "utf-8")
    print(minidom.parseString(rough_string).toprettyxml(indent="  "))
    fold_end()

    fold_end()


def read_spelling_dict():
    """
    :rtype: list[str]
    """
    return open("%s/spelling.dic" % my_dir).read().splitlines()


def create_spelling_dict_xml(src_dir):
    """
    Need to create this on-the-fly for the current user.
    """
    # Example:
    """
  <component name="ProjectDictionaryState">
  <dictionary name="az">
    <words>
      <w>dtype</w>
      <w>idxs</w>
      <w>keepdims</w>
      ...
    </words>
  </dictionary>
  </component>
  """
    from returnn.util.basic import get_login_username

    user_name = get_login_username()
    root = ElementTree.Element("component", name="ProjectDictionaryState")
    dict_ = ElementTree.SubElement(root, "dictionary", name=user_name)
    words = ElementTree.SubElement(dict_, "words")
    for w in read_spelling_dict():
        ElementTree.SubElement(words, "w").text = w
    et = ElementTree.ElementTree(root)
    print("Save XML.")
    xml_filename = "%s/.idea/dictionaries/%s.xml" % (src_dir, user_name)
    os.makedirs(os.path.dirname(xml_filename), exist_ok=True)
    et.write(xml_filename, encoding="UTF-8")


def prepare_src_dir(files=None):
    """
    New clean source dir, where we symlink only the relevant src files.

    With an explicit `files` list (``--files``), the whole package tree is still
    laid out -- only the listed files are real copies, everything else is a symlink to
    the original. The inspection then sees a COMPLETE project (imports resolve, the
    type inference matches a full run) while the report step can filter to the listed
    files. Copying the listed files (instead of symlinking them like the rest) keeps
    the original tree untouched no matter what the IDE does to its project files.

    :param list[str]|None files: relative paths, e.g. ["returnn/datasets/hdf.py"]
    :return: src dir
    :rtype: str
    """
    fold_start("script.prepare")
    print("Prepare project source files...")
    explicit_files = list(files) if files else None
    top_level = ["returnn", "tools", "demos", "rnn.py", "setup.py", "__init__.py"]
    src_tmp_dir = "%s/returnn" % tempfile.mkdtemp()
    os.mkdir(src_tmp_dir)
    shutil.copytree("%s/PyCharm.idea" % my_dir, "%s/.idea" % src_tmp_dir, symlinks=True)
    for fn in top_level:
        src = "%s/%s" % (root_dir, fn)
        dst = "%s/%s" % (src_tmp_dir, fn)
        if os.path.isdir(src):
            shutil.copytree(src, dst, symlinks=True)
        else:
            shutil.copy(src, dst)
    if explicit_files:
        # keep the paths (a flattened copy breaks the package structure and every import in it)
        for fn in explicit_files:
            src = "%s/%s" % (root_dir, fn)
            dst = "%s/%s" % (src_tmp_dir, fn)
            assert os.path.isfile(src), "--files: %s does not exist" % src
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy(src, dst)
    create_spelling_dict_xml(src_tmp_dir)
    print("All source files:")
    sys.stdout.flush()
    subprocess.check_call(["ls", "-la", src_tmp_dir])
    fold_end()
    return src_tmp_dir


def run_inspect(pycharm_dir, src_dir, skip_pycharm_inspect=False, scope_dir=None):
    """
    :param str pycharm_dir:
    :param str src_dir:
    :param bool skip_pycharm_inspect:
    :param str|None scope_dir: relative dir to limit the ANALYSIS to (inspect.sh -d),
        while the whole project stays indexed, so type inference matches a full run.
        Used by ``--files`` to keep that mode fast without changing what is inferred.
    :return: dir of xml files
    :rtype: str
    """
    out_tmp_dir = tempfile.mkdtemp()

    fold_start("script.inspect")
    if not skip_pycharm_inspect:
        fold_start("script.inspect.pycharm_inspect.sh.content")
        with open("%s/bin/inspect.sh" % pycharm_dir) as f:
            content = f.read()
            print("Content of inspect.sh:")
            print(content)
        fold_end()

        fold_start("script.inspect.pycharm.sh.content")
        with open("%s/bin/pycharm.sh" % pycharm_dir) as f:
            content = f.read()
            print("Content of pycharm.sh:")
            print(content)
        fold_end()

        fold_start("script.inspect.vmoptions.content")
        fns = glob("%s/bin/*.vmoptions" % pycharm_dir)
        if fns:
            for fn in fns:
                print("Content of %s:" % fn)
                with open(fn) as f:
                    content = f.read().splitlines(keepends=True)
                print("".join(content))
                if any(line.startswith("-Xmx") for line in content):
                    # 8g: indexing modern site-packages (torch+TF+...) fails half-way at the old 4g
                    # (partially-unresolved core torch members), and silently so
                    print("Note: Patching Xmx settings...")
                    # overridable for heap experiments (e.g. checking whether unresolved-reference
                    # noise is a memory-pressure artifact of the batch inspection)
                    xmx = os.environ.get("RETURNN_PYCHARM_INSPECT_XMX", "8000m")
                    content = [f"-Xmx{xmx}\n" if line.startswith("-Xmx") else line for line in content]
                    with open(fn, "w") as f:
                        f.write("".join(content))
        else:
            print("No *.vmoptions found, not printing content.")
        fold_end()

        # Note: Will not run if PyCharm is already running.
        # Maybe we can find some workaround for this?
        # See here: https://stackoverflow.com/questions/55339010/run-pycharm-inspect-sh-even-if-pycharm-is-already-runn
        # And here: https://github.com/albertz/pycharm-inspect
        # Also: https://stackoverflow.com/questions/55323910/pycharm-code-style-check-via-command-line
        cmd = [
            "%s/bin/inspect.sh" % pycharm_dir,
            src_dir,
            "%s/PyCharm-inspection-profile.xml" % my_dir,
            out_tmp_dir,
            "-v2",
        ]
        if scope_dir:
            cmd += ["-d", "%s/%s" % (src_dir, scope_dir)]
        env = dict(os.environ)
        if os.environ.get("PYCHARM_INSPECT_CONFIG_DIR"):
            # match the env-override dirs of setup_pycharm_python_interpreter:
            # point the IDE itself at them via a properties file (PYCHARM_PROPERTIES)
            props_fn = "%s/pycharm-inspect.properties" % out_tmp_dir
            with open(props_fn, "w") as f:
                f.write("idea.config.path=%s\n" % os.environ["PYCHARM_INSPECT_CONFIG_DIR"])
                if os.environ.get("PYCHARM_INSPECT_SYSTEM_DIR"):
                    f.write("idea.system.path=%s\n" % os.environ["PYCHARM_INSPECT_SYSTEM_DIR"])
            env["PYCHARM_PROPERTIES"] = props_fn
            # ... and pin the vmoptions: otherwise the IDE ALSO loads the user's real
            # pycharm64.vmoptions (jb.vmOptionsFile), whose -Xmx comes last and WINS
            # (observed: our 8000m silently reduced to the user's 3072m -> indexing
            # under-provisioned -> core torch members unresolved).
            vmopts_fn = "%s/pycharm-inspect.vmoptions" % out_tmp_dir
            with open("%s/bin/pycharm64.vmoptions" % pycharm_dir) as f_in, open(vmopts_fn, "w") as f_out:
                f_out.write(f_in.read())
                # extra JVM options for experiments, whitespace-separated
                # (e.g. -Djava.util.concurrent.ForkJoinPool.common.parallelism=1
                #  to serialize the concurrent inspection engine when chasing per-file races)
                for opt in os.environ.get("RETURNN_PYCHARM_INSPECT_VM_EXTRA", "").split():
                    f_out.write(opt + "\n")
            env["PYCHARM_VM_OPTIONS"] = vmopts_fn
        # Headless index/skeleton prebuild (the remote-dev "warmup" command) BEFORE inspecting:
        # inspect.sh otherwise analyzes files concurrently with indexing the site-packages
        # (torch alone is huge), and files analyzed before the relevant index part is complete
        # get nondeterministic unresolved-reference noise (measured: three identical runs gave
        # 1182/920/1087 "Cannot find reference ... in 'torch'" findings).
        warmup_cmd = ["%s/bin/pycharm.sh" % pycharm_dir, "warmup", "--project-dir=%s" % src_dir]
        fold_start("script.inspect.warmup")
        print("$ %s" % " ".join(warmup_cmd))
        subprocess.check_call(warmup_cmd, stderr=subprocess.STDOUT, env=env)
        fold_end()
        print("$ %s" % " ".join(cmd))
        subprocess.check_call(cmd, stderr=subprocess.STDOUT, env=env)

    # PyCharm does not do PEP8 code style checks by itself but uses the (bundled) pycodestyle tool.
    # https://youtrack.jetbrains.com/issue/PY-43901
    # Do that now. pycodestyle must be in the env (CI installs it via --install_py_deps;
    # never skip silently -- a missing module here means a broken env, and skipping
    # would hide all PEP8 problems from the report).
    subprocess.check_output([sys.executable, "-m", "pycodestyle", "--version"], stderr=subprocess.STDOUT)
    root = ElementTree.Element("problems")
    from lint_common import find_all_py_source_files

    for py_src_file in find_all_py_source_files():
        ignore_codes = "E121,E123,E126,E226,E24,E704,W503,W504"  # PyCharm defaults
        ignore_codes += (
            ",E203"  # https://black.readthedocs.io/en/stable/faq.html#why-are-flake8-s-e203-and-w503-violated
        )
        indent_size = 4
        cmd = [
            sys.executable,
            "-m",
            "pycodestyle",
            py_src_file,
            "--ignore=%s" % ignore_codes,
            "--indent-size=%i" % indent_size,
            "--max-line-length=120",
        ]
        print("$ %s" % " ".join(cmd))
        sys.stdout.flush()
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        stdout, _ = proc.communicate()
        problem_count = 0
        # We do not check returncode, as this is always non-zero if there is any warning.
        for line in stdout.decode("utf8").splitlines():
            # Example line: demos/demo-record-and-push-to-webserver.py:48:1: E302 expected 2 blank lines, found 1
            m = re.match("^(.*):([0-9]+):([0-9]+): ([EW][0-9]+) (.+)$", line)
            assert m, "unexpected line %r" % line
            fn_, line_nr, col_nr, warn_id, description = m.groups()
            assert fn_ == py_src_file, "unexpected line %r" % line
            line_nr, col_nr = int(line_nr), int(col_nr)
            description = "%s: %s" % (warn_id, description)
            prob = ElementTree.SubElement(root, "problem")
            # Note: We do not aim to have this complete. This is just enough such that report_inspect_xml can read it.
            ElementTree.SubElement(prob, "file").text = "file://$PROJECT_DIR$/%s" % py_src_file
            ElementTree.SubElement(prob, "line").text = str(line_nr)
            ElementTree.SubElement(prob, "offset").text = str(col_nr)
            ElementTree.SubElement(prob, "problem_class", severity="WEAK WARNING", id=warn_id).text = description
            ElementTree.SubElement(prob, "description").text = description
            problem_count += 1
        if proc.returncode != 0:
            assert problem_count > 0, "pycodestyle returned error but did not list any problems"
    et = ElementTree.ElementTree(root)
    et.write("%s/Pep8CodeStyle.xml" % out_tmp_dir, encoding="UTF-8")

    fold_end()
    return out_tmp_dir


def report_inspect_xml(fn):
    """
    :param str fn:
    :return: list of (filename, line, problem_severity, inspect_class, description)
    :rtype: list[(str,int,str,str,str)]
    """
    # Example PyPackageRequirementsInspection.xml:
    """
  <problems is_local_tool="true">
  <problem>
    <file>file://$PROJECT_DIR$/TFUtil.py</file>
    <line>1</line>
    <module>returnn</module>
    <entry_point TYPE="file" FQNAME="file://$PROJECT_DIR$/TFUtil.py" />
    <problem_class severity="WARNING" attribute_key="WARNING_ATTRIBUTES">Package requirements</problem_class>
    <description>Package requirements 'h5py', 'theano==0.9' are not satisfied</description>
  </problem>
  </problems>
  """
    inspect_class = os.path.splitext(os.path.basename(fn))[0]  # e.g. "PyPackageRequirementsInspection"
    root = ElementTree.parse(fn).getroot()
    assert isinstance(root, ElementTree.Element)
    if root.tag != "problems":
        # e.g. DuplicatedCode_aggregate.xml (PyCharm 2026.2): a summary artifact, not a problem list
        print("Skipping %s (root tag %r, not a problems list)" % (os.path.basename(fn), root.tag))
        return []
    assert root.tag == "problems"
    result = []
    for problem in root.findall("./problem"):
        assert isinstance(problem, ElementTree.Element)
        assert problem.tag == "problem"
        filename = problem.find("./file").text.strip()
        if filename.startswith("file://$PROJECT_DIR$/"):
            filename = filename[len("file://$PROJECT_DIR$/") :]
        line = int(problem.find("./line").text.strip())
        problem_severity = problem.find("./problem_class").attrib["severity"]
        description = problem.find("./description").text.strip()

        # Do some filtering for false positives. This is ugly, but the other solution would be to ignore all of them.
        possible_false_positive = False
        if inspect_class == "PyArgumentListInspection" and "'d0' unfilled" in description:  # Numpy false positive
            possible_false_positive = True
        if inspect_class == "PyArgumentListInspection" and "'d1' unfilled" in description:  # Numpy false positive
            possible_false_positive = True
        if inspect_class == "PyArgumentListInspection" and "'self' unfilled" in description:  # Numpy false positive
            possible_false_positive = True
        if inspect_class == "PyStringFormatInspection" and "Unexpected type None" in description:
            possible_false_positive = True
        if possible_false_positive:
            problem_severity = "POSSIBLE-FALSE %s" % problem_severity

        result.append((filename, line, problem_severity, inspect_class, description))

    return result


def report_inspect_dir(
    inspect_xml_dir,
    inspect_class_blacklist=None,
    inspect_class_not_counted=None,
    inspect_class_msg_not_counted=None,
    ignore_count_for_files=(),
):
    """
    :param str inspect_xml_dir:
    :param set[str]|None inspect_class_blacklist:
    :param set[str]|None inspect_class_not_counted:
    :param list[(str,str)]|None inspect_class_msg_not_counted: (inspect class, description regex) pairs:
        matching findings are reported but not counted.
        More precise than inspect_class_not_counted when only one sub-analysis of an otherwise
        useful inspection is noisy. How many findings each pattern filtered is reported at the end,
        so drift stays visible (a filter that suddenly matches thousands more is a red flag).
    :param set[str]|tuple[str]|None ignore_count_for_files:
    :return: count of reports
    :rtype: int
    """
    import re

    msg_not_counted = [(cls, re.compile(pat)) for cls, pat in (inspect_class_msg_not_counted or [])]
    msg_not_counted_hits = {(cls, pat.pattern): 0 for cls, pat in msg_not_counted}

    def _msg_match(inspect_class_, description_):
        for cls_, pat_ in msg_not_counted:
            if inspect_class_ == cls_ and pat_.search(description_):
                return cls_, pat_.pattern
        return None

    if os.path.isfile(inspect_xml_dir):
        assert inspect_xml_dir.endswith(".xml")
        inspect_xml_files = [inspect_xml_dir]
    else:
        assert os.path.isdir(inspect_xml_dir)
        inspect_xml_files = list(glob(inspect_xml_dir + "/*.xml"))
        assert inspect_xml_files

    inspections = []
    for fn in inspect_xml_files:
        inspections.extend(report_inspect_xml(fn))
    inspections.sort()
    inspections.append((None, None, None, None, None))  # final marker

    # copy
    inspect_class_blacklist = set(inspect_class_blacklist or ())
    inspect_class_not_counted = set(inspect_class_not_counted or ())

    # maybe update inspect_class_not_counted
    from lint_common import find_all_py_source_files

    returnn_py_source_files = set(find_all_py_source_files())
    all_files = set()
    relevant_inspections_for_file = set()
    explicitly_ignored_files = ignore_count_for_files
    ignore_count_for_files = set(ignore_count_for_files)
    for filename, line, problem_severity, inspect_class, description in inspections:
        all_files.add(filename)
        if filename not in returnn_py_source_files:
            continue
        if inspect_class in inspect_class_blacklist:
            continue
        if inspect_class in inspect_class_not_counted:
            continue
        if _msg_match(inspect_class, description):
            continue
        if problem_severity.startswith("POSSIBLE-FALSE "):
            continue
        relevant_inspections_for_file.add(filename)
    for filename in all_files:
        if filename not in relevant_inspections_for_file:
            ignore_count_for_files.add(filename)

    print("Reporting individual files.")
    color = better_exchook.Color()
    # Files with zero reports are listed as such at the end:
    # "no reports at all" for most files usually means the inspection did not really run
    # (e.g. unresolved interpreter/profile), NOT that the code is clean --
    # silently skipping them made exactly that failure look like a pass.
    total_relevant_count = 0
    file_count = None
    last_filename = None
    for filename, line, problem_severity, inspect_class, description in inspections:
        if filename and filename not in returnn_py_source_files:
            continue  # for now, to not spam Travis too much
        if inspect_class in inspect_class_blacklist:
            continue

        if filename != last_filename:
            if last_filename:
                if last_filename in explicitly_ignored_files:
                    msg = color.color("This file is on the ignore list.", color=gray_color)
                elif last_filename not in returnn_py_source_files:
                    msg = color.color(
                        "This file is not part of the official RETURNN Python source code.", color=gray_color
                    )
                elif last_filename in ignore_count_for_files:
                    msg = color.color("The inspection reports for this file are all non critical.", color=gray_color)
                else:
                    msg = color.color("The inspection reports for this file are fatal!", color="red")
                print(msg)
                fold_end()
            if filename:
                file_msg = color.color(
                    "File: %s" % filename, color=gray_color if filename in ignore_count_for_files else "red"
                )
                if github_env:
                    fold_start(file_msg)
                else:
                    fold_start("inspect.%s" % filename)
                    print(file_msg)
            last_filename = filename
            file_count = 0
        if not filename:
            continue
        if filename in ignore_count_for_files and file_count >= 10:
            if file_count == 10:
                print("... (further warnings skipped)")
            file_count += 1
            continue

        msg = "%s:%i: %s %s: %s" % (filename, line, problem_severity, inspect_class, description)
        msg_counted = True
        if inspect_class in inspect_class_not_counted:
            msg_counted = False
        matched_pattern = _msg_match(inspect_class, description)
        if matched_pattern:
            msg_not_counted_hits[matched_pattern] += 1
            msg_counted = False
        if problem_severity.startswith("POSSIBLE-FALSE "):
            msg_counted = False
        if msg_counted:
            print(color.color(msg, color="red"))
            if filename not in ignore_count_for_files:
                total_relevant_count += 1
        else:
            print(color.color(msg, color=gray_color))
        file_count += 1

    files_without_reports = sorted(returnn_py_source_files - all_files)
    fold_start("inspect.files_without_reports")
    print(
        "RETURNN Python source files WITHOUT any inspection report (%i of %i):"
        % (len(files_without_reports), len(returnn_py_source_files))
    )
    for filename in files_without_reports:
        print("File: %s (No reports for this file.)" % filename)
    fold_end()
    if len(files_without_reports) == len(returnn_py_source_files):
        print("WARNING: NO file got any inspection report -- the inspection likely did not run properly.")

    if msg_not_counted:
        print("Findings filtered as not-counted by message pattern:")
        for (cls_, pattern_), n_ in sorted(msg_not_counted_hits.items()):
            print("  %6i  %s: /%s/" % (n_, cls_, pattern_))

    print("Total relevant inspection reports:", total_relevant_count)
    return total_relevant_count


def main():
    """
    Main entry point for this script.
    """
    if not os.environ.get("GITHUB_ACTIONS") and not os.environ.get("PYCHARM_INSPECT_CONFIG_DIR"):
        # Local (non-CI) run: NEVER default to the user's real IDE config --
        # a running IDE holds the single-instance lock (the inspect then just dies)
        # and would get its jdk.table.xml edited underneath it.
        # Persistent cache dir (not per-run tmp) so the generated python stubs survive across runs.
        # PER-INTERPRETER subdir: config+system MUST NOT be shared across envs -- the IDE
        # rewrites jdk.table.xml on shutdown from its own cached model, silently reverting a
        # freshly registered interpreter to the previous env's (observed: every "torch2.12"
        # inspection actually ran against the first run's torch2.7 site-packages).
        import hashlib

        _env_tag = hashlib.sha1(sys.executable.encode()).hexdigest()[:10]
        _base = os.path.expanduser("~/.cache/returnn-pycharm-inspect") + "/" + _env_tag
        os.environ["PYCHARM_INSPECT_CONFIG_DIR"] = _base + "/config"
        os.environ.setdefault("PYCHARM_INSPECT_SYSTEM_DIR", _base + "/system")
        os.makedirs(os.environ["PYCHARM_INSPECT_CONFIG_DIR"], exist_ok=True)
        os.makedirs(os.environ["PYCHARM_INSPECT_SYSTEM_DIR"], exist_ok=True)
        print("PyCharm inspect isolated dirs:", _base, "(interpreter: %s)" % sys.executable)
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--xml")
    arg_parser.add_argument("--pycharm")
    arg_parser.add_argument("--setup_pycharm_only", action="store_true")
    arg_parser.add_argument(
        "--install_py_deps",
        action="store_true",
        help="pip-install TF and further packages the inspection expects (CI passes this; "
        "without it, the current env is used as-is and never modified)",
    )
    arg_parser.add_argument("--skip_setup_pycharm", action="store_true")
    arg_parser.add_argument("--skip_pycharm_inspect", action="store_true", help="only PEP8")
    arg_parser.add_argument("--files", nargs="*")
    args = arg_parser.parse_args()

    from lint_common import ignore_count_for_files

    inspect_kwargs = dict(
        inspect_class_blacklist={},
        inspect_class_msg_not_counted=[
            # PyCharm 2026.2 Optional/None type-flow analysis (mypy union-attr / pyright
            # reportOptionalMemberAccess equivalent): technically right, but the codebase predates
            # strict Optional narrowing, so most hits are guarded-by-invariant false alarms.
            # Message-level, NOT class-level: real unresolved names must stay counted.
            # Mid-term plan: narrow Optionals (assert x is not None) in code we touch, then drop this.
            ("PyUnresolvedReferencesInspection", r"^Member 'None' of "),
            ("PyStringConversionWithoutDunderMethodInspection", r"^Type 'None' doesn't define "),
            # historic Dataset-family API loosening (param names/extras differ from the base);
            # aligning ~50 signatures is API churn, not a lint fix
            ("PyMethodOverridingInspection", r"^Signature of method "),
            # same Optional-flow family as the Member-'None' filter above
            ("PyCallingNonCallableInspection", r"^'None' object is not callable"),
            # indexing artifact: Cython's bundled numpy .pxd shadow joins the union type
            # whenever Cython is installed in the env; not a property of our code
            ("PyUnresolvedReferencesInspection", r"^Member 'Cython\.Includes\.numpy' of "),
            # Stub gaps, no code fix possible (attribute exists at runtime, stub omits it):
            # librosa does not re-export .feature in its stub; torch stubs miss the dtype alias.
            ("PyUnresolvedReferencesInspection", r"^Cannot find reference '(?:feature|__version__)' in 'librosa'"),
            ("PyUnresolvedReferencesInspection", r"^Cannot find reference 'dtype' in 'torch'"),
            # optional deps, imported behind guards / lazily:
            ("PyUnresolvedReferencesInspection", r"^Module '(?:transformers|load_file)' not found"),
            ("PyUnresolvedReferencesInspection", r"^Unresolved reference 'safetensors'"),
            ("PyUnresolvedReferencesInspection", r"^No module named '(?:orbax|seaborn|tensor2tensor)'"),
            # Two filters used to sit here:
            # "Parameter 'in_spatial_dim' unfilled" and "for class '(Tensor, Dim)'".
            # They worked around PyCharm checking an rf.Module construction
            # against the base class' __call__ instead of __init__,
            # which only happened because the I*Encoder interfaces derived from abc.ABC.
            # That ABC was inert and is gone, see returnn/frontend/encoder/base.py.
            # Measured on demos/demo-rf-pt-benchmark.py, the only file either filter matched:
            # 23 findings from that root cause with ABC, 0 without.
            # union-member attr findings (member is a real class, union has '|'): triaged 2026-08-06
            # (205 distinct sites incl. every crash-looking candidate read individually) -- all were
            # duck-typing idioms (type[X] | X, Dim | str), guarded branches PyCharm cannot correlate,
            # heterogeneous-list or loose annotations, or indexing artifacts ('torch | torch').
            # Keep AFTER the Member-'None' pattern so that family keeps its own count.
            ("PyUnresolvedReferencesInspection", r"^Member '[^']+' of '[^']*\|[^']*' does not have attribute "),
            # f-string format specs on inferred numeric unions (float | int | Any, numpy
            # signedinteger): PyCharm falls back to object.__format__ and rejects the spec,
            # but every such value formats fine at runtime. Triaged 2026-08-07: all 8 sites
            # (graph_capture GiB prints, file_cache ages, util stats) verified by execution.
            ("PyStringFormatInspection", r"^Format spec is not supported for "),
            # Sequence[...] is an ABC that declares no __str__/__repr__/__format__ of its own, so
            # PyCharm flags every interpolation of a value annotated that way -- even explicit !r.
            # At runtime these are list/tuple (spot-checked across tensor_dict, array_,
            # _tensor_extra, jax/_backend, decoder/transformer: dims, perm, padding in error and
            # assert messages), whose repr is informative and delegates to the elements.
            ("PyStringConversionWithoutDunderMethodInspection", r"^Type 'Sequence\["),
            # numpy scalar/array types: the bundled stubs do not DECLARE __repr__, but the runtime
            # classes all define it (verified by execution: dtype -> dtype('float32'), ndarray ->
            # array([...]), float64 -> np.float64(1.5)). Same stub-artifact class as the Cython
            # numpy filter above.
            (
                "PyStringConversionWithoutDunderMethodInspection",
                r"^Type '(?:ndarray|dtype|number|bool_|float\d+|int\d+|uint\d+|complex\d+)\b",
            ),
            # interpolating a CLASS gives "<class 'module.Name'>", which is precisely what the
            # type-mismatch messages these sites live in want to say. Sampled 12 of the 60 sites,
            # spread over 12 different files: every one is `type(x)` (or str(type(x))) inside a
            # TypeError / assert message. Unlike most inspections this one can never indicate a
            # runtime fault -- the worst case is a message that reads awkwardly.
            ("PyStringConversionWithoutDunderMethodInspection", r"^Type 'type' string value"),
            # the "string value might not be useful" sub-check bypasses the profile's
            # ignoredTypes (measured 2026-08-11); same rationale as 'type' above:
            # str() of a callable/object in a message is exactly what we want printed
            (
                "PyStringConversionWithoutDunderMethodInspection",
                r"^Type '(?:function|FunctionType|BuiltinFunctionType|MethodType|object)' string value",
            ),
            # `return _sdpa_no(...)` / `return _flex_no(...)` is a DELIBERATE, documented idiom:
            # both helpers warn once and return None so an `-> Optional[Tensor]` fast path can bail
            # out in one line (see _sdpa_no's own docstring in _packed_backend.py). 28 of the 36
            # findings of this class are those two helpers; rewriting every call site to
            # `_sdpa_no(...); return None` would be churn against the documented intent.
            ("PyNoneFunctionAssignmentInspection", r"^Function '_(?:sdpa|flex)_no' doesn't return anything"),
            # A note carried by shutil.which's STUB, not a property of our call: it fires even on
            # `shutil.which("cc")` with a string literal (all 9 findings are the 9 shutil.which
            # calls in native_code_compiler.py, incl. literal args). We never pass a PathLike, and
            # RETURNN does not support Windows anyway.
            ("PyDeprecationInspection", r"^On Windows before Python 3\.12, using a PathLike as `cmd`"),
        ],
        inspect_class_not_counted={
            # Here we disable more than what you would do in the IDE.
            # The aim is that any left over warnings are always indeed important and should be fixed.
            # False alarms.
            "PyTypeCheckerInspection",  # too much false alarms: https://youtrack.jetbrains.com/issue/PY-34893
            # Not critical.
            "SpellCheckingInspection",  # way too much for now...
            "GrazieInspection",  # grammar
            # GrazieStyle is a SEPARATE class from GrazieInspection, so excluding the latter left
            # 89 of these counted. Triaged 2026-08-08 by message: 27x "etc. requires a period",
            # ~35x "long sentence (40-66 words)", the rest redundant phrases, British-vs-American
            # spelling and adverb placement -- prose style in comments and docstrings, none of it
            # a defect. The Grazie findings that WOULD be worth acting on (repeated word, unpaired
            # bracket, two consecutive dots) are GrazieInspection and already not counted.
            "GrazieStyle",  # prose style (as above)
            "PyClassHasNoInitInspection",  # not relevant?
            "PyMethodMayBeStaticInspection",  # not critical
            "DuplicatedCode",  # fires on ancient demo scripts; dedup there has no value
            # Does not work correctly here?
            "PyPackageRequirementsInspection",  # TODO only with newer PyCharm versions?
        },
        ignore_count_for_files=ignore_count_for_files,
    )

    if args.xml:
        if report_inspect_dir(args.xml, **inspect_kwargs) > 0:
            sys.exit(1)
        return

    if args.pycharm:
        pycharm_dir = args.pycharm
        check_pycharm_dir(pycharm_dir)
    else:
        pycharm_dir = install_pycharm()

    if not args.skip_setup_pycharm and not args.skip_pycharm_inspect:
        setup_pycharm_python_interpreter(pycharm_dir=pycharm_dir, install_py_deps=args.install_py_deps)
    if args.setup_pycharm_only:
        return

    src_dir = prepare_src_dir(files=args.files)
    # --files: analyze only the smallest dir covering them (the rest of the project stays
    # indexed for type inference), and report only those files
    scope_dir = os.path.commonpath([os.path.dirname(f) for f in args.files]) if args.files else None
    res_dir = run_inspect(
        pycharm_dir=pycharm_dir,
        src_dir=src_dir,
        skip_pycharm_inspect=args.skip_pycharm_inspect,
        scope_dir=scope_dir,
    )
    if report_inspect_dir(res_dir, **inspect_kwargs) > 0:
        sys.exit(1)


if __name__ == "__main__":
    better_exchook.install()
    main()
