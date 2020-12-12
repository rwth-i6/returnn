
"""
Usage:

Create ~/.pypirc with info:

    [distutils]
    index-servers =
        pypi

    [pypi]
    repository: https://upload.pypi.org/legacy/
    username: ...
    password: ...

(Not needed anymore) Registering the project: python3 setup.py register
New release: python3 setup.py sdist upload

I had some trouble at some point, and this helped:
pip3 install --user twine
python3 setup.py sdist
twine upload dist/*.tar.gz

See also MANIFEST.in for included files.

For debugging this script:

python3 setup.py sdist
pip3 install --user dist/*.tar.gz -v
(Without -v, all stdout/stderr from here will not be shown.)

"""

import os
import shutil
from returnn.__setup__ import get_version_str, debug_print_file


def main():
  """
  Setup main entry
  """
  # Do not use current time as fallback for the version anymore,
  # as this would result in a version which can be bigger than what we actually have,
  # so this would not be useful at all.
  long_version = get_version_str(verbose=True, fallback="1.0.0+setup-fallback-version", long=True)
  version = long_version[:long_version.index("+")]

  if os.environ.get("DEBUG", "") == "1":
    debug_print_file(".")
    debug_print_file("PKG-INFO")
    debug_print_file("pip-egg-info")
    debug_print_file("pip-egg-info/returnn.egg-info")
    debug_print_file("pip-egg-info/returnn.egg-info/SOURCES.txt")  # like MANIFEST

  if os.path.exists("PKG-INFO"):
    if os.path.exists("MANIFEST"):
      print("package_data, found PKG-INFO and MANIFEST")
      package_data = open("MANIFEST").read().splitlines() + ["PKG-INFO"]
    else:
      print("package_data, found PKG-INFO, no MANIFEST, use *")
      # Currently the setup will ignore all other data except in returnn/.
      # At least make the version available.
      shutil.copy("PKG-INFO", "returnn/")
      shutil.copy("_setup_info_generated.py", "returnn/")
      # Just using package_data = ["*"] would only take files from current dir.
      package_data = []
      for root, dirs, files in os.walk('.'):
        for file in files:
          package_data.append(os.path.join(root, file))
  else:
    print("dummy package_data, does not matter, likely you are running sdist")
    with open("_setup_info_generated.py", "w") as f:
      f.write("version = %r\n" % version)
      f.write("long_version = %r\n" % long_version)
    package_data = ["MANIFEST", "_setup_info_generated.py"]

  from distutils.core import setup
  setup(
    name='returnn',
    version=version,
    packages=['returnn'],
    include_package_data=True,
    package_data={'returnn': package_data},  # filtered via MANIFEST.in
    description='The RWTH extensible training framework for universal recurrent neural networks',
    author='Albert Zeyer',
    author_email='albzey@gmail.com',
    url='https://github.com/rwth-i6/returnn/',
    license='RETURNN license',
    long_description=open('README.rst').read(),
    # https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
      'Development Status :: 5 - Production/Stable',
      'Environment :: Console',
      'Environment :: GPU',
      'Environment :: GPU :: NVIDIA CUDA',
      'Intended Audience :: Developers',
      'Intended Audience :: Education',
      'Intended Audience :: Science/Research',
      'License :: Other/Proprietary License',
      'Operating System :: MacOS :: MacOS X',
      'Operating System :: Microsoft :: Windows',
      'Operating System :: POSIX',
      'Operating System :: Unix',
      'Programming Language :: Python',
      'Programming Language :: Python :: 3',
      'Topic :: Scientific/Engineering',
      'Topic :: Scientific/Engineering :: Artificial Intelligence',
      'Topic :: Software Development :: Libraries :: Python Modules',
    ]
  )


if __name__ == "__main__":
  main()
