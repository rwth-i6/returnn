
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

from returnn.__setup__ import main


if __name__ == "__main__":
  main()
