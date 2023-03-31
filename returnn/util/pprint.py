"""
Alternative to the original pprint module.
This one has different behavior for indentation, specifically for dicts.
Also the order of dict items are kept as-is
(which is fine for newer Python versions, which will be the insertion order).
Compare (via our ``pprint``)::
  {
    'melgan': {
      'class': 'subnetwork',
      'from': 'data',
      'subnetwork': {
        'l0': {'class': 'pad', 'mode': 'reflect', 'axes': 'spatial', 'padding': (3, 3), 'from': 'data'},
        'la1': {
          'class': 'conv',
          'from': 'l0',
          'activation': None,
          'with_bias': True,
          'n_out': 384,
          'filter_size': (7,),
          'padding': 'valid',
          'strides': (1,),
          'dilation_rate': (1,)
        },
        'lay2': {'class': 'eval', 'eval': 'tf.nn.leaky_relu(source(0), alpha=0.2)', 'from': 'la1'},
        'layer3_xxx': {
          'class': 'transposed_conv',
          'from': 'lay2',
          'activation': None,
          'with_bias': True,
          'n_out': 192,
          'filter_size': (10,),
          'strides': (5,),
          'padding': 'valid',
          'output_padding': (1,),
          'remove_padding': (3,)
        },
        'output': {'class': 'copy', 'from': 'layer3_xxx'}
      }
    },
    'output': {'class': 'copy', 'from': 'melgan'}
  }
Vs (via original ``pprint``)::
  {'melgan': {'class': 'subnetwork',
              'from': 'data',
              'subnetwork': {'l0': {'axes': 'spatial',
                                    'class': 'pad',
                                    'from': 'data',
                                    'mode': 'reflect',
                                    'padding': (3, 3)},
                             'la1': {'activation': None,
                                     'class': 'conv',
                                     'dilation_rate': (1,),
                                     'filter_size': (7,),
                                     'from': 'l0',
                                     'n_out': 384,
                                     'padding': 'valid',
                                     'strides': (1,),
                                     'with_bias': True},
                             'lay2': {'class': 'eval',
                                      'eval': 'tf.nn.leaky_relu(source(0), '
                                              'alpha=0.2)',
                                      'from': 'la1'},
                             'layer3_xxx': {'activation': None,
                                            'class': 'transposed_conv',
                                            'filter_size': (10,),
                                            'from': 'lay2',
                                            'n_out': 192,
                                            'output_padding': (1,),
                                            'padding': 'valid',
                                            'remove_padding': (3,),
                                            'strides': (5,),
                                            'with_bias': True},
                             'output': {'class': 'copy', 'from': 'layer3_xxx'}}},
   'output': {'class': 'copy', 'from': 'melgan'}}
This is a very simple implementation.
There are other similar alternatives:
* [Rich](https://github.com/willmcgugan/rich)
* [pprint++](https://github.com/wolever/pprintpp)
"""

from __future__ import annotations
from typing import Any
import sys
import numpy


def pprint(obj: Any, *, file=None, prefix="", postfix="", line_prefix="", line_postfix="\n") -> None:
    """
    Pretty-print a Python object.
    """
    if file is None:
        file = sys.stdout
    if "\n" in line_postfix and _type_simplicity_score(obj) <= _type_simplicity_limit:
        prefix = f"{line_prefix}{prefix}"
        line_prefix = ""
        postfix = postfix + line_postfix
        line_postfix = ""

    def _sub_pprint(obj_: Any, prefix_="", postfix_="", inc_indent=True):
        multi_line = "\n" in line_postfix
        if not multi_line and postfix_.endswith(","):
            postfix_ += " "
        pprint(
            obj_,
            file=file,
            prefix=prefix_,
            postfix=postfix_,
            line_prefix=(line_prefix + "  " * inc_indent) if multi_line else "",
            line_postfix=line_postfix,
        )

    def _print(s: str, is_end: bool = False):
        nonlocal prefix  # no need for is_begin, just reset prefix
        file.write(line_prefix)
        file.write(prefix)
        file.write(s)
        if is_end:
            file.write(postfix)
        file.write(line_postfix)
        if "\n" in line_postfix:
            file.flush()
        prefix = ""

    def _print_list():
        for i_, v_ in enumerate(obj):
            _sub_pprint(v_, postfix_="," if i_ < len(obj) - 1 else "")

    if isinstance(obj, list):
        if len(obj) == 0:
            _print("[]", is_end=True)
            return
        _print("[")
        _print_list()
        _print("]", is_end=True)
        return

    if isinstance(obj, tuple):
        if len(obj) == 0:
            _print("()", is_end=True)
            return
        if len(obj) == 1:
            _sub_pprint(obj[0], prefix_=f"{prefix}(", postfix_=f",){postfix}", inc_indent=False)
            return
        _print("(")
        _print_list()
        _print(")", is_end=True)
        return

    if isinstance(obj, set):
        if len(obj) == 0:
            _print("set()", is_end=True)
            return
        _print("{")
        _print_list()
        _print("}", is_end=True)
        return

    if isinstance(obj, dict):
        if len(obj) == 0:
            _print("{}", is_end=True)
            return
        _print("{")
        for i, (k, v) in enumerate(obj.items()):
            _sub_pprint(v, prefix_=f"{k!r}: ", postfix_="," if i < len(obj) - 1 else "")
        _print("}", is_end=True)
        return

    if isinstance(obj, numpy.ndarray):
        _sub_pprint(
            obj.tolist(),
            prefix_=f"{prefix}numpy.array(",
            postfix_=f", dtype=numpy.{obj.dtype}){postfix}",
            inc_indent=False,
        )
        return

    # fallback
    _print(repr(obj), is_end=True)


def pformat(obj: Any) -> str:
    """
    Pretty-format a Python object.
    """
    import io

    s = io.StringIO()
    pprint(obj, file=s)
    return s.getvalue()


_type_simplicity_limit = 120.0  # magic number


def _type_simplicity_score(obj: Any, _offset=0.0) -> float:
    """
    :param Any obj:
    :param float _offset:
    :return: a score, which is a very rough estimate of len(repr(o)), calculated efficiently
    """
    _spacing = 2.0
    if isinstance(obj, bool):
        return 4.0 + _offset
    if isinstance(obj, (int, numpy.integer)):
        if obj == 0:
            return 1.0 + _offset
        return 1.0 + numpy.log10(abs(obj)) + _offset
    if isinstance(obj, str):
        return 2.0 + len(obj) + _offset
    if isinstance(obj, (float, complex, numpy.number)):
        return len(repr(obj)) + _offset
    if isinstance(obj, (tuple, list, set)):
        for x in obj:
            _offset = _type_simplicity_score(x, _offset=_offset + _spacing)
            if _offset > _type_simplicity_limit:
                break
        return _offset
    if isinstance(obj, dict):
        for x in obj.values():  # ignore keys...
            _offset = _type_simplicity_score(x, _offset=_offset + 10.0 + _spacing)  # +10 for key
            if _offset > _type_simplicity_limit:
                break
        return _offset
    if isinstance(obj, numpy.ndarray):
        _offset += 10.0  # prefix/postfix
        if obj.size * 2.0 + _offset > _type_simplicity_limit:  # too big already?
            return obj.size * 2.0 + _offset
        if str(obj.dtype).startswith("int"):
            a = _type_simplicity_score(numpy.max(numpy.abs(obj))) + _spacing
            return obj.size * a + _offset
        a = max([_type_simplicity_score(x) for x in obj.flatten()]) + _spacing
        return obj.size * a + _offset
    # Unknown object. Fallback > _type_simplicity_limit.
    return _type_simplicity_limit + 1.0 + _offset
