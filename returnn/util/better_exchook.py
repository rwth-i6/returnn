
# -*- coding: utf-8 -*-
# Copyright (c) 2011-2021, Albert Zeyer, www.az2000.de
# All rights reserved.
# file created 2011-04-15


# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
https://github.com/albertz/py_better_exchook

This is a simple replacement for the standard Python exception handler (sys.excepthook).
In addition to what the standard handler does, it also prints all referenced variables
(no matter if local, global or builtin) of the code line of each stack frame.
See below for some examples and some example output.

See these functions:

- better_exchook
- format_tb / print_tb
- iter_traceback
- get_current_frame
- dump_all_thread_tracebacks
- install
- replace_traceback_format_tb

Although there might be a few more useful functions, thus we export all of them.

Also see the demo/tests at the end.
"""

from __future__ import print_function

import sys
import os
import os.path
import threading
import keyword
import inspect
import contextlib
try:
    import typing
except ImportError:
    typing = None

try:
    from traceback import StackSummary, FrameSummary
except ImportError:
    class _Dummy:
        pass
    StackSummary = FrameSummary = _Dummy

# noinspection PySetFunctionToLiteral,SpellCheckingInspection
py_keywords = set(keyword.kwlist) | set(["None", "True", "False"])

_cur_pwd = os.getcwd()
_threading_main_thread = threading.main_thread() if hasattr(threading, "main_thread") else None

try:
    # noinspection PyUnresolvedReferences,PyUnboundLocalVariable
    unicode
except NameError:  # Python3
    unicode = str   # Python 3 compatibility

try:
    # noinspection PyUnresolvedReferences,PyUnboundLocalVariable
    raw_input
except NameError:  # Python3
    raw_input = input


PY3 = sys.version_info[0] >= 3


def parse_py_statement(line):
    """
    Parse Python statement into tokens.
    Note that this is incomplete.
    It should be simple and fast and just barely enough for what we need here.

    Reference:
    https://docs.python.org/3/reference/lexical_analysis.html

    :param str line:
    :return: yields (type, value)
    :rtype: typing.Iterator[typing.Tuple[str,str]]
    """
    state = 0
    cur_token = ""
    spaces = " \t\n"
    ops = ".,;:+-*/%&!=|(){}[]^<>"
    i = 0

    def _escape_char(_c):
        if _c == "n":
            return "\n"
        elif _c == "t":
            return "\t"
        else:
            return _c

    while i < len(line):
        c = line[i]
        i += 1
        if state == 0:
            if c in spaces:
                pass
            elif c in ops:
                yield "op", c
            elif c == "#":
                state = 6
            elif c == '"':
                state = 1
            elif c == "'":
                state = 2
            else:
                cur_token = c
                state = 3
        elif state == 1:  # string via "
            if c == "\\":
                state = 4
            elif c == "\"":
                yield "str", cur_token
                cur_token = ""
                state = 0
            else:
                cur_token += c
        elif state == 2:  # string via '
            if c == "\\":
                state = 5
            elif c == "'":
                yield "str", cur_token
                cur_token = ""
                state = 0
            else:
                cur_token += c
        elif state == 3:  # identifier
            if c in spaces + ops + "#":
                yield "id", cur_token
                cur_token = ""
                state = 0
                i -= 1
            elif c == '"':  # identifier is string prefix
                cur_token = ""
                state = 1
            elif c == "'":  # identifier is string prefix
                cur_token = ""
                state = 2
            else:
                cur_token += c
        elif state == 4:  # escape in "
            cur_token += _escape_char(c)
            state = 1
        elif state == 5:  # escape in '
            cur_token += _escape_char(c)
            state = 2
        elif state == 6:  # comment
            cur_token += c
    if state == 3:
        yield "id", cur_token
    elif state == 6:
        yield "comment", cur_token


def parse_py_statements(source_code):
    """
    :param str source_code:
    :return: via :func:`parse_py_statement`
    :rtype: typing.Iterator[typing.Tuple[str,str]]
    """
    for line in source_code.splitlines():
        for t in parse_py_statement(line):
            yield t


def grep_full_py_identifiers(tokens):
    """
    :param typing.Iterable[(str,str)] tokens:
    :rtype: typing.Iterator[str]
    """
    global py_keywords
    tokens = list(tokens)
    i = 0
    while i < len(tokens):
        token_type, token = tokens[i]
        i += 1
        if token_type != "id":
            continue
        while i+1 < len(tokens) and tokens[i] == ("op", ".") and tokens[i+1][0] == "id":
            token += "." + tokens[i+1][1]
            i += 2
        if token == "":
            continue
        if token in py_keywords:
            continue
        if token[0] in ".0123456789":
            continue
        yield token


def set_linecache(filename, source):
    """
    The :mod:`linecache` module has some cache of the source code for the current source.
    Sometimes it fails to find the source of some files.
    We can explicitly set the source for some filename.

    :param str filename:
    :param str source:
    :return: nothing
    """
    import linecache
    linecache.cache[filename] = None, None, [line+'\n' for line in source.splitlines()], filename


# noinspection PyShadowingBuiltins
def simple_debug_shell(globals, locals):
    """
    :param dict[str] globals:
    :param dict[str] locals:
    :return: nothing
    """
    try:
        import readline
    except ImportError:
        pass  # ignore
    compile_string_fn = "<simple_debug_shell input>"
    while True:
        try:
            s = raw_input("> ")
        except (KeyboardInterrupt, EOFError):
            print("breaked debug shell: " + sys.exc_info()[0].__name__)
            break
        if s.strip() == "":
            continue
        try:
            c = compile(s, compile_string_fn, "single")
        except Exception as e:
            print("%s : %s in %r" % (e.__class__.__name__, str(e), s))
        else:
            set_linecache(compile_string_fn, s)
            # noinspection PyBroadException
            try:
                ret = eval(c, globals, locals)
            except (KeyboardInterrupt, SystemExit):
                print("debug shell exit: " + sys.exc_info()[0].__name__)
                break
            except Exception:
                print("Error executing %r" % s)
                better_exchook(*sys.exc_info(), autodebugshell=False)
            else:
                # noinspection PyBroadException
                try:
                    if ret is not None:
                        print(ret)
                except Exception:
                    print("Error printing return value of %r" % s)
                    better_exchook(*sys.exc_info(), autodebugshell=False)


# keep non-PEP8 argument name for compatibility
# noinspection PyPep8Naming
def debug_shell(user_ns, user_global_ns, traceback=None, execWrapper=None):
    """
    Spawns some interactive shell. Tries to use IPython if available.
    Falls back to :func:`pdb.post_mortem` or :func:`simple_debug_shell`.

    :param dict[str] user_ns:
    :param dict[str] user_global_ns:
    :param traceback:
    :param execWrapper:
    :return: nothing
    """
    ipshell = None
    try:
        # noinspection PyPackageRequirements
        import IPython
        have_ipython = True
    except ImportError:
        have_ipython = False

    if not ipshell and traceback and have_ipython:
        # noinspection PyBroadException
        try:
            # noinspection PyPackageRequirements,PyUnresolvedReferences
            from IPython.core.debugger import Pdb
            # noinspection PyPackageRequirements,PyUnresolvedReferences
            from IPython.terminal.debugger import TerminalPdb
            # noinspection PyPackageRequirements,PyUnresolvedReferences
            from IPython.terminal.ipapp import TerminalIPythonApp
            ipapp = TerminalIPythonApp.instance()
            ipapp.interact = False  # Avoid output (banner, prints)
            ipapp.initialize(argv=[])
            def_colors = ipapp.shell.colors
            pdb_obj = TerminalPdb(def_colors)
            pdb_obj.botframe = None  # not sure. exception otherwise at quit

            def ipshell():
                """
                Run the IPython shell.
                """
                pdb_obj.interaction(None, traceback=traceback)

        except Exception:
            print("IPython Pdb exception:")
            better_exchook(*sys.exc_info(), autodebugshell=False, file=sys.stdout)

    if not ipshell and have_ipython:
        # noinspection PyBroadException
        try:
            # noinspection PyPackageRequirements,PyUnresolvedReferences
            import IPython
            # noinspection PyPackageRequirements,PyUnresolvedReferences
            import IPython.terminal.embed

            class DummyMod(object):
                """Dummy module"""
            module = DummyMod()
            module.__dict__ = user_global_ns
            module.__name__ = "_DummyMod"
            if "__name__" not in user_ns:
                user_ns = user_ns.copy()
                user_ns["__name__"] = "_DummyUserNsMod"
            ipshell = IPython.terminal.embed.InteractiveShellEmbed.instance(
                user_ns=user_ns, user_module=module)
        except Exception:
            print("IPython not available:")
            better_exchook(*sys.exc_info(), autodebugshell=False, file=sys.stdout)
        else:
            if execWrapper:
                old = ipshell.run_code
                ipshell.run_code = lambda code: execWrapper(lambda: old(code))
    if ipshell:
        ipshell()
    else:
        print("Use simple pdb debug shell:")
        if traceback:
            import pdb
            pdb.post_mortem(traceback)
        else:
            simple_debug_shell(user_global_ns, user_ns)


def output_limit():
    """
    :return: num chars
    :rtype: int
    """
    return 300


def fallback_findfile(filename):
    """
    :param str filename:
    :return: try to find the full filename, e.g. in modules, etc
    :rtype: str|None
    """
    mods = [m for m in list(sys.modules.values()) if m and getattr(m, "__file__", None) and filename in m.__file__]
    if len(mods) == 0:
        return None
    alt_fn = mods[0].__file__
    if alt_fn[-4:-1] == ".py":
        alt_fn = alt_fn[:-1]  # *.pyc or whatever
    if not os.path.exists(alt_fn) and alt_fn.startswith("./"):
        # Maybe current dir changed.
        alt_fn2 = _cur_pwd + alt_fn[1:]
        if os.path.exists(alt_fn2):
            return alt_fn2
        # Try dirs of some other mods.
        for m in ["__main__", "better_exchook"]:
            if hasattr(sys.modules.get(m), "__file__"):
                alt_fn2 = os.path.dirname(sys.modules[m].__file__) + alt_fn[1:]
                if os.path.exists(alt_fn2):
                    return alt_fn2
    return alt_fn


def is_source_code_missing_brackets(source_code, prioritize_missing_open=False):
    """
    We check whether this source code snippet (e.g. one line) is complete/even w.r.t. opening/closing brackets.

    :param str source_code:
    :param bool prioritize_missing_open: once we found any missing open bracket, directly return -1
    :return: 1 if missing_close, -1 if missing_open, 0 otherwise.
        I.e. whether there are missing open/close brackets.
        E.g. this would mean that you might want to include the prev/next source code line as well in the stack trace.
    :rtype: int
    """
    open_brackets = "[{("
    close_brackets = "]})"
    last_bracket = [-1]  # stack
    counters = [0] * len(open_brackets)
    missing_open = False
    for t_type, t_content in list(parse_py_statements(source_code)):
        if t_type != "op":
            continue  # we are from now on only interested in ops (including brackets)
        if t_content in open_brackets:
            idx = open_brackets.index(t_content)
            counters[idx] += 1
            last_bracket.append(idx)
        elif t_content in close_brackets:
            idx = close_brackets.index(t_content)
            if last_bracket[-1] == idx:
                counters[idx] -= 1
                del last_bracket[-1]
            else:
                if prioritize_missing_open:
                    return -1
                missing_open = True
    missing_close = not all([c == 0 for c in counters])
    if missing_close:
        return 1
    if missing_open:
        return -1
    return 0


def is_source_code_missing_open_brackets(source_code):
    """
    We check whether this source code snippet (e.g. one line) is complete/even w.r.t. opening/closing brackets.

    :param str source_code:
    :return: whether there are missing open brackets.
        E.g. this would mean that you might want to include the previous source code line as well in the stack trace.
    :rtype: bool
    """
    return is_source_code_missing_brackets(source_code, prioritize_missing_open=True) < 0


def get_source_code(filename, lineno, module_globals=None):
    """
    :param str filename:
    :param int lineno:
    :param dict[str]|None module_globals:
    :return: source code of that line (including newline)
    :rtype: str
    """
    import linecache
    linecache.checkcache(filename)
    source_code = linecache.getline(filename, lineno, module_globals)
    # In case of a multi-line statement, lineno is usually the last line.
    # We are checking for missing open brackets and add earlier code lines.
    start_line = end_line = lineno
    lines = None
    while True:
        missing_bracket_level = is_source_code_missing_brackets(source_code)
        if missing_bracket_level == 0:
            break
        if not lines:
            lines = linecache.getlines(filename, module_globals)
        if missing_bracket_level < 0:  # missing open bracket, add prev line
            start_line -= 1
            if start_line < 1:  # 1-indexed
                break
        else:
            end_line += 1
            if end_line > len(lines):  # 1-indexed
                break
        source_code = "".join(lines[start_line - 1:end_line])  # 1-indexed
    return source_code


def str_visible_len(s):
    """
    :param str s:
    :return: len without escape chars
    :rtype: int
    """
    import re
    # via: https://github.com/chalk/ansi-regex/blob/master/index.js
    s = re.sub("[\x1b\x9b][\\[()#;?]*(?:[0-9]{1,4}(?:;[0-9]{0,4})*)?[0-9A-PRZcf-nqry=><]", "", s)
    return len(s)


def add_indent_lines(prefix, s):
    """
    :param str prefix:
    :param str s:
    :return: s with prefix indent added to all lines
    :rtype: str
    """
    if not s:
        return prefix
    prefix_len = str_visible_len(prefix)
    lines = s.splitlines(True)
    return "".join([prefix + lines[0]] + [" " * prefix_len + line for line in lines[1:]])


def get_indent_prefix(s):
    """
    :param str s:
    :return: the indent spaces of s
    :rtype: str
    """
    return s[:len(s) - len(s.lstrip())]


def get_same_indent_prefix(lines):
    """
    :param list[] lines:
    :rtype: str|None
    """
    if not lines:
        return ""
    prefix = get_indent_prefix(lines[0])
    if not prefix:
        return ""
    if all([line.startswith(prefix) for line in lines]):
        return prefix
    return None


def remove_indent_lines(s):
    """
    :param str s:
    :return: remove as much indentation as possible
    :rtype: str
    """
    if not s:
        return ""
    lines = s.splitlines(True)
    prefix = get_same_indent_prefix(lines)
    if prefix is None:  # not in expected format. just lstrip all lines
        return "".join([line.lstrip() for line in lines])
    return "".join([line[len(prefix):] for line in lines])


def replace_tab_indent(s, replace="    "):
    """
    :param str s: string with tabs
    :param str replace: e.g. 4 spaces
    :rtype: str
    """
    prefix = get_indent_prefix(s)
    return prefix.replace("\t", replace) + s[len(prefix):]


def replace_tab_indents(s, replace="    "):
    """
    :param str s: multi-line string with tabs
    :param str replace: e.g. 4 spaces
    :rtype: str
    """
    lines = s.splitlines(True)
    return "".join([replace_tab_indent(line, replace) for line in lines])


def to_bool(s, fallback=None):
    """
    :param str s: str to be converted to bool, e.g. "1", "0", "true", "false"
    :param T fallback: if s is not recognized as a bool
    :return: boolean value, or fallback
    :rtype: bool|T
    """
    if not s:
        return fallback
    s = s.lower()
    if s in ["1", "true", "yes", "y"]:
        return True
    if s in ["0", "false", "no", "n"]:
        return False
    return fallback


class Color:
    """
    Helper functions provided to perform terminal coloring.
    """

    ColorIdxTable = {k: i for (i, k) in enumerate([
        "black", "red", "green", "yellow", "blue", "magenta", "cyan", "white"])}

    @classmethod
    def get_global_color_enabled(cls):
        """
        :rtype: bool
        """
        return to_bool(os.environ.get("CLICOLOR", ""), fallback=True)

    @classmethod
    def is_dark_terminal_background(cls):
        """
        :return: Whether we have a dark Terminal background color, or None if unknown.
            We currently just check the env var COLORFGBG,
            which some terminals define like "<foreground-color>:<background-color>",
            and if <background-color> in {0,1,2,3,4,5,6,8}, then we have some dark background.
            There are many other complex heuristics we could do here, which work in some cases but not in others.
            See e.g. `here <https://stackoverflow.com/questions/2507337/terminals-background-color>`__.
            But instead of adding more heuristics, we think that explicitly setting COLORFGBG would be the best thing,
            in case it's not like you want it.
        :rtype: bool|None
        """
        if os.environ.get("COLORFGBG", None):
            parts = os.environ["COLORFGBG"].split(";")
            try:
                last_number = int(parts[-1])
                if 0 <= last_number <= 6 or last_number == 8:
                    return True
                else:
                    return False
            except ValueError:  # not an integer?
                pass
        return None  # unknown (and bool(None) == False, i.e. expect light by default)

    def __init__(self, enable=None):
        """
        :param bool|None enable:
        """
        if enable is None:
            enable = self.get_global_color_enabled()
        self.enable = enable
        self._dark_terminal_background = self.is_dark_terminal_background()
        # Set color palettes (will be used sometimes as bold, sometimes as normal).
        # 5 colors, for: code/general, error-msg, string, comment, line-nr.
        # Try to set them in a way such that if we guessed the terminal background color wrongly,
        # it is still not too bad (although people might disagree here...).
        if self._dark_terminal_background:
            self.fg_colors = ["yellow", "red", "cyan", "white", "magenta"]
        else:
            self.fg_colors = ["blue", "red", "cyan", "white", "magenta"]

    def color(self, s, color=None, bold=False):
        """
        :param str s:
        :param str|None color: sth in self.ColorIdxTable
        :param bool bold:
        :return: s optionally wrapped with ansi escape codes
        :rtype: str
        """
        if not self.enable:
            return s
        code_seq = []
        if color:
            code_seq += [30 + self.ColorIdxTable[color]]  # foreground color
        if bold:
            code_seq += [1]
        if not code_seq:
            return s
        start = "\x1b[%sm" % ";".join(map(str, code_seq))
        end = "\x1b[0m"
        while s[:1] == " ":  # move prefix spaces outside
            start = " " + start
            s = s[1:]
        while s[-1:] == " ":  # move postfix spaces outside
            end += " "
            s = s[:-1]
        return start + s + end

    def __call__(self, *args, **kwargs):
        return self.color(*args, **kwargs)

    def py_syntax_highlight(self, s):
        """
        :param str s:
        :rtype: str
        """
        if not self.enable:
            return s
        state = 0
        spaces = " \t\n"
        ops = ".,;:+-*/%&!=|(){}[]^<>"
        i = 0
        cur_token = ""
        color_args = {0: {}, len(s): {}}  # type: typing.Dict[int,typing.Dict[str]]  # pos in s -> color kwargs

        def finish_identifier():
            """
            Reset color to standard for current identifier.
            """
            if cur_token in py_keywords:
                color_args[max([k for k in color_args.keys() if k < i])] = {"color": self.fg_colors[0]}

        while i < len(s):
            c = s[i]
            i += 1
            if c == "\n":
                if state == 3:
                    finish_identifier()
                color_args[i] = {}
                state = 0
            elif state == 0:
                if c in spaces:
                    pass
                elif c in ops:
                    color_args[i - 1] = {"color": self.fg_colors[0]}
                    color_args[i] = {}
                elif c == "#":
                    color_args[i - 1] = {"color": self.fg_colors[3]}
                    state = 6
                elif c == '"':
                    color_args[i - 1] = {"color": self.fg_colors[2]}
                    state = 1
                elif c == "'":
                    color_args[i - 1] = {"color": self.fg_colors[2]}
                    state = 2
                else:
                    cur_token = c
                    color_args[i - 1] = {}
                    state = 3
            elif state == 1:  # string via "
                if c == "\\":
                    state = 4
                elif c == "\"":
                    color_args[i] = {}
                    state = 0
            elif state == 2:  # string via '
                if c == "\\":
                    state = 5
                elif c == "'":
                    color_args[i] = {}
                    state = 0
            elif state == 3:  # identifier
                if c in spaces + ops + "#\"'":
                    finish_identifier()
                    color_args[i] = {}
                    state = 0
                    i -= 1
                else:
                    cur_token += c
            elif state == 4:  # escape in "
                state = 1
            elif state == 5:  # escape in '
                state = 2
            elif state == 6:  # comment
                pass
        if state == 3:
            finish_identifier()
        out = ""
        i = 0
        while i < len(s):
            j = min([k for k in color_args.keys() if k > i])
            out += self.color(s[i:j], **color_args[i])
            i = j
        return out


class DomTerm:
    """
    DomTerm (https://github.com/PerBothner/DomTerm/) is a terminal emulator
    with many extended escape codes, such as folding text away, or even generic HTML.
    We can make use of some of these features (currently just folding text).
    """

    _is_domterm = None

    @classmethod
    def is_domterm(cls):
        """
        :return: whether we are inside DomTerm
        :rtype: bool
        """
        import os
        if cls._is_domterm is not None:
            return cls._is_domterm
        if not os.environ.get("DOMTERM"):
            cls._is_domterm = False
            return False
        cls._is_domterm = True
        return True

    @contextlib.contextmanager
    def logical_block(self, file=sys.stdout):
        """
        :param io.TextIOBase|io.StringIO file:
        """
        file.write("\033]110\007")
        yield
        file.write("\033]111\007")

    @contextlib.contextmanager
    def hide_button_span(self, mode, file=sys.stdout):
        """
        :param int mode: 1 or 2
        :param io.TextIOBase|io.StringIO file:
        """
        file.write("\033[83;%iu" % mode)
        yield
        file.write("\033[83;0u")

    # noinspection PyMethodMayBeStatic
    def indentation(self, file=sys.stdout):
        """
        :param io.TextIOBase|io.StringIO file:
        """
        file.write("\033]114;\"│\"\007")

    # noinspection PyMethodMayBeStatic
    def hide_button(self, file=sys.stdout):
        """
        :param io.TextIOBase|io.StringIO file:
        """
        file.write("\033[16u▶▼\033[17u")

    @contextlib.contextmanager
    def _temp_replace_attrib(self, obj, attr, new_value):
        old_value = getattr(obj, attr)
        setattr(obj, attr, new_value)
        yield old_value
        setattr(obj, attr, old_value)

    @contextlib.contextmanager
    def fold_text_stream(self, prefix, postfix="", hidden_stream=None, **kwargs):
        """
        :param str prefix: always visible
        :param str postfix: always visible, right after.
        :param io.TextIOBase|io.StringIO hidden_stream: sys.stdout by default.
            If this is sys.stdout, it will replace that stream,
            and collect the data during the context (in the `with` block).
        """
        import io
        if hidden_stream is None:
            hidden_stream = sys.stdout
        assert isinstance(hidden_stream, io.IOBase)
        assert hidden_stream is sys.stdout, "currently not supported otherwise"
        hidden_buf = io.StringIO()
        with self._temp_replace_attrib(sys, "stdout", hidden_buf):
            yield
        self.fold_text(prefix=prefix, postfix=postfix, hidden=hidden_buf.getvalue(), **kwargs)

    def fold_text(self, prefix, hidden, postfix="", file=None, align=0):
        """
        :param str prefix: always visible
        :param str hidden: hidden
            If this is sys.stdout, it will replace that stream,
            and collect the data during the context (in the `with` block).
        :param str postfix: always visible, right after. "" by default.
        :param io.TextIOBase|io.StringIO file: sys.stdout by default.
        :param int align: remove this number of initial chars from hidden
        """
        if file is None:
            file = sys.stdout
        # Extra logic: Multi-line hidden. Add initial "\n" if not there.
        if "\n" in hidden:
            if hidden[:1] != "\n":
                hidden = "\n" + hidden
        # Extra logic: A final "\n" of hidden, make it always visible such that it looks nicer.
        if hidden[-1:] == "\n":
            hidden = hidden[:-1]
            postfix += "\n"
        if self.is_domterm():
            with self.logical_block(file=file):
                self.indentation(file=file)
                self.hide_button(file=file)
                file.write(prefix)
                if prefix.endswith("\x1b[0m"):
                    file.write(" ")  # bug in DomTerm?
                with self.hide_button_span(2, file=file):
                    hidden_ls = hidden.split("\n")
                    hidden_ls = [s[align:] for s in hidden_ls]
                    hidden = "\033]118\007".join(hidden_ls)
                    file.write(hidden)
        else:
            file.write(prefix)
            file.write(hidden.replace("\n", "\n "))
        file.write(postfix)
        file.flush()

    def fold_text_string(self, prefix, hidden, **kwargs):
        """
        :param str prefix:
        :param str hidden:
        :param kwargs: passed to :func:`fold_text`
        :rtype: str
        """
        import io
        output_buf = io.StringIO()
        self.fold_text(prefix=prefix, hidden=hidden, file=output_buf, **kwargs)
        return output_buf.getvalue()


def is_at_exit():
    """
    Some heuristics to figure out whether this is called at a stage where the Python interpreter is shutting down.

    :return: whether the Python interpreter is currently in the process of shutting down
    :rtype: bool
    """
    if _threading_main_thread is not None:
        if not hasattr(threading, "main_thread"):
            return True
        if threading.main_thread() != _threading_main_thread:
            return True
        if not _threading_main_thread.is_alive():
            return True
    return False


class _OutputLinesCollector:
    def __init__(self, color):
        """
        :param Color color:
        """
        self.color = color
        self.lines = []
        self.dom_term = DomTerm() if DomTerm.is_domterm() else None

    def __call__(self, s1, s2=None, **kwargs):
        """
        Adds to self.lines.
        This strange function signature is for historical reasons.

        :param str s1:
        :param str|None s2:
        :param kwargs: passed to self.color
        """
        if kwargs:
            s1 = self.color(s1, **kwargs)
        if s2 is not None:
            s1 = add_indent_lines(s1, s2)
        self.lines.append(s1 + "\n")

    @contextlib.contextmanager
    def fold_text_ctx(self, line):
        """
        Folds text, via :class:`DomTerm`, if available.
        Notes that this temporarily overwrites self.lines.

        :param str line: always visible
        """
        if not self.dom_term:
            self.__call__(line)
            yield
            return
        self.lines, old_lines = [], self.lines  # overwrite self.lines
        yield  # collect output (in new self.lines)
        self.lines, new_lines = old_lines, self.lines  # recover self.lines
        hidden_text = "".join(new_lines)
        import io
        output_buf = io.StringIO()
        prefix = ""
        while line[:1] == " ":
            prefix += " "
            line = line[1:]
        self.dom_term.fold_text(line, hidden=hidden_text, file=output_buf, align=len(prefix))
        output_text = prefix[1:] + output_buf.getvalue()
        self.lines.append(output_text)

    def _pp_extra_info(self, obj, depth_limit=3):
        """
        :param typing.Any|typing.Sized obj:
        :param int depth_limit:
        :rtype: str
        """
        # We want to exclude array types from Numpy, TensorFlow, PyTorch, etc.
        # Getting __getitem__ or __len__ on them even could lead to unexpected results or deadlocks,
        # depending on the context (e.g. inside a TF session run, extending the graph is unexpected).
        if hasattr(obj, "shape"):
            return ""
        s = []
        if hasattr(obj, "__len__"):
            # noinspection PyBroadException
            try:
                if type(obj) in (str, unicode, list, tuple, dict) and len(obj) <= 5:
                    pass  # don't print len in this case
                else:
                    s += ["len = " + str(obj.__len__())]
            except Exception:
                pass
        if depth_limit > 0 and hasattr(obj, "__getitem__"):
            # noinspection PyBroadException
            try:
                if type(obj) in (str, unicode):
                    pass  # doesn't make sense to get subitems here
                else:
                    subobj = obj.__getitem__(0)  # noqa
                    extra_info = self._pp_extra_info(subobj, depth_limit - 1)
                    if extra_info != "":
                        s += ["_[0]: {" + extra_info + "}"]
            except Exception:
                pass
        return ", ".join(s)

    def pretty_print(self, obj):
        """
        :param object obj:
        :rtype: str
        """
        s = repr(obj)
        limit = output_limit()
        if len(s) > limit:
            if self.dom_term:
                s = self.color.py_syntax_highlight(s)
                s = self.dom_term.fold_text_string("", s)
            else:
                s = s[:limit - 3]  # cut before syntax highlighting, to avoid missing color endings
                s = self.color.py_syntax_highlight(s)
                s += "..."
        else:
            s = self.color.py_syntax_highlight(s)
        extra_info = self._pp_extra_info(obj)
        if extra_info != "":
            s += ", " + self.color.py_syntax_highlight(extra_info)
        return s


# For compatibility, we keep non-PEP8 argument names.
# noinspection PyPep8Naming
def format_tb(tb=None, limit=None, allLocals=None, allGlobals=None, withTitle=False, with_color=None, with_vars=None):
    """
    :param types.TracebackType|types.FrameType|StackSummary tb: traceback. if None, will use sys._getframe
    :param int|None limit: limit the traceback to this number of frames. by default, will look at sys.tracebacklimit
    :param dict[str]|None allLocals: if set, will update it with all locals from all frames
    :param dict[str]|None allGlobals: if set, will update it with all globals from all frames
    :param bool withTitle:
    :param bool|None with_color: output with ANSI escape codes for color
    :param bool with_vars: will print var content which are referenced in the source code line. by default enabled.
    :return: list of strings (line-based)
    :rtype: list[str]
    """
    color = Color(enable=with_color)
    output = _OutputLinesCollector(color=color)

    def format_filename(s):
        """
        :param str s:
        :rtype: str
        """
        base = os.path.basename(s)
        return (
            color('"' + s[:-len(base)], color.fg_colors[2]) +
            color(base, color.fg_colors[2], bold=True) +
            color('"', color.fg_colors[2]))

    format_py_obj = output.pretty_print
    if tb is None:
        # noinspection PyBroadException
        try:
            tb = get_current_frame()
            assert tb
        except Exception:
            output(color("format_tb: tb is None and sys._getframe() failed", color.fg_colors[1], bold=True))
            return output.lines

    def is_stack_summary(_tb):
        """
        :param StackSummary|object _tb:
        :rtype: bool
        """
        return isinstance(_tb, StackSummary)

    isframe = inspect.isframe
    if withTitle:
        if isframe(tb) or is_stack_summary(tb):
            output(color('Traceback (most recent call first):', color.fg_colors[0]))
        else:  # expect traceback-object (or compatible)
            output(color('Traceback (most recent call last):', color.fg_colors[0]))
    if with_vars is None and is_at_exit():
        # Better to not show __repr__ of some vars, as this might lead to crashes
        # when native extensions are involved.
        with_vars = False
        if withTitle:
            output("(Exclude vars because we are exiting.)")
    if with_vars is None:
        if any([f.f_code.co_name == "__del__" for f in iter_traceback()]):
            # __del__ is usually called via the Python garbage collector (GC).
            # This can happen and very random / non-deterministic places.
            # There are cases where it is not safe to access some of the vars on the stack
            # because they might be in a non-well-defined state, thus calling their __repr__ is not safe.
            # See e.g. this bug:
            # https://github.com/tensorflow/tensorflow/issues/22770
            with_vars = False
            if withTitle:
                output("(Exclude vars because we are on a GC stack.)")
    if with_vars is None:
        with_vars = True
    # noinspection PyBroadException
    try:
        if limit is None:
            if hasattr(sys, 'tracebacklimit'):
                limit = sys.tracebacklimit
        n = 0
        _tb = tb

        class NotFound(Exception):
            """
            Identifier not found.
            """

        def _resolve_identifier(namespace, keys):
            """
            :param dict[str] namespace:
            :param tuple[str] keys:
            :return: namespace[name[0]][name[1]]...
            """
            if keys[0] not in namespace:
                raise NotFound()
            obj = namespace[keys[0]]
            for part in keys[1:]:
                obj = getattr(obj, part)
            return obj

        # noinspection PyShadowingNames
        def _try_set(old, prefix, func):
            """
            :param None|str old:
            :param str prefix:
            :param func:
            :return: old
            """
            if old is not None:
                return old
            try:
                return add_indent_lines(prefix, func())
            except NotFound:
                return old
            except Exception as e:
                return prefix + "!" + e.__class__.__name__ + ": " + str(e)

        while _tb is not None and (limit is None or n < limit):
            if isframe(_tb):
                f = _tb
            elif is_stack_summary(_tb):
                if isinstance(_tb[0], ExtendedFrameSummary):
                    f = _tb[0].tb_frame
                else:
                    f = DummyFrame.from_frame_summary(_tb[0])
            else:
                f = _tb.tb_frame
            if allLocals is not None:
                allLocals.update(f.f_locals)
            if allGlobals is not None:
                allGlobals.update(f.f_globals)
            if hasattr(_tb, "tb_lineno"):
                lineno = _tb.tb_lineno
            elif is_stack_summary(_tb):
                lineno = _tb[0].lineno
            else:
                lineno = f.f_lineno
            co = f.f_code
            filename = co.co_filename
            name = get_func_str_from_code_object(co)
            file_descr = "".join([
                '  ',
                color("File ", color.fg_colors[0], bold=True), format_filename(filename), ", ",
                color("line ", color.fg_colors[0]), color("%d" % lineno, color.fg_colors[4]), ", ",
                color("in ", color.fg_colors[0]), name])
            with output.fold_text_ctx(file_descr):
                if not os.path.isfile(filename):
                    alt_fn = fallback_findfile(filename)
                    if alt_fn:
                        output(
                            color("    -- couldn't find file, trying this instead: ", color.fg_colors[0]) +
                            format_filename(alt_fn))
                        filename = alt_fn
                source_code = get_source_code(filename, lineno, f.f_globals)
                if source_code:
                    source_code = remove_indent_lines(replace_tab_indents(source_code)).rstrip()
                    output("    line: ", color.py_syntax_highlight(source_code), color=color.fg_colors[0])
                    if not with_vars:
                        pass
                    elif isinstance(f, DummyFrame) and not f.have_vars_available:
                        pass
                    else:
                        with output.fold_text_ctx(color('    locals:', color.fg_colors[0])):
                            already_printed_locals = set()  # type: typing.Set[typing.Tuple[str,...]]
                            for token_str in grep_full_py_identifiers(parse_py_statement(source_code)):
                                splitted_token = tuple(token_str.split("."))
                                for token in [splitted_token[0:i] for i in range(1, len(splitted_token) + 1)]:
                                    if token in already_printed_locals:
                                        continue
                                    token_value = None
                                    token_value = _try_set(
                                        token_value, color("<local> ", color.fg_colors[0]),
                                        lambda: format_py_obj(_resolve_identifier(f.f_locals, token)))
                                    token_value = _try_set(
                                        token_value, color("<global> ", color.fg_colors[0]),
                                        lambda: format_py_obj(_resolve_identifier(f.f_globals, token)))
                                    token_value = _try_set(
                                        token_value, color("<builtin> ", color.fg_colors[0]),
                                        lambda: format_py_obj(_resolve_identifier(f.f_builtins, token)))
                                    token_value = token_value or color("<not found>", color.fg_colors[0])
                                    prefix = (
                                        '      %s ' % color(".", color.fg_colors[0], bold=True).join(token) +
                                        color("= ", color.fg_colors[0], bold=True))
                                    output(prefix, token_value)
                                    already_printed_locals.add(token)
                            if len(already_printed_locals) == 0:
                                output(color("       no locals", color.fg_colors[0]))
                else:
                    output(color('    -- code not available --', color.fg_colors[0]))
            if isframe(_tb):
                _tb = _tb.f_back
            elif is_stack_summary(_tb):
                _tb = StackSummary.from_list(_tb[1:])
                if not _tb:
                    _tb = None
            else:
                _tb = _tb.tb_next
            n += 1

    except Exception:
        output(color("ERROR: cannot get more detailed exception info because:", color.fg_colors[1], bold=True))
        import traceback
        for line in traceback.format_exc().split("\n"):
            output("   " + line)

    return output.lines


def print_tb(tb, file=None, **kwargs):
    """
    :param types.TracebackType|types.FrameType|StackSummary tb:
    :param io.TextIOBase|io.StringIO|typing.TextIO|None file: stderr by default
    :return: nothing, prints to ``file``
    """
    if file is None:
        file = sys.stderr
    for line in format_tb(tb=tb, **kwargs):
        file.write(line)
    file.flush()


def better_exchook(etype, value, tb,
                   debugshell=False, autodebugshell=True,
                   file=None, with_color=None, with_preamble=True):
    """
    Replacement for sys.excepthook.

    :param etype: exception type
    :param value: exception value
    :param tb: traceback
    :param bool debugshell: spawn a debug shell at the context of the exception
    :param bool autodebugshell: if env DEBUG is an integer != 0, it will spawn a debug shell
    :param io.TextIOBase|io.StringIO|typing.TextIO|None file: output stream where we will print the traceback
        and exception information. stderr by default.
    :param bool|None with_color: whether to use ANSI escape codes for colored output
    :param bool with_preamble: print a short preamble for the exception
    """
    if file is None:
        file = sys.stderr

    color = Color(enable=with_color)
    output = _OutputLinesCollector(color=color)

    rec_args = dict(autodebugshell=False, file=file, with_color=with_color, with_preamble=with_preamble)
    if getattr(value, "__cause__", None):
        better_exchook(type(value.__cause__), value.__cause__, value.__cause__.__traceback__, **rec_args)
        output("")
        output("The above exception was the direct cause of the following exception:")
        output("")
    elif getattr(value, "__context__", None):
        better_exchook(type(value.__context__), value.__context__, value.__context__.__traceback__, **rec_args)
        output("")
        output("During handling of the above exception, another exception occurred:")
        output("")

    def format_filename(s):
        """
        :param str s:
        :rtype: str
        """
        base = os.path.basename(s)
        return (
            color('"' + s[:-len(base)], color.fg_colors[2]) +
            color(base, color.fg_colors[2], bold=True) +
            color('"', color.fg_colors[2]))

    if with_preamble:
        output(color("EXCEPTION", color.fg_colors[1], bold=True))
    all_locals, all_globals = {}, {}
    if tb is not None:
        output.lines.extend(
            format_tb(
                tb=tb, allLocals=all_locals, allGlobals=all_globals, withTitle=True, with_color=color.enable))
    else:
        output(color("better_exchook: traceback unknown", color.fg_colors[1]))

    if isinstance(value, SyntaxError):
        # The standard except hook will also print the source of the SyntaxError,
        # so do it in a similar way here as well.
        filename = value.filename
        # Keep the output somewhat consistent with format_tb.
        file_descr = "".join([
            '  ',
            color("File ", color.fg_colors[0], bold=True), format_filename(filename), ", ",
            color("line ", color.fg_colors[0]), color("%d" % value.lineno, color.fg_colors[4])])
        with output.fold_text_ctx(file_descr):
            if not os.path.isfile(filename):
                alt_fn = fallback_findfile(filename)
                if alt_fn:
                    output(
                        color("    -- couldn't find file, trying this instead: ", color.fg_colors[0]) +
                        format_filename(alt_fn))
                    filename = alt_fn
            source_code = get_source_code(filename, value.lineno)
            if source_code:
                # Similar to remove_indent_lines.
                # But we need to know the indent-prefix such that we can use the syntax-error offset.
                source_code = replace_tab_indents(source_code)
                lines = source_code.splitlines(True)
                indent_prefix = get_same_indent_prefix(lines)
                if indent_prefix is None:
                    indent_prefix = ""
                source_code = "".join([line[len(indent_prefix):] for line in lines])
                source_code = source_code.rstrip()
                prefix = "    line: "
                output(prefix, color.py_syntax_highlight(source_code), color=color.fg_colors[0])
                output(" " * (len(prefix) + value.offset - len(indent_prefix) - 1) + "^", color=color.fg_colors[4])

    import types

    # noinspection PyShadowingNames
    def _some_str(value):
        """
        :param object value:
        :rtype: str
        """
        # noinspection PyBroadException
        try:
            return str(value)
        except Exception:
            return '<unprintable %s object>' % type(value).__name__

    # noinspection PyShadowingNames
    def _format_final_exc_line(etype, value):
        value_str = _some_str(value)
        if value is None or not value_str:
            line = color("%s" % etype, color.fg_colors[1])
        else:
            line = color("%s" % etype, color.fg_colors[1]) + ": %s" % (value_str,)
        return line

    # noinspection PyUnresolvedReferences
    if (isinstance(etype, BaseException) or
            (hasattr(types, "InstanceType") and isinstance(etype, types.InstanceType)) or
            etype is None or type(etype) is str):
        output(_format_final_exc_line(etype, value))
    else:
        output(_format_final_exc_line(etype.__name__, value))

    for line in output.lines:
        file.write(line)
    file.flush()

    if autodebugshell:
        # noinspection PyBroadException
        try:
            debugshell = int(os.environ["DEBUG"]) != 0
        except Exception:
            pass
    if debugshell:
        output("---------- DEBUG SHELL -----------")
        debug_shell(user_ns=all_locals, user_global_ns=all_globals, traceback=tb)


def dump_all_thread_tracebacks(exclude_thread_ids=None, file=None):
    """
    Prints the traceback of all threads.

    :param set[int]|list[int]|None exclude_thread_ids: threads to exclude
    :param io.TextIOBase|io.StringIO|typing.TextIO|None file: output stream
    """
    if exclude_thread_ids is None:
        exclude_thread_ids = []
    if not file:
        file = sys.stdout
    import threading

    if hasattr(sys, "_current_frames"):
        print("", file=file)
        threads = {t.ident: t for t in threading.enumerate()}
        # noinspection PyProtectedMember
        for tid, stack in sys._current_frames().items():
            if tid in exclude_thread_ids:
                continue
            # This is a bug in earlier Python versions.
            # https://bugs.python.org/issue17094
            # Note that this leaves out all threads not created via the threading module.
            if tid not in threads:
                continue
            tags = []
            thread = threads.get(tid)
            if thread:
                assert isinstance(thread, threading.Thread)
                if thread is threading.currentThread():
                    tags += ["current"]
                # noinspection PyProtectedMember,PyUnresolvedReferences
                if isinstance(thread, threading._MainThread):
                    tags += ["main"]
                tags += [str(thread)]
            else:
                tags += ["unknown with id %i" % tid]
            print("Thread %s:" % ", ".join(tags), file=file)
            print_tb(stack, file=file)
            print("", file=file)
        print("That were all threads.", file=file)
    else:
        print("Does not have sys._current_frames, cannot get thread tracebacks.", file=file)


def get_current_frame():
    """
    :return: current frame object (excluding this function call)
    :rtype: types.FrameType

    Uses sys._getframe if available, otherwise some trickery with sys.exc_info and a dummy exception.
    """
    if hasattr(sys, "_getframe"):
        # noinspection PyProtectedMember
        return sys._getframe(1)
    try:
        raise ZeroDivisionError
    except ZeroDivisionError:
        return sys.exc_info()[2].tb_frame.f_back


def get_func_str_from_code_object(co):
    """
    :param types.CodeType co:
    :return: co.co_name as fallback, but maybe sth better like the full func name if possible
    :rtype: str
    """
    f = get_func_from_code_object(co)
    if f:
        if hasattr(f, "__qualname__"):
            return f.__qualname__
        return str(f)
    return co.co_name


def get_func_from_code_object(co):
    """
    :param types.CodeType co:
    :return: function, such that ``func.__code__ is co``, or None
    :rtype: types.FunctionType

    This is CPython specific (to some degree; it uses the `gc` module to find references).
    Inspired from: https://stackoverflow.com/questions/12787108/getting-the-python-function-for-a-code-object
    """
    import types
    if not isinstance(co, types.CodeType):
        return None
    import gc
    candidates = gc.get_referrers(co)
    candidates = [f for f in candidates if getattr(f, "__code__" if PY3 else "func_code", None) is co]
    if candidates:
        return candidates[0]
    return None


def iter_traceback(tb=None, enforce_most_recent_call_first=False):
    """
    Iterates a traceback of various formats:
      - traceback (types.TracebackType)
      - frame object (types.FrameType)
      - stack summary (traceback.StackSummary)

    :param types.TracebackType|types.FrameType|StackSummary|None tb: traceback. if None, will use sys._getframe
    :param bool enforce_most_recent_call_first:
        Frame or stack summery: most recent call first (top of the stack is the first entry in the result)
        Traceback: most recent call last
        If True, and we get traceback, will unroll and reverse, such that we have always the most recent call first.
    :return: yields the frames (types.FrameType)
    :rtype: list[types.FrameType|DummyFrame]
    """
    if tb is None:
        tb = get_current_frame()

    def is_stack_summary(_tb):
        """
        :param StackSummary|object _tb:
        :rtype: bool
        """
        return isinstance(_tb, StackSummary)

    is_frame = inspect.isframe
    is_traceback = inspect.istraceback
    assert is_traceback(tb) or is_frame(tb) or is_stack_summary(tb)
    # Frame or stack summery: most recent call first
    # Traceback: most recent call last
    if is_traceback(tb) and enforce_most_recent_call_first:
        frames = list(iter_traceback(tb))
        for frame in frames[::-1]:
            yield frame
        return

    _tb = tb
    while _tb is not None:
        if is_frame(_tb):
            frame = _tb
        elif is_stack_summary(_tb):
            if isinstance(_tb[0], ExtendedFrameSummary):
                frame = _tb[0].tb_frame
            else:
                frame = DummyFrame.from_frame_summary(_tb[0])
        else:
            frame = _tb.tb_frame
        yield frame
        if is_frame(_tb):
            _tb = _tb.f_back
        elif is_stack_summary(_tb):
            _tb = StackSummary.from_list(_tb[1:])
            if not _tb:
                _tb = None
        else:
            _tb = _tb.tb_next


class ExtendedFrameSummary(FrameSummary):
    """
    Extends :class:`FrameSummary` by ``self.tb_frame``.
    """
    def __init__(self, frame, **kwargs):
        super(ExtendedFrameSummary, self).__init__(**kwargs)
        self.tb_frame = frame


class DummyFrame:
    """
    This class has the same attributes as a code and a frame object
    and is intended to be used as a dummy replacement.
    """

    @classmethod
    def from_frame_summary(cls, f):
        """
        :param FrameSummary f:
        :rtype: DummyFrame
        """
        return cls(filename=f.filename, lineno=f.lineno, name=f.name, f_locals=f.locals)

    def __init__(self, filename, lineno, name, f_locals=None, f_globals=None, f_builtins=None):
        self.lineno = lineno
        self.tb_lineno = lineno
        self.f_lineno = lineno
        self.f_code = self
        self.filename = filename
        self.co_filename = filename
        self.name = name
        self.co_name = name
        self.f_locals = f_locals or {}
        self.f_globals = f_globals or {}
        self.f_builtins = f_builtins or {}
        self.have_vars_available = (f_locals is not None or f_globals is not None or f_builtins is not None)


# noinspection PyPep8Naming,PyUnusedLocal
def _StackSummary_extract(frame_gen, limit=None, lookup_lines=True, capture_locals=False):
    """
    Replacement for :func:`StackSummary.extract`.

    Create a StackSummary from a traceback or stack object.
    Very simplified copy of the original StackSummary.extract().
    We want always to capture locals, that is why we overwrite it.
    Additionally, we also capture the frame.
    This is a bit hacky and also not like this is originally intended (to not keep refs).

    :param frame_gen: A generator that yields (frame, lineno) tuples to
        include in the stack.
    :param limit: None to include all frames or the number of frames to
        include.
    :param lookup_lines: If True, lookup lines for each frame immediately,
        otherwise lookup is deferred until the frame is rendered.
    :param capture_locals: If True, the local variables from each frame will
        be captured as object representations into the FrameSummary.
    """
    result = StackSummary()
    for f, lineno in frame_gen:
        co = f.f_code
        filename = co.co_filename
        name = co.co_name
        result.append(ExtendedFrameSummary(
            frame=f, filename=filename, lineno=lineno, name=name, lookup_line=False))
    return result


def install():
    """
    Replaces sys.excepthook by our better_exchook.
    """
    sys.excepthook = better_exchook


def replace_traceback_format_tb():
    """
    Replaces these functions from the traceback module by our own:

    - traceback.format_tb
    - traceback.StackSummary.format
    - traceback.StackSummary.extract

    Note that this kind of monkey patching might not be safe under all circumstances
    and is not officially supported by Python.
    """
    import traceback
    traceback.format_tb = format_tb
    if hasattr(traceback, "StackSummary"):
        traceback.StackSummary.format = format_tb
        traceback.StackSummary.extract = _StackSummary_extract
