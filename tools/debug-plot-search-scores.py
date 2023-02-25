#!/usr/bin/env python3

"""
Use ``debug-dump-search-scores.py`` to dump all information to some HDF file.
The you can use this script here just for visualization/plotting.
"""

from __future__ import annotations

from math import ceil
import typing
import os
from pprint import pprint
import argparse
import h5py
import numpy
from glob import glob

import _setup_returnn_env  # noqa
from returnn.datasets.lm import Lexicon
import returnn.util.basic as util


class Alignment:
    """Phone/word alignment"""

    class Item:
        """Phone or word"""

        def __init__(self, value, start_frame, end_frame=None, word_start=True, word_end=True):
            """
            :param str value:
            :param int start_frame:
            :param int|None end_frame:
            :param bool word_start:
            :param bool word_end:
            """
            self.value = value
            self.word_start = word_start
            self.word_end = word_end
            self.start_frame = start_frame
            if end_frame is None:
                end_frame = self.start_frame
            self.end_frame = end_frame

        @property
        def num_frames(self):
            return self.end_frame - self.start_frame + 1

        def copy(self):
            """
            :rtype: Alignment.Item
            """
            return Alignment.Item(
                value=self.value,
                start_frame=self.start_frame,
                end_frame=self.end_frame,
                word_start=self.word_start,
                word_end=self.word_end,
            )

        __repr__ = util.simple_obj_repr

        def __eq__(self, other):
            """
            :param Alignment.Item other:
            :rtype: bool
            """
            return repr(self) == repr(other)

        def __ne__(self, other):
            return not (self == other)

    def __init__(self):
        self.items = []  # type: typing.List[Alignment.Item]

    def __repr__(self):
        return "Alignment{%s}" % ", ".join(
            ["%r %i-%i" % (item.value, item.start_frame, item.end_frame) for item in self.items]
        )

    def __eq__(self, other):
        """
        :param Alignment other:
        :rtype: bool
        """
        return self.items == other.items

    def __ne__(self, other):
        return not (self == other)

    def copy(self):
        """
        :rtype: Alignment
        """
        res = Alignment()
        res.items = [item.copy() for item in self.items]
        return res

    def copy_extended(self, new_item):
        """
        :param Alignment.Item new_item:
        :rtype: Alignment
        """
        res = self.copy()
        res.items.append(new_item)
        return res


def parse_phone_alignment(filename):
    """
    :param str filename:
    :rtype: Alignment
    """
    lines = open(filename, "r").read().splitlines()
    # Example:
    """
  <?xml version="1.0" encoding="ISO-8859-1"?>
  <sprint>
    time= 0       emission=       10989   allophone=      ow{#+k}@i       index=  10989   state=  0
    time= 1       emission=       10989   allophone=      ow{#+k}@i       index=  10989   state=  0
    time= 2       emission=       10989   allophone=      ow{#+k}@i       index=  10989   state=  0
  """
    import re

    res = Alignment()
    line_re = re.compile(
        "^time=\\s*([0-9]+)\\s*"
        "emission=\\s*([0-9]+)\\s*"
        "allophone=\\s*([A-Za-z{}\\[\\]+#@]+)\\s*"
        "index=\\s*([0-9]+)\\s*"
        "state=\\s*([0-9]+)$"
    )
    allophone_re = re.compile("^([A-Za-z\\[\\]]+){([#a-z]+)\\+([#a-z]+)}([@a-z]*)$")
    time_i = 0
    last_state_i = -1
    last_flags_s = None
    for line in lines:
        line = line.strip()
        if not line or line.startswith("<"):
            continue
        line_match = line_re.match(line)
        assert line_match, "line: %r" % line
        time_s, emission_s, allophone_s, index_s, state_s = line_match.groups()
        state_i = int(state_s)
        assert int(time_s) == time_i
        allophone_match = allophone_re.match(allophone_s)
        assert allophone_match, "allophone: %r" % allophone_s
        phone_s, phone_left_s, phone_right_s, phone_flags_s = allophone_match.groups()
        if res.items and res.items[-1].value == phone_s and state_i >= last_state_i and phone_flags_s == last_flags_s:
            res.items[-1].end_frame += 1
        else:
            res.items.append(
                Alignment.Item(
                    value=phone_s, word_start="@i" in phone_flags_s, word_end="@f" in phone_flags_s, start_frame=time_i
                )
            )
        last_state_i = state_i
        last_flags_s = phone_flags_s
        time_i += 1
    assert sum([p.num_frames for p in res.items]) == time_i
    return res


def parse_phone_alignments(pattern):
    """
    :param str pattern: e.g. 'align-*.txt'. stdout from RASR archiver
    :return: name -> list of phones
    :rtype: dict[str,Alignment]
    """
    res = {}
    assert pattern.count("*") == 1
    prefix, postfix = pattern.split("*")
    for fn in glob(pattern):
        assert fn.startswith(prefix) and fn.endswith(postfix)
        name = fn[len(prefix) : -len(postfix)]
        res[name] = parse_phone_alignment(fn)
    return res


def select_phone_alignment(phone_alignments, seq_tag):
    """
    :param dict[str,Alignment] phone_alignments:
    :param str seq_tag:
    :rtype: Alignment
    """
    if seq_tag in phone_alignments:
        phone_alignment = phone_alignments[seq_tag]
    elif os.path.basename(seq_tag) in phone_alignments:
        phone_alignment = phone_alignments[os.path.basename(seq_tag)]
    else:
        raise Exception(
            "Did not find phone alignment for seq tag %r. Available: %r" % (seq_tag, list(phone_alignments.keys()))
        )
    return phone_alignment


def phone_alignment_to_word_alignment(lexicon, words, phone_alignment):
    """
    :param Lexicon lexicon:
    :param list[str] words:
    :param Alignment phone_alignment:
    :return: list of words, and number of time frames (time aligned)
    :rtype: Alignment|None
    """
    sil_phone = "[SILENCE]"
    num_non_silence_frames = sum([p.num_frames for p in phone_alignment.items if p.value != sil_phone])

    class SingleState:
        """Single state"""

        def __init__(self, word_i, lemma_phones=None, lemma_phone_i=-1, word_alignment=None):
            """
            :param int word_i:
            :param list[str]|None lemma_phones:
            :param int lemma_phone_i:
            :param Alignment word_alignment:
            """
            if not word_alignment:
                word_alignment = Alignment()
            self.word_i = word_i
            self.lemma_phones = lemma_phones
            self.lemma_phone_i = lemma_phone_i
            self.word_alignment = word_alignment

        __repr__ = util.simple_obj_repr

        def is_lemma_finished(self):
            if not self.lemma_phones:
                return True
            if self.lemma_phone_i >= len(self.lemma_phones):
                return True
            return False

        def get_with_new_lemma(self, phone):
            """
            :param Alignment.Item phone:
            :rtype: typing.Generator[SingleState]
            """
            assert self.is_lemma_finished()
            if not phone.word_start:
                return
            if phone.value == sil_phone:
                yield self
                return
            word_i = self.word_i + 1
            if word_i >= len(words):
                return
            assert word_i < len(words)
            word = words[word_i]
            new_word_alignment = self.word_alignment.copy_extended(
                Alignment.Item(value=word, start_frame=phone.start_frame, end_frame=phone.end_frame)
            )
            lemma = lexicon.lemmas[word]
            lemma_phones_opts = lemma["phons"]
            assert isinstance(lemma_phones_opts, list)
            for lemma_phones in lemma_phones_opts:
                lemma_phones = lemma_phones["phon"]
                assert isinstance(lemma_phones, str) and len(lemma_phones) > 0
                lemma_phones = lemma_phones.split()
                assert isinstance(lemma_phones, list) and len(lemma_phones) > 0 and isinstance(lemma_phones[0], str)
                if lemma_phones[0] == phone.value:
                    yield SingleState(
                        word_i=word_i,
                        lemma_phones=lemma_phones,
                        lemma_phone_i=1,
                        word_alignment=new_word_alignment.copy(),
                    )

        def get_inc_phone_in_lemma(self, phone):
            """
            :param Alignment.Item phone:
            :rtype: typing.Generator[SingleState]
            """
            if self.lemma_phones[self.lemma_phone_i] != phone.value:
                return
            if self.lemma_phone_i == len(self.lemma_phones) - 1:
                if not phone.word_end:
                    return
            new_word_alignment = self.word_alignment.copy()
            new_word_alignment.items[-1].end_frame += phone.num_frames
            yield SingleState(
                word_i=self.word_i,
                lemma_phones=self.lemma_phones,
                lemma_phone_i=self.lemma_phone_i + 1,
                word_alignment=new_word_alignment,
            )

    possible_states = [SingleState(word_i=-1, lemma_phones=None, lemma_phone_i=-1, word_alignment=Alignment())]

    for phone in phone_alignment.items:
        next_possible_states = []  # type: typing.List[SingleState]
        for state in possible_states:
            if state.is_lemma_finished():
                next_possible_states.extend(state.get_with_new_lemma(phone=phone))
            else:
                next_possible_states.extend(state.get_inc_phone_in_lemma(phone=phone))

        if not next_possible_states:
            break
        possible_states = next_possible_states

    final_possible_states = []  # type: typing.List[SingleState]
    for state in possible_states:
        if state.is_lemma_finished() and state.word_i == len(words) - 1:
            final_possible_states.append(state)

    if not final_possible_states:
        return None
    assert len(final_possible_states) == 1
    word_alignment = final_possible_states[0].word_alignment
    assert sum([w.num_frames for w in word_alignment.items]) == num_non_silence_frames
    return word_alignment


class AlignmentPlotter:
    """
    Using matplotlib.
    Can be hard or soft alignments.
    """

    def __init__(self):
        self.recent_seq_tag = None  # type: typing.Optional[str]
        self.plot_by_seq_tag = {}  # type: typing.Dict[str,AlignmentPlotter.Plot]

    def set_recent_seq_tag(self, seq_tag):
        """
        :param str seq_tag:
        """
        self.recent_seq_tag = seq_tag
        if seq_tag in self.plot_by_seq_tag:
            return
        self.plot_by_seq_tag[seq_tag] = AlignmentPlotter.Plot(seq_tag=seq_tag, parent=self)

    @property
    def recent_plot(self):
        return self.plot_by_seq_tag[self.recent_seq_tag]

    class Alignment:
        """
        (Soft) alignment to plot, e.g. attention weights.
        """

        def __init__(
            self,
            plot,
            key,
            human,
            alignment,
            time_reduction=1,
            add_extra_time=0,
            add_extra_time_red=0,
            cmap=None,
            cmap_palette=None,
            colorbar=False,
        ):
            """
            :param AlignmentPlotter plot:
            :param str key:
            :param str human:
            :param list[int]|numpy.ndarray|Alignment alignment:
            :param int time_reduction:
            :param int add_extra_time:
            :param int add_extra_time_red:
            :param cmap:
            :param int|None cmap_palette:
            :param bool colorbar:
            """
            self.plot = plot
            self.seq_tag = plot.recent_seq_tag
            self.key = key
            self.colorbar = colorbar
            self.human = human
            self.cmap = cmap
            self.cmap_palette = cmap_palette
            self.alignment = alignment
            self.time_reduction = time_reduction
            self.add_extra_time = add_extra_time
            self.add_extra_time_red = add_extra_time_red
            self.enabled = True
            self.mapped_alignment = None  # type: typing.Optional[numpy.ndarray]

        def __repr__(self):
            return "AlignmentPlotter.Alignment(human=%r, key=%r, seq_tag=%r)" % (self.human, self.key, self.seq_tag)

        def add_to_plot(self, overwrite=False):
            assert self.plot.recent_plot.seq_tag == self.seq_tag
            if not overwrite:
                assert self.key not in self.plot.recent_plot.alignments
            self.plot.recent_plot.alignments[self.key] = self

        def _check_alignment(self):
            """
            :return: whether to use this
            :rtype: bool
            """
            plot = self.plot.plot_by_seq_tag[self.seq_tag]  # recent_plot might be different
            time_length = plot.time_length + self.add_extra_time
            if isinstance(self.alignment, numpy.ndarray):
                assert self.alignment.shape == (
                    len(plot.output),
                    ceil(time_length / self.time_reduction) + self.add_extra_time_red,
                ), "%s: output len %i, orig len %i, time red %s, extra %i, extra red %i, align shape %r" % (
                    self,
                    len(plot.output),
                    plot.time_length,
                    self.time_reduction,
                    self.add_extra_time,
                    self.add_extra_time_red,
                    self.alignment.shape,
                )
                if self.time_reduction != 1:
                    assert isinstance(self.time_reduction, int) and self.time_reduction > 1  # else not implemented...
                    new_align = numpy.repeat(self.alignment, self.time_reduction, axis=1)
                    assert new_align.shape[1] >= time_length
                    self.alignment = new_align
            elif isinstance(self.alignment, list):
                assert len(self.alignment) == len(plot.output)
                alignment = numpy.zeros(
                    (len(plot.output), ceil(time_length / self.time_reduction) + self.add_extra_time_red),
                    dtype="float32",
                )
                for i, j in enumerate(self.alignment):
                    alignment[i, j] = 1.0
                alignment = numpy.repeat(alignment, self.time_reduction, axis=1)
                assert alignment.shape[1] >= time_length
                self.alignment = alignment
            elif isinstance(self.alignment, Alignment):
                if len(self.alignment.items) + 1 != len(plot.output):  # except EOS
                    return False
                assert self.time_reduction == 1 and self.add_extra_time == 0 and self.add_extra_time_red == 0
                alignment = numpy.zeros((len(plot.output), time_length), dtype="float32")
                for i, item in enumerate(self.alignment.items):
                    alignment[i, item.start_frame : item.end_frame + 1] = 1.0
                self.alignment = alignment
            else:
                raise TypeError("%s: unexpected alignment %r, type %s" % (self, self.alignment, type(self.alignment)))
            assert isinstance(self.alignment, numpy.ndarray)
            return True

        def set_mapped_alignment(self):
            """
            :return: whether to use this
            :rtype: bool
            """
            import seaborn
            from seaborn import light_palette

            if not self._check_alignment():
                return False
            if self.cmap is None:
                if self.cmap_palette is not None:
                    self.cmap = seaborn.color_palette()[self.cmap_palette]
                else:
                    plot = self.plot.plot_by_seq_tag[self.seq_tag]  # recent_plot might be different
                    self.cmap = plot._get_next_cmap()
            cmap = light_palette(self.cmap, as_cmap=True, n_colors=50)
            self.mapped_alignment = cmap(self.alignment, alpha=1)  # type: numpy.ndarray
            return True

    class Plot:
        def __init__(self, seq_tag, parent):
            """
            :param str seq_tag:
            :param AlignmentPlotter parent:
            """
            self.parent = parent
            self.seq_tag = seq_tag
            self.model_names = []  # type: typing.List[str]
            self.use_tex = False
            self.time_length = None  # type: typing.Optional[int]
            self.output = None  # type: typing.Optional[typing.List[str]]  # e.g. BPE units by the decoder
            self.alignments = {}  # type: typing.Dict[str,AlignmentPlotter.Alignment]
            self.time_alignment = None  # type: typing.Optional[Alignment]
            self._palette_idx = -1
            self.style_type = "paper"
            self.style_figsize = (8, 5)
            self.style_titlesize = None

        def set_time_alignment(self, time_alignment):
            """
            :param Alignment time_alignment:
            :return: whether this was added
            :rtype: bool
            """
            if self.time_alignment:
                assert self.time_alignment == time_alignment
                return False
            self.time_alignment = time_alignment
            AlignmentPlotter.Alignment(
                plot=self.parent, key="00_00_hybrid", human="Hybrid", alignment=time_alignment, cmap_palette=2
            ).add_to_plot()
            return True

        def add_model_name(self, name):
            """
            :param str name:
            :return: key prefix if new, else None
            :rtype: str|None
            """
            if name in self.model_names:
                return None
            self.model_names.append(name)
            return "%02i" % len(self.model_names)

        def _get_next_cmap(self):
            import seaborn

            self._palette_idx += 1
            return [seaborn.color_palette()[i] for i in [0, 5]][self._palette_idx]

        def _set_style(self):
            """
            Sets the matplotlib style according to the parameters.
            """
            style = self.style_type
            figsize = self.style_figsize
            titlesize = self.style_titlesize
            import matplotlib.pyplot as plt

            assert style in ("talk", "paper")
            import warnings

            # matplotlib warns about the font family,
            # but the rendering is done by latex.
            warnings.filterwarnings(
                "ignore", category=UserWarning, module="matplotlib", message=r"findfont: Font family.*"
            )
            # seaborn-paper: axes.titlesize: 9.6
            # seaborn-talk: axes.titlesize: 15.6
            if titlesize is None:
                if style == "paper":
                    titlesize = 10
                elif style == "talk":
                    titlesize = 16
            legendsize = titlesize * 0.6
            labelsize = titlesize * 0.8

            # noinspection PyTypeChecker
            plt.style.use(
                [
                    "seaborn-%s" % style,
                    {
                        "text.usetex": self.use_tex,
                        "text.latex.preamble": [r"\usepackage{times}"],
                        "font.family": "arial",
                        "font.size": labelsize,
                        "legend.facecolor": "white",
                        "figure.dpi": 120,
                        "figure.figsize": figsize,
                        "figure.autolayout": True,
                        "axes.titlesize": titlesize,
                        "axes.labelsize": labelsize,
                        "xtick.labelsize": labelsize,
                        "ytick.labelsize": labelsize,
                        "legend.fontsize": legendsize,
                    },
                ]
            )

        def _string_latex_compatible(self, output):
            """
            :param str output:
            :rtype: str
            """
            if not self.use_tex:
                return output
            return output.replace("<", r"\textless").replace(">", r"\textgreater").replace("_", r"\_")

        def _list_latex_compatible(self, output):
            """
            :param list[str] output:
            :rtype: list[str]
            """
            assert isinstance(output, list)
            return [self._string_latex_compatible(o) for o in output]

        def _add_plot_seq_title(self, ax, prepend="", **title_kws):
            """
            Adds a sequence title to the plot

            :param matplotlib.pyplot.Axes ax:
            :param str prepend:
            """
            from textwrap import wrap

            output = " ".join(self.output)
            output = output.replace("@@ ", "")
            output = output.replace("</s>", "")
            output = output.replace("<s>", "")
            title = '%s%s:\n"%s"' % (prepend, self.seq_tag, "\n".join(wrap(output, 60)))
            title = self._string_latex_compatible(title)
            ax.set_title(title, **title_kws)

        def _add_plot_time_title(self, ax, rotate_45=False, **font_kwargs):
            """
            Show phonemes or words above the plot, aligned to their time position.

            :param matplotlib.pyplot.Axes ax:
            :param bool rotate_45:
            """
            start_y = -1.0
            assert self.time_alignment
            for i, item in enumerate(self.time_alignment.items):
                phoneme = item.value
                phoneme = phoneme.lower().replace("[silence]", "sil")
                x1 = item.start_frame
                x2 = item.end_frame + 1.0
                p = start_y

                t = ax.text(x=x1, y=p, s=phoneme, **font_kwargs)
                if rotate_45:
                    t.set_rotation(45)

                ax.axvline(x=x1 - 0.5, color="gray", linestyle=":")
                ax.axvline(x=x2 - 0.5, color="gray", linestyle=":")
            ax.axvline(x=self.time_length - 0.5, color="black", linestyle="-")

        def _add_plot_decoder_output(self, ax, hide_prev_label=True, stride=1, draw_lines=True):
            """
            Shows the decoder output on the left axis.

            :param matplotlib.pyplot.Axes ax:
            :param bool hide_prev_label:
            :param int stride:
            :param bool draw_lines:
            """
            from matplotlib.ticker import AutoMinorLocator, NullFormatter, FixedFormatter

            if hide_prev_label:
                ax.set_ylabel("")
            assert self.output is not None
            output = self._list_latex_compatible(self.output)
            ax.set_yticklabels(output, verticalalignment="bottom")

            ticks = numpy.arange(len(output) * stride + stride)[::stride] - 0.5
            ax.set_yticks(ticks)

            # tick labels, we have to adjust so that the label is centered
            # minor ticks are in-between major ticks
            ax.yaxis.set_major_formatter(NullFormatter())
            ax.yaxis.set_minor_locator(AutoMinorLocator(2))
            ax.yaxis.set_minor_formatter(FixedFormatter(output))
            ax.yaxis.set_ticks_position("none")  # hide labels only

            # draw output boundaries (horizontal lines)
            if draw_lines:
                for i, output_bpe in enumerate(output):
                    ax.axhline(y=i * stride + stride - 0.5, color="gray", linestyle="-")

        def do_plot(self, save_filename=None, show=None, with_title=True):
            """
            Plot.

            :param str|None save_filename:
            :param bool|None show:
            :param bool with_title:
            """
            import matplotlib.pyplot as plt
            from seaborn import light_palette
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            import matplotlib.patches as mpatches
            from matplotlib.cm import ScalarMappable
            import matplotlib.patches as patches

            for align_type, d in sorted(self.alignments.items()):
                assert isinstance(d, AlignmentPlotter.Alignment)
                if not d.set_mapped_alignment():
                    self.alignments.pop(align_type)

            stride = len(self.alignments)
            self._set_style()

            fig, ax = plt.subplots()
            assert isinstance(ax, plt.Axes)

            # mix alignment matrices (interleave)
            # we have matrices of form [dec_len, enc_len]
            # for 4x alignments, we get [4*dec_len, enc_len]
            multi_dec = stride * len(self.output)

            assert self.time_length is not None
            max_time_len = max([d.mapped_alignment.shape[1] for _, d in self.alignments.items()])
            alignment_matrix = numpy.ones(shape=(multi_dec, max_time_len, 4))  # RGBA
            offset = 0
            for align_type, d in sorted(self.alignments.items()):
                assert isinstance(d, AlignmentPlotter.Alignment)
                alignment_matrix[offset::stride, : d.mapped_alignment.shape[1]] = d.mapped_alignment
                offset += 1

            # fixes the issue where the background is slightly colored
            # the value for clipping is related to the cmap levels
            alignment_matrix[alignment_matrix > 0.9] = 1

            ax.imshow(alignment_matrix, aspect="auto")

            ax.xaxis.tick_bottom()
            ax.set_xlabel("Input sequence")
            ax.set_ylabel("Output sequence")

            divider = make_axes_locatable(ax)
            legend_patches = []
            # legend_patches.append(dotted_line)
            prev_colorbar_palettes = set()  # type: typing.Set[int]
            idx = 0
            for align_type, d in sorted(self.alignments.items()):
                assert isinstance(d, AlignmentPlotter.Alignment)
                patch = mpatches.Patch(color=d.cmap, label=d.human)
                legend_patches.append(patch)
                if not d.colorbar:
                    continue
                if d.cmap_palette is not None:
                    if d.cmap_palette in prev_colorbar_palettes:
                        continue
                    prev_colorbar_palettes.add(d.cmap_palette)
                cax = divider.append_axes("right", size="5%", pad=0.1 if idx == 0 else 0)
                cmap = light_palette(d.cmap, as_cmap=True)
                mappable = ScalarMappable(cmap=cmap)
                mappable.set_array(d.mapped_alignment)
                ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1]
                plt.colorbar(mappable, cax=cax, ticks=ticks)
                idx += 1

            ax.legend(handles=legend_patches, frameon=True, handlelength=3, loc="best")

            self._add_plot_decoder_output(ax, stride=stride, hide_prev_label=True)
            self._add_plot_time_title(ax)
            if with_title:
                self._add_plot_seq_title(ax, y=1.15)

            offset = 0
            for align_type, d in sorted(self.alignments.items()):
                assert isinstance(d, AlignmentPlotter.Alignment)
                for i in range(len(self.output)):
                    arg_max = numpy.argmax(d.alignment[i])
                    if d.alignment[i, arg_max] > 0:
                        window_left, window_right = None, None  # inclusive,exclusive
                        if d.alignment[i, 0] == 0.0:  # exact 0.0 -> likely there is a window
                            window_left = numpy.where(numpy.cumsum(d.alignment[i, :arg_max]) == 0.0)[0][-1] + 1
                            assert window_left <= arg_max and d.alignment[i, window_left - 1] == 0.0
                            assert d.alignment[i, window_left] > 0 and all(d.alignment[i, :window_left] == 0.0)
                        if d.alignment[i, -1] == 0.0:  # exact 0.0 -> likely there is a window
                            window_right = (
                                numpy.where(numpy.cumsum(d.alignment[i, arg_max + 1 :][::-1])[::-1] == 0.0)[0][0]
                                + arg_max
                                + 1
                            )
                            assert window_right > arg_max and d.alignment[i, window_right] == 0.0
                            assert d.alignment[i, window_right - 1] > 0 and all(d.alignment[i, window_right:] == 0.0)
                        if window_left is not None or window_right is not None:
                            if window_left is None:
                                window_left = 0
                            if window_right is None:
                                window_right = d.alignment.shape[1]
                            rect = patches.Rectangle(
                                (window_left - 0.5, offset + i * stride - 0.5 - 0.05),
                                window_right - window_left - 0.05,
                                1.0 - 0.05,
                                linewidth=0.5,
                                edgecolor="lightgray",
                                facecolor="none",
                            )
                            ax.add_patch(rect)
                offset += 1

            if save_filename:
                plt.savefig(save_filename)
                print("Saved to file:", save_filename)
            if show is None:
                show = not save_filename
            if show:
                plt.show(block=True)


def read_hdf(
    hdf_filename,
    rec_layer_name="output",
    lexicon=None,
    apply_bpe_merge=False,
    remove_sentence_end=None,
    phone_alignment_file_pattern=None,
    pos_choice=None,
    pos_time_reduction_factor=1,
    add_extra_time_frames=0,
    add_extra_time_red_frames=0,
    plot_replace_names=None,
    plotter=None,
    plot_name_prefix="",
):
    """
    :param str hdf_filename:
    :param str rec_layer_name:
    :param str|None lexicon:
    :param bool apply_bpe_merge:
    :param str|None remove_sentence_end:
    :param str|None phone_alignment_file_pattern:
    :param str|None pos_choice:
    :param int pos_time_reduction_factor:
    :param int add_extra_time_frames:
    :param int add_extra_time_red_frames:
    :param bool|None|AlignmentPlotter plotter:
    :param str plot_name_prefix:
    :param dict[str,str]|None plot_replace_names:
    :return: plotter
    :rtype: AlignmentPlotter|None
    """
    assert os.path.exists(hdf_filename), "hdf file not dumped?"

    if lexicon:
        lexicon = Lexicon(filename=lexicon)

    phone_alignments = None  # name -> ...
    if phone_alignment_file_pattern:
        phone_alignments = parse_phone_alignments(phone_alignment_file_pattern)

    f = h5py.File(hdf_filename, "r")
    num_steps = f["seqLengths"].shape[0]  # we used dump_whole_batches
    print("Num steps:", num_steps)
    keys = sorted(f["targets/size"].attrs.keys())  # type: typing.List[str]
    keys.insert(0, "data")
    print("Data keys:")
    pprint(keys)
    assert len(keys) == f["seqLengths"].shape[1]  # seq-lens for each data-key and 'data'
    offsets = {key: 0 for key in keys}
    key_prefix = "%s__" % rec_layer_name
    assert all([key.startswith(key_prefix) for key in keys if key not in ["data", "seq_sizes", "seq_tags"]])
    labels = f["labels"]
    print("Labels dim:", labels.shape[0])
    if not labels.shape[0]:
        labels = None

    if plotter:
        if isinstance(plotter, bool):
            plotter = AlignmentPlotter()
        assert isinstance(plotter, AlignmentPlotter)
    if plot_replace_names is None:
        plot_replace_names = {}

    for step in range(num_steps):
        print("Step:", step)

        raw_values = {}
        for key_idx, key in enumerate(keys):
            raw_num_frames = f["seqLengths"][step, key_idx]
            offset = offsets[key]
            offsets[key] += raw_num_frames
            if key == "data":
                group = f["inputs"]
            else:
                group = f["targets/data/%s" % key]
            raw_value = group[offset:raw_num_frames]
            raw_values[key] = raw_value

        seq_sizes = raw_values["seq_sizes"]
        seq_tags = raw_values["seq_tags"]
        num_seqs = seq_sizes.shape[0]
        assert seq_tags.shape == (num_seqs,) == seq_sizes.shape[:1]
        print("  Num seqs:", num_seqs)

        choice_layer_names = {key[: -len("_src_beams_raw")] for key in keys if key.endswith("_src_beams_raw")}
        assert all([name.startswith(key_prefix) for name in choice_layer_names])
        choice_layer_names = {name[len(key_prefix) :] for name in choice_layer_names}
        choice_layer_names = sorted(choice_layer_names)
        if "output" in choice_layer_names:
            # Add it at the end. This is an implicit assumption about the choice order.
            choice_layer_names.remove("output")
            choice_layer_names.append("output")
        print("  Choice layers:", choice_layer_names)
        if pos_choice:
            assert pos_choice in choice_layer_names
        choices_num_seqs = {
            name: raw_values["%s%s_src_beams_raw_seq_lens" % (key_prefix, name)].shape[0] for name in choice_layer_names
        }
        assert all([n % num_seqs == 0 for (_, n) in choices_num_seqs.items()])
        choices_beam_sizes = {name: n // num_seqs for (name, n) in choices_num_seqs.items()}
        print("  Choice beam sizes:", choices_beam_sizes)
        output_dim = f["targets/size"].attrs["%s%s_value_final" % (key_prefix, choice_layer_names[-1])][0]
        print("  %r dim:" % choice_layer_names[-1], output_dim)
        if labels is not None:
            assert labels.shape[0] == output_dim

        for seq_idx, seq_tag in enumerate(seq_tags):
            print("  Seq idx:", seq_idx)
            print("    Seq tag:", seq_tag)

            last_beam_size = choices_beam_sizes[choice_layer_names[-1]]
            last_choice_seq_lens = raw_values["%s%s_beam_scores_raw_seq_lens" % (key_prefix, choice_layer_names[-1])]
            assert last_choice_seq_lens.shape == (num_seqs * last_beam_size, 1)
            last_choice_seq_lens = numpy.reshape(last_choice_seq_lens, (num_seqs, last_beam_size))[
                seq_idx
            ]  # (beam_size,)
            print("    N-best seq lens:", list(last_choice_seq_lens))
            max_seq_len = max(last_choice_seq_lens)

            if plotter:
                plotter.set_recent_seq_tag(seq_tag)

            for i in range(max_seq_len):
                print("    Frame i:", i)

                for choice_name in choice_layer_names:
                    beam_size = choices_beam_sizes[choice_name]
                    print("      Choice:", choice_name, ", beam size:", beam_size)
                    choice_seq_lens = raw_values["%s%s_beam_scores_raw_seq_lens" % (key_prefix, choice_name)]
                    assert choice_seq_lens.shape == (num_seqs * beam_size, 1)
                    choice_max_seq_len = max(choice_seq_lens.flat)

                    beam_scores = raw_values["%s%s_beam_scores_raw" % (key_prefix, choice_name)]
                    assert beam_scores.shape == (num_seqs * beam_size * choice_max_seq_len,)
                    beam_scores = numpy.reshape(beam_scores, (num_seqs, beam_size, choice_max_seq_len))[seq_idx, :, i]
                    print("        Beam scores:", list(beam_scores))

                    src_beams = raw_values["%s%s_src_beams_raw" % (key_prefix, choice_name)]
                    assert src_beams.shape == (num_seqs * beam_size * choice_max_seq_len,)
                    src_beams = numpy.reshape(src_beams, (num_seqs, beam_size, choice_max_seq_len))[seq_idx, :, i]
                    print("        Src beam indices:", list(src_beams))

                    choice_values = raw_values["%s%s_value_raw" % (key_prefix, choice_name)]
                    assert choice_values.shape == (num_seqs * beam_size * choice_max_seq_len,)
                    choice_values = numpy.reshape(choice_values, (num_seqs, beam_size, choice_max_seq_len))[
                        seq_idx, :, i
                    ]
                    print("        Choice values:", list(choice_values))

            plot_key_prefix = None
            if plotter:
                plot_key_prefix = plotter.recent_plot.add_model_name(plot_name_prefix)
                assert plot_key_prefix

            for n in range(last_beam_size):
                final_beam_scores = raw_values["%sfinal_beam_scores" % key_prefix]
                assert final_beam_scores.shape == (num_seqs * last_beam_size,)
                final_beam_scores = numpy.reshape(final_beam_scores, (num_seqs, last_beam_size))
                print("    Final hyp %i (out of %i), score %f:" % (n, last_beam_size, final_beam_scores[seq_idx, n]))
                for choice_name in choice_layer_names:
                    choice_seq_lens = raw_values["%s%s_beam_scores_final_seq_lens" % (key_prefix, choice_name)]
                    assert choice_seq_lens.shape == (num_seqs * last_beam_size, 1)
                    choice_max_seq_len = max(choice_seq_lens.flat)

                    choice_values = raw_values["%s%s_value_final" % (key_prefix, choice_name)]
                    assert choice_values.shape == (num_seqs * last_beam_size * choice_max_seq_len,)
                    choice_values = numpy.reshape(choice_values, (num_seqs, last_beam_size, choice_max_seq_len))
                    choice_values = list(choice_values[seq_idx, n, : last_choice_seq_lens[n]])
                    print("      %s:" % choice_name, choice_values)

                    if choice_name == pos_choice:
                        hard_att_align = [v * pos_time_reduction_factor for v in choice_values]
                        print("      %s pos with factor %i:" % (choice_name, pos_time_reduction_factor), hard_att_align)
                        if plotter and plot_key_prefix:
                            AlignmentPlotter.Alignment(
                                plot=plotter,
                                key="%s_02_hard_att_%s" % (plot_key_prefix, choice_name),
                                human="%s%s" % (plot_name_prefix, plot_replace_names.get(choice_name, choice_name)),
                                cmap_palette=4,
                                alignment=choice_values,
                                time_reduction=pos_time_reduction_factor,
                                add_extra_time=add_extra_time_frames,
                                add_extra_time_red=add_extra_time_red_frames,
                            ).add_to_plot(overwrite=True)

                    if choice_name == choice_layer_names[-1] and labels is not None:
                        label_str = " ".join([labels[l] for l in choice_values])
                        print("      %s labels:" % choice_name, label_str)
                        if plotter:
                            plotter.recent_plot.output = label_str.split()
                        if remove_sentence_end and label_str.endswith(" %s" % remove_sentence_end):
                            label_str = label_str[: -len(remove_sentence_end) - 1]
                        if apply_bpe_merge:
                            label_str = label_str.replace("@@ ", "")
                            print("      %s labels (after BPE merge):" % choice_name, label_str)
                        if lexicon:
                            words = label_str.split()
                            phone_alignment = select_phone_alignment(phone_alignments=phone_alignments, seq_tag=seq_tag)
                            if plotter:
                                plotter.recent_plot.time_length = phone_alignment.items[-1].end_frame + 1
                            word_alignment = phone_alignment_to_word_alignment(
                                lexicon=lexicon, words=words, phone_alignment=phone_alignment
                            )
                            print("      word alignment:", word_alignment)
                            # We found word alignment -> this output seq is ground truth.
                            if plotter and word_alignment and plot_key_prefix:
                                plotter.recent_plot.set_time_alignment(word_alignment)
                                spatial_sm_value_keys = {
                                    key[: -len("_spatial_sm_value_final")]
                                    for key in keys
                                    if key.endswith("_spatial_sm_value_final")
                                }
                                assert all([name.startswith(key_prefix) for name in spatial_sm_value_keys])
                                spatial_sm_value_keys = {name[len(key_prefix) :] for name in spatial_sm_value_keys}
                                print("      Spatial softmax keys:", spatial_sm_value_keys)
                                for key in spatial_sm_value_keys:
                                    spatial_seq_lens = raw_values[
                                        "%s%s_spatial_sm_value_final_seq_lens" % (key_prefix, key)
                                    ]
                                    assert spatial_seq_lens.shape[0] == num_seqs * last_beam_size
                                    assert spatial_seq_lens.ndim == 2
                                    assert spatial_seq_lens.shape[1] in {
                                        2,
                                        3,
                                    }  # (dec-len,enc-len) tuples, or (dec-len,enc-len,head)
                                    max_dec_len = max(spatial_seq_lens[:, 0])
                                    max_enc_len = max(spatial_seq_lens[:, 1])
                                    spatial_seq_lens = numpy.reshape(
                                        spatial_seq_lens, (num_seqs, last_beam_size, spatial_seq_lens.shape[-1])
                                    )
                                    if spatial_seq_lens.shape[-1] == 2:
                                        dec_len, enc_len = spatial_seq_lens[seq_idx, n]
                                        num_heads = 1
                                    else:
                                        dec_len, enc_len, num_heads = spatial_seq_lens[seq_idx, n]
                                    spatial_sm_value = raw_values["%s%s_spatial_sm_value_final" % (key_prefix, key)]
                                    assert spatial_sm_value.shape == (
                                        num_seqs * last_beam_size * max_dec_len * max_enc_len * num_heads,
                                    )
                                    spatial_sm_value = numpy.reshape(
                                        spatial_sm_value,
                                        (num_seqs, last_beam_size, max_dec_len, max_enc_len, num_heads),
                                    )
                                    spatial_sm_value = spatial_sm_value[seq_idx, n, :dec_len, :enc_len]
                                    if spatial_seq_lens.shape[-1] == 2 or num_heads == 1:
                                        AlignmentPlotter.Alignment(
                                            plot=plotter,
                                            key="%s_01_sm_%s" % (plot_key_prefix, key),
                                            human="%s%s" % (plot_name_prefix, plot_replace_names.get(key, key)),
                                            alignment=spatial_sm_value[:, :, 0],
                                            time_reduction=pos_time_reduction_factor,
                                            add_extra_time=add_extra_time_frames,
                                            add_extra_time_red=add_extra_time_red_frames,
                                            cmap_palette=0,
                                            colorbar=True,
                                        ).add_to_plot()
                                    else:
                                        for h in range(num_heads):
                                            AlignmentPlotter.Alignment(
                                                plot=plotter,
                                                key="%s_01_sm_%s_%02i" % (plot_key_prefix, key, h),
                                                human="%s%s, head %i"
                                                % (plot_name_prefix, plot_replace_names.get(key, key), h),
                                                alignment=spatial_sm_value[:, :, h],
                                                time_reduction=pos_time_reduction_factor,
                                                add_extra_time=add_extra_time_frames,
                                                add_extra_time_red=add_extra_time_red_frames,
                                                cmap_palette=0,
                                                colorbar=True,
                                            ).add_to_plot()
                                plot_key_prefix = None

    return plotter


def main():
    arg_parser = argparse.ArgumentParser(description="Visualize search scores")
    arg_parser.add_argument("hdf", help="dumped via debug-dump-search-scores.py")
    arg_parser.add_argument("--rec_layer_name", default="output")
    arg_parser.add_argument("--lexicon", help="xml file")
    arg_parser.add_argument("--apply_bpe_merge", action="store_true", help="replace '@@ ' by ''")
    arg_parser.add_argument("--remove_sentence_end", help="e.g. '<s>'")
    arg_parser.add_argument("--phone_alignment_file_pattern", help="e.g. 'align-*.txt', stdout from RASR archiver")
    arg_parser.add_argument("--pos_choice", help="e.g. 't' in decoder")
    arg_parser.add_argument("--pos_time_reduction_factor", type=int, default=1)
    arg_parser.add_argument("--add_extra_time_frames", type=int, default=0)
    arg_parser.add_argument("--plot", action="store_true")
    args = arg_parser.parse_args()

    plotter = read_hdf(
        hdf_filename=args.hdf,
        rec_layer_name=args.rec_layer_name,
        lexicon=args.lexicon,
        apply_bpe_merge=args.apply_bpe_merge,
        remove_sentence_end=args.remove_sentence_end,
        phone_alignment_file_pattern=args.phone_alignment_file_pattern,
        pos_choice=args.pos_choice,
        pos_time_reduction_factor=args.pos_time_reduction_factor,
        add_extra_time_frames=args.add_extra_time_frames,
        plotter=args.plot,
    )

    if plotter:
        for seq_tag, plot in sorted(plotter.plot_by_seq_tag.items()):
            assert isinstance(plot, AlignmentPlotter.Plot)
            plot.do_plot()


if __name__ == "__main__":
    try:
        from returnn.util import better_exchook

        better_exchook.install()
    except ImportError:
        better_exchook = None
    main()
