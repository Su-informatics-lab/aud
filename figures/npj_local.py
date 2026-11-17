#!/usr/bin/env python3
"""Type sizes, palette and helpers for the panels drawn from aggregate tables; the values are those of downstream/npj/npj_style.py.
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.font_manager                                        # noqa: F401

SANS = ["Arial", "Helvetica", "Liberation Sans", "Nimbus Sans", "DejaVu Sans"]

# Tier 1: category. One meaning per colour across every figure in the paper.
C_INDEX = "#D55E00"      # the LLM
C_CODE = "#0072B2"       # the AUD diagnosis code
C_REF = "#1F1E1D"        # the clinicians, and the reference standard

# Tier 2: value. Foreground against background, no category.
INK, MUTED, RULE, HAIR = "#1F1E1D", "#55534E", "#B9B7B2", "#DCDAD5"

# Tier 3: the ramps, one figure each.
RAMP_INDEX = ["#8C3D00", "#D55E00", "#F0A878", "#F7DCCC"]
RAMP_CODE = ["#00507D", "#0072B2", "#7FB6D9", "#CCE3F0"]
RAMP_REF = ["#0F0E0D", "#4A4844", "#8C8981", "#E6E3DC"]
# The record-silent stratum, and the row tint that marks it. Neutral grey, because orange
# names the LLM and blue names the AUD diagnosis code in every figure.
C_SILENT = "#4D4D4D"
TINT_SILENT = "#EBEBEB"

FS_BODY, FS_SMALL, FS_PANEL = 7.0, 6.0, 8.0
LW_MIN = 1.0
MM = 1 / 25.4

W_DOUBLE, W_HALF, W_SINGLE = 183.0, 120.0, 89.0
H_MAX = 247.0
GUTTER = 5.0
LETTER_X, LETTER_Y, LETTER_BOX_W = 1.0, 1.0, 6.0

# The separator between blocks of rows, as Figure 1f draws it.
DOT = (0, (2.2, 2.2))
LW_HAIR = 0.6


def apply():
    resolved = mpl.font_manager.FontProperties(
        fname=mpl.font_manager.findfont(
            mpl.font_manager.FontProperties(family=SANS))).get_name()
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": SANS,
        "font.size": FS_BODY,
        "text.color": INK,
        "axes.linewidth": LW_MIN,
        "lines.linewidth": LW_MIN,
        "patch.linewidth": LW_MIN,
        "hatch.linewidth": LW_MIN,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": None,
        "savefig.pad_inches": 0.0,
        "mathtext.fontset": "custom",
        "mathtext.rm": resolved,
        "mathtext.it": "%s:italic" % resolved,
        "mathtext.bf": "%s:bold" % resolved,
        "mathtext.cal": "%s:italic" % resolved,
    })
    return resolved


def panel_letter(fig, letter, visible=False):
    """Reserve the letter's place; draw it only when asked.

    Panels are exported on their own and lettered where they are assembled, so the
    placeholder is drawn at alpha 0 by default. The space is reserved either way.
    """
    w = fig.get_size_inches()[0] * 25.4
    h = fig.get_size_inches()[1] * 25.4
    return fig.text(LETTER_X / w, 1.0 - LETTER_Y / h, letter, fontsize=FS_PANEL,
                    fontweight="bold", color=INK, ha="left", va="top",
                    alpha=1.0 if visible else 0.0)


def check(fig, w_mm, h_mm):
    """Warn for text that leaves the panel or sits in the letter's reserved box."""
    fig.canvas.draw()
    items = list(fig.texts)
    for ax in fig.axes:
        items += list(ax.texts)
    bad, seen, margin = 0, 0, w_mm
    for t in items:
        if not t.get_text() or t.get_alpha() == 0.0:
            continue
        bb = t.get_window_extent()
        if bb.x1 / fig.dpi * 25.4 < -w_mm or bb.y1 / fig.dpi * 25.4 < -h_mm:
            continue          # an axes parked off the canvas so one panel can be drawn
        seen += 1
        x0, x1 = bb.x0 / fig.dpi * 25.4, bb.x1 / fig.dpi * 25.4
        y0, y1 = bb.y0 / fig.dpi * 25.4, bb.y1 / fig.dpi * 25.4
        margin = min(margin, w_mm - x1, x0)
        if x1 > w_mm or x0 < 0.0 or y1 > h_mm or y0 < 0.0:
            print("  WARNING %r spans %.1f-%.1f mm across and %.1f-%.1f mm up a "
                  "%.0f x %.0f mm panel" % (t.get_text(), x0, x1, y0, y1, w_mm, h_mm))
            bad += 1
        elif x0 < LETTER_BOX_W and y1 > h_mm - LETTER_BOX_W / 2:
            print("  WARNING %r sits in the panel letter's reserved box" % t.get_text())
            bad += 1
    print("  %d text items inside the panel, the tightest %.1f mm from an edge"
          % (seen, margin))
    return bad
