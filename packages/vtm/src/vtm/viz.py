"""Notebook visualisation: terminal frames with role overlays, metric plots, A2UI replay."""

from __future__ import annotations

import html

import numpy as np

from vtcore.frame import BOLD, DEFAULT_COLOR, REVERSE, UNDERLINE, Frame
from vtcore.roles import ROLE_COLORS, ROLES
from vttui.style import XTERM_PALETTE

DEF_FG, DEF_BG = "#d4d4d4", "#1e1e1e"


def frame_html(f: Frame, roles: np.ndarray | None = None, title: str = "", alpha: float = 0.38, font_px: int = 11) -> str:
    """Terminal frame as coloured HTML; with ``roles``, each cell is tinted by its role colour."""
    h, w = f.shape
    rows = []
    for r in range(h):
        spans, c = [], 0
        while c < w:
            key = (f.fg[r, c], f.bg[r, c], f.attr[r, c], -1 if roles is None else roles[r, c])
            e = c + 1
            while e < w and (f.fg[r, e], f.bg[r, e], f.attr[r, e], -1 if roles is None else roles[r, e]) == key:
                e += 1
            fg, bg, at, role = key
            fgc = XTERM_PALETTE[fg] if fg < 16 else DEF_FG
            bgc = XTERM_PALETTE[bg] if bg < 16 else DEF_BG
            if at & REVERSE:
                fgc, bgc = bgc, fgc
            st = f"color:{fgc};background:{bgc};"
            if role is not None and role > 0:
                col = ROLE_COLORS[role]
                st += f"box-shadow:inset 0 0 0 100vmax {col}{int(alpha * 255):02x};"
            if at & BOLD:
                st += "font-weight:700;"
            if at & UNDERLINE:
                st += "text-decoration:underline;"
            txt = "".join(chr(x) if x else "" for x in f.cp[r, c:e])
            spans.append(f'<span style="{st}">{html.escape(txt)}</span>')
            c = e
        rows.append("".join(spans))
    cur = f"cursor r{f.cursor[0]} c{f.cursor[1]}" if f.cursor[0] >= 0 else "cursor hidden"
    head = f'<div style="font:12px system-ui;color:#888;margin:2px 0">{html.escape(title)} <span style="opacity:.6">{h}x{w} · {cur}</span></div>'
    return (f'<div style="display:inline-block;vertical-align:top;margin:4px">{head}'
            f'<pre style="margin:0;padding:6px;background:{DEF_BG};font:{font_px}px/1.2 ui-monospace,Menlo,monospace;'
            f'border-radius:6px;overflow:auto;max-width:100%">{"<br>".join(rows)}</pre></div>')


def legend_html() -> str:
    items = "".join(
        f'<span style="display:inline-block;margin:2px 6px;font:12px system-ui"><span style="display:inline-block;width:10px;height:10px;'
        f'background:{ROLE_COLORS[i]};border-radius:2px;margin-right:4px"></span>{r}</span>'
        for i, r in enumerate(ROLES) if i)
    return f'<div style="margin:4px 0">{items}</div>'


def compare_html(f: Frame, maps: dict[str, np.ndarray], font_px: int = 9) -> str:
    return legend_html() + "".join(frame_html(f, m, title=k, font_px=font_px) for k, m in maps.items())


def show(html_str: str):
    from IPython.display import HTML, display

    display(HTML(html_str))


def plot_iou(reports: dict[str, dict], ax=None):
    import matplotlib.pyplot as plt

    names = [r for r in ROLES[1:]]
    ax = ax or plt.subplots(figsize=(12, 3.5))[1]
    width = 0.8 / max(len(reports), 1)
    for i, (k, rep) in enumerate(reports.items()):
        vals = [rep["per_class"][n]["iou"] for n in names]
        ax.bar(np.arange(len(names)) + i * width, vals, width, label=f"{k} (mIoU {rep['macro_iou']:.2f})")
    ax.set_xticks(np.arange(len(names)) + width * (len(reports) - 1) / 2, names, rotation=40, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("IoU")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=.3)
    return ax


def plot_confusion(cm: np.ndarray, ax=None, title: str = ""):
    import matplotlib.pyplot as plt

    ax = ax or plt.subplots(figsize=(7, 6))[1]
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(ROLES)), ROLES, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(ROLES)), ROLES, fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("gold")
    ax.set_title(title)
    for i in range(len(ROLES)):
        for j in range(len(ROLES)):
            if norm[i, j] >= 0.1:
                ax.text(j, i, f"{norm[i, j]:.1f}", ha="center", va="center", fontsize=6, color="white" if norm[i, j] > .5 else "black")
    return ax


def replay(frames: list[Frame], roles: list[np.ndarray], msgs: list[list[dict]]):
    """Slider over keyframes: terminal (with roles) | A2UI native render | A2UI faithful render | JSONL delta."""
    import ipywidgets as W
    from IPython.display import HTML, display

    from vttui.reconcile import to_jsonl
    from vttui.render import Surface

    surf, pages = Surface(), []
    for f, r, m in zip(frames, roles, msgs):
        for x in m:
            surf.apply(x)
        delta = html.escape(to_jsonl(m)[:4000])
        pages.append(
            f'<div style="display:flex;gap:12px;flex-wrap:wrap;align-items:flex-start">{frame_html(f, r, "terminal + roles", font_px=9)}'
            f'<div style="max-width:520px"><div style="font:12px system-ui;color:#888">A2UI · native</div>{surf.render(False)}</div>'
            f'<div><div style="font:12px system-ui;color:#888">A2UI · faithful (style layer)</div>'
            f'<div style="font-size:9px">{surf.render(True)}</div></div></div>'
            f'<details><summary style="font:12px system-ui">delta stream for this keyframe ({len(m)} msgs)</summary>'
            f'<pre style="font-size:10px;white-space:pre-wrap">{delta}</pre></details>')
    out = W.Output()
    sl = W.IntSlider(0, 0, max(len(pages) - 1, 0), description="keyframe", layout=W.Layout(width="60%"))

    def draw(change=None):
        out.clear_output(wait=True)
        with out:
            display(HTML(legend_html() + pages[sl.value]))

    sl.observe(draw, "value")
    draw()
    play = W.Play(min=0, max=sl.max, interval=700, description="play")
    W.jslink((play, "value"), (sl, "value"))
    display(W.VBox([W.HBox([play, sl]), out]))
    return sl
