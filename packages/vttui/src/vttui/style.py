"""Style layer: keep A2UI renders proportionate and colour-faithful to the source terminal.

Every component carries ``style``:
  area   [row0, col0, row1, col1)  in terminal cells (CSS grid lines are these + 1)
  fg/bg  ANSI palette tokens ("ansi-0".."ansi-15", "default")
  bold / inverse  booleans
The surface ``theme`` carries the palette and grid size. ``stylesheet()`` turns components into
plain CSS (a cell-unit CSS grid), so any A2UI renderer can opt into a "faithful" layout, or
ignore it and render natively.
"""

from __future__ import annotations

import numpy as np

from vtcore.frame import BOLD, DEFAULT_COLOR, REVERSE, Frame

XTERM_PALETTE = [
    "#000000", "#cd0000", "#00cd00", "#cdcd00", "#0000ee", "#cd00cd", "#00cdcd", "#e5e5e5",
    "#7f7f7f", "#ff0000", "#00ff00", "#ffff00", "#5c5cff", "#ff00ff", "#00ffff", "#ffffff",
]


def _tok(i: int) -> str:
    return "default" if i >= DEFAULT_COLOR else f"ansi-{i}"


def _mode(a: np.ndarray, default: int = DEFAULT_COLOR) -> int:
    if a.size == 0:
        return default
    return int(np.bincount(a.ravel().astype(np.int64), minlength=18).argmax())


def _style_cells(f: Frame, rows: slice, cols: slice) -> dict:
    cp, fg, bg, at = f.cp[rows, cols], f.fg[rows, cols], f.bg[rows, cols], f.attr[rows, cols]
    ink = cp != 32
    s = {"fg": _tok(_mode(fg[ink])), "bg": _tok(_mode(bg))}
    if ink.any() and (at[ink] & BOLD).mean() > 0.5:
        s["bold"] = True
    if (at & REVERSE).mean() > 0.5:
        s["inverse"] = True
    return s


def region_style(f: Frame, n) -> dict:
    s = _style_cells(f, slice(n.top, n.bottom + 1), slice(n.left, n.right + 1))
    s["area"] = [int(n.top), int(n.left), int(n.bottom) + 1, int(n.right) + 1]
    return s


def line_style(f: Frame, r: int, c0: int, c1: int) -> dict:
    s = _style_cells(f, slice(r, r + 1), slice(c0, c1))
    return {"fg": s["fg"], "bg": s["bg"], "bold": bool(s.get("bold", False))}


def theme(f: Frame, palette: list[str] = XTERM_PALETTE, font: str = "ui-monospace, Menlo, monospace") -> dict:
    h, w = f.shape
    return {
        "grid": {"rows": int(h), "cols": int(w)},
        "palette": {f"ansi-{i}": c for i, c in enumerate(palette)} | {"default-fg": "#d4d4d4", "default-bg": "#1e1e1e"},
        "font": {"family": font, "cellWidth": "1ch", "lineHeight": 1.25},
    }


def _color(tok: str, fallback: str) -> str:
    return f"var(--{tok})" if tok and tok != "default" else f"var(--{fallback})"


def resolve_style(c: dict, data: dict | None) -> dict | None:
    st = c.get("style")
    if isinstance(st, dict) and set(st) == {"path"} and data is not None:
        node = data
        for k in st["path"].split("/")[1:]:
            node = node.get(k) if isinstance(node, dict) else None
        return node
    return st if isinstance(st, dict) else None


def stylesheet(components: dict[str, dict], th: dict, data: dict | None = None, scope: str = ".a2ui-term") -> str:
    """CSS for the 'faithful' layout: components absolutely placed in terminal-cell units, coloured
    with the source palette. Containers without an area (Column/Row) should render as
    ``display: contents`` so placement is always relative to the surface."""
    g, lh = th["grid"], th["font"]["lineHeight"]
    pal = "".join(f"--{k}:{v};" for k, v in th["palette"].items())
    css = [
        f"{scope}{{{pal}--lh:{lh}em;font-family:{th['font']['family']};line-height:var(--lh);position:relative;"
        f"width:{g['cols']}ch;height:calc({g['rows']} * var(--lh));background:var(--default-bg);color:var(--default-fg);"
        f"white-space:pre;overflow:hidden}}",
        f"{scope} .flow{{display:contents}}",
        f"{scope} [data-id]{{position:absolute;overflow:hidden}}",
        f"{scope} .card-frame{{box-sizing:border-box;border:1px solid currentColor;border-radius:4px}}",
        f"{scope} .row-item{{height:var(--lh);display:block}}",
        f"{scope} .row-item.sel{{filter:invert(1)}}",
    ]
    for cid, c in components.items():
        st = resolve_style(c, data)
        if not isinstance(st, dict) or "area" not in st:
            continue
        r0, c0, r1, c1 = st["area"]
        fg, bg = st.get("fg", "default"), st.get("bg", "default")
        if st.get("inverse"):
            fg, bg = bg, fg
        if c["component"] == "Card":  # border runs through the centre of the box-drawing cells
            r0, c0, r1, c1 = r0 + 0.5, c0 + 0.5, r1 - 0.5, c1 - 0.5
        rule = (f"top:calc({r0} * var(--lh));left:{c0}ch;width:{c1 - c0}ch;height:calc({r1 - r0} * var(--lh));"
                f"color:{_color(fg, 'default-fg')};")
        if bg != "default" or st.get("inverse"):
            rule += f"background:{_color(bg, 'default-bg')};"
        if st.get("bold"):
            rule += "font-weight:700;"
        css.append(f'{scope} [data-id="{_css_escape(cid)}"]{{{rule}}}')
    return "\n".join(css)


def _css_escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')
