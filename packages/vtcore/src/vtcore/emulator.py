"""VT emulation (pyte) -> Frame snapshots, and keyframe sampling over a cast."""

from __future__ import annotations

from typing import Iterator

import numpy as np
import pyte

from .cast import Cast
from .frame import BLINK, BOLD, DEFAULT_COLOR, ITALIC, REVERSE, STRIKE, UNDERLINE, Frame

_NAMES = ["black", "red", "green", "brown", "blue", "magenta", "cyan", "white"]
_COLOR_IDX = {n: i for i, n in enumerate(_NAMES)} | {"bright" + n: i + 8 for i, n in enumerate(_NAMES)}
_COLOR_IDX |= {"yellow": 3, "brightyellow": 11, "default": DEFAULT_COLOR}
_ANSI_RGB = np.array(
    [(0, 0, 0), (205, 0, 0), (0, 205, 0), (205, 205, 0), (0, 0, 238), (205, 0, 205), (0, 205, 205), (229, 229, 229),
     (127, 127, 127), (255, 0, 0), (0, 255, 0), (255, 255, 0), (92, 92, 255), (255, 0, 255), (0, 255, 255), (255, 255, 255)],
    dtype=np.int32,
)
_hex_cache: dict[str, int] = {}


def color_index(c: str) -> int:
    i = _COLOR_IDX.get(c)
    if i is not None:
        return i
    i = _hex_cache.get(c)
    if i is None:
        try:
            rgb = np.array([int(c[k : k + 2], 16) for k in (0, 2, 4)])
            i = int(((_ANSI_RGB - rgb) ** 2).sum(1).argmin())
        except ValueError:
            i = 17
        _hex_cache[c] = i
    return i


class VT:
    """Incremental emulator that keeps dense arrays in sync using pyte's dirty-line set."""

    def __init__(self, cols: int, rows: int):
        self.screen = pyte.Screen(cols, rows)
        self.stream = pyte.Stream(self.screen)
        self._alloc(rows, cols)

    def _alloc(self, rows: int, cols: int) -> None:
        self.cp = np.full((rows, cols), 32, np.int32)
        self.fg = np.full((rows, cols), DEFAULT_COLOR, np.uint8)
        self.bg = np.full((rows, cols), DEFAULT_COLOR, np.uint8)
        self.attr = np.zeros((rows, cols), np.uint8)
        self.screen.dirty.update(range(rows))

    def feed(self, data: str) -> None:
        try:
            self.stream.feed(data)
        except Exception:  # malformed sequences in the wild: skip rather than abort the recording
            pass

    def resize(self, cols: int, rows: int) -> None:
        self.screen.resize(rows, cols)
        self._alloc(rows, cols)

    def _sync(self) -> None:
        scr = self.screen
        rows, cols = scr.lines, scr.columns
        for y in scr.dirty:
            if y >= rows:
                continue
            line = scr.buffer[y]
            cp, fg, bg, at = self.cp[y], self.fg[y], self.bg[y], self.attr[y]
            for x in range(cols):
                ch = line[x]
                d = ch.data
                cp[x] = ord(d[0]) if d else 0
                fg[x] = color_index(ch.fg)
                bg[x] = color_index(ch.bg)
                at[x] = (BOLD * ch.bold) | (ITALIC * ch.italics) | (UNDERLINE * ch.underscore) | (REVERSE * ch.reverse) | (
                    BLINK * ch.blink) | (STRIKE * ch.strikethrough)
        scr.dirty.clear()

    def snapshot(self, t: float = 0.0, rec_id: str = "", prev: Frame | None = None) -> Frame:
        self._sync()
        c = self.screen.cursor
        cursor = (-1, -1) if c.hidden else (min(c.y, self.screen.lines - 1), min(c.x, self.screen.columns - 1))
        f = Frame(self.cp.copy(), self.fg.copy(), self.bg.copy(), self.attr.copy(), cursor, t, rec_id=rec_id)
        if prev is not None and prev.shape == f.shape:
            f.damage = (prev.cp != f.cp) | (prev.fg != f.fg) | (prev.bg != f.bg) | (prev.attr != f.attr)
        else:
            f.damage = np.ones(f.shape, bool)
        return f


def keyframes(
    cast: Cast,
    idle: float = 0.15,
    max_interval: float = 3.0,
    max_frames: int = 400,
    rec_id: str | None = None,
) -> Iterator[Frame]:
    """Yield a Frame whenever the screen *settles* (no output for ``idle`` s) or at least every
    ``max_interval`` s while output keeps streaming. Unchanged screens are skipped."""
    vt = VT(cast.cols, cast.rows)
    rid = rec_id or str(cast.meta.get("id", ""))
    prev: Frame | None = None
    last_emit = -1e9
    pending = False
    n = 0
    evs = cast.events
    for i, e in enumerate(evs):
        if e.code == "o":
            vt.feed(e.data)
            pending = True
        elif e.code == "r":
            try:
                cols, rows = (int(v) for v in e.data.lower().split("x"))
                vt.resize(cols, rows)
                pending = True
            except ValueError:
                pass
        if not pending:
            continue
        nxt = evs[i + 1].t if i + 1 < len(evs) else float("inf")
        if nxt - e.t >= idle or e.t - last_emit >= max_interval:
            f = vt.snapshot(e.t, rid, prev)
            pending = False
            last_emit = e.t
            if prev is not None and not f.damage.any():
                continue
            yield f
            prev = f
            n += 1
            if n >= max_frames:
                return
