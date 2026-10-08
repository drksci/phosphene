"""Grid encodings: model features, layout signatures (dedupe), and compact text for LLM labelers."""

from __future__ import annotations

import hashlib
import unicodedata

import numpy as np

from .frame import BOLD, DEFAULT_COLOR, REVERSE, UNDERLINE, Frame

# ---------------------------------------------------------------------------
# char vocabulary: ~300 ids that keep the structurally meaningful distinctions
PAD, SPACE, WIDE_CONT = 0, 1, 2
ASCII0 = 3  # 0x21..0x7E -> 3..96
BOX0 = 97  # U+2500..257F individually -> 97..224
BLOCK0 = 225  # U+2580..259F individually -> 225..256
BRAILLE = 257
ARROWS_SHAPES0 = 258  # U+2190..21FF, U+25A0..25FF, U+2700..27BF hashed into 16 buckets -> 258..273
CAT0 = 274  # unicode general category buckets
_CATS = {"L": 0, "N": 1, "P": 2, "S": 3, "M": 4, "Z": 5, "C": 6}
CJK = 281
PRIVATE_USE = 282  # powerline / nerd-font glyphs
CHAR_VOCAB = 288


def _char_id(cp: int) -> int:
    if cp == 0:
        return WIDE_CONT
    if cp == 32:
        return SPACE
    if 0x21 <= cp <= 0x7E:
        return ASCII0 + cp - 0x21
    if 0x2500 <= cp <= 0x257F:
        return BOX0 + cp - 0x2500
    if 0x2580 <= cp <= 0x259F:
        return BLOCK0 + cp - 0x2580
    if 0x2800 <= cp <= 0x28FF:
        return BRAILLE
    if 0x2190 <= cp <= 0x21FF or 0x25A0 <= cp <= 0x25FF or 0x2700 <= cp <= 0x27BF:
        return ARROWS_SHAPES0 + cp % 16
    if 0xE000 <= cp <= 0xF8FF or cp >= 0xF0000:
        return PRIVATE_USE
    try:
        ch = chr(cp)
    except ValueError:
        return CAT0 + 6
    if unicodedata.east_asian_width(ch) in ("W", "F"):
        return CJK
    return CAT0 + _CATS.get(unicodedata.category(ch)[0], 6)


_LUT = np.array([_char_id(c) for c in range(0x10000)], dtype=np.int16)


def char_ids(cp: np.ndarray) -> np.ndarray:
    out = _LUT[np.clip(cp, 0, 0xFFFF)]
    hi = cp > 0xFFFF
    if hi.any():
        out = out.copy()
        out[hi] = [_char_id(int(c)) for c in cp[hi]]
    return out


def features(f: Frame) -> dict[str, np.ndarray]:
    """Per-cell integer features consumed by the VTM model."""
    h, w = f.shape
    cur = np.zeros((h, w), np.uint8)
    r, c = f.cursor
    if r >= 0:
        cur[r, :] = 1  # cursor row
        cur[r, c] = 2  # cursor cell
    return {
        "char": char_ids(f.cp).astype(np.int64),
        "fg": f.fg.astype(np.int64),
        "bg": f.bg.astype(np.int64),
        "attr": f.attr.astype(np.int64),
        "damage": (f.damage if f.damage is not None else np.ones((h, w), bool)).astype(np.int64),
        "cursor": cur.astype(np.int64),
    }


# ---------------------------------------------------------------------------
# layout signature: robust near-duplicate key (ignores which letters/digits are shown)
def _shape_class(cid: np.ndarray) -> np.ndarray:
    cls = np.full(cid.shape, ord("?"), np.uint8)
    cls[cid == SPACE] = ord(" ")
    a = (cid >= ASCII0) & (cid < BOX0)
    cls[a] = ord("a")
    cls[(cid >= BOX0) & (cid < BLOCK0)] = ord("+")
    cls[(cid >= BLOCK0) & (cid <= BRAILLE)] = ord("#")
    return cls


def layout_signature(f: Frame, coarse: int = 4) -> str:
    """Hash of the coarse shape: char class per cell + reverse/bg-coloured spans, run-length
    collapsed per row, with column positions quantised to ``coarse`` cells."""
    cls = _shape_class(char_ids(f.cp))
    hl = ((f.attr & REVERSE) > 0) | (f.bg != DEFAULT_COLOR)
    h = hashlib.blake2b(digest_size=10)
    h.update(bytes(f.shape))
    for r in range(f.shape[0]):
        row = cls[r].copy()
        row[hl[r]] = ord("R")
        runs, start = [], 0
        for c in range(1, len(row) + 1):
            if c == len(row) or row[c] != row[start]:
                runs.append(f"{chr(row[start])}{start // coarse}")
                start = c
        h.update(("|".join(runs) + "\n").encode())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# compact text rendering for LLM labelers
_COLOR_NAMES = ["black", "red", "green", "yellow", "blue", "magenta", "cyan", "white"]


def _cname(i: int) -> str:
    if i < 8:
        return _COLOR_NAMES[i]
    if i < 16:
        return "br" + _COLOR_NAMES[i - 8]
    return "def" if i == DEFAULT_COLOR else "other"


def render_for_llm(f: Frame, max_style_runs: int = 60) -> str:
    """Rows prefixed with a 2-digit index (so the LLM can cite rows/cols exactly), a column ruler,
    and a compact list of styled spans (reverse / bg colour / fg colour / bold) on the side."""
    h, w = f.shape
    out = [f"size {h}x{w}  cursor={'hidden' if f.cursor[0] < 0 else f'r{f.cursor[0]} c{f.cursor[1]}'}"]
    out.append("   " + "".join(str((c // 10) % 10) if c % 10 == 0 else " " for c in range(w)).rstrip())
    out.append("   " + "".join(str(c % 10) for c in range(w)))
    r = 0
    while r < h:  # collapse runs of empty rows to save tokens
        txt = f.row_text(r).rstrip()
        if not txt and not (f.bg[r] != DEFAULT_COLOR).any() and not (f.attr[r] & REVERSE).any():
            e = r
            while e + 1 < h and not f.row_text(e + 1).strip() and not (f.bg[e + 1] != DEFAULT_COLOR).any() and not (f.attr[e + 1] & REVERSE).any():
                e += 1
            out.append(f"{r:02d}|" if e == r else f"{r:02d}-{e:02d}| (empty)")
            r = e + 1
            continue
        out.append(f"{r:02d}|" + txt)
        r += 1
    styles = []
    keys = (f.attr & (REVERSE | BOLD | UNDERLINE)).astype(np.int32) * 400 + f.fg.astype(np.int32) * 20 + f.bg.astype(np.int32)
    for r in range(h):
        row = keys[r]
        c0 = 0
        for c in range(1, w + 1):
            if c == w or row[c] != row[c0]:
                k = int(row[c0])
                at, fg, bg = k // 400, (k // 20) % 20, k % 20
                nonblank = (f.cp[r, c0:c] != 32).any()
                if (at & REVERSE) or bg != DEFAULT_COLOR or ((fg != DEFAULT_COLOR or at & BOLD) and nonblank):
                    tags = [t for t, b in (("rev", REVERSE), ("bold", BOLD), ("ul", UNDERLINE)) if at & b]
                    if fg != DEFAULT_COLOR:
                        tags.append(f"fg={_cname(fg)}")
                    if bg != DEFAULT_COLOR:
                        tags.append(f"bg={_cname(bg)}")
                    styles.append(f"r{r} c{c0}-{c - 1} {' '.join(tags)}")
                c0 = c
    if styles:
        if len(styles) > max_style_runs:
            styles = styles[:max_style_runs] + [f"... {len(styles) - max_style_runs} more"]
        out.append("styles:")
        out.extend(styles)
    return "\n".join(out)
