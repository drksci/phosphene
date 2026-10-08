"""Labeling functions: free, instant weak labels for every frame.

Each rule writes (role, confidence) into cell maps; a later rule only overwrites a cell when its
confidence is higher. The per-frame mean confidence over non-blank cells drives active selection of
frames to send to the (paid) LLM labeler: the frames the rules are least sure about.
"""

from __future__ import annotations

import re

import numpy as np

from vtcore.frame import BOLD, DEFAULT_COLOR, REVERSE, Frame
from vtcore.roles import FILL_ROLES, ROLE_ID

R = ROLE_ID

RE_PROMPT = re.compile(
    r"^(?:\S+@\S+[:\s][^\s$#%]*[\]\)]?\s?[$#%>] |\[[^\]]+\][$#] |(?:\S+ )?[❯➜λ›»] |[$#%] |>>> |\.\.\. |In \[\d+\]: |PS [A-Z]:\\[^>]*> |irb\(\S+\):\d+:\d+[>*] |mysql> |postgres=# |\w+> )"
)
RE_LOG = re.compile(
    r"^\s*(?:\[?\d{4}-\d\d-\d\d[T ]\d\d:\d\d|\[?\d\d:\d\d:\d\d|\[?(?:INFO|WARN(?:ING)?|DEBUG|TRACE|ERROR|ERR|FATAL)\]?[: ]|[IWEF]\d{4} \d\d:)"
)
RE_ERROR = re.compile(r"(?i)\b(error|fatal|exception|traceback|failed|failure|panic|denied|not found|segmentation fault)\b|^\s*(?:E\s|!\s|✗|✘|×)")
RE_WARN = re.compile(r"(?i)\b(warn(?:ing)?|deprecated)\b")
RE_PROGRESS = re.compile(
    r"\[[#=\-|>.\s*]{4,}\]|\d{1,3}(?:\.\d)?%|[█▉▊▋▌▍▎▏▓▒░■□⣿]{3,}|[━─]{3,}╸?\s*\d|[⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏⣾⣽⣻⢿⡿⣟⣯⣷◐◓◑◒]\s|\d+(?:\.\d+)?\s?[KMG]i?B/s|ETA\s"
)
RE_KEYHINT = re.compile(
    r"(?:\^[A-Z\\_\]]\s?[A-Z][\w ]{1,12})|(?:\bF\d{1,2}\s?[A-Z][a-z]+)|(?:<\s*[A-Z][\w ]{0,10}\s*>)|(?:\[\s*[A-Z][\w ]{0,10}\s*\])|(?:\b(?:q|Esc|Enter|Tab|Space|↑↓|j/k|\?)\s(?:quit|help|select|back|search|filter|menu|exit|close)\b)"
)
RE_GUTTER = re.compile(r"^\s{0,4}\d{1,5}[ │|:]")
BOX = set(range(0x2500, 0x2580))
BLOCKS = set(range(0x2580, 0x25A0))


def label(f: Frame) -> tuple[np.ndarray, np.ndarray]:
    h, w = f.shape
    role = np.zeros((h, w), np.int8)
    conf = np.zeros((h, w), np.float32)
    nonblank = f.cp != 32
    lines = f.lines()

    def put(mask, rid, c):
        m = mask & (conf < c)
        role[m] = rid
        conf[m] = c

    def row_span(r, c0, c1, rid, c):
        m = np.zeros((h, w), bool)
        m[r, c0:c1] = True
        put(m, rid, c)

    # base: any glyph is text
    put(nonblank, R["text"], 0.3)
    # colour/attr cues
    red = (f.fg == 1) | (f.fg == 9)
    put(nonblank & red, R["error"], 0.45)

    # box drawing / ascii frames
    box = np.isin(f.cp, list(BOX))
    put(box, R["border"], 0.9)
    for r, line in enumerate(lines):
        for m in re.finditer(r"[+][-=+]{3,}[+]|[-=]{8,}|_{8,}", line):
            row_span(r, m.start(), m.end(), R["border"], 0.7)
    pipes = f.cp == ord("|")
    vert = pipes & np.roll(pipes, 1, 0) & np.roll(pipes, -1, 0)
    put(vert, R["border"], 0.6)

    # titles embedded in frame top edges: text between box chars on a border row
    for r, line in enumerate(lines):
        if box[r].sum() > w * 0.3:
            for m in re.finditer(r"[┤\]]?\s?([^\s─━═┌┐╭╮╔╗┏┓┤├].*?[^\s─━═┌┐╭╮╔╗┏┓┤├])\s?[├\[]?", line):
                if not np.isin(f.cp[r, m.start(1):m.end(1)], list(BOX)).any():
                    row_span(r, m.start(1), m.end(1), R["title"], 0.75)

    # highlighted spans: status bars (full-width, top/bottom) vs selected items (inner rows)
    hl = ((f.attr & REVERSE) > 0) | (f.bg != DEFAULT_COLOR)
    bg_mode = np.bincount(f.bg.ravel(), minlength=18).argmax()  # dialog-style coloured background
    hl &= ~((f.bg == bg_mode) & ((f.attr & REVERSE) == 0))
    for r in range(h):
        cover = hl[r].mean()
        if cover >= 0.6 and (r <= 1 or r >= h - 2):
            row_span(r, 0, w, R["status_bar"], 0.8)
        elif cover >= 0.6 and nonblank[r].any():
            row_span(r, 0, w, R["selected"], 0.55)  # could be a table header too: low conf
        else:
            # runs of >= 8 highlighted cells in the middle of the screen
            run, c0 = 0, 0
            for c in range(w + 1):
                if c < w and hl[r, c]:
                    if run == 0:
                        c0 = c
                    run += 1
                else:
                    if run >= 8 and 1 < r < h - 2 and nonblank[r, c0:c0 + run].any():
                        row_span(r, c0, c0 + run, R["selected"], 0.6)
                    run = 0

    for r, line in enumerate(lines):
        s = line.rstrip()
        if not s:
            continue
        # prompt + input on the cursor row
        m = RE_PROMPT.match(s) if box[r].sum() == 0 else None
        if m:
            row_span(r, 0, m.end(), R["prompt"], 0.8 if r == f.cursor[0] else 0.65)
            if r == f.cursor[0]:
                row_span(r, m.end(), max(len(s), f.cursor[1] + 1), R["input"], 0.75)
            else:
                row_span(r, m.end(), len(s), R["text"], 0.5)
        if RE_LOG.match(s):
            row_span(r, 0, len(s), R["log"], 0.7)
        if RE_ERROR.search(s) and not RE_LOG.match(s) or re.search(r"\b(ERROR|ERR|FATAL)\b", s[:40]):
            row_span(r, len(s) - len(s.lstrip()), len(s), R["error"], 0.72)
        elif RE_WARN.search(s):
            row_span(r, len(s) - len(s.lstrip()), len(s), R["error"], 0.6)
        for m in RE_PROGRESS.finditer(line):
            row_span(r, m.start(), m.end(), R["progress"], 0.78)
        for m in RE_KEYHINT.finditer(line):
            row_span(r, m.start(), m.end(), R["key_hint"], 0.7 if r >= h - 3 else 0.55)
        if line.lstrip().startswith("~") and len(s.strip()) == 1:
            row_span(r, 0, len(s), R["code"], 0.7)

    # progress bars made of block elements / meters: extend through the whole bracketed run
    blocks = np.isin(f.cp, list(BLOCKS)) | (f.cp == 0x2800) | ((f.cp >= 0x2801) & (f.cp <= 0x28FF))
    put(blocks, R["progress"], 0.7)

    # editor gutters: >= 3 consecutive rows with incrementing line numbers
    _gutter(f, lines, put)
    # tables: >= 3 consecutive rows sharing >= 2 aligned whitespace column gaps
    _tables(lines, nonblank, put)

    role[~nonblank & ~np.isin(role, list(FILL_ROLES))] = 0
    conf[role == 0] = 1.0
    return role, conf


def _gutter(f, lines, put):
    h, w = f.shape
    nums = []
    for r, line in enumerate(lines):
        m = RE_GUTTER.match(line)
        nums.append(int(re.findall(r"\d+", line[: m.end()])[0]) if m else None)
    r = 0
    while r < h:
        if nums[r] is None:
            r += 1
            continue
        e = r
        while e + 1 < h and nums[e + 1] is not None and nums[e + 1] == nums[e] + 1:
            e += 1
        if e - r >= 2:
            m = np.zeros((h, w), bool)
            m[r:e + 1] = f.cp[r:e + 1] != 32
            put(m, R["code"], 0.72)
        r = e + 1


def _tables(lines, nonblank, put):
    h, w = nonblank.shape
    gaps = np.zeros((h, w), bool)  # cell is part of a >=2-space gap between tokens
    for r, line in enumerate(lines):
        for m in re.finditer(r"(?<=\S)\s{2,}(?=\S)", line):
            gaps[r, m.start():m.end()] = True
    r = 0
    while r < h - 2:
        common = gaps[r] & gaps[r + 1] & gaps[r + 2]
        if _n_runs(common) >= 2:
            e = r + 2
            while e + 1 < h and _n_runs(common & gaps[e + 1]) >= 2:
                common &= gaps[e + 1]
                e += 1
            m = np.zeros((h, w), bool)
            m[r:e + 1] = nonblank[r:e + 1]
            put(m, R["table"], 0.62)
            r = e + 1
        else:
            r += 1


def _n_runs(mask: np.ndarray) -> int:
    return int(np.sum(mask[1:] & ~mask[:-1]) + (1 if mask[0] else 0))


def frame_confidence(role: np.ndarray, conf: np.ndarray) -> float:
    m = role != 0
    return float(conf[m].mean()) if m.any() else 1.0
