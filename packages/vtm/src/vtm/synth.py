"""Procedural TUI scenes with *exact* per-cell role labels — the cheapest labels there are.

Each scene is painted on a Canvas (chars + colours + attrs + role), serialised to real ANSI escape
sequences and fed through the same VT emulator used for the asciinema corpus, so the model sees
exactly what a real recording would produce. Animations (typing, progress, scrolling selection)
produce realistic damage masks.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from vtcore.cast import Cast, Event
from vtcore.emulator import VT
from vtcore.frame import BOLD, DEFAULT_COLOR, REVERSE, UNDERLINE, Frame
from vtcore.roles import APP_ID, FILL_ROLES, ROLE_ID

R = ROLE_ID
D = DEFAULT_COLOR
BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(8)

BOX_STYLES = [
    "┌┐└┘─│", "╭╮╰╯─│", "╔╗╚╝═║", "┏┓┗┛━┃", "++++-|",
]
WORDS = ("alpha beta gamma delta server client build deploy config cache index worker queue node docker "
         "kube pod main src lib test release update fetch merge commit branch module package install "
         "network memory disk process thread socket token session request response user admin data").split()
EXTS = ["py", "rs", "go", "ts", "js", "md", "toml", "json", "yaml", "c", "h", "sh", "txt", "lock"]


@dataclass
class Canvas:
    rows: int
    cols: int

    def __post_init__(self):
        self.ch = np.full((self.rows, self.cols), " ", dtype="<U1")
        self.fg = np.full((self.rows, self.cols), D, np.uint8)
        self.bg = np.full((self.rows, self.cols), D, np.uint8)
        self.attr = np.zeros((self.rows, self.cols), np.uint8)
        self.role = np.zeros((self.rows, self.cols), np.int8)
        self.cursor = (-1, -1)

    def copy(self) -> "Canvas":
        c = Canvas.__new__(Canvas)
        c.rows, c.cols = self.rows, self.cols
        for k in ("ch", "fg", "bg", "attr", "role"):
            setattr(c, k, getattr(self, k).copy())
        c.cursor = self.cursor
        return c

    def put(self, r: int, c: int, text: str, role: str, fg=D, bg=D, attr=0) -> int:
        if not (0 <= r < self.rows):
            return c
        for ch in text:
            if 0 <= c < self.cols:
                self.ch[r, c], self.fg[r, c], self.bg[r, c], self.attr[r, c] = ch, fg, bg, attr
                self.role[r, c] = R[role]
            c += 1
        return c

    def fill_row(self, r: int, role: str, fg=D, bg=D, attr=0, c0=0, c1=None):
        c1 = self.cols if c1 is None else c1
        self.put(r, c0, " " * (c1 - c0), role, fg, bg, attr)

    def box(self, r0, c0, r1, c1, style: str, title: str | None = None, fg=D, bg=D):
        tl, tr, bl, br, hz, vt = style
        self.put(r0, c0, tl + hz * (c1 - c0 - 1) + tr, "border", fg, bg)
        self.put(r1, c0, bl + hz * (c1 - c0 - 1) + br, "border", fg, bg)
        for r in range(r0 + 1, r1):
            self.put(r, c0, vt, "border", fg, bg)
            self.put(r, c1, vt, "border", fg, bg)
            if bg != D:
                self.put(r, c0 + 1, " " * (c1 - c0 - 1), "blank", D, bg)
        if title:
            self.put(r0, c0 + 2, f" {title} ", "title", fg, bg, BOLD)

    def labels(self) -> np.ndarray:
        role = self.role.copy()
        blank = (self.ch == " ") & ~np.isin(role, list(FILL_ROLES))
        role[blank] = 0
        return role

    def to_ansi(self, prev: "Canvas | None" = None) -> str:
        out = [] if prev is not None else ["\x1b[0m\x1b[H\x1b[2J"]
        for r in range(self.rows):
            if prev is not None and (prev.ch[r] == self.ch[r]).all() and (prev.fg[r] == self.fg[r]).all() and (
                    prev.bg[r] == self.bg[r]).all() and (prev.attr[r] == self.attr[r]).all():
                continue
            out.append(f"\x1b[{r + 1};1H")
            last = None
            for c in range(self.cols):
                key = (self.fg[r, c], self.bg[r, c], self.attr[r, c])
                if key != last:
                    out.append(_sgr(*key))
                    last = key
                out.append(self.ch[r, c])
            out.append("\x1b[0m")
        if self.cursor[0] >= 0:
            out.append(f"\x1b[{self.cursor[0] + 1};{self.cursor[1] + 1}H\x1b[?25h")
        else:
            out.append("\x1b[?25l")
        return "".join(out)


def _sgr(fg, bg, attr) -> str:
    p = ["0"]
    for bit, code in ((BOLD, "1"), (UNDERLINE, "4"), (REVERSE, "7")):
        if attr & bit:
            p.append(code)
    if fg != D:
        p.append(str(30 + fg if fg < 8 else 90 + fg - 8))
    if bg != D:
        p.append(str(40 + bg if bg < 8 else 100 + bg - 8))
    return f"\x1b[{';'.join(p)}m"


# ---------------------------------------------------------------------------
# helpers
def _w(rng) -> str:
    return rng.choice(WORDS)


def _fname(rng) -> str:
    return f"{_w(rng)}{rng.choice(['', '_', '-'])}{_w(rng) if rng.random() < .4 else ''}.{rng.choice(EXTS)}"


def _prompt(rng) -> tuple[list[tuple[str, int, int]], str]:
    user, host, d = rng.choice(["dev", "root", "ubuntu", "alice", "bob"]), rng.choice(["box", "mbp", "srv1", "ci"]), _w(rng)
    styles = [
        [(f"{user}@{host}", GREEN, BOLD), (":", D, 0), (f"~/{d}", BLUE, BOLD), ("$ ", D, 0)],
        [("❯ ", MAGENTA, BOLD)],
        [("➜  ", GREEN, BOLD), (d, CYAN, BOLD), (" git:(", BLUE, 0), ("main", RED, 0), (") ", BLUE, 0)],
        [("$ ", D, 0)],
        [(f"[{user}@{host} {d}]# ", D, 0)],
        [(f"{d} ", BLUE, BOLD), ("λ ", YELLOW, 0)],
    ]
    return rng.choice(styles), "shell"


def _command(rng) -> str:
    return rng.choice([
        f"ls -la {_w(rng)}/", f"git status", f"cat {_fname(rng)}", f"npm install {_w(rng)}", f"cargo build --release",
        f"docker ps", f"python {_fname(rng)}", f"grep -rn {_w(rng)} .", f"kubectl get pods -n {_w(rng)}", "make test",
        f"pip install {_w(rng)}-{_w(rng)}", f"tail -f /var/log/{_w(rng)}.log", f"cd {_w(rng)}",
    ])


def _put_prompt(cv: Canvas, r: int, parts, cmd: str | None, active=False) -> int:
    c = 0
    for txt, fg, at in parts:
        c = cv.put(r, c, txt, "prompt", fg, D, at)
    if cmd is not None:
        cv.put(r, c, cmd, "input" if active else "text")
        if active:
            cv.put(r, c + len(cmd), " ", "input")
            cv.cursor = (r, min(c + len(cmd), cv.cols - 1))
    return c


def _table_lines(rng, width: int, n: int) -> list[str]:
    ncol = rng.randint(3, 6)
    widths = [rng.randint(5, 14) for _ in range(ncol)]
    hdr = rng.sample(["NAME", "STATUS", "AGE", "SIZE", "PID", "CPU%", "MEM", "USER", "PORTS", "IMAGE", "READY", "TIME"], ncol)
    rows = ["  ".join(h.ljust(w) for h, w in zip(hdr, widths))]
    for _ in range(n):
        cells = []
        for w in widths:
            v = rng.choice([_w(rng), str(rng.randint(0, 99999)), f"{rng.random() * 100:.1f}", f"{rng.randint(1, 59)}m",
                            rng.choice(["Running", "Exited", "Pending", "ok"])])
            cells.append(v[:w].ljust(w))
        rows.append("  ".join(cells))
    return [r[:width] for r in rows]


def _log_line(rng, width: int) -> tuple[str, str, int]:
    lvl = rng.choices(["INFO", "DEBUG", "WARN", "ERROR"], [6, 3, 2, 1])[0]
    ts = f"2025-0{rng.randint(1, 9)}-{rng.randint(10, 28)} {rng.randint(10, 23)}:{rng.randint(10, 59)}:{rng.randint(10, 59)}"
    style = rng.choice(["{ts} [{lvl}] {msg}", "[{ts}] {lvl:5} {msg}", "{lvl}: {ts} {msg}"])
    msg = " ".join(_w(rng) for _ in range(rng.randint(3, 9)))
    line = style.format(ts=ts, lvl=lvl, msg=msg)[:width]
    if lvl == "ERROR":
        return line, "error", RED
    return line, "log", YELLOW if lvl == "WARN" else D


def _bar(rng, width: int, frac: float) -> str:
    style = rng.choice(["hash", "eq", "block", "pipe"])
    n = int(frac * width)
    if style == "hash":
        return "[" + "#" * n + " " * (width - n) + "]"
    if style == "eq":
        return "[" + "=" * max(0, n - 1) + (">" if n else "") + " " * (width - n) + "]"
    if style == "pipe":
        return "[" + "|" * n + " " * (width - n) + "]"
    return "█" * n + "░" * (width - n)


# ---------------------------------------------------------------------------
# scenes: each returns (list[Canvas] animation, app_kind)
def scene_shell(rng, rows, cols):
    cv = Canvas(rows, cols)
    parts, _ = _prompt(rng)
    r = 0
    while r < rows - 3:
        cmd = _command(rng)
        _put_prompt(cv, r, parts, cmd)
        r += 1
        kind = rng.choice(["text", "table", "error", "log", "ls", "none"])
        n = rng.randint(1, 6)
        if kind == "table":
            for line in _table_lines(rng, cols, n)[: rows - r - 2]:
                cv.put(r, 0, line, "table")
                r += 1
        elif kind == "ls":
            names = [_fname(rng) for _ in range(rng.randint(4, 14))]
            colw = max(len(x) for x in names) + 2
            per = max(1, cols // colw)
            for i in range(0, len(names), per):
                if r >= rows - 2:
                    break
                for j, nm in enumerate(names[i:i + per]):
                    cv.put(r, j * colw, nm, "table", BLUE if nm.endswith(("sh",)) else D, D, BOLD if rng.random() < .2 else 0)
                r += 1
        elif kind == "error":
            for line in [f"error: {_w(rng)} {_w(rng)} not found", f"  --> {_fname(rng)}:{rng.randint(1, 300)}:{rng.randint(1, 80)}",
                         "Traceback (most recent call last):", f'  File "{_fname(rng)}", line {rng.randint(1, 500)}, in {_w(rng)}'][:n]:
                if r < rows - 2:
                    cv.put(r, 0, line[:cols], "error", RED if rng.random() < .7 else D)
                    r += 1
        elif kind == "log":
            for _ in range(n):
                if r < rows - 2:
                    line, role, fg = _log_line(rng, cols)
                    cv.put(r, 0, line, role, fg)
                    r += 1
        elif kind == "text":
            for _ in range(n):
                if r < rows - 2:
                    cv.put(r, 0, " ".join(_w(rng) for _ in range(rng.randint(2, 12)))[:cols], "text")
                    r += 1
    final = _command(rng)
    frames = []
    for k in range(0, len(final) + 1, max(1, len(final) // 4)):
        f = cv.copy()
        _put_prompt(f, r, parts, final[:k], active=True)
        frames.append(f)
    return frames, "shell"


def scene_monitor(rng, rows, cols):
    """htop/btop-like: meters, process table with selected row, key hints bar."""
    frames = []
    nproc = rows - 8
    procs = _table_lines(rng, cols - 2, nproc)
    sel = rng.randint(1, max(1, nproc - 2))
    vals = [rng.random() for _ in range(4)]
    for step in range(rng.randint(3, 6)):
        cv = Canvas(rows, cols)
        r = 0
        if rng.random() < .5:
            cv.fill_row(0, "status_bar", BLACK, CYAN)
            cv.put(0, 1, f"{_w(rng)}top - up {rng.randint(1, 99)} days, load average: 0.{rng.randint(10, 99)}", "status_bar", BLACK, CYAN)
            r = 1
        for i, v in enumerate(vals):
            v = min(1.0, max(0.0, v + rng.uniform(-.15, .15)))
            vals[i] = v
            label = f"{i:>2}"
            c = cv.put(r + i, 1, label, "text", CYAN)
            barw = cols // 2 - 12
            n = int(v * barw)
            c = cv.put(r + i, c, "[", "progress", D, D, BOLD)
            c = cv.put(r + i, c, "|" * n, "progress", GREEN if v < .6 else RED)
            c = cv.put(r + i, c, " " * (barw - n), "progress")
            cv.put(r + i, c, f"{v * 100:5.1f}%]", "progress", D, D, BOLD)
        top = r + 5
        cv.fill_row(top, "table", BLACK, GREEN)
        cv.put(top, 1, procs[0], "table", BLACK, GREEN)
        sel = max(1, min(nproc - 1, sel + rng.choice([-1, 0, 1, 1])))
        for i, line in enumerate(procs[1:], start=1):
            rr = top + i
            if rr >= rows - 1:
                break
            if i == sel:
                cv.fill_row(rr, "selected", BLACK, CYAN)
                cv.put(rr, 1, line, "selected", BLACK, CYAN)
            else:
                cv.put(rr, 1, line, "table")
        _key_hints(rng, cv, rows - 1, fkeys=True)
        frames.append(cv)
    return frames, "monitor"


def _key_hints(rng, cv: Canvas, r: int, fkeys=True):
    if fkeys:
        c = 0
        for i, name in enumerate(rng.sample(["Help", "Setup", "Search", "Filter", "Tree", "SortBy", "Nice", "Kill", "Quit"], 6)):
            c = cv.put(r, c, f"F{i + 1}", "key_hint", D, D, 0)
            c = cv.put(r, c, f"{name:<6}", "key_hint", BLACK, CYAN)
        cv.put(r, c, " " * (cv.cols - c), "key_hint", BLACK, CYAN)
    else:
        items = rng.sample(["^G Get Help", "^O Write Out", "^W Where Is", "^K Cut", "^X Exit", "^J Justify", "^R Read File", "^U Paste"], 6)
        for row, chunk in ((r - 1, items[:3]), (r, items[3:])):
            c = 0
            for it in chunk:
                k, name = it.split(" ", 1)
                cv.put(row, c, k, "key_hint", D, D, REVERSE)
                cv.put(row, c + len(k) + 1, name, "key_hint")
                c += 16


def scene_dialog(rng, rows, cols):
    """dialog/whiptail-like menu or gauge on a coloured background."""
    frames = []
    bgc = rng.choice([BLUE, D, CYAN])
    items = [f"{_w(rng).capitalize()} {_w(rng)}" for _ in range(rng.randint(4, min(10, rows - 10)))]
    style = rng.choice(BOX_STYLES)
    gauge = rng.random() < .3
    sel = 0
    w = min(cols - 4, rng.randint(36, 60))
    h = (len(items) + 7) if not gauge else 7
    r0, c0 = max(1, (rows - h) // 2), (cols - w) // 2
    for step in range(rng.randint(3, 5)):
        cv = Canvas(rows, cols)
        if bgc != D:
            for r in range(rows):
                cv.fill_row(r, "blank", D, bgc)
        cv.put(0, 1, f"{_w(rng).capitalize()} Configuration", "title", WHITE if bgc != D else D, bgc, BOLD)
        cv.box(r0, c0, r0 + h - 1, c0 + w - 1, style, title=f"{_w(rng).capitalize()} {_w(rng)}", fg=BLACK, bg=WHITE)
        cv.put(r0 + 1, c0 + 2, "Choose an option:" if not gauge else f"Installing {_w(rng)}...", "text", BLACK, WHITE)
        if gauge:
            frac = min(1.0, (step + 1) / 5 + rng.random() * .1)
            bw = w - 8
            n = int(frac * bw)
            pct = f"{int(frac * 100)}%"
            bar = list(" " * bw)
            for i, ch in enumerate(pct):
                bar[bw // 2 - 1 + i] = ch
            for i, ch in enumerate(bar):
                cv.put(r0 + 3, c0 + 4 + i, ch, "progress", WHITE if i < n else BLUE, BLUE if i < n else WHITE)
        else:
            sel = min(len(items) - 1, sel + (1 if step else 0))
            for i, it in enumerate(items):
                rr = r0 + 3 + i
                if i == sel:
                    cv.put(rr, c0 + 4, f" {i + 1}  {it} ".ljust(w - 8), "selected", WHITE, BLUE)
                else:
                    cv.put(rr, c0 + 4, f" {i + 1}  {it}", "menu_item", BLACK, WHITE)
            br = r0 + h - 2
            cv.put(br, c0 + w // 2 - 12, "<  OK  >", "key_hint", WHITE if step % 2 == 0 else BLACK, RED if step % 2 == 0 else WHITE)
            cv.put(br, c0 + w // 2 + 3, "<Cancel>", "key_hint", BLACK, WHITE)
        frames.append(cv)
    return frames, "installer" if gauge else "menu"


def scene_editor(rng, rows, cols):
    cv = Canvas(rows, cols)
    nano = rng.random() < .5
    body_top = 0
    if nano:
        cv.fill_row(0, "title", BLACK, WHITE, REVERSE)
        cv.put(0, 2, "GNU nano 7.2", "title", BLACK, WHITE, REVERSE)
        cv.put(0, cols // 2 - 5, _fname(rng), "title", BLACK, WHITE, REVERSE)
        body_top = 2
    nlines = rng.randint(4, rows - 6)
    numbers = rng.random() < .6
    kw = ["def", "fn", "let", "const", "return", "if", "for", "import", "class", "struct"]
    for i in range(nlines):
        r = body_top + i
        if r >= rows - 3:
            break
        c = 0
        if numbers:
            c = cv.put(r, 0, f"{i + 1:>4} ", "code", YELLOW)
        indent = " " * (4 * rng.randint(0, 2))
        line = f"{indent}{rng.choice(kw)} {_w(rng)}({_w(rng)}, {_w(rng)}):" if rng.random() < .4 else f"{indent}{_w(rng)} = {_w(rng)}.{_w(rng)}({rng.randint(0, 99)})"
        c2 = cv.put(r, c, line[: cols - c], "code")
        if rng.random() < .3:
            cv.put(r, c + len(indent), line.split()[0], "code", MAGENTA, D, BOLD)
    if not nano:
        for r in range(body_top + nlines, rows - 2):
            cv.put(r, 0, "~", "code", BLUE)
        cv.fill_row(rows - 2, "status_bar", BLACK, WHITE, REVERSE)
        cv.put(rows - 2, 1, f"{_fname(rng)} [+]", "status_bar", BLACK, WHITE, REVERSE)
        cv.put(rows - 2, cols - 18, f"{rng.randint(1, 99)},{rng.randint(1, 40)}  All", "status_bar", BLACK, WHITE, REVERSE)
        cv.put(rows - 1, 0, rng.choice(["-- INSERT --", "", f'"{_fname(rng)}" {rng.randint(5, 300)}L written']), "text", D, D, BOLD)
    else:
        _key_hints(rng, cv, rows - 1, fkeys=False)
    frames = []
    for k in range(3):
        f = cv.copy()
        rr = body_top + rng.randint(0, max(0, min(nlines, rows - 4) - 1))
        f.cursor = (rr, rng.randint(0, cols // 2))
        frames.append(f)
    return frames, "editor"


def scene_installer(rng, rows, cols):
    frames = []
    cv = Canvas(rows, cols)
    parts, _ = _prompt(rng)
    _put_prompt(cv, 0, parts, f"{rng.choice(['pip', 'npm', 'cargo', 'apt'])} install {_w(rng)}")
    r = 1
    pkgs = [f"{_w(rng)}-{_w(rng)}" for _ in range(rng.randint(2, 5))]
    spinner = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    for i, p in enumerate(pkgs):
        if r >= rows - 1:
            break
        cv.put(r, 0, f"Collecting {p}=={rng.randint(0, 9)}.{rng.randint(0, 20)}", "text")
        r += 1
        if r >= rows - 1:
            break
        for step in range(3):
            f = cv.copy()
            frac = (step + 1) / 3
            bw = rng.choice([30, 40])
            c = f.put(r, 2, _bar(rng, bw, frac), "progress", GREEN)
            f.put(r, c + 1, f"{frac * 100:.0f}% {rng.uniform(1, 9):.1f}/{rng.uniform(9, 20):.1f} MB {rng.uniform(1, 9):.1f} MB/s", "progress")
            if r + 1 < rows:
                f.put(r + 1, 0, f"{spinner[step]} Resolving {_w(rng)}", "progress", CYAN)
            frames.append(f)
        cv = frames[-1].copy()
        if r + 1 < rows:
            cv.put(r + 1, 0, " " * cols, "blank")
        r += 1
        if rng.random() < .3 and r < rows - 1:
            cv.put(r, 0, f"WARNING: {_w(rng)} {_w(rng)} is deprecated", "error", YELLOW)
            r += 1
    return frames or [cv], "installer"


def scene_pager(rng, rows, cols):
    frames = []
    lines = [_log_line(rng, cols) for _ in range(rows * 3)]
    off = 0
    for step in range(3):
        cv = Canvas(rows, cols)
        for i in range(rows - 1):
            line, role, fg = lines[off + i]
            cv.put(i, 0, line, role, fg)
        cv.put(rows - 1, 0, rng.choice([":", "(END)", f"lines {off + 1}-{off + rows - 1}"]), "status_bar", D, D, REVERSE)
        cv.cursor = (rows - 1, 1)
        frames.append(cv)
        off += rng.randint(1, rows // 2)
    return frames, "pager"


def scene_dashboard(rng, rows, cols):
    frames = []
    spark = "▁▂▃▄▅▆▇█"
    series = [[rng.random() for _ in range(cols)] for _ in range(2)]
    style = rng.choice(BOX_STYLES)
    for step in range(3):
        cv = Canvas(rows, cols)
        mid, hr = cols // 2, rows // 2
        cv.box(0, 0, hr - 1, mid - 1, style, title=_w(rng).upper(), fg=CYAN)
        cv.box(0, mid, hr - 1, cols - 1, style, title=_w(rng).upper(), fg=CYAN)
        cv.box(hr, 0, rows - 2, cols - 1, style, title="Processes", fg=CYAN)
        for k, c0 in enumerate((1, mid + 1)):
            s = series[k][step:step + mid - 3]
            for rr in range(2, hr - 2):
                pass
            cv.put(hr - 3, c0 + 1, "".join(spark[int(v * 7.99)] for v in s), "progress", GREEN)
            cv.put(2, c0 + 1, f"{_w(rng)}: {rng.randint(0, 9999)} req/s", "text")
        for i, line in enumerate(_table_lines(rng, cols - 4, rows - hr - 4)):
            cv.put(hr + 1 + i, 2, line, "table", D, D, BOLD if i == 0 else 0)
        cv.put(rows - 1, 0, "q quit  ↑↓ select  / filter", "key_hint")
        frames.append(cv)
    return frames, "dashboard"


SCENES = [scene_shell, scene_monitor, scene_dialog, scene_editor, scene_installer, scene_pager, scene_dashboard]


def synth_scene(seed: int):
    rng = random.Random(seed)
    rows, cols = rng.choice([(24, 80), (30, 100), (40, 120), (24, 100), (35, 90), (50, 160)])
    scene = rng.choice(SCENES)
    return scene(rng, rows, cols)


def synth_frames(seed: int) -> list[tuple[Frame, np.ndarray, int]]:
    """Render one synthetic scene through the emulator. Returns [(frame, roles HxW int8, app_id)]."""
    canvases, kind = synth_scene(seed)
    vt = VT(canvases[0].cols, canvases[0].rows)
    out, prev_cv, prev_f = [], None, None
    for i, cv in enumerate(canvases):
        vt.feed(cv.to_ansi(prev_cv))
        f = vt.snapshot(t=float(i), rec_id=f"synth-{seed}", prev=prev_f)
        f.meta["source"] = "synth"
        out.append((f, cv.labels(), APP_ID[kind]))
        prev_cv, prev_f = cv, f
    return out


def synth_cast(seed: int) -> Cast:
    canvases, kind = synth_scene(seed)
    cast = Cast(cols=canvases[0].cols, rows=canvases[0].rows, meta={"id": f"synth-{seed}", "kind": kind})
    prev = None
    for i, cv in enumerate(canvases):
        cast.events.append(Event(float(i), "o", cv.to_ansi(prev)))
        prev = cv
    return cast
