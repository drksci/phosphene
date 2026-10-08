"""Pre-generate the web demo: find recordings of popular TUI apps, export traces, benchmark the gateway.

    python -m vtm.demos MODEL.pt RECORDINGS_DIR OUT_DIR [N_BENCH]

1. Finds recordings by app signature in the raw output stream (no emulation needed): htop, top,
   vim, nano, mc, ncdu, tig, emacs, dialog/whiptail, less, tmux.
2. For each app, picks the recording with a usable size and length and the most settled screens,
   and exports a trace (vtm.trace) with the trained model as matcher.
3. Benchmarks the gateway on N_BENCH random recordings (locked share, model calls, bytes).
Writes OUT_DIR/index.json, OUT_DIR/<app>.json and OUT_DIR/bench.json.
"""

from __future__ import annotations

import glob
import json
import os
import random
import re
import sys
import time

from vtcore import keyframes, load_cast
from vttui import Gateway, Reconciler, TemplateCache, msg_bytes

from .trace import export_trace
from .train import load, predict

APPS = {  # label: (description, regexes that must all appear in the output stream)
    "htop": ("process monitor", [r"Load average", r"Tasks:", r"F10\s*Quit|F10Quit|Quit"]),
    "vim": ("editor", [r"-- INSERT --|:wq|:q!", r"\x1b\[\d+;\d+H~"]),
    "mc": ("Midnight Commander file manager", [r"Left\s+File\s+Command\s+Options\s+Right"]),
    "ncdu": ("disk usage browser", [r"ncdu \d", r"Total disk usage"]),
    "nano": ("editor", [r"GNU nano", r"\^X|Exit"]),
    "top": ("process monitor", [r"top - \d\d:\d\d", r"PID\s+USER"]),
    "dialog": ("dialog / whiptail menus", [r"<\s*OK\s*>|< Yes >|<Cancel>"]),
    "tig": ("git browser", [r"\[main\]|\[log\]|\[status\]", r"tig"]),
    "emacs": ("editor", [r"-UU[U-]?:|-\*\*-|Fundamental|\(Lisp Interaction\)"]),
    "less": ("pager", [r"\(END\)", r"lines \d+-\d+"]),
    "tmux": ("terminal multiplexer", [r"\x1b\[\d+;1H\x1b\[[\d;]*m\[\d+\] \d+:"]),
}


def find_demos(paths: list[str], per_app: int = 3) -> dict[str, list[tuple[str, int]]]:
    hits: dict[str, list[tuple[str, int]]] = {a: [] for a in APPS}
    for p in paths:
        try:
            text = open(p, encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        if not 2_000 < len(text) < 600_000:
            continue
        for app, (_, rxs) in APPS.items():
            if len(hits[app]) < 40 and all(re.search(r, text) for r in rxs):
                hits[app].append((p, len(text)))
    return hits


def pick(cands: list[tuple[str, int]]) -> str | None:
    """Prefer 80-140 col terminals, 20-50 rows, with many settled screens."""
    best, score = None, -1.0
    for p, _ in cands[:25]:
        try:
            c = load_cast(p)
        except Exception:
            continue
        if not (70 <= c.cols <= 160 and 18 <= c.rows <= 55) or not c.events or c.events[-1].t > 900:
            continue
        n = sum(1 for _ in keyframes(c, max_frames=200))
        s = min(n, 160) - abs(c.cols - 100) / 20
        if n >= 25 and s > score:
            best, score = p, s
    return best


def bench(paths: list[str], seg, n: int) -> dict:
    cache, rows = TemplateCache(), []
    for p in paths[:n]:
        try:
            cast = load_cast(p)
        except Exception:
            continue
        if cast.cols * cast.rows > 80 * 250 or not cast.events or cast.events[-1].t > 900:
            continue
        gw, vt, full = Gateway(seg, cache=cache), 0, 0
        evs = iter(e for e in cast.events if e.code == "o")
        pend = next(evs, None)
        for f in keyframes(cast, max_frames=150):
            while pend is not None and pend.t <= f.t:
                vt += len(pend.data.encode("utf-8", "replace"))
                pend = next(evs, None)
            gw.step(f)
            full += msg_bytes(Reconciler().step(f, gw.rec.roles))
        s = gw.stats.summary()
        s.update(path=p, vt_bytes=vt, full_bytes=full)
        rows.append(s)
    keys = ("keyframes", "locked", "partial", "raw", "model_calls", "cache_hits", "drift", "structural_msgs", "a2ui_bytes",
            "vt_bytes", "full_bytes")
    tot = {k: sum(r[k] for r in rows) for k in keys}
    tot["recordings"] = len(rows)
    return {"total": tot, "rows": rows}


def main(model_path: str, rec_dir: str, out: str, n_bench: int = 200):
    t0 = time.time()
    os.makedirs(out, exist_ok=True)
    model = load(model_path, device="cuda")
    seg = lambda f: predict(model, f)[0]  # noqa: E731
    paths = sorted(glob.glob(f"{rec_dir}/**/record.*", recursive=True))
    hits = find_demos(paths)
    print({a: len(h) for a, h in hits.items()}, flush=True)
    index = []
    for app, cands in hits.items():
        p = pick(cands)
        if p is None:
            continue
        tr = export_trace(load_cast(p), seg, title=app, max_frames=160)
        json.dump(tr, open(f"{out}/{app}.json", "w"), separators=(",", ":"))
        index.append({"label": app, "title": f"{app} · {APPS[app][0]}", "file": f"{app}.json", "source": os.path.basename(os.path.dirname(p)),
                      "stats": tr["stats"], "cols": tr["cols"], "rows": tr["rows"]})
        print(app, p, tr["cols"], tr["rows"], len(tr["frames"]), "keyframes", tr["stats"], f"{time.time() - t0:.0f}s", flush=True)
    json.dump({"recordings": index}, open(f"{out}/index.json", "w"), indent=1)
    random.Random(7).shuffle(paths)
    b = bench(paths, seg, n_bench)
    json.dump(b, open(f"{out}/bench.json", "w"))
    print("BENCH", json.dumps(b["total"]), f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 200)
