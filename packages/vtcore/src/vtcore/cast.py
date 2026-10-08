"""asciicast v1 / v2 / v3 parsing into a uniform event list.

v1: one JSON object, ``stdout`` = [[delay, data], ...] with *relative* delays.
v2: header line + ``[time, code, data]`` lines with *absolute* times.
v3: header line (``term.cols/rows``) + ``[interval, code, data]`` lines with *relative* intervals.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Event:
    t: float  # absolute seconds since start
    code: str  # "o" output, "i" input, "r" resize, "m" marker, "x" exit
    data: str


@dataclass
class Cast:
    cols: int
    rows: int
    events: list[Event] = field(default_factory=list)
    version: int = 2
    meta: dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return self.events[-1].t if self.events else 0.0

    def output_bytes(self) -> int:
        return sum(len(e.data.encode("utf-8", "replace")) for e in self.events if e.code == "o")


def parse_cast(text: str) -> Cast:
    text = text.lstrip("﻿")
    first_nl = text.find("\n")
    head_txt = text if first_nl < 0 else text[:first_nl]
    try:
        head = json.loads(head_txt)
    except json.JSONDecodeError:
        head = json.loads(text)  # v1 is a single (possibly pretty-printed) JSON document
    if head.get("version") == 1 or "stdout" in head:
        return _parse_v1(head if "stdout" in head else json.loads(text))
    version = head.get("version", 2)
    if version == 3:
        term = head.get("term", {})
        cols, rows = term.get("cols", 80), term.get("rows", 24)
    else:
        cols, rows = head.get("width", 80), head.get("height", 24)
    cast = Cast(cols=int(cols), rows=int(rows), version=version, meta=head)
    t = 0.0
    for line in text[first_nl + 1 :].splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            ts, code, data = json.loads(line)[:3]
        except (json.JSONDecodeError, ValueError):
            continue
        t = t + float(ts) if version == 3 else float(ts)
        cast.events.append(Event(t, str(code), data if isinstance(data, str) else str(data)))
    return cast


def _parse_v1(doc: dict) -> Cast:
    cast = Cast(cols=int(doc.get("width", 80)), rows=int(doc.get("height", 24)), version=1, meta={k: v for k, v in doc.items() if k != "stdout"})
    t = 0.0
    for item in doc.get("stdout", []):
        delay, data = item[0], item[1]
        t += float(delay)
        cast.events.append(Event(t, "o", data))
    return cast


def load_cast(path: str | Path) -> Cast:
    p = Path(path)
    cast = parse_cast(p.read_text(encoding="utf-8", errors="replace"))
    cast.meta.setdefault("id", p.parent.name if p.stem == "record" else p.stem)
    return cast


def find_casts(root: str | Path) -> list[Path]:
    """Dataset layout: recordings/<id>/record.{cast,json}."""
    root = Path(root)
    return sorted([*root.rglob("record.cast"), *root.rglob("record.json")] or [*root.rglob("*.cast")])
