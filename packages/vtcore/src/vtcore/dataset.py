"""Turn a directory of recordings into a deduplicated keyframe corpus."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable, Iterator

from .cast import find_casts, load_cast
from .emulator import keyframes
from .encode import layout_signature
from .frame import Frame


def iter_corpus(
    paths: Iterable[str | Path],
    per_recording: int = 12,
    max_cells: int = 80 * 250,
    max_output_bytes: int = 5_000_000,
    seen: set[str] | None = None,
    seed: int = 0,
) -> Iterator[Frame]:
    """Keyframes from many recordings, deduplicated by layout signature across the whole corpus.

    ``per_recording`` frames are kept per recording (evenly spread over its keyframes), which keeps
    long sessions from dominating. Oversized terminals / huge recordings are skipped."""
    seen = set() if seen is None else seen
    rng = random.Random(seed)
    for p in paths:
        try:
            cast = load_cast(p)
        except Exception:
            continue
        if cast.cols * cast.rows > max_cells or cast.cols < 20 or cast.rows < 5 or cast.output_bytes() > max_output_bytes:
            continue
        uniq: list[Frame] = []
        for f in keyframes(cast):
            if f.shape[0] * f.shape[1] > max_cells or not (f.cp != 32).any():
                continue
            sig = layout_signature(f)
            if sig in seen:
                continue
            seen.add(sig)
            f.meta["sig"] = sig
            uniq.append(f)
        if len(uniq) > per_recording:
            idx = sorted(rng.sample(range(len(uniq)), per_recording))
            uniq = [uniq[i] for i in idx]
        yield from uniq


def sample_recordings(root: str | Path, n: int, seed: int = 0) -> list[Path]:
    paths = find_casts(root)
    random.Random(seed).shuffle(paths)
    return paths[:n]
