"""Active selection: spend the LLM budget on the frames that teach the model the most."""

from __future__ import annotations

import numpy as np

from vtcore.encode import char_ids
from vtcore.frame import DEFAULT_COLOR, REVERSE, Frame
from vtcore.roles import N_ROLES


def layout_vector(f: Frame, roles: np.ndarray | None = None) -> np.ndarray:
    """Cheap fixed-size descriptor for diversity sampling: ink/box/highlight profiles + role histogram."""
    h, w = f.shape
    cid = char_ids(f.cp)
    ink = (f.cp != 32).astype(np.float32)
    box = ((cid >= 97) & (cid < 225)).astype(np.float32)
    hl = (((f.attr & REVERSE) > 0) | (f.bg != DEFAULT_COLOR)).astype(np.float32)

    def prof(a, axis, n=8):
        p = a.mean(axis)
        idx = np.linspace(0, len(p), n + 1).astype(int)
        return np.array([p[idx[i]:max(idx[i + 1], idx[i] + 1)].mean() for i in range(n)])

    parts = [prof(x, ax) for x in (ink, box, hl) for ax in (0, 1)]
    if roles is not None:
        hist = np.bincount(roles.ravel(), minlength=N_ROLES)[1:].astype(np.float32)
        parts.append(hist / max(hist.sum(), 1))
    parts.append(np.array([h / 60, w / 200]))
    return np.concatenate(parts).astype(np.float32)


def select_frames(
    frames: list[Frame], confidences: list[float], k: int, roles: list[np.ndarray] | None = None, uncertain_frac: float = 0.6,
    seed: int = 0,
) -> list[int]:
    """Pick ``k`` frame indices: ``uncertain_frac`` of the budget by lowest confidence (heuristic or
    model), the rest by k-center greedy over layout vectors for coverage of rare layouts."""
    n = len(frames)
    if k >= n:
        return list(range(n))
    rng = np.random.default_rng(seed)
    order = np.argsort(confidences)
    n_unc = int(k * uncertain_frac)
    chosen = list(order[:n_unc])
    X = np.stack([layout_vector(f, roles[i] if roles else None) for i, f in enumerate(frames)])
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    d = np.full(n, np.inf)
    for i in chosen:
        d = np.minimum(d, ((X - X[i]) ** 2).sum(1))
    if not chosen:
        i = int(rng.integers(n))
        chosen.append(i)
        d = ((X - X[i]) ** 2).sum(1)
    while len(chosen) < k:
        i = int(d.argmax())
        chosen.append(i)
        d = np.minimum(d, ((X - X[i]) ** 2).sum(1))
    return sorted(set(int(i) for i in chosen))


def disagreement(a: np.ndarray, b: np.ndarray) -> tuple[float, list[int]]:
    """Fraction of non-blank cells where two labelings differ, and the rows with most disagreement."""
    m = (a != 0) | (b != 0)
    diff = (a != b) & m
    frac = float(diff.sum() / max(m.sum(), 1))
    rows = [int(r) for r in np.argsort(-diff.sum(1)) if diff[r].any()]
    return frac, rows
