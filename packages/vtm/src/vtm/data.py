"""Corpus building (frames + labels -> .npz shards) and the torch Dataset/collate used for training.

Label sources, by trust:
  synth   exact labels from procedurally generated TUIs          weight 1.0
  llm     LLM region labels (reviewed where flagged)             weight 1.0
  weak    heuristic labeling functions, weighted per cell by their confidence
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable

import numpy as np

from vtcore.encode import features
from vtcore.frame import Frame, load_frames, save_frames
from vtcore.roles import APP_ID

from .labeling.heuristics import label as heuristic_label

SOURCES = {"weak": 0, "synth": 1, "llm": 2}


def build_synth_shard(path: str | Path, seeds: Iterable[int]) -> int:
    from .synth import synth_frames

    frames, roles, conf, apps = [], [], [], []
    for s in seeds:
        for f, lab, app in synth_frames(s):
            f.meta.update(source="synth", app=int(app))
            frames.append(f)
            roles.append(lab.astype(np.int8))
            conf.append(np.ones(lab.shape, np.float16))
    save_frames(path, frames, {"roles": roles, "conf": conf})
    return len(frames)


def build_weak_shard(path: str | Path, frames: list[Frame]) -> int:
    roles, conf = [], []
    for f in frames:
        r, c = heuristic_label(f)
        f.meta.setdefault("source", "weak")
        f.meta.setdefault("app", -1)
        roles.append(r)
        conf.append(c.astype(np.float16))
    save_frames(path, frames, {"roles": roles, "conf": conf})
    return len(frames)


class GridDataset:
    """Loads shards fully into memory (frames are small: a 24x80 frame is ~2k cells)."""

    def __init__(self, shards: list[str | Path], max_rows: int = 64, max_cols: int = 200, weak_weight: float = 0.5,
                 augment: bool = True):
        self.items = []
        for p in shards:
            frames, extra = load_frames(p)
            for f, r, c in zip(frames, extra["roles"], extra["conf"]):
                src = f.meta.get("source", "weak")
                w = c.astype(np.float32) * (weak_weight if src == "weak" else 1.0)
                self.items.append((f, r.astype(np.int64), w, int(f.meta.get("app", -1))))
        self.max_rows, self.max_cols, self.augment = max_rows, max_cols, augment

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        f, roles, w, app = self.items[i]
        feats = features(f)
        h, wd = roles.shape
        # random crop to the size cap (keeps bottom rows more often: status bars / key hints live there)
        r0 = 0 if h <= self.max_rows else (h - self.max_rows if self.augment and random.random() < .5 else
                                           (random.randint(0, h - self.max_rows) if self.augment else 0))
        c0 = 0 if wd <= self.max_cols else (random.randint(0, wd - self.max_cols) if self.augment else 0)
        sl = (slice(r0, r0 + self.max_rows), slice(c0, c0 + self.max_cols))
        out = {k: v[sl] for k, v in feats.items()}
        out["roles"], out["weight"], out["app"] = roles[sl], w[sl], app
        if self.augment and random.random() < 0.15:  # pretend it's the first frame: no damage info
            out["damage"] = np.ones_like(out["damage"])
        return out


def collate(batch):
    import torch

    h = max(b["char"].shape[0] for b in batch)
    w = max(b["char"].shape[1] for b in batch)
    out = {}
    for k in ("char", "fg", "bg", "attr", "damage", "cursor", "roles"):
        pad = -100 if k == "roles" else 0
        arr = np.full((len(batch), h, w), pad, np.int64)
        for i, b in enumerate(batch):
            x = b[k]
            arr[i, : x.shape[0], : x.shape[1]] = x
        out[k] = torch.from_numpy(arr)
    wt = np.zeros((len(batch), h, w), np.float32)
    mask = np.zeros((len(batch), h, w), bool)
    for i, b in enumerate(batch):
        x = b["weight"]
        wt[i, : x.shape[0], : x.shape[1]] = x
        mask[i, : x.shape[0], : x.shape[1]] = True
    out["weight"], out["mask"] = torch.from_numpy(wt), torch.from_numpy(mask)
    out["fg"].clamp_(max=17)
    out["bg"].clamp_(max=17)
    out["app"] = torch.tensor([b["app"] for b in batch])
    return out


def app_id(name: str) -> int:
    return APP_ID.get(name, APP_ID["other"])
