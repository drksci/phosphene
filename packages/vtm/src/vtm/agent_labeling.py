"""Agent labeling inside a Colab kernel (no API key).

Claude Code subagents drive the notebook through colab-mcp and call these helpers in short cells
(each well under the 30 s MCP limit). Frame text goes straight into each subagent's context, never
through the orchestrator's:

    s = Session(DATA, frames_by_id)          # once, in the kernel
    s.show_guide(); s.show_tasks(k)          # labeler reads chunk k
    s.save_labels(k, jsonl)                  # validates coverage/bounds, writes label_KK.jsonl
    s.route_review(n_review, n_gold)         # disagreement vs heuristics + random gold set
    s.show_review(k); s.save_review(k, jsonl)
    s.bundle("round1")                       # zip of labels + frames for the repo's labels/
"""

from __future__ import annotations

import glob
import json
import os
import random
import shutil
import zipfile

import numpy as np

from vtcore import ROLES, save_frames
from vtcore.frame import Frame

from . import labeling
from .labeling import llm
from .labeling.select import disagreement

LETTERS = dict(zip(ROLES, ".tpibTSmsxPcleK"))


class Session:
    def __init__(self, data_dir: str, frames: dict[str, Frame], heuristic: dict[str, np.ndarray] | None = None,
                 chunk: int = 25, review_chunk: int = 10):
        self.dir = f"{data_dir}/agent_labels"
        os.makedirs(self.dir, exist_ok=True)
        self.frames = frames
        self.heur = heuristic or {cid: labeling.label(f)[0] for cid, f in frames.items()}
        ids = list(frames)
        self.chunks = [ids[i:i + chunk] for i in range(0, len(ids), chunk)]
        self.review_chunk = review_chunk
        self.rchunks: list[list[str]] = []

    # -- labeling ---------------------------------------------------------
    def show_guide(self):
        print(llm.guide_markdown())

    def show_tasks(self, k: int):
        for cid in self.chunks[k]:
            print(f"===== id={cid} =====\n{llm.render_for_llm(self.frames[cid])}\n")

    def _store(self, kind: str, k: int, jsonl: str, want: set[str]) -> dict[str, dict]:
        tmp = f"{self.dir}/.tmp"
        with open(tmp, "w") as fh:
            fh.write(jsonl)
        got = {c: v for c, v in llm.import_labels(tmp).items() if c in want}
        with open(f"{self.dir}/{kind}_{k:02d}.jsonl", "w") as fh:
            for cid, lab in got.items():
                fh.write(json.dumps({"id": cid, **lab}) + "\n")
        return got

    def save_labels(self, k: int, jsonl: str):
        want = set(self.chunks[k])
        got = self._store("label", k, jsonl, want)
        issues = []
        for cid, lab in got.items():
            f = self.frames[cid]
            h, w = f.shape
            oob = sum(1 for r in lab["regions"] if r["r0"] > r["r1"] or r["c0"] > r["c1"] or r["r1"] >= h or r["c1"] >= w)
            explicit = llm.rasterize(lab["regions"], f.shape)
            ink = f.cp != 32
            cov = float((explicit[ink] != 0).mean()) if ink.any() else 1.0
            if cov < 0.95 or oob:
                issues.append(f"{cid}: ink covered {cov:.0%}, out-of-bounds regions {oob}")
        print(f"chunk {k}: saved {len(got)}/{len(want)}; missing {sorted(want - set(got))}")
        print("\n".join(issues) or "all frames ≥95% ink covered, no out-of-bounds regions")

    # -- review -----------------------------------------------------------
    def load(self, kind: str) -> dict[str, dict]:
        out = {}
        for fn in sorted(glob.glob(f"{self.dir}/{kind}_*.jsonl")):
            for line in open(fn):
                d = json.loads(line)
                out[d["id"]] = d
        return out

    def raster(self, cid: str, lab: dict) -> np.ndarray:
        f = self.frames[cid]
        return llm.rasterize(lab["regions"], f.shape, f)

    def route_review(self, n_review: int, n_gold: int, seed: int = 0) -> dict:
        labels = self.load("label")
        self._dis = {c: disagreement(self.raster(c, labels[c]), self.heur[c]) for c in labels}
        order = sorted(self._dis, key=lambda c: -self._dis[c][0])
        review = order[:n_review]
        gold = random.Random(seed).sample(order[n_review:], min(n_gold, len(order) - n_review))
        allr = review + gold
        self.rchunks = [allr[i:i + self.review_chunk] for i in range(0, len(allr), self.review_chunk)]
        split = {"review": review, "gold": gold}
        json.dump(split, open(f"{self.dir}/review_split.json", "w"))
        print(f"{len(labels)} labeled · median disagreement {np.median([d for d, _ in self._dis.values()]):.2f} · "
              f"{len(self.rchunks)} review chunks")
        return split

    def show_review(self, k: int):
        labels = self.load("label")
        legend = " ".join(f"{v}={r}" for r, v in LETTERS.items())
        for cid in self.rchunks[k]:
            f, lab = self.frames[cid], labels[cid]
            r = self.raster(cid, lab)
            print(f"===== id={cid} =====\n{llm.render_for_llm(f)}\n--- proposed labeling (app={lab['app']}):\n"
                  f"{json.dumps(lab['regions'], separators=(',', ':'))}\n"
                  f"--- rows where heuristics disagree most: {self._dis[cid][1][:12]}\n--- proposal as role letters ({legend}):")
            for i in range(f.shape[0]):
                if r[i].any():
                    print(f"{i:02d}|{''.join(LETTERS[ROLES[x]] for x in r[i]).rstrip('.')}")
            print()

    def save_review(self, k: int, jsonl: str):
        labels = self.load("label")
        want = set(self.rchunks[k])
        got = self._store("review", k, jsonl, want)
        changed = sum(not np.array_equal(self.raster(c, got[c]), self.raster(c, labels[c])) for c in got)
        print(f"review chunk {k}: saved {len(got)}/{len(want)}, missing {sorted(want - set(got))}, changed vs proposal: {changed}")

    # -- persistence ------------------------------------------------------
    def bundle(self, name: str, extra_files: list[str] = ()) -> str:
        """Zip labels + the labeled frames; download it and unzip into the repo's labels/ folder."""
        out = f"/content/bundle_{name}"
        shutil.rmtree(out, ignore_errors=True)
        os.makedirs(out)
        ids = list(self.frames)
        save_frames(f"{out}/frames.npz", [self.frames[c] for c in ids], {"heuristic_roles": [self.heur[c] for c in ids]})
        json.dump({"ids": ids}, open(f"{out}/ids.json", "w"))
        for fn in glob.glob(f"{self.dir}/*.json*") + list(extra_files):
            shutil.copy(fn, out)
        zp = f"/content/labels_{name}.zip"
        with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
            for fn in sorted(os.listdir(out)):
                z.write(f"{out}/{fn}", f"{name}/{fn}")
        return zp
