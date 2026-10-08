"""Corpus build as a background job (python -m vtm.prep DATA_DIR N_RECORDINGS).

Synthetic shards, a size-filtered subset of the asciinema archive, keyframes deduplicated across the
whole corpus (per-recording work in a process pool, global dedupe in the parent), weak-label shards.
Resumable: finished steps are skipped. Logs progress lines suitable for `tail`.
"""

from __future__ import annotations

import glob
import os
import sys
import time
from multiprocessing import Pool

PER_REC, SHARD = 12, 4000


def _work(p):
    from vtcore import iter_corpus

    try:
        return list(iter_corpus([p], per_recording=PER_REC))
    except Exception as e:  # one bad recording must not kill the run
        print("skip", p, type(e).__name__, e, flush=True)
        return []


def main(data: str, n_rec: int, workers: int = os.cpu_count() or 2, n_synth: int = 3000, n_synth_val: int = 300):
    from .corpus import fetch_recordings
    from .data import build_synth_shard, build_weak_shard

    t0 = time.time()

    def log(*a):
        print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)

    os.makedirs(data, exist_ok=True)
    if not os.path.exists(f"{data}/synth_val.npz"):
        log("synth train", build_synth_shard(f"{data}/synth_train.npz", range(n_synth)))
        log("synth val", build_synth_shard(f"{data}/synth_val.npz", range(10**6, 10**6 + n_synth_val)))
    rec_dir = f"{os.path.dirname(data.rstrip('/'))}/recordings"
    paths = sorted(glob.glob(f"{rec_dir}/**/record.*", recursive=True)) or fetch_recordings(rec_dir, n=n_rec)
    log("recordings", len(paths))
    if glob.glob(f"{data}/weak_*.npz"):
        log("weak shards exist, skipping")
        return
    seen, buf, n_shard = set(), [], 0
    with Pool(workers) as pool:
        for n, frames in enumerate(pool.imap_unordered(_work, paths, chunksize=8), 1):
            for f in frames:
                if f.meta["sig"] not in seen:
                    seen.add(f.meta["sig"])
                    buf.append(f)
            if len(buf) >= SHARD:
                build_weak_shard(f"{data}/weak_{n_shard:03d}.npz", buf[:SHARD])
                buf, n_shard = buf[SHARD:], n_shard + 1
            if n % 200 == 0:
                log(f"{n}/{len(paths)} recordings, {len(seen)} unique frames")
    if buf:
        build_weak_shard(f"{data}/weak_{n_shard:03d}.npz", buf)
    log("done", len(seen), "unique frames")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
