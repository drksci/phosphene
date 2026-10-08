"""Fetch a subset of James4Ever0/asciinema_terminal_recordings (one 1 GB solid .7z, 79k casts,
18 GB uncompressed) without extracting everything.

Recordings early in the archive extract in seconds; ``spread=True`` samples across the whole
archive (decompresses the full stream once, ~5-10 min on Colab, but still writes only the subset).
"""

from __future__ import annotations

import random
from pathlib import Path

REPO = "James4Ever0/asciinema_terminal_recordings"


def fetch_recordings(dest: str | Path, n: int = 4000, min_bytes: int = 2_000, max_bytes: int = 400_000,
                     spread: bool = False, seed: int = 0, archive: str | None = None) -> list[Path]:
    import py7zr
    from huggingface_hub import hf_hub_download

    dest = Path(dest)
    archive = archive or hf_hub_download(REPO, "recordings.7z", repo_type="dataset")
    with py7zr.SevenZipFile(archive) as z:
        infos = [i for i in z.list() if i.filename.endswith(("record.cast", "record.json")) and min_bytes <= i.uncompressed <= max_bytes]
    if spread:
        random.Random(seed).shuffle(infos)
        picked = infos[:n]
    else:
        picked = infos[:n]
    with py7zr.SevenZipFile(archive) as z:
        z.extract(path=dest, targets=[i.filename for i in picked])
    return sorted(dest.rglob("record.*"))
