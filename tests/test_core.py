import json

import numpy as np
import pytest

from vtcore import FILL_ROLES, ROLE_ID, VT, keyframes, layout_signature, parse_cast, render_for_llm
from vtm.labeling.heuristics import label
from vtm.labeling.llm import estimate_cost, rasterize
from vtm.synth import synth_cast, synth_frames
from vttui import Reconciler, Surface, compile_frame, msg_bytes


def test_parse_versions():
    v2 = '{"version": 2, "width": 40, "height": 10}\n[0.1, "o", "hi\\r\\n"]\n[0.5, "o", "$ "]\n'
    c = parse_cast(v2)
    assert (c.cols, c.rows, len(c.events)) == (40, 10, 2) and c.events[1].t == 0.5
    v3 = '{"version": 3, "term": {"cols": 50, "rows": 12}}\n[0.1, "o", "a"]\n[0.2, "o", "b"]\n[0.1, "r", "60x20"]\n'
    c = parse_cast(v3)
    assert c.cols == 50 and abs(c.events[1].t - 0.3) < 1e-9 and c.events[2].code == "r"
    v1 = json.dumps({"version": 1, "width": 30, "height": 8, "stdout": [[0.2, "x"], [0.3, "y"]]}, indent=2)
    c = parse_cast(v1)
    assert c.version == 1 and abs(c.events[1].t - 0.5) < 1e-9


def test_emulator_colors_and_cursor():
    vt = VT(20, 4)
    vt.feed("\x1b[1;31mERR\x1b[0m ok\r\n\x1b[7mbar\x1b[0m")
    f = vt.snapshot()
    assert f.row_text(0).startswith("ERR ok") and f.fg[0, 0] == 1 and f.attr[0, 0] & 1
    assert f.attr[1, 0] & 8 and f.cursor == (1, 3)


def test_keyframes_skip_unchanged():
    c = parse_cast('{"version": 2, "width": 20, "height": 4}\n[0.0, "o", "a"]\n[1.0, "o", ""]\n[2.0, "o", "b"]\n')
    fs = list(keyframes(c))
    assert len(fs) == 2 and fs[1].damage.sum() == 1


@pytest.mark.parametrize("seed", range(25))
def test_synth_labels_align(seed):
    for f, lab, app in synth_frames(seed):
        assert f.shape == lab.shape
        # whitespace is blank unless it belongs to a role that owns its whitespace
        assert (lab[(f.cp == 32) & ~np.isin(lab, list(FILL_ROLES))] == 0).all()


def test_heuristics_prompt_and_border():
    vt = VT(40, 6)
    vt.feed("┌──────┐\r\n│ hi   │\r\n└──────┘\r\nuser@box:~/x$ ls -la")
    r, c = label(vt.snapshot())
    assert r[0, 0] == ROLE_ID["border"] and r[3, 0] == ROLE_ID["prompt"] and r[3, 16] == ROLE_ID["input"]


@pytest.mark.parametrize("seed", range(40))
def test_incremental_stream_reconstructs_full_ui(seed):
    rec, surf = Reconciler(), Surface()
    for f, lab, _ in synth_frames(seed):
        msgs = rec.step(f, lab)
        for m in msgs:
            surf.apply(m)
        comps, data = compile_frame(f, rec.roles)
        assert surf.components == comps
        assert surf.data == data
        assert surf.render(False) and surf.render(True)


def test_delta_smaller_than_full():
    inc = full = 0
    for seed in range(30):
        rec = Reconciler()
        for i, (f, lab, _) in enumerate(synth_frames(seed)):
            m = rec.step(f, lab)
            if i:
                inc += msg_bytes(m)
                full += msg_bytes(Reconciler().step(f, lab))
    assert inc < 0.6 * full


def test_style_layer_present():
    f, lab, _ = synth_frames(3)[0]
    comps, data = compile_frame(f, lab)
    bound = [c for c in comps.values() if c.get("style", {}).get("path", "").startswith("/s/")]
    assert bound and all(len(data["s"][c["id"]]["area"]) == 4 for c in bound)
    rec = Reconciler()
    rec.step(f, lab)
    assert "position:relative" in rec.stylesheet() and "data-id" in rec.stylesheet()


def test_growing_log_streams_data_only():
    from vtcore import parse_cast

    lines = "".join(f'[{i}.0, "o", "[2025-01-01 10:00:0{i}] INFO worker {i} ok\\r\\n"]\n' for i in range(6))
    cast = parse_cast('{"version": 2, "width": 60, "height": 12}\n' + lines)
    rec = Reconciler()
    structural = []
    for i, f in enumerate(keyframes(cast)):
        msgs = rec.step(f, label(f)[0])
        structural.append(any("updateComponents" in m for m in msgs))
    assert structural[0] and not any(structural[2:])


def test_rasterize_and_cost():
    f, lab, _ = synth_frames(1)[0]
    h, w = f.shape
    r = rasterize([{"role": "border", "r0": 0, "c0": 0, "r1": h - 1, "c1": w - 1},
                   {"role": "title", "r0": 0, "c0": 2, "r1": 0, "c1": 10}], f.shape, f)
    assert r.shape == f.shape and (r[f.cp == 32] == 0).all()
    est = estimate_cost([f] * 1000)
    assert 0 < est["usd"] < 50
    assert "size" in render_for_llm(f)


def test_layout_signature_ignores_text_changes():
    a, b = VT(20, 3), VT(20, 3)
    a.feed("hello 123")
    b.feed("world 987")
    assert layout_signature(a.snapshot()) == layout_signature(b.snapshot())


def test_synth_cast_roundtrip():
    c = synth_cast(5)
    assert len(list(keyframes(c))) >= 1
