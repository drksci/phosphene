import numpy as np
import pytest

from vtcore import VT, ROLE_ID
from vtm.synth import synth_frames
from vttui import RAW, Gateway, Surface, TemplateCache, compile_frame, edit_bytes, key_bytes, select_bytes


def gold_segmenter(seed):
    table = {f.content_hash(): lab for f, lab, _ in synth_frames(seed)}
    return lambda f: table[f.content_hash()]


def run(seed, cache=None, lock_after=2):
    gw = Gateway(gold_segmenter(seed), cache=cache, lock_after=lock_after)
    surf = Surface()
    kinds = []
    for f, _, _ in synth_frames(seed):
        for m in gw.step(f):
            surf.apply(m)
        comps, data = compile_frame(f, gw.rec.roles)
        assert surf.reachable() == comps and surf.data == data  # client state == server truth
        kinds.append(gw.state)
        surf.render(True)
    return gw, surf, kinds


@pytest.mark.parametrize("seed", range(30))
def test_gateway_stream_is_consistent(seed):
    gw, _, _ = run(seed)
    st = gw.stats
    assert st.keyframes == st.locked + st.partial + st.raw
    assert st.model_calls <= st.keyframes


def test_first_frame_is_raw_terminal():
    gw = Gateway(gold_segmenter(0))
    f = synth_frames(0)[0][0]
    msgs = gw.step(f)
    comps = [c for m in msgs if "updateComponents" in m for c in m["updateComponents"]["components"]]
    assert any(c["component"] == "Terminal" for c in comps)
    assert gw.state == "matching" and gw.stats.raw == 1


def test_locking_saves_model_calls_and_cache_carries_over():
    cache = TemplateCache()
    calls_first = calls_second = frames = 0
    for seed in range(40):
        gw1, _, _ = run(seed, cache=cache)
        gw2, _, _ = run(seed, cache=cache)
        calls_first += gw1.stats.model_calls
        calls_second += gw2.stats.model_calls
        frames += gw1.stats.keyframes
    assert calls_first < frames  # locked frames skip the matcher
    assert calls_second < calls_first  # templates learned in session 1 are reused in session 2


def test_drift_falls_back_to_raw_then_relocks():
    f0, lab0, _ = synth_frames(2)[0]
    gw = Gateway(lambda f: lab0 if f.shape == f0.shape else None, lock_after=1)
    gw.step(f0)
    assert gw.state == "locked"
    vt = VT(f0.shape[1], f0.shape[0])
    # same screen plus ink where the template has no slot
    h, w = f0.shape
    free = np.argwhere(gw.template.slots == 0)
    r, c = free[len(free) // 2]
    f1 = f0.__class__(f0.cp.copy(), f0.fg, f0.bg, f0.attr, f0.cursor, 1.0, damage=np.zeros(f0.shape, bool))
    f1.cp[r, c] = ord("Z")
    f1.damage[r, c] = True
    gw.step(f1)
    assert gw.stats.drift == 1 and gw.state in ("matching", "locked")
    assert (gw.rec.roles == RAW).any()


def test_keys():
    assert key_bytes("^X") == "\x18" and key_bytes("^G") == "\x07"
    assert key_bytes("F1") == "\x1bOP" and key_bytes("F10") == "\x1b[21~"
    assert key_bytes("M-x") == "\x1bx" and key_bytes("<Enter>") == "\r" and key_bytes("[ OK ]") == "\r"
    assert key_bytes("q") == "q" and key_bytes("Esc") == "\x1b"
    assert edit_bytes("ls -l", "ls -la") == "a"
    assert edit_bytes("git sta", "git log", submit=True) == "\x7f\x7f\x7flog\r"
    assert select_bytes(2, 5) == "\x1b[B" * 3 and select_bytes(4, 1, activate=True) == "\x1b[A" * 3 + "\r"


def test_select_action_uses_current_selection():
    for seed in range(200):
        frames = synth_frames(seed)
        f, lab, _ = frames[0]
        if (lab == ROLE_ID["selected"]).any():
            break
    gw = Gateway(lambda x: lab, lock_after=1)
    gw.step(f)
    lists = {k: v for k, v in gw.rec.data["r"].items() if any(r.get("selected") for r in v.get("rows", []))}
    assert lists
    k, v = next(iter(lists.items()))
    cur = next(i for i, r in enumerate(v["rows"]) if r["selected"])
    tgt = (cur + 1) % len(v["rows"])
    keys = gw.action({"name": "select", "context": {"list": k, "row": tgt}})
    assert keys == select_bytes(cur, tgt)
