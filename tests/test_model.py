import pytest

torch = pytest.importorskip("torch")

from vtm.data import collate  # noqa: E402
from vtm.model import VTM  # noqa: E402
from vtm.synth import synth_frames  # noqa: E402
from vtcore import N_ROLES, features  # noqa: E402


def test_forward_shapes_and_padding():
    items = []
    for seed in (0, 1):
        f, lab, app = synth_frames(seed)[0]
        d = features(f)
        d.update(roles=lab.astype("int64"), weight=lab.astype("float32") * 0 + 1, app=app)
        items.append(d)
    b = collate(items)
    m = VTM(d=32, layers=1, heads=2)
    logits, app = m(b["char"], b["fg"], b["bg"], b["attr"], b["damage"], b["cursor"], b["mask"])
    assert logits.shape == (*b["char"].shape, N_ROLES) and app.shape[0] == 2
    assert torch.isfinite(logits).all()
