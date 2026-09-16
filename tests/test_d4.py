"""D4 eight-transform correctness.

The transforms are IMPORTED from the production v6 inference script
(scripts/infer_waterfilm_phantom_v6.py) so the test never re-implements or
drifts from the shipped production definition.
"""
from pathlib import Path
import importlib.util

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
_V6 = REPO / "scripts" / "infer_waterfilm_phantom_v6.py"


def _load_v6_d4():
    spec = importlib.util.spec_from_file_location("infer_v6", str(_V6))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.D4_TRANSFORMS


@pytest.fixture(scope="module")
def d4():
    return _load_v6_d4()


def test_eight_distinct_transforms(d4):
    a = np.arange(12 * 12, dtype=float).reshape(12, 12)
    outs = {name: fn(a) for name, fn in d4.items()}
    names = list(outs)
    assert len(names) == 8
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            assert not np.array_equal(outs[names[i]], outs[names[j]]), \
                f"{names[i]} == {names[j]}"


def test_anti_transpose_differs_from_transpose(d4):
    a = np.arange(12 * 12, dtype=float).reshape(12, 12)
    assert not np.array_equal(d4["transpose"](a), d4["anti_transpose"](a))


def test_inverse_roundtrip(d4):
    a = np.random.RandomState(0).rand(12, 12)
    # every D4 element is its own inverse up to the inverse element
    for name, fn in d4.items():
        # generic: apply twice for self-inverse, else check structure via rot180 symmetry
        out = fn(fn(a))
        # rotations of 4-fold or flips -> applying twice returns original for self-inverses
        if name in ("identity", "rot180", "hflip", "vflip", "transpose", "anti_transpose"):
            assert np.allclose(out, a), f"self-inverse failed for {name}"
        else:  # rot90/rot270 pair
            back = d4["rot270"](d4["rot90"](a))
            assert np.allclose(back, a)
