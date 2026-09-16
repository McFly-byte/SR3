"""Runtime re-degradation dataset for water-film SR3 fine-tuning.

It wraps the existing offline MRSISR3Dataset but *re-generates* the LR
condition from the stored HR at every call, so the retained k-space block can
cover the real acquisitions (8/12 px on the 64 grid) that the original
16..32-px range never saw.  It also applies structural-condition dropout so the
model learns to ignore T1/FLAIR when they are unreliable (water film has no
brain).

Held-out rule: this dataset only reads the *train* split of the existing
simulated dataset.  The two evaluation .mat files are never opened here.
"""
from __future__ import annotations

import random
from typing import Dict

import numpy as np
import torch

from data.MRSI_SR3_dataset import MRSISR3Dataset
from core.improved_degradation import (
    degrade_hr_to_lr_condition,
    sample_kspace_block,
)


class ImprovedWaterfilmDataset(MRSISR3Dataset):
    def __init__(
        self,
        *args,
        min_block: int = 8,
        max_block: int = 32,
        struct_dropout_prob: float = 0.1,
        window: str = "hamming",
        seed: int = 0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.min_block = int(min_block)
        self.max_block = int(max_block)
        self.struct_dropout_prob = float(struct_dropout_prob)
        self.window = window
        self._rng = random.Random(seed)
        # locate channel offsets inside the concatenated condition tensor
        self._offsets = self._layout_offsets()

    def _layout_offsets(self) -> Dict[str, tuple[int, int]]:
        offs: Dict[str, tuple[int, int]] = {}
        c = 0
        for name in self.condition_layout:
            if name == "met_onehot":
                width = 4
            else:
                width = 1
            offs[name] = (c, c + width)
            c += width
        return offs

    def __getitem__(self, index):
        out = super().__getitem__(index)
        hr_m11 = out["HR"]                    # (1,H,W) in [-1,1]
        hr_01 = ((hr_m11 + 1.0) * 0.5).clamp(0.0, 1.0)
        block = sample_kspace_block(
            np.random.RandomState(self._rng.randint(0, 2 ** 31 - 1)),
            self.min_block, self.max_block,
        )
        lr_01 = degrade_hr_to_lr_condition(
            hr_01.unsqueeze(0), block, window=self.window
        ).squeeze(0)                          # (1,H,W) in [0,1]
        lr_m11 = lr_01 * 2.0 - 1.0

        sr = out["SR"].clone()
        a0, a1 = self._offsets["lr"]
        sr[a0:a1] = lr_m11

        # structural dropout: zero the T1/FLAIR channels
        if (
            self.split == "train"
            and self.struct_dropout_prob > 0
            and self._rng.random() < self.struct_dropout_prob
        ):
            for k in ("t1", "flair"):
                if k in self._offsets:
                    b0, b1 = self._offsets[k]
                    sr[b0:b1] = -1.0   # constant -1 == (0 in [0,1])

        out["SR"] = sr
        out["LR"] = lr_m11
        # Native-resolution LR center-padded to the 64 canvas, in [0,1] -- this
        # is the contract expected by native_acquisition_l1_sum (pred in [-1,1],
        # lr_native in [0,1]).  HR/SR/LR stay in [-1,1].
        out["LR_NATIVE"] = lr_01.detach().cpu()
        out["HAS_NATIVE_LR"] = torch.tensor(1.0, dtype=torch.float32)
        out["LR_MATRIX"] = torch.tensor(block, dtype=torch.int64)
        out["BLOCK"] = torch.tensor(block, dtype=torch.int64)
        return out
