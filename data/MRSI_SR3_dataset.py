import csv
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset


def _to_minus1_1(x: torch.Tensor) -> torch.Tensor:
    return x * 2.0 - 1.0


class MRSISR3Dataset(Dataset):
    """
    Dataset for offline-prepared MRSI SR3 samples (.npz + manifest.csv).

    Expected npz fields:
      - hr:        (1,H,W), float32 in [0,1]
      - lr:        (1,H,W), float32 in [0,1]
      - t1:        (1,H,W), float32 in [0,1]
      - flair:     (1,H,W), float32 in [0,1]
      - met_onehot:(4,H,W), float32 in {0,1}
      - mask:      (1,H,W), float32 in {0,1}
      - met_id / patient_id / slice_idx / lowres: scalar
    """

    def __init__(
        self,
        dataroot,
        split: str = "train",
        data_len: int = -1,
        hflip: bool = True,
        vflip: bool = False,
    ):
        self.split = split
        self.hflip = bool(hflip)
        self.vflip = bool(vflip)

        root = Path(dataroot)
        if (root / "manifest.csv").exists():
            self.split_root = root
        elif (root / split / "manifest.csv").exists():
            self.split_root = root / split
        else:
            raise FileNotFoundError(
                f"manifest.csv not found under {root} or {root / split}"
            )

        manifest = self.split_root / "manifest.csv"
        records: List[Dict[str, str]] = []
        with manifest.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append(row)
        if data_len is not None and int(data_len) > 0:
            records = records[: int(data_len)]
        self.records = records

    def __len__(self):
        return len(self.records)

    def _maybe_flip(self, tensors: List[torch.Tensor]) -> List[torch.Tensor]:
        if self.split == "train" and self.hflip and random.random() < 0.5:
            tensors = [torch.flip(t, dims=[-1]) for t in tensors]
        if self.split == "train" and self.vflip and random.random() < 0.5:
            tensors = [torch.flip(t, dims=[-2]) for t in tensors]
        return tensors

    def __getitem__(self, index):
        rec = self.records[index]
        npz_path = self.split_root / rec["npz"]
        data = np.load(str(npz_path))

        hr = torch.from_numpy(data["hr"]).float()
        lr = torch.from_numpy(data["lr"]).float()
        t1 = torch.from_numpy(data["t1"]).float()
        flair = torch.from_numpy(data["flair"]).float()
        met_onehot = torch.from_numpy(data["met_onehot"]).float()
        mask = torch.from_numpy(data["mask"]).float()

        hr, lr, t1, flair, met_onehot, mask = self._maybe_flip([hr, lr, t1, flair, met_onehot, mask])

        # Conditional tensor: LR + T1 + FLAIR + met onehot => 7 channels.
        sr_cond = torch.cat([lr, t1, flair, met_onehot], dim=0)

        out = {
            "HR": _to_minus1_1(hr),
            "SR": _to_minus1_1(sr_cond),
            "LR": _to_minus1_1(lr),
            "MASK": mask,  # keep in 0~1 for frequency-domain metrics
            "Index": index,
            "MET_ID": int(data["met_id"]),
            "PATIENT_ID": int(data["patient_id"]),
            "SLICE_IDX": int(data["slice_idx"]),
            "LOWRES": int(data["lowres"]),
        }
        return out


