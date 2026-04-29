import csv
import random
from pathlib import Path
from typing import Dict, List, Tuple

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
        use_lr: bool = True,
        use_t1: bool = True,
        use_flair: bool = True,
        use_met_onehot: bool = True,
        use_mask_channel: bool = False,
        strict_check: bool = False,
    ):
        self.split = split
        self.hflip = bool(hflip)
        self.vflip = bool(vflip)
        self.use_lr = bool(use_lr)
        self.use_t1 = bool(use_t1)
        self.use_flair = bool(use_flair)
        self.use_met_onehot = bool(use_met_onehot)
        self.use_mask_channel = bool(use_mask_channel)
        self.strict_check = bool(strict_check)
        self.condition_layout = self._build_condition_layout()

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

    def _build_condition_layout(self) -> List[str]:
        layout = []
        if self.use_lr:
            layout.append("lr")
        if self.use_t1:
            layout.append("t1")
        if self.use_flair:
            layout.append("flair")
        if self.use_met_onehot:
            layout.extend([f"met_onehot_{i}" for i in range(4)])
        if self.use_mask_channel:
            layout.append("mask")
        if not layout:
            raise ValueError("At least one MRSI condition channel must be enabled.")
        return layout

    def __len__(self):
        return len(self.records)

    def _maybe_flip(self, tensors: List[torch.Tensor]) -> List[torch.Tensor]:
        if self.split == "train" and self.hflip and random.random() < 0.5:
            tensors = [torch.flip(t, dims=[-1]) for t in tensors]
        if self.split == "train" and self.vflip and random.random() < 0.5:
            tensors = [torch.flip(t, dims=[-2]) for t in tensors]
        return tensors

    @staticmethod
    def _check_range(name: str, arr: np.ndarray, min_v: float = 0.0, max_v: float = 1.0, tol: float = 1e-4) -> None:
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} contains non-finite values.")
        arr_min = float(np.min(arr))
        arr_max = float(np.max(arr))
        if arr_min < min_v - tol or arr_max > max_v + tol:
            raise ValueError(f"{name} range [{arr_min:.6g}, {arr_max:.6g}] is outside [{min_v}, {max_v}].")

    def _validate_npz(self, data, npz_path: Path) -> None:
        required = ["hr", "lr", "t1", "flair", "met_onehot", "mask", "met_id", "patient_id", "slice_idx", "lowres"]
        missing = [key for key in required if key not in data.files]
        if missing:
            raise KeyError(f"{npz_path} is missing required fields: {missing}")

        shapes: Dict[str, Tuple[int, ...]] = {key: tuple(data[key].shape) for key in ["hr", "lr", "t1", "flair", "mask", "met_onehot"]}
        ref_shape = shapes["hr"]
        if len(ref_shape) != 3 or ref_shape[0] != 1:
            raise ValueError(f"{npz_path}: hr must have shape (1,H,W), got {ref_shape}")
        for key in ["lr", "t1", "flair", "mask"]:
            if shapes[key] != ref_shape:
                raise ValueError(f"{npz_path}: {key} shape {shapes[key]} does not match hr shape {ref_shape}")
        if shapes["met_onehot"] != (4, ref_shape[1], ref_shape[2]):
            raise ValueError(f"{npz_path}: met_onehot must have shape (4,H,W), got {shapes['met_onehot']}")

        for key in ["hr", "lr", "t1", "flair", "mask", "met_onehot"]:
            self._check_range(key, data[key])

        met_sum = data["met_onehot"].sum(axis=0)
        if not np.allclose(met_sum, 1.0, atol=1e-4):
            raise ValueError(f"{npz_path}: met_onehot must sum to 1 at each pixel.")

    def __getitem__(self, index):
        rec = self.records[index]
        npz_path = self.split_root / rec["npz"]
        data = np.load(str(npz_path))
        if self.strict_check:
            self._validate_npz(data, npz_path)

        hr = torch.from_numpy(data["hr"]).float()
        lr = torch.from_numpy(data["lr"]).float()
        t1 = torch.from_numpy(data["t1"]).float()
        flair = torch.from_numpy(data["flair"]).float()
        met_onehot = torch.from_numpy(data["met_onehot"]).float()
        mask = torch.from_numpy(data["mask"]).float()

        hr, lr, t1, flair, met_onehot, mask = self._maybe_flip([hr, lr, t1, flair, met_onehot, mask])

        cond_tensors: List[torch.Tensor] = []
        if self.use_lr:
            cond_tensors.append(lr)
        if self.use_t1:
            cond_tensors.append(t1)
        if self.use_flair:
            cond_tensors.append(flair)
        if self.use_met_onehot:
            cond_tensors.append(met_onehot)
        if self.use_mask_channel:
            cond_tensors.append(mask)
        sr_cond = torch.cat(cond_tensors, dim=0)

        out = {
            "HR": _to_minus1_1(hr),
            "SR": _to_minus1_1(sr_cond),
            "LR": _to_minus1_1(lr),
            "MASK": mask,  # keep in 0~1 for frequency-domain metrics
            "Index": index,
            "SAMPLE_ID": int(rec.get("sample_id", index)),
            "SPLIT": rec.get("split", self.split),
            "MET_NAME": rec.get("met_name", str(int(data["met_id"]))),
            "COND_LAYOUT": "|".join(self.condition_layout),
            "MET_ID": int(data["met_id"]),
            "PATIENT_ID": int(data["patient_id"]),
            "SLICE_IDX": int(data["slice_idx"]),
            "LOWRES": int(data["lowres"]),
        }
        return out


