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
        use_native_lr_consistency: bool = False,
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
        self.use_native_lr_consistency = bool(use_native_lr_consistency)
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

        if "lr_native" in data.files:
            lr_native_shape = tuple(data["lr_native"].shape)
            lr_matrix = (
                int(np.asarray(data["lr_matrix"]).item())
                if "lr_matrix" in data.files else int(lr_native_shape[-1])
            )
            if lr_native_shape != (1, lr_matrix, lr_matrix):
                raise ValueError(
                    f"{npz_path}: lr_native must have shape (1,{lr_matrix},{lr_matrix}), "
                    f"got {lr_native_shape}"
                )
            self._check_range("lr_native", data["lr_native"])

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

        hr_matrix = int(hr.shape[-1])
        lr_matrix = int(np.asarray(data["lr_matrix"]).item()) if "lr_matrix" in data.files else int(data["lowres"]) * 2
        has_native_lr = int("lr_native" in data.files)
        if self.use_native_lr_consistency and not has_native_lr:
            raise KeyError(
                f"{npz_path} has no lr_native field, but use_native_lr_consistency=true. "
                "Use the quantitative-v2 dataset or disable acquisition consistency."
            )
        lr_native_padded = torch.zeros_like(hr)
        if has_native_lr:
            lr_native = torch.from_numpy(data["lr_native"]).float()
            native_h, native_w = int(lr_native.shape[-2]), int(lr_native.shape[-1])
            if (native_h, native_w) != (lr_matrix, lr_matrix):
                raise ValueError(
                    f"{npz_path}: lr_native shape {(native_h, native_w)} does not match "
                    f"lr_matrix={lr_matrix}"
                )
            y0 = hr_matrix // 2 - native_h // 2
            x0 = int(hr.shape[-1]) // 2 - native_w // 2
            lr_native_padded[:, y0 : y0 + native_h, x0 : x0 + native_w] = lr_native

        hr, lr, t1, flair, met_onehot, mask, lr_native_padded = self._maybe_flip(
            [hr, lr, t1, flair, met_onehot, mask, lr_native_padded]
        )

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
            "LR_NATIVE": lr_native_padded,
            "HAS_NATIVE_LR": torch.tensor(has_native_lr, dtype=torch.float32),
            "LR_MATRIX": torch.tensor(lr_matrix, dtype=torch.int64),
            "HR_MATRIX": torch.tensor(hr_matrix, dtype=torch.int64),
            "NORMALIZATION_SCALE": torch.tensor(
                float(np.asarray(data["normalization_scale"]).item())
                if "normalization_scale" in data.files else 1.0,
                dtype=torch.float32,
            ),
            "QUANTITY_NAME": (
                str(np.asarray(data["quantity_name"]).item())
                if "quantity_name" in data.files else "concentration_like_signal"
            ),
            "QUANTITY_UNIT": (
                str(np.asarray(data["quantity_unit"]).item())
                if "quantity_unit" in data.files else "simulation_arbitrary_unit"
            ),
            "VOXEL_SEMANTICS": (
                str(np.asarray(data["voxel_semantics"]).item())
                if "voxel_semantics" in data.files else "intensive"
            ),
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
