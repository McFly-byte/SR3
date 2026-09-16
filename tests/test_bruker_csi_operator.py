import sys
import unittest
from pathlib import Path

import numpy as np
import torch


_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from core.bruker_csi_operator import (  # noqa: E402
    BrukerCSIReconstructionContract,
    BrukerCSIReconstructionOperator,
)
import validate_rawdata_to_fidproc as validator  # noqa: E402


_SCAN = Path(r"D:\LMC\data\invivo_zlx\zlx_healthy_rats_data\R001\RAW\57")


def _contract_and_encoded(scan: Path):
    method = validator.read_text(scan / "method")
    reco = validator.read_text(scan / "pdata" / "1" / "reco")
    counts = np.asarray(
        validator.parse_numbers(validator.bruker_raw(method, "AverageList")),
        dtype=np.int64,
    )
    maps = np.asarray(
        validator.parse_numbers(validator.bruker_raw(reco, "RecoSortMaps"))[:305],
        dtype=np.int64,
    )
    user_window = np.asarray(
        validator.parse_numbers(validator.bruker_raw(reco, "RECO_usr_wdw")),
        dtype=np.float64,
    ).reshape(4, 256) / 2147483647.0
    rotate = validator.parse_numbers(validator.bruker_raw(reco, "RECO_rotate"))
    pc = validator.parse_numbers(validator.bruker_raw(reco, "RECO_pc_lin"))
    pc_first_order = pc[1::2]
    contract = BrukerCSIReconstructionContract(
        sort_maps=maps.tolist(),
        window_z=user_window[3, :5].tolist(),
        window_y=user_window[2, :9].tolist(),
        window_x=user_window[1, :9].tolist(),
        rotate_z=rotate[3],
        rotate_y=rotate[2],
        rotate_x=rotate[1],
        phase_first_order_z_deg=pc_first_order[3],
        phase_first_order_y_deg=pc_first_order[2],
        phase_first_order_x_deg=pc_first_order[1],
    )
    raw = validator.decode_raw(scan / "rawdata.job0", "ri")
    encoded = validator.aggregate_contiguous(raw, counts, "mean").T
    return contract, encoded


class BrukerCSIOperatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not _SCAN.is_dir():
            raise FileNotFoundError(f"Required audited scan not found: {_SCAN}")
        cls.contract, cls.encoded_numpy = _contract_and_encoded(_SCAN)
        cls.operator = BrukerCSIReconstructionOperator(cls.contract)

    def test_real_scan_matches_retained_fidproc_complex_values(self):
        encoded = torch.from_numpy(self.encoded_numpy).to(torch.complex128)
        actual = self.operator(encoded).detach().cpu().numpy()
        expected = validator.decode_vendor_fid(_SCAN / "pdata" / "1" / "fid_proc.64")
        relative_l2 = np.linalg.norm(actual - expected) / np.linalg.norm(expected)
        self.assertLess(relative_l2, 1e-12)
        self.assertLess(np.max(np.abs(actual - expected)), 1e-9)

    def test_adjoint_inner_product_identity(self):
        generator = torch.Generator().manual_seed(20260904)
        encoded = torch.complex(
            torch.randn((2, 7, 305), generator=generator, dtype=torch.float64),
            torch.randn((2, 7, 305), generator=generator, dtype=torch.float64),
        )
        voxel_fids = torch.complex(
            torch.randn((2, 7, 5, 9, 9), generator=generator, dtype=torch.float64),
            torch.randn((2, 7, 5, 9, 9), generator=generator, dtype=torch.float64),
        )
        left = torch.vdot(self.operator(encoded).reshape(-1), voxel_fids.reshape(-1))
        right = torch.vdot(encoded.reshape(-1), self.operator.adjoint(voxel_fids).reshape(-1))
        relative_error = torch.abs(left - right) / torch.maximum(
            torch.maximum(torch.abs(left), torch.abs(right)),
            torch.tensor(1.0, dtype=torch.float64),
        )
        self.assertLess(float(relative_error), 1e-12)

    def test_gradients_flow_without_magnitude_or_clamping(self):
        encoded = torch.from_numpy(self.encoded_numpy[:4]).to(torch.complex128).requires_grad_(True)
        output = self.operator(encoded)
        loss = output.abs().square().mean()
        loss.backward()
        self.assertIsNotNone(encoded.grad)
        self.assertTrue(torch.isfinite(encoded.grad.real).all())
        self.assertTrue(torch.isfinite(encoded.grad.imag).all())

    def test_contract_rejects_duplicate_sort_positions(self):
        values = list(self.contract.sort_maps)
        values[1] = values[0]
        bad = BrukerCSIReconstructionContract(
            sort_maps=values,
            window_z=self.contract.window_z,
            window_y=self.contract.window_y,
            window_x=self.contract.window_x,
            rotate_z=self.contract.rotate_z,
            rotate_y=self.contract.rotate_y,
            rotate_x=self.contract.rotate_x,
            phase_first_order_z_deg=self.contract.phase_first_order_z_deg,
            phase_first_order_y_deg=self.contract.phase_first_order_y_deg,
            phase_first_order_x_deg=self.contract.phase_first_order_x_deg,
        )
        with self.assertRaisesRegex(ValueError, "duplicate"):
            BrukerCSIReconstructionOperator(bad)


if __name__ == "__main__":
    unittest.main()
