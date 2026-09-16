import unittest

import numpy as np
import torch
from torch import nn

from model.model import DDPM
from model.base_model import BaseModel
from model.sr3_modules.diffusion import GaussianDiffusion, make_ddim_timesteps
from core.mrsi_physics import (
    crop_padded_native,
    mrsi_native_forward_batch,
    refine_native_data_consistency,
)
from core.quantitative_normalization import denormalize_quantity, normalize_quantity
from core.sr_metrics import hfen_2d, native_acquisition_consistency_2d
from model.sr3_modules.losses import native_acquisition_l1_sum, roi_mean_consistency_sum
from sr import _safe_mean, _validation_seed


class _ZeroDenoiser(nn.Module):
    def forward(self, x, noise_level):
        return torch.zeros(
            (x.shape[0], 1, x.shape[-2], x.shape[-1]),
            dtype=x.dtype,
            device=x.device,
        )


def _diffusion(min_snr_gamma=0.0):
    model = GaussianDiffusion(
        _ZeroDenoiser(),
        image_size=4,
        channels=1,
        loss_type='l1',
        conditional=False,
        min_snr_gamma=min_snr_gamma,
    )
    model.set_loss(torch.device('cpu'))
    model.set_new_noise_schedule(
        {
            'schedule': 'linear',
            'n_timestep': 10,
            'linear_start': 1e-4,
            'linear_end': 2e-2,
        },
        torch.device('cpu'),
    )
    return model


class TrainingFixTests(unittest.TestCase):
    def test_ddim_schedule_covers_both_endpoints(self):
        steps = make_ddim_timesteps(1000, 20)
        self.assertEqual(len(steps), 20)
        self.assertEqual(steps[0], 999)
        self.assertEqual(steps[-1], 0)
        self.assertTrue(all(left > right for left, right in zip(steps, steps[1:])))

    def test_min_snr_reweights_epsilon_loss(self):
        batch = {'HR': torch.zeros(4, 1, 4, 4)}
        noise = torch.ones_like(batch['HR'])
        baseline = _diffusion(min_snr_gamma=0.0)
        weighted = _diffusion(min_snr_gamma=5.0)

        np.random.seed(7)
        baseline_loss = float(baseline.p_losses(batch, noise=noise).item())
        np.random.seed(7)
        weighted_loss = float(weighted.p_losses(batch, noise=noise).item())

        self.assertGreater(baseline_loss, 0.0)
        self.assertGreater(weighted_loss, 0.0)
        self.assertLess(weighted_loss, baseline_loss)
        self.assertIn('train/min_snr_weight', weighted.last_loss_dict)

    def test_eval_network_can_explicitly_select_raw_or_ema(self):
        holder = DDPM.__new__(DDPM)
        holder.netG = object()
        holder.netG_EMA = object()
        holder.eval_network = 'auto'
        self.assertIs(holder._get_eval_network('raw'), holder.netG)
        self.assertIs(holder._get_eval_network('ema'), holder.netG_EMA)
        self.assertIs(holder._get_eval_network(), holder.netG_EMA)

    def test_validation_seed_is_stable_per_sample(self):
        opt = {'fixed_seed': 11, 'seed_mode': 'sample_id'}
        self.assertEqual(_validation_seed(opt, 9792, 1), 9803)
        self.assertEqual(_validation_seed(opt, 9792, 99), 9803)
        self.assertEqual(_validation_seed({'fixed_seed': 11, 'seed_mode': 'constant'}, 9792, 1), 11)

    def test_missing_metric_is_not_reported_as_zero(self):
        self.assertIsNone(_safe_mean([]))
        self.assertEqual(_safe_mean([1.0, 3.0]), 2.0)

    def test_set_device_moves_modules_as_well_as_tensors(self):
        holder = BaseModel.__new__(BaseModel)
        holder.device = torch.device('cpu')
        module = nn.Linear(2, 1)
        self.assertIs(holder.set_device(module), module)
        self.assertEqual(next(module.parameters()).device.type, 'cpu')

    def test_hfen_kernel_has_valid_conv2d_shape(self):
        pred = torch.zeros(1, 64, 64)
        target = torch.ones(1, 1, 64, 64)
        mask = torch.ones(1, 1, 64, 64)
        result = hfen_2d(pred, target, mask=mask)
        self.assertIn('hfen_nrmse', result)
        self.assertTrue(np.isfinite(result['hfen_nrmse']))

    def test_roi_mean_loss_has_finite_gradient_near_zero_mean(self):
        pred = torch.tensor([[[[-0.1, 0.1], [0.1, -0.1]]]], requires_grad=True)
        target = torch.tensor([[[[-0.1, 0.1], [-0.1, 0.1]]]])
        loss = roi_mean_consistency_sum(pred, target, mask=torch.ones_like(target))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(pred.grad).all())

    def test_native_forward_preserves_constant_concentration(self):
        hr = torch.full((3, 1, 64, 64), 0.4)
        padded = mrsi_native_forward_batch(hr, torch.tensor([16, 24, 32]), window='hamming')
        for sample_idx, matrix in enumerate((16, 24, 32)):
            native = crop_padded_native(padded[sample_idx:sample_idx + 1], matrix)
            self.assertTrue(torch.allclose(native, torch.full_like(native, 0.4), atol=2e-5))

    def test_native_forward_conserves_integral_with_voxel_area(self):
        torch.manual_seed(3)
        hr = torch.rand(1, 1, 64, 64)
        padded = mrsi_native_forward_batch(hr, [16], window='none', clamp_nonnegative=False)
        native = crop_padded_native(padded, 16)
        hr_integral = hr.sum() / float(64 * 64)
        native_integral = native.sum() / float(16 * 16)
        self.assertTrue(torch.allclose(hr_integral, native_integral, atol=1e-5))

    def test_native_acquisition_loss_is_zero_for_matching_observation(self):
        pred_01 = torch.linspace(0.0, 1.0, 64 * 64).reshape(1, 1, 64, 64)
        target = mrsi_native_forward_batch(pred_01, [16], window='hamming')
        pred_x0 = pred_01 * 2.0 - 1.0
        loss = native_acquisition_l1_sum(
            pred_x0,
            target,
            torch.tensor([16]),
            mask=torch.ones_like(pred_01),
            valid=torch.ones(1),
            window='hamming',
        )
        self.assertLess(float(loss.item()), 1e-4)

    def test_native_acquisition_metric_detects_bias(self):
        pred = torch.full((1, 1, 64, 64), 0.5)
        target = mrsi_native_forward_batch(torch.full_like(pred, 0.4), [16])
        result = native_acquisition_consistency_2d(pred, target, 16)
        self.assertGreater(result['native_acquisition_l1'], 0.09)
        self.assertGreater(result['native_acquisition_mean_bias'], 0.09)

    def test_quantity_normalization_is_reversible_and_preserves_zero(self):
        quantity = np.array([[0.0, 0.25], [0.5, 1.0]], dtype=np.float32)
        normalized = normalize_quantity(quantity, 2.0, clip=False)
        restored = denormalize_quantity(normalized, 2.0)
        self.assertEqual(float(normalized[0, 0]), 0.0)
        np.testing.assert_allclose(restored, quantity, atol=1e-7)

    def test_inference_projection_reduces_native_residual(self):
        initial_01 = torch.full((1, 1, 32, 32), 0.55)
        target = mrsi_native_forward_batch(torch.full_like(initial_01, 0.4), [8])
        before = native_acquisition_consistency_2d(initial_01, target, 8)['native_acquisition_l1']
        refined = refine_native_data_consistency(
            initial_01 * 2.0 - 1.0,
            target,
            torch.tensor([8]),
            hr_mask=torch.ones_like(initial_01),
            iterations=8,
            learning_rate=0.02,
            anchor_weight=0.1,
        )
        after = native_acquisition_consistency_2d(refined, target, 8)['native_acquisition_l1']
        self.assertLess(after, before)


if __name__ == '__main__':
    unittest.main()
