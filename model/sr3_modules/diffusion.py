import math
import inspect
import torch
from torch import device, nn, einsum
import torch.nn.functional as F
from inspect import isfunction
from functools import partial
import numpy as np
from tqdm import tqdm

from .losses import (
    degradation_l1_sum,
    frequency_l1_sum,
    gradient_l1_sum,
    masked_l1_sum,
    roi_mean_consistency_sum,
)


def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(
        linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas


def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == 'quad':
        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5,
                            n_timestep, dtype=np.float64) ** 2
    elif schedule == 'linear':
        betas = np.linspace(linear_start, linear_end,
                            n_timestep, dtype=np.float64)
    elif schedule == 'warmup10':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.1)
    elif schedule == 'warmup50':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.5)
    elif schedule == 'const':
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(n_timestep,
                                 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) /
            n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(schedule)
    return betas


# gaussian diffusion trainer class

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        image_size,
        channels=3,
        loss_type='l1',
        conditional=True,
        schedule_opt=None,
        sampler_type='ddpm',
        sample_num_steps=None,
        mask_loss_weight=2.0,
        freq_loss_weight=0.0,
        # T1 cross-attention: index of the T1 channel within the condition tensor
        # Set to -1 (default) to disable; set to e.g. 1 for MRSI (LR=0, T1=1)
        t1_channel_idx=-1,
        x0_loss_weight=0.0,
        roi_mean_loss_weight=0.0,
        grad_loss_weight=0.0,
        freq_x0_loss_weight=0.0,
        degradation_loss_weight=0.0,
        degradation_window="hamming",
        condition_dropout_prob=0.0,
        condition_layout=None,
        condition_adapter=None,
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.denoise_fn = denoise_fn
        self.loss_type = loss_type
        self.conditional = conditional
        self.sampler_type = sampler_type
        self.sample_num_steps = sample_num_steps
        # ROI mask weight: pixels inside mask get this weight, background gets 1.0
        self.mask_loss_weight = float(mask_loss_weight)
        # frequency-domain L1 loss weight (0 = disabled)
        self.freq_loss_weight = float(freq_loss_weight)
        # Index of the T1 image channel within the stacked condition tensor
        # Used to extract T1 before forwarding to UNet's T1Encoder
        self.t1_channel_idx = int(t1_channel_idx)
        self.x0_loss_weight = float(x0_loss_weight)
        self.roi_mean_loss_weight = float(roi_mean_loss_weight)
        self.grad_loss_weight = float(grad_loss_weight)
        self.freq_x0_loss_weight = float(freq_x0_loss_weight)
        self.degradation_loss_weight = float(degradation_loss_weight)
        self.degradation_window = degradation_window
        self.condition_dropout_prob = float(condition_dropout_prob)
        self.condition_layout = condition_layout or {
            "lr": [0, 1],
            "t1": [1, 2],
            "flair": [2, 3],
            "met_onehot": [3, 7],
            "mask": None,
        }
        self.condition_adapter = condition_adapter or {}
        self.use_condition_adapter = bool(self.condition_adapter.get("enabled", False))
        denoise_fn = self.denoise_fn.module if hasattr(self.denoise_fn, 'module') else self.denoise_fn
        try:
            forward_params = inspect.signature(denoise_fn.forward).parameters
            self._denoise_accepts_cond = "cond" in forward_params
            self._denoise_accepts_t1_feat = "t1_feat" in forward_params
        except (TypeError, ValueError):
            self._denoise_accepts_cond = False
            self._denoise_accepts_t1_feat = False
        self.last_loss_dict = {}
        if schedule_opt is not None:
            pass
            # self.set_new_noise_schedule(schedule_opt)

    def _slice_condition(self, condition_x, key):
        spec = self.condition_layout.get(key)
        if condition_x is None or spec is None:
            return None
        if isinstance(spec, int):
            start, end = spec, spec + 1
        else:
            start, end = int(spec[0]), int(spec[1])
        if start < 0 or end > condition_x.shape[1] or end <= start:
            return None
        return condition_x[:, start:end, :, :]

    def _build_cond_dict(self, condition_x, mask=None):
        if not self.use_condition_adapter:
            return None
        cond = {
            "lr": self._slice_condition(condition_x, "lr"),
            "t1": self._slice_condition(condition_x, "t1"),
            "flair": self._slice_condition(condition_x, "flair"),
            "met_onehot": self._slice_condition(condition_x, "met_onehot"),
            "mask": self._slice_condition(condition_x, "mask"),
        }
        if cond["mask"] is None and mask is not None:
            cond["mask"] = mask
        return cond

    def _call_denoise_fn(self, model_input, noise_level, t1_feat=None, cond=None):
        kwargs = {}
        if self._denoise_accepts_t1_feat:
            kwargs["t1_feat"] = t1_feat
        if cond is not None and self._denoise_accepts_cond:
            kwargs["cond"] = cond
        return self.denoise_fn(model_input, noise_level, **kwargs)

    def _get_t1_feat(self, condition_x):
        """Extract T1 features via UNet's T1Encoder if cross-attention is enabled."""
        if self.t1_channel_idx < 0:
            return None
        if condition_x is None or self.t1_channel_idx + 1 > condition_x.shape[1]:
            return None
        denoise_fn = self.denoise_fn.module if hasattr(self.denoise_fn, 'module') else self.denoise_fn
        if not getattr(denoise_fn, 'use_t1_cross_attn', False):
            return None
        t1_img = condition_x[:, self.t1_channel_idx:self.t1_channel_idx + 1, :, :]
        return denoise_fn.t1_encoder(t1_img)

    def _maybe_drop_condition(self, condition_x):
        if (not self.training) or self.condition_dropout_prob <= 0.0:
            return condition_x
        keep = torch.rand((condition_x.shape[0], 1, 1, 1), device=condition_x.device) >= self.condition_dropout_prob
        return condition_x * keep.to(condition_x.dtype)

    def set_loss(self, device):
        if self.loss_type == 'l1':
            self.loss_func = nn.L1Loss(reduction='sum').to(device)
            self.loss_func_elementwise = nn.L1Loss(reduction='none').to(device)
        elif self.loss_type == 'l2':
            self.loss_func = nn.MSELoss(reduction='sum').to(device)
            self.loss_func_elementwise = nn.MSELoss(reduction='none').to(device)
        else:
            raise NotImplementedError()

    def set_new_noise_schedule(self, schedule_opt, device):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)

        betas = make_beta_schedule(
            schedule=schedule_opt['schedule'],
            n_timestep=schedule_opt['n_timestep'],
            linear_start=schedule_opt['linear_start'],
            linear_end=schedule_opt['linear_end'])
        betas = betas.detach().cpu().numpy() if isinstance(
            betas, torch.Tensor) else betas
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        self.sqrt_alphas_cumprod_prev = np.sqrt(
            np.append(1., alphas_cumprod))

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev',
                             to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod',
                             to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod',
                             to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * \
            (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance',
                             to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(
            np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

    def predict_start_from_noise(self, x_t, t, noise):
        return self.sqrt_recip_alphas_cumprod[t] * x_t - \
            self.sqrt_recipm1_alphas_cumprod[t] * noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = self.posterior_mean_coef1[t] * \
            x_start + self.posterior_mean_coef2[t] * x_t
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]
        return posterior_mean, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool, condition_x=None):
        batch_size = x.shape[0]
        noise_level = torch.FloatTensor(
            [self.sqrt_alphas_cumprod_prev[t+1]]).repeat(batch_size, 1).to(x.device)
        if condition_x is not None:
            t1_feat = self._get_t1_feat(condition_x)
            cond = self._build_cond_dict(condition_x)
            x_recon = self.predict_start_from_noise(
                x,
                t=t,
                noise=self._call_denoise_fn(
                    torch.cat([condition_x, x], dim=1),
                    noise_level,
                    t1_feat=t1_feat,
                    cond=cond,
                ),
            )
        else:
            x_recon = self.predict_start_from_noise(
                x, t=t, noise=self._call_denoise_fn(x, noise_level))

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, condition_x=None, generator=None):
        model_mean, model_log_variance = self.p_mean_variance(
            x=x, t=t, clip_denoised=clip_denoised, condition_x=condition_x)
        noise = torch.randn(
            x.shape, device=x.device, dtype=x.dtype, generator=generator
        ) if t > 0 else torch.zeros_like(x)
        return model_mean + noise * (0.5 * model_log_variance).exp()

    @torch.no_grad()
    def ddim_p_sample_loop(self, x_in, ddim_steps=50, eta=0.0, continous=False, seed=None):
        """DDIM deterministic sampler (Song et al. 2020).

        eta=0  → fully deterministic (recommended for medical reproducibility)
        eta>0  → stochastic with partial noise injection
        ddim_steps << num_timesteps gives 20-50x speed-up over full DDPM.
        """
        device = self.betas.device

        # Build uniformly-spaced subsequence of timesteps
        c = max(self.num_timesteps // ddim_steps, 1)
        # Descending order: T-1, T-1-c, ..., 0
        timesteps = list(reversed(range(0, self.num_timesteps, c)))[:ddim_steps]

        generator = None
        if seed is not None:
            if device.type == 'cuda':
                generator = torch.Generator(device=device)
            else:
                generator = torch.Generator()
            generator.manual_seed(int(seed))

        if self.conditional:
            x = x_in
            shape = x.shape
            img = torch.randn(
                (shape[0], self.channels, shape[2], shape[3]),
                device=device,
                generator=generator,
            )
            t1_feat = self._get_t1_feat(x)
            cond = self._build_cond_dict(x)
        else:
            shape = x_in
            img = torch.randn(shape, device=device, generator=generator)
            x = None
            t1_feat = None
            cond = None

        ret_img = img
        sample_inter = max(1, len(timesteps) // 10)

        for i, t in enumerate(tqdm(timesteps, desc='DDIM sampling')):
            t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else -1

            alpha_t = self.alphas_cumprod[t]
            alpha_t_prev = self.alphas_cumprod[t_prev] if t_prev >= 0 else torch.ones(1, device=device)

            # noise_level matches the convention used during training
            noise_level = torch.FloatTensor(
                [self.sqrt_alphas_cumprod_prev[t + 1]]
            ).repeat(img.shape[0], 1).to(device)

            if x is not None:
                pred_noise = self._call_denoise_fn(
                    torch.cat([x, img], dim=1),
                    noise_level,
                    t1_feat=t1_feat,
                    cond=cond,
                )
            else:
                pred_noise = self._call_denoise_fn(img, noise_level)

            # Predict x_0 from x_t and predicted noise
            pred_x0 = (img - (1.0 - alpha_t).sqrt() * pred_noise) / alpha_t.sqrt()
            pred_x0 = pred_x0.clamp(-1.0, 1.0)

            # DDIM update
            sigma = eta * (
                (1.0 - alpha_t_prev) / (1.0 - alpha_t) * (1.0 - alpha_t / alpha_t_prev)
            ).sqrt()
            dir_xt = (1.0 - alpha_t_prev - sigma ** 2).clamp(min=0.0).sqrt() * pred_noise
            rand_noise = (
                sigma * torch.randn(img.shape, device=device, generator=generator)
                if eta > 0 else 0.0
            )
            img = alpha_t_prev.sqrt() * pred_x0 + dir_xt + rand_noise

            if continous and i % sample_inter == 0:
                ret_img = torch.cat([ret_img, img], dim=0)

        if continous:
            return ret_img
        return img

    @torch.no_grad()
    def p_sample_loop(self, x_in, continous=False, seed=None, sample_num_steps=None):
        device = self.betas.device
        sample_num_steps = int(sample_num_steps or self.sample_num_steps or self.num_timesteps)

        # Route to DDIM when fewer steps are requested or sampler_type is 'ddim'
        if self.sampler_type == 'ddim' or sample_num_steps < self.num_timesteps:
            return self.ddim_p_sample_loop(
                x_in, ddim_steps=sample_num_steps, eta=0.0,
                continous=continous, seed=seed,
            )

        # Full DDPM sampling
        sample_inter = (1 | (self.num_timesteps // 10))
        generator = None
        if seed is not None:
            if device.type == 'cuda':
                generator = torch.Generator(device=device)
            else:
                generator = torch.Generator()
            generator.manual_seed(int(seed))

        if not self.conditional:
            shape = x_in
            img = torch.randn(shape, device=device, generator=generator)
            ret_img = img
            for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
                img = self.p_sample(img, i, generator=generator)
                if i % sample_inter == 0:
                    ret_img = torch.cat([ret_img, img], dim=0)
        else:
            x = x_in
            shape = x.shape
            img = torch.randn(
                (shape[0], self.channels, shape[2], shape[3]),
                device=device,
                generator=generator,
            )
            ret_img = img
            for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
                img = self.p_sample(img, i, condition_x=x, generator=generator)
                if i % sample_inter == 0:
                    ret_img = torch.cat([ret_img, img], dim=0)

        if continous:
            return ret_img
        return img

    @torch.no_grad()
    def sample(self, batch_size=1, continous=False, seed=None, sample_num_steps=None):
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop(
            (batch_size, channels, image_size, image_size),
            continous,
            seed=seed,
            sample_num_steps=sample_num_steps,
        )

    @torch.no_grad()
    def super_resolution(self, x_in, continous=False, seed=None, sample_num_steps=None):
        return self.p_sample_loop(x_in, continous, seed=seed, sample_num_steps=sample_num_steps)

    def q_sample(self, x_start, continuous_sqrt_alpha_cumprod, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
            continuous_sqrt_alpha_cumprod * x_start +
            (1 - continuous_sqrt_alpha_cumprod ** 2).sqrt() * noise
        )

    def _freq_loss(self, noise_target, noise_pred):
        """Frequency-domain L1 loss on the 2-D FFT magnitude spectrum."""
        fft_target = torch.fft.rfft2(noise_target, norm='ortho')
        fft_pred = torch.fft.rfft2(noise_pred, norm='ortho')
        mag_target = torch.abs(fft_target)
        mag_pred = torch.abs(fft_pred)
        return F.l1_loss(mag_pred, mag_target, reduction='sum')

    def _predict_x0_from_continuous_noise(self, x_noisy, pred_noise, sqrt_alpha):
        return (x_noisy - (1.0 - sqrt_alpha ** 2).sqrt() * pred_noise) / sqrt_alpha.clamp_min(1e-8)

    def p_losses(self, x_in, noise=None):
        x_start = x_in['HR']
        [b, c, h, w] = x_start.shape

        # Per-sample independent timestep sampling (fixes high gradient variance
        # caused by the original single global t shared across the entire batch).
        t_batch = np.random.randint(1, self.num_timesteps + 1, size=b)
        continuous_sqrt_alpha_cumprod = torch.FloatTensor([
            np.random.uniform(
                self.sqrt_alphas_cumprod_prev[t - 1],
                self.sqrt_alphas_cumprod_prev[t],
            )
            for t in t_batch
        ]).to(x_start.device)
        # Shape for noise-level embedding: (B, 1)
        continuous_sqrt_alpha_cumprod_emb = continuous_sqrt_alpha_cumprod.view(b, 1)
        # Shape for q_sample broadcast: (B, 1, 1, 1)
        continuous_sqrt_alpha_cumprod_spatial = continuous_sqrt_alpha_cumprod.view(b, 1, 1, 1)

        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(
            x_start=x_start,
            continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod_spatial,
            noise=noise,
        )

        if not self.conditional:
            x_recon = self._call_denoise_fn(x_noisy, continuous_sqrt_alpha_cumprod_emb)
        else:
            condition_sr = self._maybe_drop_condition(x_in['SR'])
            t1_feat = self._get_t1_feat(condition_sr)
            cond = self._build_cond_dict(condition_sr, mask=x_in.get('MASK', None))
            x_recon = self._call_denoise_fn(
                torch.cat([condition_sr, x_noisy], dim=1),
                continuous_sqrt_alpha_cumprod_emb,
                t1_feat=t1_feat,
                cond=cond,
            )

        # Mask-weighted pixel loss: ROI pixels receive higher gradient weight
        mask = x_in.get('MASK', None)
        if mask is not None and self.mask_loss_weight != 1.0:
            # weight in [1, mask_loss_weight]; background=1, ROI=mask_loss_weight
            weight = 1.0 + (self.mask_loss_weight - 1.0) * mask.to(x_start.device)
            pixel_loss = (self.loss_func_elementwise(noise, x_recon) * weight).sum()
        else:
            pixel_loss = self.loss_func(noise, x_recon)

        loss = pixel_loss
        loss_terms = {"loss/noise": pixel_loss.detach() / float(x_start.numel())}

        if self.freq_loss_weight > 0.0:
            freq_noise = self._freq_loss(noise, x_recon)
            loss = loss + self.freq_loss_weight * freq_noise
            loss_terms["loss/freq_noise"] = freq_noise.detach() / float(x_start.numel())

        needs_x0 = any(
            weight > 0.0 for weight in [
                self.x0_loss_weight,
                self.roi_mean_loss_weight,
                self.grad_loss_weight,
                self.freq_x0_loss_weight,
                self.degradation_loss_weight,
            ]
        )
        if needs_x0:
            x0_hat = self._predict_x0_from_continuous_noise(
                x_noisy,
                x_recon,
                continuous_sqrt_alpha_cumprod_spatial,
            ).clamp(-1.0, 1.0)
            if self.x0_loss_weight > 0.0:
                x0_l1 = masked_l1_sum(x0_hat, x_start, mask=mask)
                loss = loss + self.x0_loss_weight * x0_l1
                loss_terms["loss/x0_l1"] = x0_l1.detach() / float(x_start.numel())
            if self.roi_mean_loss_weight > 0.0:
                roi_mean = roi_mean_consistency_sum(x0_hat, x_start, mask=mask)
                loss = loss + self.roi_mean_loss_weight * roi_mean
                loss_terms["loss/roi_mean"] = roi_mean.detach() / float(x_start.numel())
            if self.grad_loss_weight > 0.0:
                grad_loss = gradient_l1_sum(x0_hat, x_start, mask=mask)
                loss = loss + self.grad_loss_weight * grad_loss
                loss_terms["loss/grad"] = grad_loss.detach() / float(x_start.numel())
            if self.freq_x0_loss_weight > 0.0:
                freq_x0 = frequency_l1_sum(x0_hat, x_start, mask=mask)
                loss = loss + self.freq_x0_loss_weight * freq_x0
                loss_terms["loss/freq_x0"] = freq_x0.detach() / float(x_start.numel())
            if self.degradation_loss_weight > 0.0 and 'LR' in x_in:
                lowres = x_in.get('LOWRES', None)
                deg_loss = degradation_l1_sum(
                    x0_hat,
                    x_in['LR'],
                    lowres_half=lowres,
                    mask=mask,
                    window=self.degradation_window,
                )
                loss = loss + self.degradation_loss_weight * deg_loss
                loss_terms["loss/degradation"] = deg_loss.detach() / float(x_start.numel())

        self.last_loss_dict = {
            key: float(val.item() if torch.is_tensor(val) else val)
            for key, val in loss_terms.items()
        }
        # DataParallel warns when every replica returns a scalar; keeping a
        # length-1 tensor preserves the existing sum() reduction downstream.
        return loss.view(1)

    def forward(self, x, *args, **kwargs):
        return self.p_losses(x, *args, **kwargs)
