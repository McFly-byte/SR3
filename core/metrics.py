import os
import math
import numpy as np
import torch
try:
    import cv2
except Exception:
    cv2 = None
from PIL import Image


def _make_grid_fallback(tensor, nrow=8, padding=2, pad_value=0):
    if tensor.dim() != 4:
        raise ValueError(f"Expected 4D tensor, got {tuple(tensor.shape)}")
    n_img, channels, h, w = tensor.shape
    nrow = max(1, min(int(nrow), n_img))
    ncol = int(math.ceil(float(n_img) / nrow))
    grid_h = ncol * h + padding * (ncol - 1)
    grid_w = nrow * w + padding * (nrow - 1)
    grid = torch.full((channels, grid_h, grid_w), float(pad_value), dtype=tensor.dtype, device=tensor.device)
    for idx in range(n_img):
        r = idx // nrow
        c = idx % nrow
        y0 = r * (h + padding)
        x0 = c * (w + padding)
        grid[:, y0:y0 + h, x0:x0 + w] = tensor[idx]
    return grid


def tensor2img(tensor, out_type=np.uint8, min_max=(-1, 1)):
    '''
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    '''
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
    tensor = (tensor - min_max[0]) / \
        (min_max[1] - min_max[0])  # to range [0,1]
    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = _make_grid_fallback(
            tensor, nrow=int(math.sqrt(n_img)), padding=2, pad_value=0
        ).numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type)


def save_img(img, img_path, mode='RGB'):
    if img.ndim == 3 and img.shape[2] == 1:
        img = img[:, :, 0]
    if cv2 is not None:
        if img.ndim == 2:
            cv2.imwrite(img_path, img)
            return
        if img.ndim == 3 and img.shape[2] >= 3:
            cv2.imwrite(img_path, cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2BGR))
            return
        raise ValueError(f"Unsupported image shape for save_img: {img.shape}")
    # PIL fallback when OpenCV is unavailable.
    if img.ndim == 2:
        Image.fromarray(img.astype(np.uint8), mode='L').save(img_path)
        return
    if img.ndim == 3 and img.shape[2] >= 3:
        Image.fromarray(img[:, :, :3].astype(np.uint8), mode='RGB').save(img_path)
        return
    raise ValueError(f"Unsupported image shape for save_img without cv2: {img.shape}")


def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def ssim(img1, img2):
    if cv2 is None:
        # Fallback to skimage when OpenCV is unavailable.
        from skimage.metrics import structural_similarity as sk_ssim
        data_range = float(max(1.0, img2.max() - img2.min()))
        return float(sk_ssim(img1.astype(np.float64), img2.astype(np.float64), data_range=data_range))

    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1[:, :, i], img2[:, :, i]))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')
