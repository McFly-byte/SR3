# Fix for Windows: TCPStore/libuv — conda 等 Windows 版 PyTorch 常未带 libuv
import os
if os.name == 'nt':
    os.environ['USE_LIBUV'] = '0'

import torch
import torch.distributed as dist

from core.win_tcpstore_patch import apply_tcpstore_no_libuv_patch

apply_tcpstore_no_libuv_patch()

import data as Data
import model as Model
import argparse
import logging
import csv
import json
import core.logger as Logger
import core.metrics as Metrics
from core.sr_metrics import (
    degradation_consistency_2d,
    dmi_quant_metrics,
    false_hotspot_rate,
    frc_2d,
    hfen_2d,
    native_acquisition_consistency_2d,
)
from core.wandb_logger import WandbLogger
try:
    from tensorboardX import SummaryWriter
except Exception:
    from torch.utils.tensorboard import SummaryWriter
import numpy as np
try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

try:
    import lpips
except Exception:
    lpips = None


def _safe_mean(vals):
    if len(vals) == 0:
        return None
    return float(np.mean(vals))


def _metric_text(value):
    return 'N/A' if value is None else '{:.4e}'.format(float(value))


def _defined_metrics(metrics):
    return {key: value for key, value in metrics.items() if value is not None}


def _validation_seed(validation_opt, sample_id, index):
    """Return a stable per-sample seed for comparable checkpoint evaluation."""
    fixed_seed = validation_opt.get('fixed_seed', None)
    if fixed_seed is None:
        return None
    base_seed = int(fixed_seed)
    seed_mode = str(validation_opt.get('seed_mode', 'sample_id') or 'sample_id').lower()
    if seed_mode == 'constant':
        return base_seed
    if seed_mode == 'index':
        return base_seed + int(index)
    if seed_mode == 'sample_id':
        try:
            return base_seed + int(sample_id)
        except (TypeError, ValueError):
            return base_seed + int(index)
    raise ValueError(
        "validation.seed_mode must be one of: constant, index, sample_id; "
        "got {!r}.".format(seed_mode)
    )


def _stratified_validation_subset(dataset, max_samples, validation_opt):
    """Build a deterministic, approximately balanced validation subset."""
    max_samples = int(max_samples)
    if max_samples <= 0 or max_samples >= len(dataset):
        return dataset
    records = getattr(dataset, 'records', None)
    if not records:
        return torch.utils.data.Subset(dataset, list(range(max_samples)))

    fields = validation_opt.get(
        'stratify_fields', ['scenario_type', 'met_name', 'lowres']
    )
    fields = [str(field) for field in fields]
    buckets = {}
    for index, record in enumerate(records):
        key = tuple(str(record.get(field, '')) for field in fields)
        buckets.setdefault(key, []).append(index)

    rng = np.random.default_rng(int(validation_opt.get('subset_seed', 0) or 0))
    for indices in buckets.values():
        rng.shuffle(indices)

    selected = []
    ordered_keys = sorted(buckets)
    while len(selected) < max_samples:
        added = False
        for key in ordered_keys:
            if buckets[key]:
                selected.append(buckets[key].pop())
                added = True
                if len(selected) >= max_samples:
                    break
        if not added:
            break
    selected.sort()
    logging.getLogger('base').info(
        'Using deterministic stratified validation subset: %d/%d rows, fields=%s.',
        len(selected),
        len(dataset),
        ','.join(fields),
    )
    return torch.utils.data.Subset(dataset, selected)


def _as_scalar(data, key, default=-1):
    if key not in data:
        return default
    val = data[key]
    if torch.is_tensor(val):
        return val.view(-1)[0].item()
    if isinstance(val, (list, tuple)):
        return val[0] if val else default
    return val


def _masked_ssim_from_imgs(sr_img, hr_img, mask_t):
    if mask_t is None:
        return None
    mask_np = mask_t.squeeze().detach().float().cpu().numpy()
    mask_np = (mask_np > 0.5).astype(np.float32)
    if sr_img.ndim == 3:
        mask_np = mask_np[:, :, None]
    sr_masked = (sr_img.astype(np.float32) * mask_np).astype(sr_img.dtype)
    hr_masked = (hr_img.astype(np.float32) * mask_np).astype(hr_img.dtype)
    try:
        return float(Metrics.calculate_ssim(sr_masked, hr_masked))
    except Exception:
        return None


@torch.no_grad()
def _compute_dmi_metrics(
    sr_t,
    hr_t,
    lr_t,
    mask_t,
    lowres,
    kspace_window="hamming",
    *,
    lr_native_t=None,
    lr_matrix=None,
    normalization_scale=1.0,
):
    out = {}
    try:
        out.update(
            dmi_quant_metrics(
                sr_t,
                hr_t,
                mask=mask_t,
                normalization_scale=float(normalization_scale),
            )
        )
    except Exception:
        pass
    try:
        out.update(false_hotspot_rate(sr_t, hr_t, mask=mask_t))
    except Exception:
        pass
    try:
        lowres_half = None if lowres in (None, -1) else int(lowres)
        out.update(degradation_consistency_2d(sr_t, lr_t, mask=mask_t, lowres_half=lowres_half, window=kspace_window))
    except Exception:
        pass
    if lr_native_t is not None and lr_matrix not in (None, -1):
        try:
            out.update(
                native_acquisition_consistency_2d(
                    sr_t,
                    lr_native_t,
                    int(lr_matrix),
                    mask=mask_t,
                    window=kspace_window,
                )
            )
        except Exception:
            pass
    return out


def _to_4d(t):
    if t.dim() == 4:
        return t
    if t.dim() == 3:
        return t.unsqueeze(0)
    if t.dim() == 2:
        return t.unsqueeze(0).unsqueeze(0)
    raise ValueError(f"Unsupported tensor dim: {t.dim()}")


def _to_3ch_hwc(img_np):
    if img_np.ndim == 2:
        return np.repeat(img_np[:, :, None], 3, axis=2)
    if img_np.ndim == 3 and img_np.shape[2] == 1:
        return np.repeat(img_np, 3, axis=2)
    if img_np.ndim == 3 and img_np.shape[2] >= 3:
        return img_np[:, :, :3]
    raise ValueError(f"Unsupported image shape: {img_np.shape}")


def _concat_for_tb(left, mid, right):
    a = _to_3ch_hwc(left)
    b = _to_3ch_hwc(mid)
    c = _to_3ch_hwc(right)
    return np.transpose(np.concatenate((a, b, c), axis=1), [2, 0, 1])


def _error_map_img(sr_t, hr_t):
    sr01 = ((sr_t.detach().float().cpu() + 1.0) * 0.5).clamp(0.0, 1.0)
    hr01 = ((hr_t.detach().float().cpu() + 1.0) * 0.5).clamp(0.0, 1.0)
    return Metrics.tensor2img(torch.abs(sr01 - hr01), min_max=(0, 1))


def _build_lpips_model(opt, logger):
    mcfg = opt.get('metrics', {})
    if not mcfg.get('enable_lpips', True):
        return None
    if lpips is None:
        logger.warning('lpips is not installed; LPIPS metric will be skipped.')
        return None
    net_name = mcfg.get('lpips_net', 'alex')
    model = lpips.LPIPS(net=net_name)
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    logger.info(f'LPIPS metric enabled (net={net_name}, device={device}).')
    return model


@torch.no_grad()
def _compute_lpips(lpips_model, sr_t, hr_t, mask_t=None):
    if lpips_model is None:
        return None
    sr4 = _to_4d(sr_t).float()
    hr4 = _to_4d(hr_t).float()
    if mask_t is not None:
        m4 = _to_4d(mask_t).float()
        sr4 = sr4 * m4
        hr4 = hr4 * m4
    sr3 = sr4.repeat(1, 3, 1, 1)
    hr3 = hr4.repeat(1, 3, 1, 1)
    dev = next(lpips_model.parameters()).device
    val = lpips_model(sr3.to(dev), hr3.to(dev))
    return float(val.mean().item())


@torch.no_grad()
def _compute_frc(sr_t, hr_t, mask_t=None, apodize=True):
    try:
        out = frc_2d(sr_t.squeeze(), hr_t.squeeze(), mask=(mask_t.squeeze() if mask_t is not None else None), apodize=apodize)
        return float(out['frc_auc_w']), float(out['frc_hf_mean']), float(out['frc_cutoff_1_7'])
    except Exception:
        return None, None, None


@torch.no_grad()
def _compute_hfen(sr_t, hr_t, mask_t=None, sigma=1.5, trunc=3.0):
    try:
        out = hfen_2d(
            sr_t.squeeze(),
            hr_t.squeeze(),
            mask=(mask_t.squeeze() if mask_t is not None else None),
            sigma=sigma,
            trunc=trunc,
        )
        return float(out['hfen_nrmse'])
    except Exception:
        return None


def _write_metrics_csv(rows, csv_path):
    preferred_fields = [
        'index', 'patient_id', 'slice_idx', 'met_id',
        'sample_id', 'split', 'met_name', 'lowres',
        'lr_matrix', 'normalization_scale', 'quantity_unit',
        'eval_network', 'sample_seed', 'sample_steps',
        'psnr', 'ssim', 'masked_ssim', 'lpips', 'hfen', 'frc_aucw', 'frc_hf', 'frc_cut',
        'masked_psnr', 'masked_mae',
        'roi_mean_abs_err', 'roi_mean_rel_err',
        'roi_std_abs_err', 'roi_std_rel_err',
        'roi_sum_abs_err', 'roi_sum_rel_err',
        'roi_mean_abs_err_quantity',
        'concentration_integral_abs_err', 'concentration_integral_rel_err',
        'false_hotspot_rate', 'false_hotspot_precision_err',
        'degradation_l1', 'degradation_rmse',
        'native_acquisition_l1', 'native_acquisition_rmse',
        'native_acquisition_mean_bias', 'native_acquisition_rel_l1',
    ]
    present = {key for row in rows for key in row}
    fieldnames = [key for key in preferred_fields if key in present]
    fieldnames.extend(sorted(present.difference(fieldnames)))
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _init_dist_if_needed(opt):
    if not opt.get('distributed', False):
        return False
    if dist.is_available() and dist.is_initialized():
        return True
    if not torch.cuda.is_available():
        raise RuntimeError('DDP requires CUDA for this project.')
    
    if os.name == 'nt':
        os.environ['USE_LIBUV'] = '0'
        # 集群主机名在部分 Windows/C10d 场景下能解析仍报 10049；单机训练强制回环（多机设 DDP_USE_CLUSTER_MASTER=1）
        if os.environ.get('DDP_USE_CLUSTER_MASTER', '').lower() not in ('1', 'true', 'yes'):
            os.environ['MASTER_ADDR'] = '127.0.0.1'

    local_rank = int(opt.get('local_rank', 0))
    torch.cuda.set_device(local_rank)

    # Windows 下优先使用 NCCL（若可用），避免 gloo 的网络设备选择问题。
    if os.name == 'nt':
        backend = 'nccl' if getattr(dist, 'is_nccl_available', lambda: False)() else 'gloo'
    else:
        backend = 'nccl'
    
    init_method = 'env://?use_libuv=0' if os.name == 'nt' else 'env://'
    dist.init_process_group(backend=backend, init_method=init_method)

    return True


def _cleanup_dist():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _reduce_mean(value):
    if not (dist.is_available() and dist.is_initialized()):
        return value
    tensor = torch.tensor(float(value), device=torch.device('cuda', torch.cuda.current_device()))
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor = tensor / float(dist.get_world_size())
    return float(tensor.item())


def _reduce_log_dict(logs):
    return {k: _reduce_mean(v) for k, v in logs.items()}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/sr_sr3_16_128.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'],
                        help='Run either train(training) or val(generation)', default='train')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_wandb_ckpt', action='store_true')
    parser.add_argument('-log_eval', action='store_true')
    parser.add_argument('--local_rank', type=int, default=int(os.environ.get('LOCAL_RANK', 0)))

    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)
    if args.phase == 'train' and len(opt.get('gpu_ids', [])) > 1 and not opt.get('distributed', False):
        # On some Windows+PyTorch builds, DDP (gloo/libuv) may fail.
        # In that case, we can still train with a single process + DataParallel.
        print('[WARN] Multiple GPUs requested but DDP not launched; using DataParallel instead.')
    _init_dist_if_needed(opt)

    # logging
    train_opt = opt.get('train', {}) or {}
    seed = int(train_opt.get('seed', 0) or 0)
    deterministic = bool(train_opt.get('deterministic', False))
    Logger.set_random_seed(seed, deterministic=deterministic)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = bool(train_opt.get('cudnn_benchmark', not deterministic))
    torch.backends.cudnn.deterministic = bool(train_opt.get('cudnn_deterministic', deterministic))
    allow_tf32 = bool(train_opt.get('allow_tf32', True))
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32

    logger = logging.getLogger('base')
    logger_val = logging.getLogger('val')
    if opt.get('is_main_process', True):
        Logger.setup_logger(None, opt['path']['log'],
                            'train', level=logging.INFO, screen=True)
        Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
        logger.info(Logger.dict2str(opt))
        tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])
    else:
        tb_logger = None

    # Initialize WandbLogger
    if opt['enable_wandb'] and opt.get('is_main_process', True):
        import wandb
        wandb_logger = WandbLogger(opt)
        wandb.define_metric('validation/val_step')
        wandb.define_metric('epoch')
        wandb.define_metric("validation/*", step_metric="val_step")
        val_step = 0
    else:
        wandb_logger = None
    lpips_model = _build_lpips_model(opt, logger) if opt.get('is_main_process', True) else None

    # dataset
    val_loader = None
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train' and args.phase != 'val':
            train_set = Data.create_dataset(dataset_opt, phase)
            train_loader = Data.create_dataloader(
                train_set, dataset_opt, phase)
        elif phase == 'val' and (opt.get('is_main_process', True) or args.phase == 'val'):
            val_set = Data.create_dataset(dataset_opt, phase)
            val_opt = opt.get('validation', {}) or {}
            if args.phase == 'val':
                val_limit = int(val_opt.get('max_samples', -1) or -1)
            else:
                val_limit = int((opt.get('train', {}) or {}).get('val_max_samples', -1) or -1)
            val_set = _stratified_validation_subset(val_set, val_limit, val_opt)
            val_loader = Data.create_dataloader(
                val_set, dataset_opt, phase)
    logger.info('Initial Dataset Finished')

    # model
    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')

    # Train
    current_step = diffusion.begin_step
    current_epoch = diffusion.begin_epoch
    n_iter = opt['train']['n_iter']

    if opt['path']['resume_state']:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            current_epoch, current_step))

    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])
    if opt['phase'] == 'train':
        pbar = None
        progress_bar_refresh = max(1, int(opt.get('train', {}).get('progress_bar_refresh', 10) or 10))
        progress_bar_mininterval = float(opt.get('train', {}).get('progress_bar_mininterval', 5.0) or 5.0)
        if opt.get('is_main_process', True) and tqdm is not None and bool(opt.get('train', {}).get('progress_bar', True)):
            pbar = tqdm(total=n_iter, initial=current_step, dynamic_ncols=True, mininterval=progress_bar_mininterval, desc='train')
        while current_step < n_iter:
            current_epoch += 1
            if hasattr(train_loader, 'sampler') and hasattr(train_loader.sampler, 'set_epoch'):
                train_loader.sampler.set_epoch(current_epoch)
            for _, train_data in enumerate(train_loader):
                current_step += 1
                if current_step > n_iter:
                    break
                diffusion.current_step = current_step
                diffusion.feed_data(train_data)
                optimize_ok = diffusion.optimize_parameters()
                if pbar is not None:
                    pbar.update(1)
                    if current_step % progress_bar_refresh == 0 or current_step == n_iter:
                        logs_for_bar = diffusion.get_current_log()
                        postfix = {'epoch': current_epoch}
                        for key in ('l_pix', 'grad_norm', 'skipped_step'):
                            if key in logs_for_bar:
                                try:
                                    postfix[key] = '{:.3g}'.format(float(logs_for_bar[key]))
                                except Exception:
                                    postfix[key] = logs_for_bar[key]
                        pbar.set_postfix(postfix, refresh=False)
                # log
                if current_step % opt['train']['print_freq'] == 0:
                    logs = _reduce_log_dict(diffusion.get_current_log())
                    if opt.get('is_main_process', True):
                        message = '<epoch:{:3d}, iter:{:8,d}> '.format(
                            current_epoch, current_step)
                        for k, v in logs.items():
                            message += '{:s}: {:.4e} '.format(k, v)
                            tb_logger.add_scalar(k, v, current_step)
                        logger.info(message)

                        if wandb_logger:
                            wandb_logger.log_metrics(logs)

                # validation
                if current_step % opt['train']['val_freq'] == 0:
                    if dist.is_available() and dist.is_initialized():
                        dist.barrier()
                    if opt.get('is_main_process', True):
                        psnr_vals = []
                        ssim_vals = []
                        lpips_vals = []
                        hfen_vals = []
                        frc_auc_vals = []
                        frc_hf_vals = []
                        frc_cut_vals = []
                        dmi_metric_keys = [
                            'masked_psnr', 'masked_mae', 'masked_ssim',
                            'roi_mean_abs_err', 'roi_mean_rel_err',
                            'roi_std_abs_err', 'roi_std_rel_err',
                            'roi_sum_abs_err', 'roi_sum_rel_err',
                            'roi_mean_abs_err_quantity',
                            'concentration_integral_abs_err', 'concentration_integral_rel_err',
                            'false_hotspot_rate', 'false_hotspot_precision_err',
                            'degradation_l1', 'degradation_rmse',
                            'native_acquisition_l1', 'native_acquisition_rmse',
                            'native_acquisition_mean_bias', 'native_acquisition_rel_l1',
                        ]
                        metric_rows = []
                        idx = 0
                        result_path = '{}/{}'.format(opt['path']
                                                     ['results'], current_epoch)
                        os.makedirs(result_path, exist_ok=True)
                        val_max_samples = int(opt['train'].get('val_max_samples', -1))
                        metrics_opt = opt.get('metrics', {}) or {}
                        val_opt = opt.get('validation', {}) or {}
                        frc_apodize = bool(metrics_opt.get('frc_apodize', True))
                        enable_hfen = bool(metrics_opt.get('enable_hfen', True))
                        enable_frc = bool(metrics_opt.get('enable_frc', True))
                        enable_dmi_metrics = bool(metrics_opt.get('enable_dmi_metrics', True))
                        val_sample_steps = int(val_opt.get('sample_num_steps', 0) or 0)
                        save_image_count = int(val_opt.get('save_image_count', -1))
                        eval_network = str(val_opt.get('eval_network', 'auto') or 'auto').lower()

                        diffusion.set_new_noise_schedule(
                            opt['model']['beta_schedule']['val'], schedule_phase='val')
                        for _,  val_data in enumerate(val_loader):
                            if val_max_samples > 0 and idx >= val_max_samples:
                                break
                            idx += 1
                            patient_id = _as_scalar(val_data, 'PATIENT_ID', -1)
                            slice_idx = _as_scalar(val_data, 'SLICE_IDX', -1)
                            met_id = _as_scalar(val_data, 'MET_ID', -1)
                            lowres = _as_scalar(val_data, 'LOWRES', -1)
                            lr_matrix = _as_scalar(val_data, 'LR_MATRIX', -1)
                            normalization_scale = _as_scalar(val_data, 'NORMALIZATION_SCALE', 1.0)
                            quantity_unit = _as_scalar(val_data, 'QUANTITY_UNIT', 'normalized_unit')
                            sample_id = _as_scalar(val_data, 'SAMPLE_ID', idx)
                            split_name = _as_scalar(val_data, 'SPLIT', 'val')
                            met_name = _as_scalar(val_data, 'MET_NAME', str(met_id))
                            sample_seed = _validation_seed(val_opt, sample_id, idx)
                            diffusion.feed_data(val_data)
                            diffusion.test(
                                continous=False,
                                seed=sample_seed,
                                sample_num_steps=(val_sample_steps if val_sample_steps > 0 else None),
                                network=eval_network,
                            )
                            visuals = diffusion.get_current_visuals()

                            sr_t = visuals['SR']
                            if sr_t.dim() == 4 and sr_t.shape[0] > 1:
                                sr_t = sr_t[-1:]
                            hr_t = visuals['HR']
                            lr_t = visuals['LR']
                            mask_t = None
                            if isinstance(diffusion.data, dict) and 'MASK' in diffusion.data:
                                mask_t = diffusion.data['MASK'].detach().float().cpu()
                            lr_native_t = None
                            if (
                                isinstance(diffusion.data, dict)
                                and 'LR_NATIVE' in diffusion.data
                                and float(_as_scalar(diffusion.data, 'HAS_NATIVE_LR', 0.0)) > 0.0
                            ):
                                lr_native_t = diffusion.data['LR_NATIVE'].detach().float().cpu()

                            sr_img = Metrics.tensor2img(sr_t)  # uint8
                            hr_img = Metrics.tensor2img(visuals['HR'])  # uint8
                            lr_img = Metrics.tensor2img(lr_t)  # uint8
                            if save_image_count < 0 or idx <= save_image_count:
                                err_img = _error_map_img(sr_t, hr_t)
                                Metrics.save_img(
                                    hr_img, '{}/{}_{}_hr.png'.format(result_path, current_step, idx))
                                Metrics.save_img(
                                    sr_img, '{}/{}_{}_sr.png'.format(result_path, current_step, idx))
                                Metrics.save_img(
                                    err_img, '{}/{}_{}_err.png'.format(result_path, current_step, idx))
                                Metrics.save_img(
                                    lr_img, '{}/{}_{}_lr.png'.format(result_path, current_step, idx))
                                tb_logger.add_image(
                                    'Iter_{}'.format(current_step),
                                    _concat_for_tb(lr_img, sr_img, hr_img),
                                    idx)

                            eval_psnr = Metrics.calculate_psnr(sr_img, hr_img)
                            eval_ssim = Metrics.calculate_ssim(sr_img, hr_img)
                            eval_masked_ssim = _masked_ssim_from_imgs(sr_img, hr_img, mask_t)
                            eval_lpips = _compute_lpips(lpips_model, sr_t, hr_t, mask_t=mask_t)
                            eval_hfen = _compute_hfen(sr_t, hr_t, mask_t=mask_t) if enable_hfen else None
                            if enable_frc:
                                eval_frc_auc, eval_frc_hf, eval_frc_cut = _compute_frc(
                                    sr_t, hr_t, mask_t=mask_t, apodize=frc_apodize)
                            else:
                                eval_frc_auc, eval_frc_hf, eval_frc_cut = None, None, None
                            dmi_metrics = (
                                _compute_dmi_metrics(
                                    sr_t,
                                    hr_t,
                                    lr_t,
                                    mask_t,
                                    lowres,
                                    lr_native_t=lr_native_t,
                                    lr_matrix=lr_matrix,
                                    normalization_scale=normalization_scale,
                                )
                                if enable_dmi_metrics else {}
                            )
                            if eval_masked_ssim is not None:
                                dmi_metrics['masked_ssim'] = eval_masked_ssim

                            psnr_vals.append(float(eval_psnr))
                            ssim_vals.append(float(eval_ssim))
                            if eval_lpips is not None and np.isfinite(eval_lpips):
                                lpips_vals.append(float(eval_lpips))
                            if eval_hfen is not None and np.isfinite(eval_hfen):
                                hfen_vals.append(float(eval_hfen))
                            if eval_frc_auc is not None and np.isfinite(eval_frc_auc):
                                frc_auc_vals.append(float(eval_frc_auc))
                            if eval_frc_hf is not None and np.isfinite(eval_frc_hf):
                                frc_hf_vals.append(float(eval_frc_hf))
                            if eval_frc_cut is not None and np.isfinite(eval_frc_cut):
                                frc_cut_vals.append(float(eval_frc_cut))
                            metric_row = {
                                'index': idx,
                                'patient_id': patient_id,
                                'slice_idx': slice_idx,
                                'met_id': met_id,
                                'sample_id': sample_id,
                                'split': split_name,
                                'met_name': met_name,
                                'lowres': lowres,
                                'lr_matrix': lr_matrix,
                                'normalization_scale': normalization_scale,
                                'quantity_unit': quantity_unit,
                                'eval_network': diffusion.get_eval_network_name(eval_network),
                                'sample_seed': sample_seed,
                                'sample_steps': (val_sample_steps if val_sample_steps > 0 else None),
                                'psnr': float(eval_psnr),
                                'ssim': float(eval_ssim),
                                'masked_ssim': (None if eval_masked_ssim is None else float(eval_masked_ssim)),
                                'lpips': (None if eval_lpips is None else float(eval_lpips)),
                                'hfen': (None if eval_hfen is None else float(eval_hfen)),
                                'frc_aucw': (None if eval_frc_auc is None else float(eval_frc_auc)),
                                'frc_hf': (None if eval_frc_hf is None else float(eval_frc_hf)),
                                'frc_cut': (None if eval_frc_cut is None else float(eval_frc_cut)),
                            }
                            metric_row.update({key: dmi_metrics.get(key) for key in dmi_metric_keys if key != 'masked_ssim'})
                            metric_rows.append(metric_row)

                            if wandb_logger:
                                wandb_logger.log_image(
                                    f'validation_{idx}',
                                    np.concatenate((_to_3ch_hwc(lr_img), _to_3ch_hwc(sr_img), _to_3ch_hwc(hr_img)), axis=1)
                                )

                        avg_psnr = _safe_mean(psnr_vals)
                        avg_ssim = _safe_mean(ssim_vals)
                        avg_lpips = _safe_mean(lpips_vals)
                        avg_hfen = _safe_mean(hfen_vals)
                        avg_frc_auc = _safe_mean(frc_auc_vals)
                        avg_frc_hf = _safe_mean(frc_hf_vals)
                        avg_frc_cut = _safe_mean(frc_cut_vals)
                        avg_dmi = {
                            key: _safe_mean([
                                float(row[key]) for row in metric_rows
                                if row.get(key) not in ('', 'None', None) and np.isfinite(float(row[key]))
                            ])
                            for key in dmi_metric_keys
                        }
                        diffusion.set_new_noise_schedule(
                            opt['model']['beta_schedule']['train'], schedule_phase='train')
                        summary_metrics = {
                            'PSNR': avg_psnr,
                            'SSIM': avg_ssim,
                            'LPIPS': avg_lpips,
                            'HFEN': avg_hfen,
                            'FRC_AUCw': avg_frc_auc,
                            'FRC_HF': avg_frc_hf,
                            'FRC_cut': avg_frc_cut,
                        }
                        logger.info(
                            '# Validation # network=%s %s',
                            diffusion.get_eval_network_name(eval_network),
                            ' '.join('{}: {}'.format(key, _metric_text(value)) for key, value in summary_metrics.items()),
                        )
                        logger_val.info(
                            '<epoch:{:3d}, iter:{:8,d}> network: {}, {}'.format(
                                current_epoch,
                                current_step,
                                diffusion.get_eval_network_name(eval_network),
                                ', '.join('{}: {}'.format(key.lower(), _metric_text(value)) for key, value in summary_metrics.items()),
                            )
                        )
                        tb_metrics = {
                            'psnr': avg_psnr,
                            'val/psnr': avg_psnr,
                            'val/ssim': avg_ssim,
                            'val/lpips': avg_lpips,
                            'val/hfen': avg_hfen,
                            'val/frc_aucw': avg_frc_auc,
                            'val/frc_hf': avg_frc_hf,
                            'val/frc_cut': avg_frc_cut,
                            **{f'val/{key}': value for key, value in avg_dmi.items()},
                        }
                        for key, value in _defined_metrics(tb_metrics).items():
                            tb_logger.add_scalar(key, value, current_step)

                        csv_path = os.path.join(result_path, f'{current_step}_metrics.csv')
                        json_path = os.path.join(result_path, f'{current_step}_metrics.json')
                        _write_metrics_csv(metric_rows, csv_path)
                        with open(json_path, 'w', encoding='utf-8') as f:
                            json.dump(
                                {
                                    'epoch': current_epoch,
                                    'iter': current_step,
                                    'count': idx,
                                    'eval_network': diffusion.get_eval_network_name(eval_network),
                                    'fixed_seed': val_opt.get('fixed_seed', None),
                                    'seed_mode': val_opt.get('seed_mode', 'sample_id'),
                                    'sample_num_steps': (val_sample_steps if val_sample_steps > 0 else None),
                                    'psnr': avg_psnr,
                                    'ssim': avg_ssim,
                                    'lpips': avg_lpips,
                                    'hfen': avg_hfen,
                                    'frc_aucw': avg_frc_auc,
                                    'frc_hf': avg_frc_hf,
                                    'frc_cut': avg_frc_cut,
                                    **avg_dmi,
                                },
                                f,
                                ensure_ascii=False,
                                indent=2,
                            )

                        if wandb_logger:
                            wandb_logger.log_metrics(_defined_metrics({
                                'validation/val_psnr': avg_psnr,
                                'validation/val_ssim': avg_ssim,
                                'validation/val_lpips': avg_lpips,
                                'validation/val_hfen': avg_hfen,
                                'validation/val_frc_aucw': avg_frc_auc,
                                'validation/val_frc_hf': avg_frc_hf,
                                'validation/val_frc_cut': avg_frc_cut,
                                **{f'validation/val_{key}': value for key, value in avg_dmi.items()},
                                'validation/val_step': val_step
                            }))
                            val_step += 1
                    if dist.is_available() and dist.is_initialized():
                        dist.barrier()

                if current_step % opt['train']['save_checkpoint_freq'] == 0:
                    if opt.get('is_main_process', True):
                        logger.info('Saving models and training states.')
                        diffusion.save_network(current_epoch, current_step)

                        if wandb_logger and opt['log_wandb_ckpt']:
                            wandb_logger.log_checkpoint(current_epoch, current_step)

            if wandb_logger and opt.get('is_main_process', True):
                wandb_logger.log_metrics({'epoch': current_epoch-1})

        if pbar is not None:
            pbar.close()
        # save model
        if opt.get('is_main_process', True):
            logger.info('End of training.')
    else:
        if opt.get('distributed', False) and not opt.get('is_main_process', True):
            _cleanup_dist()
            raise SystemExit(0)
        logger.info('Begin Model Evaluation.')
        psnr_vals = []
        ssim_vals = []
        lpips_vals = []
        hfen_vals = []
        frc_auc_vals = []
        frc_hf_vals = []
        frc_cut_vals = []
        dmi_metric_keys = [
            'masked_psnr', 'masked_mae', 'masked_ssim',
            'roi_mean_abs_err', 'roi_mean_rel_err',
            'roi_std_abs_err', 'roi_std_rel_err',
            'roi_sum_abs_err', 'roi_sum_rel_err',
            'roi_mean_abs_err_quantity',
            'concentration_integral_abs_err', 'concentration_integral_rel_err',
            'false_hotspot_rate', 'false_hotspot_precision_err',
            'degradation_l1', 'degradation_rmse',
            'native_acquisition_l1', 'native_acquisition_rmse',
            'native_acquisition_mean_bias', 'native_acquisition_rel_l1',
        ]
        metric_rows = []
        idx = 0
        result_path = '{}'.format(opt['path']['results'])
        os.makedirs(result_path, exist_ok=True)
        metrics_opt = opt.get('metrics', {}) or {}
        val_opt = opt.get('validation', {}) or {}
        frc_apodize = bool(metrics_opt.get('frc_apodize', True))
        enable_hfen = bool(metrics_opt.get('enable_hfen', True))
        enable_frc = bool(metrics_opt.get('enable_frc', True))
        enable_dmi_metrics = bool(metrics_opt.get('enable_dmi_metrics', True))
        val_sample_steps = int(val_opt.get('sample_num_steps', 0) or 0)
        save_process = bool(val_opt.get('save_process', False))
        save_image_count = int(val_opt.get('save_image_count', -1))
        eval_network = str(val_opt.get('eval_network', 'auto') or 'auto').lower()
        for _,  val_data in enumerate(val_loader):
            idx += 1
            patient_id = _as_scalar(val_data, 'PATIENT_ID', -1)
            slice_idx = _as_scalar(val_data, 'SLICE_IDX', -1)
            met_id = _as_scalar(val_data, 'MET_ID', -1)
            lowres = _as_scalar(val_data, 'LOWRES', -1)
            lr_matrix = _as_scalar(val_data, 'LR_MATRIX', -1)
            normalization_scale = _as_scalar(val_data, 'NORMALIZATION_SCALE', 1.0)
            quantity_unit = _as_scalar(val_data, 'QUANTITY_UNIT', 'normalized_unit')
            sample_id = _as_scalar(val_data, 'SAMPLE_ID', idx)
            split_name = _as_scalar(val_data, 'SPLIT', 'val')
            met_name = _as_scalar(val_data, 'MET_NAME', str(met_id))
            sample_seed = _validation_seed(val_opt, sample_id, idx)
            diffusion.feed_data(val_data)
            diffusion.test(
                continous=save_process,
                seed=sample_seed,
                sample_num_steps=(val_sample_steps if val_sample_steps > 0 else None),
                network=eval_network,
            )
            visuals = diffusion.get_current_visuals()

            hr_img = Metrics.tensor2img(visuals['HR'])  # uint8
            lr_img = Metrics.tensor2img(visuals['LR'])  # uint8
            cond_img = lr_img  # use LR as condition visualization for multi-channel inputs

            sr_img_mode = 'grid' if save_process else 'single'
            if sr_img_mode == 'single':
                sr_img = visuals['SR']
                sample_num = sr_img.shape[0]
                if save_image_count < 0 or idx <= save_image_count:
                    iter = sample_num - 1
                    Metrics.save_img(
                        Metrics.tensor2img(sr_img[iter]), '{}/{}_{}_sr_{}.png'.format(result_path, current_step, idx, iter))
            else:
                # grid img
                sr_img = Metrics.tensor2img(visuals['SR'])  # uint8
                if save_image_count < 0 or idx <= save_image_count:
                    Metrics.save_img(
                        sr_img, '{}/{}_{}_sr_process.png'.format(result_path, current_step, idx))
                    Metrics.save_img(
                        Metrics.tensor2img(visuals['SR'][-1]), '{}/{}_{}_sr.png'.format(result_path, current_step, idx))

            if save_image_count < 0 or idx <= save_image_count:
                Metrics.save_img(
                    hr_img, '{}/{}_{}_hr.png'.format(result_path, current_step, idx))
                Metrics.save_img(
                    lr_img, '{}/{}_{}_lr.png'.format(result_path, current_step, idx))

            # generation
            sr_last_t = visuals['SR'][-1]
            sr_last_img = Metrics.tensor2img(sr_last_t)
            err_img = _error_map_img(sr_last_t, visuals['HR'])
            if save_image_count < 0 or idx <= save_image_count:
                Metrics.save_img(
                    err_img, '{}/{}_{}_err.png'.format(result_path, current_step, idx))
            mask_t = None
            if isinstance(diffusion.data, dict) and 'MASK' in diffusion.data:
                mask_t = diffusion.data['MASK'].detach().float().cpu()
            lr_native_t = None
            if (
                isinstance(diffusion.data, dict)
                and 'LR_NATIVE' in diffusion.data
                and float(_as_scalar(diffusion.data, 'HAS_NATIVE_LR', 0.0)) > 0.0
            ):
                lr_native_t = diffusion.data['LR_NATIVE'].detach().float().cpu()

            eval_psnr = Metrics.calculate_psnr(sr_last_img, hr_img)
            eval_ssim = Metrics.calculate_ssim(sr_last_img, hr_img)
            eval_masked_ssim = _masked_ssim_from_imgs(sr_last_img, hr_img, mask_t)
            eval_lpips = _compute_lpips(lpips_model, sr_last_t, visuals['HR'], mask_t=mask_t)
            eval_hfen = _compute_hfen(sr_last_t, visuals['HR'], mask_t=mask_t) if enable_hfen else None
            if enable_frc:
                eval_frc_auc, eval_frc_hf, eval_frc_cut = _compute_frc(
                    sr_last_t, visuals['HR'], mask_t=mask_t, apodize=frc_apodize
                )
            else:
                eval_frc_auc, eval_frc_hf, eval_frc_cut = None, None, None
            dmi_metrics = (
                _compute_dmi_metrics(
                    sr_last_t,
                    visuals['HR'],
                    visuals['LR'],
                    mask_t,
                    lowres,
                    lr_native_t=lr_native_t,
                    lr_matrix=lr_matrix,
                    normalization_scale=normalization_scale,
                )
                if enable_dmi_metrics else {}
            )
            if eval_masked_ssim is not None:
                dmi_metrics['masked_ssim'] = eval_masked_ssim

            psnr_vals.append(float(eval_psnr))
            ssim_vals.append(float(eval_ssim))
            if eval_lpips is not None and np.isfinite(eval_lpips):
                lpips_vals.append(float(eval_lpips))
            if eval_hfen is not None and np.isfinite(eval_hfen):
                hfen_vals.append(float(eval_hfen))
            if eval_frc_auc is not None and np.isfinite(eval_frc_auc):
                frc_auc_vals.append(float(eval_frc_auc))
            if eval_frc_hf is not None and np.isfinite(eval_frc_hf):
                frc_hf_vals.append(float(eval_frc_hf))
            if eval_frc_cut is not None and np.isfinite(eval_frc_cut):
                frc_cut_vals.append(float(eval_frc_cut))
            metric_row = {
                'index': idx,
                'patient_id': patient_id,
                'slice_idx': slice_idx,
                'met_id': met_id,
                'sample_id': sample_id,
                'split': split_name,
                'met_name': met_name,
                'lowres': lowres,
                'lr_matrix': lr_matrix,
                'normalization_scale': normalization_scale,
                'quantity_unit': quantity_unit,
                'eval_network': diffusion.get_eval_network_name(eval_network),
                'sample_seed': sample_seed,
                'sample_steps': (val_sample_steps if val_sample_steps > 0 else None),
                'psnr': float(eval_psnr),
                'ssim': float(eval_ssim),
                'masked_ssim': (None if eval_masked_ssim is None else float(eval_masked_ssim)),
                'lpips': (None if eval_lpips is None else float(eval_lpips)),
                'hfen': (None if eval_hfen is None else float(eval_hfen)),
                'frc_aucw': (None if eval_frc_auc is None else float(eval_frc_auc)),
                'frc_hf': (None if eval_frc_hf is None else float(eval_frc_hf)),
                'frc_cut': (None if eval_frc_cut is None else float(eval_frc_cut)),
            }
            metric_row.update({key: dmi_metrics.get(key) for key in dmi_metric_keys if key != 'masked_ssim'})
            metric_rows.append(metric_row)

            if wandb_logger and opt['log_eval']:
                wandb_logger.log_eval_data(cond_img, sr_last_img, hr_img, eval_psnr, eval_ssim)

        avg_psnr = _safe_mean(psnr_vals)
        avg_ssim = _safe_mean(ssim_vals)
        avg_lpips = _safe_mean(lpips_vals)
        avg_hfen = _safe_mean(hfen_vals)
        avg_frc_auc = _safe_mean(frc_auc_vals)
        avg_frc_hf = _safe_mean(frc_hf_vals)
        avg_frc_cut = _safe_mean(frc_cut_vals)
        avg_dmi = {
            key: _safe_mean([
                float(row[key]) for row in metric_rows
                if row.get(key) not in ('', 'None', None) and np.isfinite(float(row[key]))
            ])
            for key in dmi_metric_keys
        }

        # log
        summary_metrics = {
            'PSNR': avg_psnr,
            'SSIM': avg_ssim,
            'LPIPS': avg_lpips,
            'HFEN': avg_hfen,
            'FRC_AUCw': avg_frc_auc,
            'FRC_HF': avg_frc_hf,
            'FRC_cut': avg_frc_cut,
        }
        logger.info(
            '# Validation # network=%s %s',
            diffusion.get_eval_network_name(eval_network),
            ' '.join('{}: {}'.format(key, _metric_text(value)) for key, value in summary_metrics.items()),
        )
        logger.info(
            '# Validation # masked_PSNR: %s ROI_mean_rel_err: %s false_hotspot_rate: %s degradation_l1: %s',
            _metric_text(avg_dmi.get('masked_psnr')),
            _metric_text(avg_dmi.get('roi_mean_rel_err')),
            _metric_text(avg_dmi.get('false_hotspot_rate')),
            _metric_text(avg_dmi.get('degradation_l1')),
        )
        logger_val = logging.getLogger('val')  # validation logger
        logger_val.info(
            '<epoch:{:3d}, iter:{:8,d}> network: {}, {}'.format(
                current_epoch,
                current_step,
                diffusion.get_eval_network_name(eval_network),
                ', '.join('{}: {}'.format(key.lower(), _metric_text(value)) for key, value in summary_metrics.items()),
            )
        )

        csv_path = os.path.join(result_path, 'metrics.csv')
        json_path = os.path.join(result_path, 'metrics.json')
        _write_metrics_csv(metric_rows, csv_path)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(
                {
                    'epoch': current_epoch,
                    'iter': current_step,
                    'count': idx,
                    'eval_network': diffusion.get_eval_network_name(eval_network),
                    'fixed_seed': val_opt.get('fixed_seed', None),
                    'seed_mode': val_opt.get('seed_mode', 'sample_id'),
                    'sample_num_steps': (val_sample_steps if val_sample_steps > 0 else None),
                    'psnr': avg_psnr,
                    'ssim': avg_ssim,
                    'lpips': avg_lpips,
                    'hfen': avg_hfen,
                    'frc_aucw': avg_frc_auc,
                    'frc_hf': avg_frc_hf,
                    'frc_cut': avg_frc_cut,
                    **avg_dmi,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        if wandb_logger:
            if opt['log_eval']:
                wandb_logger.log_eval_table()
            wandb_logger.log_metrics(_defined_metrics({
                'PSNR': avg_psnr,
                'SSIM': avg_ssim,
                'LPIPS': avg_lpips,
                'HFEN': avg_hfen,
                'FRC_AUCw': avg_frc_auc,
                'FRC_HF': avg_frc_hf,
                'FRC_cut': avg_frc_cut,
                **avg_dmi,
            }))
    _cleanup_dist()
