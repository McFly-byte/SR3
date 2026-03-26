import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
from core.wandb_logger import WandbLogger
from tensorboardX import SummaryWriter
import os


def _to_3ch_hwc(img_np):
    if img_np.ndim == 2:
        return img_np[:, :, None].repeat(3, axis=2)
    if img_np.ndim == 3 and img_np.shape[2] == 1:
        return img_np.repeat(3, axis=2)
    if img_np.ndim == 3 and img_np.shape[2] >= 3:
        return img_np[:, :, :3]
    raise ValueError(f'Unsupported image shape: {img_np.shape}')


def _concat_triplet(left, mid, right):
    import numpy as np
    return np.concatenate((_to_3ch_hwc(left), _to_3ch_hwc(mid), _to_3ch_hwc(right)), axis=1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/sr_sr3_64_512.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['val'], help='val(generation)', default='val')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_infer', action='store_true')
    parser.add_argument('--eval_split', type=str, choices=['val', 'test'], default=None)
    
    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(None, opt['path']['log'],
                        'train', level=logging.INFO, screen=True)
    Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))
    tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

    # Initialize WandbLogger
    if opt['enable_wandb']:
        wandb_logger = WandbLogger(opt)
    else:
        wandb_logger = None

    seed = int(opt.get('train', {}).get('seed', 0) or 0)
    deterministic = bool(opt.get('train', {}).get('deterministic', False))
    Logger.set_random_seed(seed, deterministic=deterministic)

    eval_loaders = {}
    for phase, dataset_opt in opt['datasets'].items():
        if phase in ['val', 'test']:
            dataset = Data.create_dataset(dataset_opt, phase)
            eval_loaders[phase] = Data.create_dataloader(dataset, dataset_opt, phase)
    logger.info('Initial Dataset Finished')

    # model
    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')

    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule']['val'], schedule_phase='val')
    
    logger.info('Begin Model Inference.')
    current_step = 0
    idx = 0
    validation_opt = opt.get('validation', {}) or {}
    eval_split = args.eval_split or validation_opt.get('split', 'val')
    val_loader = eval_loaders.get(eval_split)
    if val_loader is None:
        raise ValueError(f'Evaluation split [{eval_split}] is not configured.')
    save_indices = validation_opt.get('save_image_indices')
    if save_indices is None:
        save_indices = list(range(int(validation_opt.get('save_image_count', 4) or 0)))
    save_indices = set(int(v) for v in save_indices)
    save_process = bool(validation_opt.get('save_process', False))
    fixed_seed = validation_opt.get('fixed_seed')
    sample_num_steps = validation_opt.get('sample_num_steps')

    result_path = os.path.join(opt['path']['results'], eval_split)
    os.makedirs(result_path, exist_ok=True)
    for sample_idx, val_data in enumerate(val_loader):
        if save_indices and sample_idx not in save_indices:
            continue
        idx += 1
        diffusion.feed_data(val_data)
        sample_seed = None if fixed_seed is None else int(fixed_seed) + int(sample_idx)
        diffusion.test(continous=True, seed=sample_seed, sample_num_steps=sample_num_steps)
        visuals = diffusion.get_current_visuals(need_LR=True)

        hr_img = Metrics.tensor2img(visuals['HR'])  # uint8
        lr_img = Metrics.tensor2img(visuals['LR'])  # uint8

        sr_img = Metrics.tensor2img(visuals['SR'])  # uint8
        Metrics.save_img(
            _concat_triplet(lr_img, Metrics.tensor2img(visuals['SR'][-1]), hr_img),
            '{}/{}_{}_vis.png'.format(result_path, current_step, idx)
        )
        if save_process:
            Metrics.save_img(
                sr_img, '{}/{}_{}_sr_process.png'.format(result_path, current_step, idx))

        if wandb_logger and opt['log_infer']:
            wandb_logger.log_eval_data(lr_img, Metrics.tensor2img(visuals['SR'][-1]), hr_img)

    if wandb_logger and opt['log_infer']:
        wandb_logger.log_eval_table(commit=True)
