import functools
import logging
import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import modules
from torch.nn.parallel import DistributedDataParallel
logger = logging.getLogger('base')
####################
# initialize
####################


def weights_init_normal(m, std=0.02):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, std)  # BN also uses norm
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m, scale=1):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='kaiming', scale=1, std=0.02):
    # scale for 'kaiming', std for 'normal'.
    logger.info('Initialization method [{:s}]'.format(init_type))
    if init_type == 'normal':
        weights_init_normal_ = functools.partial(weights_init_normal, std=std)
        net.apply(weights_init_normal_)
    elif init_type == 'kaiming':
        weights_init_kaiming_ = functools.partial(
            weights_init_kaiming, scale=scale)
        net.apply(weights_init_kaiming_)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError(
            'initialization method [{:s}] not implemented'.format(init_type))


####################
# define network
####################


# Generator
def _resolve_mrsi_in_channel(opt):
    datasets_opt = opt.get('datasets', {})
    target_dataset_opt = None
    for phase_name in ['train', 'val', 'test']:
        dataset_opt = datasets_opt.get(phase_name)
        if dataset_opt and dataset_opt.get('mode') == 'MRSI_SR3':
            target_dataset_opt = dataset_opt
            break
    if target_dataset_opt is None:
        return None

    cond_channels = 0
    if target_dataset_opt.get('use_lr', True):
        cond_channels += 1
    if target_dataset_opt.get('use_t1', True):
        cond_channels += 1
    if target_dataset_opt.get('use_flair', True):
        cond_channels += 1
    if target_dataset_opt.get('use_met_onehot', True):
        cond_channels += 4

    target_channels = int(opt['model']['diffusion']['channels'])
    return cond_channels + target_channels


def define_G(opt):
    model_opt = opt['model']
    if model_opt['which_model_G'] == 'ddpm':
        from .ddpm_modules import diffusion, unet
    elif model_opt['which_model_G'] == 'sr3':
        from .sr3_modules import diffusion, unet
    if ('norm_groups' not in model_opt['unet']) or model_opt['unet']['norm_groups'] is None:
        model_opt['unet']['norm_groups']=32
    if ('in_channel' not in model_opt['unet']) or model_opt['unet']['in_channel'] in [None, 0]:
        resolved_in_channel = _resolve_mrsi_in_channel(opt)
        if resolved_in_channel is not None:
            model_opt['unet']['in_channel'] = resolved_in_channel
        else:
            raise ValueError('model.unet.in_channel must be provided for non-MRSI datasets.')
    model = unet.UNet(
        in_channel=model_opt['unet']['in_channel'],
        out_channel=model_opt['unet']['out_channel'],
        norm_groups=model_opt['unet']['norm_groups'],
        inner_channel=model_opt['unet']['inner_channel'],
        channel_mults=model_opt['unet']['channel_multiplier'],
        attn_res=model_opt['unet']['attn_res'],
        res_blocks=model_opt['unet']['res_blocks'],
        dropout=model_opt['unet']['dropout'],
        image_size=model_opt['diffusion']['image_size']
    )
    netG = diffusion.GaussianDiffusion(
        model,
        image_size=model_opt['diffusion']['image_size'],
        channels=model_opt['diffusion']['channels'],
        loss_type='l1',    # L1 or L2
        conditional=model_opt['diffusion']['conditional'],
        schedule_opt=model_opt['beta_schedule']['train'],
        sampler_type=model_opt['diffusion'].get('sampler_type', 'ddpm'),
        sample_num_steps=model_opt['diffusion'].get('sample_num_steps')
    )
    if opt['phase'] == 'train':
        # init_weights(netG, init_type='kaiming', scale=0.1)
        init_weights(netG, init_type='orthogonal')
    if opt.get('gpu_ids') and opt.get('distributed'):
        assert torch.cuda.is_available()
        local_rank = int(opt.get('local_rank', 0))
        device = torch.device('cuda:{}'.format(local_rank))
        netG = netG.to(device)
        netG = DistributedDataParallel(
            netG,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False
        )
    elif opt.get('gpu_ids') and (not opt.get('distributed', False)):
        # Windows 上 DDP 可能不可用时，退化为 DataParallel（单进程，多 GPU）。
        gpu_ids = [int(x) for x in opt.get('gpu_ids', [])]
        if len(gpu_ids) > 1 and torch.cuda.is_available():
            # 由于 Logger.parse 会设置 CUDA_VISIBLE_DEVICES，这里的 device_ids 需要按“可见设备序号”处理。
            device_ids = list(range(len(gpu_ids)))
            netG = netG.to(torch.device('cuda:{}'.format(device_ids[0])))
            netG = nn.DataParallel(netG, device_ids=device_ids, output_device=device_ids[0])
    return netG
