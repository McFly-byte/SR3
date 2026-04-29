import logging
from collections import OrderedDict
import math
import copy
import json

import torch
import torch.nn as nn
import os
import model.networks as networks
from .base_model import BaseModel
logger = logging.getLogger('base')


def _unwrap_module(module):
    return module.module if hasattr(module, 'module') else module


class DDPM(BaseModel):
    def __init__(self, opt):
        super(DDPM, self).__init__(opt)
        # define network and load pretrained models
        raw_net = networks.define_G(opt)
        if opt.get('distributed', False):
            self.netG = raw_net
        else:
            self.netG = self.set_device(raw_net)
        self.schedule_phase = None
        self.netG_EMA = None

        # set loss and load resume state
        self.set_loss()
        self.set_new_noise_schedule(
            opt['model']['beta_schedule']['train'], schedule_phase='train')
        self.grad_clip_norm = float(opt.get('train', {}).get('grad_clip_norm', 0.0) or 0.0)
        ema_cfg = opt.get('train', {}).get('ema_scheduler', {}) or {}
        self.use_ema = bool(ema_cfg)
        self.step_start_ema = int(ema_cfg.get('step_start_ema', 0) or 0)
        self.update_ema_every = int(ema_cfg.get('update_ema_every', 1) or 1)
        self.ema_decay = float(ema_cfg.get('ema_decay', 0.9999) or 0.9999)
        if self.use_ema:
            self.netG_EMA = copy.deepcopy(_unwrap_module(self.netG)).to(self.device)
        if self.opt['phase'] == 'train':
            self.netG.train()
            # find the parameters to optimize
            if opt['model']['finetune_norm']:
                optim_params = []
                for k, v in self.netG.named_parameters():
                    v.requires_grad = False
                    if k.find('transformer') >= 0:
                        v.requires_grad = True
                        v.data.zero_()
                        optim_params.append(v)
                        logger.info(
                            'Params [{:s}] initialized to 0 and will optimize.'.format(k))
            else:
                optim_params = list(self.netG.parameters())

            self.optG = torch.optim.Adam(
                optim_params, lr=opt['train']["optimizer"]["lr"])
            self.log_dict = OrderedDict()
        self.load_network()
        if self.use_ema:
            self.netG_EMA.eval()
            for param in self.netG_EMA.parameters():
                param.requires_grad = False
        self.print_network()

    def feed_data(self, data):
        self.data = self.set_device(data)

    def optimize_parameters(self):
        self.optG.zero_grad()
        l_pix = self.netG(self.data)
        # need to average in multi-gpu
        b, c, h, w = self.data['HR'].shape
        l_pix = l_pix.sum() / int(b * c * h * w)
        if not torch.isfinite(l_pix):
            logger.warning('Non-finite training loss detected; skip optimizer step.')
            self.log_dict['l_pix'] = float('nan')
            self.log_dict['skipped_step'] = 1.0
            if self.grad_clip_norm > 0:
                self.log_dict['grad_norm'] = float('nan')
            return False

        l_pix.backward()
        if self.grad_clip_norm > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.netG.parameters(), max_norm=self.grad_clip_norm)
            grad_norm_val = float(grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm)
            self.log_dict['grad_norm'] = grad_norm_val
            if not math.isfinite(grad_norm_val):
                logger.warning('Non-finite gradient norm detected after clipping; skip optimizer step.')
                self.optG.zero_grad()
                self.log_dict['l_pix'] = float('nan')
                self.log_dict['skipped_step'] = 1.0
                return False
        self.optG.step()
        if self.use_ema and (getattr(self, 'current_step', 0) % self.update_ema_every == 0):
            self.update_ema()

        # set log
        self.log_dict['l_pix'] = l_pix.item()
        net = _unwrap_module(self.netG)
        for key, value in getattr(net, 'last_loss_dict', {}).items():
            self.log_dict[key] = float(value)
        self.log_dict['skipped_step'] = 0.0
        return True

    def _get_eval_network(self):
        if self.netG_EMA is not None:
            return self.netG_EMA
        return self.netG

    def update_ema(self):
        if self.netG_EMA is None:
            return
        source_net = _unwrap_module(self.netG)
        target_net = _unwrap_module(self.netG_EMA)
        source_state = source_net.state_dict()
        target_state = target_net.state_dict()
        current_step = getattr(self, 'current_step', 0)
        if current_step < self.step_start_ema:
            target_net.load_state_dict(source_state)
            return
        for key, param in source_state.items():
            target_state[key].mul_(self.ema_decay).add_(param.detach(), alpha=1.0 - self.ema_decay)

    def test(self, continous=False, seed=None, sample_num_steps=None):
        eval_net = self._get_eval_network()
        eval_net.eval()
        with torch.no_grad():
            if hasattr(eval_net, 'module'):
                self.SR = eval_net.module.super_resolution(
                    self.data['SR'], continous, seed=seed, sample_num_steps=sample_num_steps)
            else:
                self.SR = eval_net.super_resolution(
                    self.data['SR'], continous, seed=seed, sample_num_steps=sample_num_steps)
        self.netG.train()

    def sample(self, batch_size=1, continous=False, seed=None, sample_num_steps=None):
        eval_net = self._get_eval_network()
        eval_net.eval()
        with torch.no_grad():
            if hasattr(eval_net, 'module'):
                self.SR = eval_net.module.sample(batch_size, continous, seed=seed, sample_num_steps=sample_num_steps)
            else:
                self.SR = eval_net.sample(batch_size, continous, seed=seed, sample_num_steps=sample_num_steps)
        self.netG.train()

    def set_loss(self):
        if hasattr(self.netG, 'module'):
            self.netG.module.set_loss(self.device)
        else:
            self.netG.set_loss(self.device)

    def set_new_noise_schedule(self, schedule_opt, schedule_phase='train'):
        if self.schedule_phase is None or self.schedule_phase != schedule_phase:
            self.schedule_phase = schedule_phase
            if hasattr(self.netG, 'module'):
                self.netG.module.set_new_noise_schedule(
                    schedule_opt, self.device)
            else:
                self.netG.set_new_noise_schedule(schedule_opt, self.device)
            if self.netG_EMA is not None:
                if hasattr(self.netG_EMA, 'module'):
                    self.netG_EMA.module.set_new_noise_schedule(schedule_opt, self.device)
                else:
                    self.netG_EMA.set_new_noise_schedule(schedule_opt, self.device)

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_LR=True, sample=False):
        out_dict = OrderedDict()
        if sample:
            out_dict['SAM'] = self.SR.detach().float().cpu()
        else:
            out_dict['SR'] = self.SR.detach().float().cpu()
            out_dict['INF'] = self.data['SR'].detach().float().cpu()
            out_dict['HR'] = self.data['HR'].detach().float().cpu()
            if need_LR and 'LR' in self.data:
                out_dict['LR'] = self.data['LR'].detach().float().cpu()
            else:
                out_dict['LR'] = out_dict['INF']
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if hasattr(self.netG, 'module'):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)

        logger.info(
            'Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        logger.info(s)

    def save_network(self, epoch, iter_step, label=None, save_optimizer=True, extra_state=None):
        if label:
            gen_name = '{}_gen.pth'.format(label)
            opt_name = '{}_opt.pth'.format(label)
            ema_name = '{}_ema_gen.pth'.format(label)
        else:
            gen_name = 'I{}_E{}_gen.pth'.format(iter_step, epoch)
            opt_name = 'I{}_E{}_opt.pth'.format(iter_step, epoch)
            ema_name = 'I{}_E{}_ema_gen.pth'.format(iter_step, epoch)
        gen_path = os.path.join(self.opt['path']['checkpoint'], gen_name)
        opt_path = os.path.join(self.opt['path']['checkpoint'], opt_name)
        ema_path = os.path.join(self.opt['path']['checkpoint'], ema_name)
        # gen
        network = _unwrap_module(self.netG)
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, gen_path)
        if self.netG_EMA is not None:
            ema_network = _unwrap_module(self.netG_EMA)
            ema_state_dict = ema_network.state_dict()
            for key, param in ema_state_dict.items():
                ema_state_dict[key] = param.cpu()
            torch.save(ema_state_dict, ema_path)
        # opt
        if save_optimizer and hasattr(self, 'optG'):
            opt_state = {'epoch': epoch, 'iter': iter_step,
                         'scheduler': None, 'optimizer': None}
            opt_state['optimizer'] = self.optG.state_dict()
            torch.save(opt_state, opt_path)
        if extra_state is not None:
            with open(os.path.join(self.opt['path']['checkpoint'], '{}_state.json'.format(label or 'last')), 'w', encoding='utf-8') as f:
                json.dump(extra_state, f, indent=2, ensure_ascii=False)

        logger.info(
            'Saved model in [{:s}] ...'.format(gen_path))

    def load_network(self):
        load_path = self.opt['path']['resume_state']
        if load_path is not None:
            logger.info(
                'Loading pretrained model for G [{:s}] ...'.format(load_path))
            gen_path = '{}_gen.pth'.format(load_path)
            opt_path = '{}_opt.pth'.format(load_path)
            # gen
            network = _unwrap_module(self.netG)
            network.load_state_dict(
                torch.load(gen_path, map_location=self.device),
                strict=(not self.opt['model']['finetune_norm'])
            )
            ema_path = '{}_ema_gen.pth'.format(load_path)
            if self.netG_EMA is not None and os.path.exists(ema_path):
                ema_network = _unwrap_module(self.netG_EMA)
                ema_network.load_state_dict(
                    torch.load(ema_path, map_location=self.device),
                    strict=(not self.opt['model']['finetune_norm'])
                )
            # network.load_state_dict(torch.load(
            #     gen_path), strict=False)
            if self.opt['phase'] == 'train':
                # optimizer
                opt = torch.load(opt_path, map_location=self.device)
                self.optG.load_state_dict(opt['optimizer'])
                for state in self.optG.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.to(self.device)
                self.begin_step = opt['iter']
                self.begin_epoch = opt['epoch']
