'''create dataset and dataloader'''
import logging
import math
from re import split
import torch
import torch.utils.data
from torch.utils.data.distributed import DistributedSampler


def create_dataloader(dataset, dataset_opt, phase):
    '''create dataloader '''
    sampler = None
    if torch.distributed.is_available() and torch.distributed.is_initialized() and phase == 'train':
        sampler = DistributedSampler(
            dataset,
            num_replicas=torch.distributed.get_world_size(),
            rank=torch.distributed.get_rank(),
            shuffle=bool(dataset_opt['use_shuffle'])
        )
        global_batch_size = int(dataset_opt['batch_size'])
        world_size = int(torch.distributed.get_world_size())
        batch_size = max(1, int(math.ceil(float(global_batch_size) / float(world_size))))
        if global_batch_size % world_size != 0 and torch.distributed.get_rank() == 0:
            logging.getLogger('base').warning(
                'Train batch_size (%d) is not divisible by world_size (%d); '
                'using per-rank batch_size=%d (effective global batch ~= %d).',
                global_batch_size, world_size, batch_size, batch_size * world_size
            )
    else:
        batch_size = dataset_opt.get('batch_size', 1)
    if phase == 'train':
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(sampler is None and dataset_opt['use_shuffle']),
            sampler=sampler,
            num_workers=dataset_opt['num_workers'],
            pin_memory=True)
    elif phase in ['val', 'test']:
        return torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
    else:
        raise NotImplementedError(
            'Dataloader [{:s}] is not found.'.format(phase))


def create_dataset(dataset_opt, phase):
    '''create dataset'''
    mode = dataset_opt['mode']
    if mode in ['HR', 'LRHR']:
        from data.LRHR_dataset import LRHRDataset as D
        dataset = D(
            dataroot=dataset_opt['dataroot'],
            datatype=dataset_opt['datatype'],
            l_resolution=dataset_opt['l_resolution'],
            r_resolution=dataset_opt['r_resolution'],
            split=phase,
            data_len=dataset_opt['data_len'],
            need_LR=(mode == 'LRHR')
        )
    elif mode == 'MRSI_SR3':
        from data.MRSI_SR3_dataset import MRSISR3Dataset as D
        dataset = D(
            dataroot=dataset_opt['dataroot'],
            split=phase,
            data_len=dataset_opt.get('data_len', -1),
            hflip=dataset_opt.get('hflip', True),
            vflip=dataset_opt.get('vflip', False),
        )
    else:
        raise NotImplementedError('Dataset mode [{:s}] is not found.'.format(mode))
    logger = logging.getLogger('base')
    logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__,
                                                           dataset_opt['name']))
    return dataset
