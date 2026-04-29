import os
import torch
import torch.nn as nn


class BaseModel():
    def __init__(self, opt):
        self.opt = opt
        if opt.get('gpu_ids'):
            if opt.get('distributed', False):
                self.device = torch.device('cuda:{}'.format(int(opt.get('local_rank', 0))))
            else:
                self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.begin_step = 0
        self.begin_epoch = 0

    def feed_data(self, data):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        pass

    def get_current_losses(self):
        pass

    def print_network(self):
        pass

    def set_device(self, x):
        if torch.is_tensor(x):
            return x.to(self.device)
        if isinstance(x, dict):
            for key, item in x.items():
                x[key] = self.set_device(item)
        elif isinstance(x, list):
            for i, item in enumerate(x):
                x[i] = self.set_device(item)
        elif isinstance(x, tuple):
            x = tuple(self.set_device(item) for item in x)
        else:
            # Keep metadata such as strings/ints/lists from DataLoader on CPU.
            return x
        return x

    def get_network_description(self, network):
        '''Get the string and total parameters of the network'''
        if hasattr(network, 'module'):
            network = network.module
        s = str(network)
        n = sum(map(lambda x: x.numel(), network.parameters()))
        return s, n
