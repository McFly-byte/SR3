"""Windows TCPStore helper for PyTorch builds without libuv support.

Some Windows PyTorch wheels default to libuv-backed TCPStore even though they
were built without libuv. Patch every module-level TCPStore binding used by
``torchrun`` and ``init_process_group(env://)`` so DDP can start with the
classic TCPStore implementation.
"""
import importlib
import os


def apply_tcpstore_no_libuv_patch():
    if os.name != 'nt':
        return
    try:
        import torch.distributed as dist
        real_tcp_store = dist.TCPStore
    except Exception:
        return

    def _TCPStore_no_libuv(*args, **kwargs):
        kwargs = dict(kwargs)
        kwargs['use_libuv'] = False
        try:
            return real_tcp_store(*args, **kwargs)
        except TypeError:
            kwargs.pop('use_libuv', None)
            return real_tcp_store(*args, **kwargs)

    dist.TCPStore = _TCPStore_no_libuv

    module_names = [
        'torch.distributed.rendezvous',
        'torch.distributed.elastic.rendezvous.c10d_rendezvous_backend',
        'torch.distributed.elastic.rendezvous.static_tcp_rendezvous',
        'torch.distributed.elastic.rendezvous.dynamic_rendezvous',
        'torch.distributed.elastic.utils.distributed',
    ]
    for module_name in module_names:
        try:
            module = importlib.import_module(module_name)
            setattr(module, 'TCPStore', _TCPStore_no_libuv)
        except Exception:
            pass
