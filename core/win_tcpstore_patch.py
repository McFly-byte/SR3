"""Windows 官方 PyTorch wheel 常无 libuv；TCPStore 默认仍会走 libuv 路径。

rendezvous 在 import 时已绑定 TCPStore，必须在各模块命名空间上替换 TCPStore 符号。"""
import os


def apply_tcpstore_no_libuv_patch():
    if os.name != 'nt':
        return
    try:
        import torch.distributed as dist
        import torch.distributed.rendezvous as rdzv
        _real = dist.TCPStore
    except Exception:
        return

    def _TCPStore_no_libuv(*args, **kwargs):
        kwargs = dict(kwargs)
        kwargs['use_libuv'] = False
        try:
            return _real(*args, **kwargs)
        except TypeError:
            kwargs.pop('use_libuv', None)
            return _real(*args, **kwargs)

    dist.TCPStore = _TCPStore_no_libuv
    rdzv.TCPStore = _TCPStore_no_libuv
    try:
        import torch.distributed.elastic.rendezvous.static_tcp_rendezvous as stcp
        stcp.TCPStore = _TCPStore_no_libuv
    except Exception:
        pass
