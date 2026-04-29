#!/usr/bin/env python
"""
Windows 单机多卡：默认不再使用 torchrun（C10d/libuv + 主机名问题多），
而是转调 train_windows_3gpu.py（单进程 nn.DataParallel，与配置里 gpu_ids 一致）。

你仍可使用：
    python train_windows_3gpu.py -p train -c config/sr3_mrsi_64.json

若坚持使用 torchrun + DDP，请设置环境变量 USE_WINDOWS_TORCHRUN_DDP=1（需本机 PyTorch/网络配合正常）。
"""
import os
import subprocess
import sys


def _prepare_windows_env():
    if os.name != 'nt':
        return
    os.environ['USE_LIBUV'] = '0'
    if os.environ.get('DDP_USE_CLUSTER_MASTER', '').lower() not in ('1', 'true', 'yes'):
        os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ.setdefault('MASTER_PORT', '29500')


def _strip_torchrun_tokens(argv):
    """去掉 --nproc_per_node 及训练入口 sr.py，得到可直接交给 train_windows_3gpu / sr 的参数。"""
    out = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in ('--nproc_per_node', '--nproc-per-node'):
            i += 2
            continue
        if a.startswith('--nproc_per_node=') or a.startswith('--nproc-per-node='):
            i += 1
            continue
        if os.path.basename(a).lower() in ('sr.py', 'sr'):
            i += 1
            continue
        out.append(a)
        i += 1
    return out


def _run_torchrun_in_process():
    _inject_master_addr_localhost_argv()
    import torch  # noqa: F401

    from core.win_tcpstore_patch import apply_tcpstore_no_libuv_patch

    apply_tcpstore_no_libuv_patch()
    from torch.distributed.run import main

    main()


def _inject_master_addr_localhost_argv():
    if os.name != 'nt':
        return
    if os.environ.get('DDP_USE_CLUSTER_MASTER', '').lower() in ('1', 'true', 'yes'):
        return
    av = sys.argv
    for a in av:
        if a in ('--master_addr', '--master-addr') or a.startswith('--master_addr=') or a.startswith('--master-addr='):
            return
    sys.argv = [av[0], '--master_addr', '127.0.0.1'] + list(av[1:])


if __name__ == '__main__':
    _prepare_windows_env()
    if os.name == 'nt' and os.environ.get('USE_WINDOWS_TORCHRUN_DDP', '').lower() not in (
        '1',
        'true',
        'yes',
    ):
        root = os.path.dirname(os.path.abspath(__file__))
        tw = os.path.join(root, 'train_windows_3gpu.py')
        fwd = _strip_torchrun_tokens(sys.argv[1:])
        cmd = [sys.executable, tw] + fwd
        print(
            '[ddp_launcher_win] Windows 默认使用 DataParallel（train_windows_3gpu），不使用 torchrun。',
            '若要坚持 DDP 请设 USE_WINDOWS_TORCHRUN_DDP=1',
        )
        print('[ddp_launcher_win]', ' '.join(cmd))
        raise SystemExit(subprocess.call(cmd, env=os.environ))

    print(
        '[ddp_launcher_win] USE_LIBUV=',
        os.environ.get('USE_LIBUV'),
        'MASTER_ADDR=',
        os.environ.get('MASTER_ADDR'),
        'MASTER_PORT=',
        os.environ.get('MASTER_PORT'),
        '| torchrun + TCPStore 补丁',
    )
    if os.name == 'nt':
        _run_torchrun_in_process()
    else:
        cmd = [sys.executable, '-m', 'torch.distributed.run'] + sys.argv[1:]
        raise SystemExit(subprocess.call(cmd, env=os.environ))
