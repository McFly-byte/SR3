#!/usr/bin/env python
"""
Windows 单机多卡训练入口（推荐）：单进程 + nn.DataParallel，不经过 torch.distributed / TCPStore。

用法（与 sr.py 相同，只是会先清掉 torchrun 残留的环境变量）：

    python train_windows_3gpu.py -p train -c config/sr3_mrsi_64.json

GPU 列表由配置文件中的 gpu_ids 决定（例如 [0,1,2]）。勿与 torchrun / ddp_launcher_win 混用。
"""
import os
import subprocess
import sys

_ENV_KEYS_CLEAR = (
    'RANK',
    'WORLD_SIZE',
    'LOCAL_RANK',
    'GROUP_RANK',
    'GROUP_WORLD_SIZE',
    'LOCAL_WORLD_SIZE',
    'ROLE_RANK',
    'ROLE_WORLD_SIZE',
    'MASTER_ADDR',
    'MASTER_PORT',
    'TORCHELASTIC_RUN_ID',
)


def main():
    env = os.environ.copy()
    for k in _ENV_KEYS_CLEAR:
        env.pop(k, None)
    env.setdefault('USE_LIBUV', '0')

    root = os.path.dirname(os.path.abspath(__file__))
    sr_py = os.path.join(root, 'sr.py')
    cmd = [sys.executable, sr_py] + sys.argv[1:]
    if len(sys.argv) == 1:
        cmd.extend(['-p', 'train', '-c', 'config/sr3_mrsi_64.json'])

    print('[train_windows_3gpu] DataParallel 单进程多卡 |', ' '.join(cmd))
    raise SystemExit(subprocess.call(cmd, env=env))


if __name__ == '__main__':
    main()
