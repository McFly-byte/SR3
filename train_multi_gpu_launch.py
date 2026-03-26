#!/usr/bin/env python
"""
Launch script using torch.distributed.launch (more stable on Windows).
This sets USE_LIBUV=0 and uses the deprecated but more stable launch method.
"""
import os
import sys
import subprocess

# Force set USE_LIBUV=0
os.environ['USE_LIBUV'] = '0'

if __name__ == '__main__':
    # Number of GPUs
    nproc = 3
    
    # Use torch.distributed.launch (deprecated but more stable on Windows)
    cmd = [
        sys.executable, '-m', 'torch.distributed.launch',
        '--nproc_per_node', str(nproc),
        '--master_port', '29500',
        'sr.py', '-p', 'train', '-c', 'config/sr3_mrsi_64.json'
    ]
    
    # Add any additional arguments
    if len(sys.argv) > 1:
        cmd.extend(sys.argv[1:])
    
    # Ensure environment variable is set
    env = os.environ.copy()
    env['USE_LIBUV'] = '0'
    
    print(f"Launching training with {nproc} GPUs using torch.distributed.launch...")
    print(f"Command: {' '.join(cmd)}")
    print(f"USE_LIBUV={env.get('USE_LIBUV')}")
    
    sys.exit(subprocess.call(cmd, env=env))


