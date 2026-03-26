#!/usr/bin/env python
"""
Launch script for multi-GPU training on Windows.
This script sets USE_LIBUV=0 before importing torch to avoid libuv errors.
"""
import os
import sys

# Set USE_LIBUV=0 BEFORE importing torch
os.environ['USE_LIBUV'] = '0'

# Now import and run torch.distributed
import subprocess

if __name__ == '__main__':
    # Get the number of GPUs from config or command line
    nproc = 3  # Default to 3 GPUs as per config
    if len(sys.argv) > 1:
        nproc = int(sys.argv[1])
    
    # Build the command
    cmd = [
        sys.executable, '-m', 'torch.distributed.run',
        '--nproc_per_node', str(nproc),
        'sr.py', '-p', 'train', '-c', 'config/sr3_mrsi_64.json'
    ]
    
    # Add any additional arguments
    if len(sys.argv) > 2:
        cmd.extend(sys.argv[2:])
    
    # Run with the environment variable set
    env = os.environ.copy()
    env['USE_LIBUV'] = '0'
    
    print(f"Launching training with {nproc} GPUs...")
    print(f"Command: {' '.join(cmd)}")
    print(f"USE_LIBUV={env.get('USE_LIBUV')}")
    
    sys.exit(subprocess.call(cmd, env=env))


