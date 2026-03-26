#!/usr/bin/env python
"""
Multi-GPU training launcher using multiprocessing (more reliable on Windows).
This bypasses torch.distributed.run and directly spawns processes.
"""
import os
import sys
import multiprocessing as mp
import runpy
import traceback

# Set USE_LIBUV=0 BEFORE any torch imports
os.environ['USE_LIBUV'] = '0'

def worker(rank, world_size, args):
    """Worker function for each GPU process"""
    # Set environment variables for this process
    os.environ['USE_LIBUV'] = '0'
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['LOCAL_RANK'] = str(rank)
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'

    # Debug: verify env seen by this subprocess before sr.py imports torch.distributed.
    print(
        f"[rank {rank}] env: USE_LIBUV={os.environ.get('USE_LIBUV')} "
        f"MASTER_ADDR={os.environ.get('MASTER_ADDR')} MASTER_PORT={os.environ.get('MASTER_PORT')} "
        f"LOCAL_RANK={os.environ.get('LOCAL_RANK')} WORLD_SIZE={os.environ.get('WORLD_SIZE')}",
        flush=True,
    )
    
    # Run sr.py as __main__ so its training entrypoint executes.
    import sys
    sys.argv = args + ['--local_rank', str(rank)]

    try:
        sr_path = os.path.join(os.path.dirname(__file__), 'sr.py')
        # argv[0] should be the script path; argparse reads sys.argv[1:]
        sys.argv = [sr_path] + args + ['--local_rank', str(rank)]
        runpy.run_path(sr_path, run_name='__main__')
    except Exception:
        # Ensure failures in child processes are visible in the parent console.
        traceback.print_exc()
        # Re-raise so the child exits non-zero (we'll report exit codes too).
        raise

if __name__ == '__main__':
    # Number of GPUs
    nproc = 3
    
    # Base arguments
    base_args = ['-p', 'train', '-c', 'config/sr3_mrsi_64.json']
    
    # Add any additional arguments
    if len(sys.argv) > 1:
        base_args.extend(sys.argv[1:])
    
    print(f"Launching {nproc} processes with USE_LIBUV=0...")
    
    # Spawn processes
    mp.set_start_method('spawn', force=True)
    processes = []
    for rank in range(nproc):
        p = mp.Process(target=worker, args=(rank, nproc, base_args))
        p.start()
        processes.append(p)
    
    # Wait for all processes to complete
    for p in processes:
        p.join()
    for i, p in enumerate(processes):
        if p.exitcode not in (0, None):
            print(f"[rank {i}] exitcode={p.exitcode}")


