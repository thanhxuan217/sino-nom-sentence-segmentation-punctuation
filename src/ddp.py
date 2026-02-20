#!/usr/bin/env python3
"""
Distributed Data Parallel (DDP) utilities for SikuBERT
"""

import os

import torch
import torch.distributed as dist


# ============================================================================
# DDP UTILITIES
# ============================================================================

def setup_ddp(rank: int, world_size: int):
    """Initialize distributed process group"""
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp():
    """Clean up distributed process group"""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    """Check if current process is the main process (rank 0)"""
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0
