"""Without a GPU, run Triton kernels under the interpreter. torch does not import triton, so asking torch first is safe;
the variable must be set before learning.linear_order decorates its kernels. Force CPU with CUDA_VISIBLE_DEVICES=''."""
import os

import torch

if not torch.cuda.is_available():
    os.environ.setdefault("TRITON_INTERPRET", "1")
