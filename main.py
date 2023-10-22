# %%
import os
import sys
from functools import partial
from typing import Any, Callable

IS_CI = os.getenv("IS_CI")
if IS_CI:
    sys.exit(0)
# %%
import numpy as np
import torch as t
import utils
from cuda_utils import Holder, ceil_divide, load_module

device = t.device("cuda:0")
# %%
mod = load_module("w1d6_kernels/intro.cu")
zero_kernel = mod.get_function("zero")
one_kernel = mod.get_function("one")
dest = t.ones(128, dtype=t.float32, device=device)
zero_kernel(Holder(dest), block=(64, 1, 1), grid=(1, 1))
# %%
t.cuda.synchronize()
print(dest)
# %%
filename = "w1d6_kernels/abc.cu"
mul_add_kernel = load_module(filename).get_function("mul_add")
size = 128
dest = t.empty(size, dtype=t.float32, device=device)
a = t.arange(size, dtype=t.float32, device=device)
b = t.arange(0, 4 * size, 4, dtype=t.float32, device=device)
c = t.arange(3, size + 3, dtype=t.float32, device=device)
mul_add_kernel(
    Holder(dest), Holder(a), Holder(b), Holder(c), block=(size, 1, 1), grid=(1, 1)
)
t.cuda.synchronize()
utils.allclose(dest, a * b + c)
# %%
filename = "w1d6_kernels/sum_atomic.cu"
sum_atomic_kernel = load_module(filename).get_function("sum_atomic")


def sum_atomic(inp: t.Tensor, dest: t.Tensor, block_size: int = 1024) -> None:
    """Call sum_atomic_kernel to sum inp into dest.

    inp: shape (*) input to be summed
    dest: shape ()
    """
    assert block_size <= 1024
    nelems = np.product(inp.shape, dtype=np.int64)
    n_blocks = int(ceil_divide(nelems, block_size))
    sum_atomic_kernel(
        Holder(inp), Holder(dest), nelems, block=(block_size, 1, 1), grid=(n_blocks, 1)
    )


inp = t.randn(256, 256, 3).to(device=device)
dest = t.tensor(0.0).to(device=device)
sum_atomic(inp, dest)
t.cuda.synchronize()
expected = inp.sum()
actual = dest
print(actual, expected)
utils.allclose_scalar(actual.item(), expected.item())
print("OK!")
# %%
filename = "w1d6_kernels/filter_atomic.cu"
filter_atomic_kernel = load_module(filename).get_function("filter_atomic")


def filter_atomic(
    inp: t.Tensor, dest: t.Tensor, thresh: float, block_size: int = 512
) -> t.Tensor:
    """Write only elements with absolute value strictly less than thresh into dest, in no particular order.

    inp: shape (N, )
    dest: shape (N, )

    Return: a slice of dest containing only the copied elements (not the zero padding at the end)
    """
    atomic = t.zeros(1, dtype=t.int32, device=inp.device)
    filter_atomic_kernel(
        Holder(inp),
        Holder(dest),
        Holder(atomic),
        np.int32(inp.size(0)),
        np.float32(thresh),
        block=(block_size, 1, 1),
        grid=(ceil_divide(inp.size(0), block_size), 1),
    )
    return dest[:atomic]


N = 2048
threshold = 512
inp = t.randint(-N // 2, N // 2, (N,), dtype=t.float32, device=device)
dest = t.zeros(N, dtype=t.float32, device=device)
filtered = filter_atomic(inp, dest, float(threshold))
t.cuda.synchronize()
assert (filtered.abs() < threshold).all()
print("Number of filtered elements (random but should be reasonable): ", len(filtered))
# %%
filename = "w1d6_kernels/index.cu"
index_kernel = load_module(filename).get_function("index")


def index(a: t.Tensor, b: t.Tensor, dest: t.Tensor, block_size=512) -> None:
    """Write dest[i] = a[b[i]] in place.

    a: shape (A, ), float32
    b: shape (B, ), int64
    dest: shape (B, ), float32
    """
    assert b.shape == dest.shape
    index_kernel(
        Holder(a),
        Holder(b),
        Holder(dest),
        np.int64(len(a)),
        np.int64(len(b)),
        block=(block_size, 1, 1),
        grid=(ceil_divide(dest.size(0), block_size), 1),
    )


aSize = 10
bSize = 100
a = t.randn(aSize, device=device)
b = t.randint(-aSize, aSize, size=(bSize,), dtype=t.int64, device=device)
dest = t.zeros(bSize, device=device)
index(a, b, dest)
t.cuda.synchronize()
expected = a[b]
expected[(b < 0) | (b >= aSize)] = 0
utils.allclose(dest, expected)
# %%
import time


def benchmark(func: Callable[[], Any], n_iters: int = 10) -> float:
    """Warmup, then call func "n_iters" times and return the time per iteration."""
    for _ in range(3):
        func()

    start_time = time.perf_counter()
    for _ in range(n_iters):
        func()
        t.cuda.synchronize()

    return (time.perf_counter() - start_time) / n_iters


def benchmark_my_sum_atomic(inp):
    dest = t.tensor(0.0).to(device=device)
    sum_atomic(inp, dest)


def benchmark_pytorch_sum_atomic(inp):
    dest = inp.sum()


inp = t.randn(1024, 1024, 256).to(device=device)
n_iters = 10
ref = benchmark(partial(benchmark_pytorch_sum_atomic, inp), n_iters=n_iters)
print(f"PyTorch: {ref:.3f}s")
yours = benchmark(partial(benchmark_my_sum_atomic, inp), n_iters=n_iters)
print(f"Yours: {yours:.3f}s")

# %%
filename = "w1d6_kernels/sum_rows.cu"
sum_rows_kernel = load_module(filename).get_function("sum_rows")


def sum_rows(inp: t.Tensor, dest: t.Tensor, block_size=1024):
    """Write the sum of each row to the corresponding element of dest."""
    assert inp.is_contiguous()
    assert dest.is_contiguous()
    (C, R) = inp.shape
    assert dest.shape == (C,)
    sum_rows_kernel(
        Holder(dest),
        Holder(inp),
        np.int64(C),
        np.int64(R),
        block=(block_size, 1, 1),
        grid=(ceil_divide(C, block_size), 1),
    )


nCols = 200
nRows = 300
inp = t.rand((nCols, nRows), device=device, dtype=t.float32)
dest = t.zeros(nCols, device=device, dtype=t.float32)
sum_rows(inp, dest)
t.cuda.synchronize()
expected = inp.sum(dim=1)
utils.allclose(dest, expected)
# %%
