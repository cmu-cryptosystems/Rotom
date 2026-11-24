"""Every ``main.py`` benchmark at n = 32768.

Each benchmark is compiled with 32768 slots, whatever n it picks for itself,
run on the toy backend, and compared with the program's own evaluation.

``bert_attention`` is compiled and lowered only. Its plan spreads each 768x768
weight over 4096 plaintext vectors, and the toy backend's packing of those
needs more memory than a 32 GB machine has, at this n and at smaller ones.
"""

import pytest

from assignment.assignment import LayoutAssignment
from benchmarks.rotom_benchmarks.bert_attention import bert_attention
from benchmarks.rotom_benchmarks.convolution.convolution import convolution
from benchmarks.rotom_benchmarks.convolution.convolution_32768 import convolution_32768
from benchmarks.rotom_benchmarks.double_matmul.double_matmul_128_64_ct_ct import (
    double_matmul_128_64_ct_ct,
)
from benchmarks.rotom_benchmarks.double_matmul.double_matmul_256_128_ct_ct import (
    double_matmul_256_128_ct_ct,
)
from benchmarks.rotom_benchmarks.logreg import logreg
from benchmarks.rotom_benchmarks.matmul.matmul_128_64 import matmul_128_64
from benchmarks.rotom_benchmarks.matmul.matmul_256_128 import matmul_256_128
from benchmarks.rotom_benchmarks.mlp_mnist_square import mlp_mnist_square
from benchmarks.rotom_benchmarks.movinet import movinet_tiny
from benchmarks.rotom_benchmarks.resnet_silu import resnet_silu, resnet_silu_one_layer
from benchmarks.rotom_benchmarks.ttm import ttm
from lower.lower import Lower
from tests.conftest import run_compiler_and_backend
from tests.test_util import get_default_args
from util.checker import check_results

N = 32768

# name -> the function main.py calls for `--benchmark name`
BENCHMARKS = {
    "matmul_128_64": matmul_128_64,
    "matmul_256_128": matmul_256_128,
    "double_matmul_128_64": double_matmul_128_64_ct_ct,
    "double_matmul_256_128": double_matmul_256_128_ct_ct,
    "convolution": convolution,
    "convolution_32768": convolution_32768,
    "logreg": logreg,
    "mlp_mnist_square": mlp_mnist_square,
    "ttm": ttm,
    "bert_attention": bert_attention,
    "resnet_silu_one_layer": resnet_silu_one_layer,
    "resnet_silu": resnet_silu,
    # movinet_a0 is left out: its layout search alone takes over half an hour
    "movinet_tiny": movinet_tiny,
}


COMPILED_ONLY = {"bert_attention"}


def _args(name, n):
    args = get_default_args()
    args.n = n
    args.rolls = True
    args.backend = "toy"
    args.benchmark = name
    return args


def run_at_n(name, build, n=N):
    """Compile the benchmark `build` returns with `n` slots, run it on the toy
    backend and check it against its evaluation. Returns the kernel."""
    term, inputs = build()[:2]  # some benchmarks also return their own n
    args = _args(name, n)
    results, kernel = run_compiler_and_backend(term, inputs, args, "toy")
    assert kernel.layout.n == n
    check_results(term, inputs, kernel, results, 0, args)
    return kernel


@pytest.mark.slow
@pytest.mark.parametrize("name", [b for b in BENCHMARKS if b not in COMPILED_ONLY])
def test_benchmark_at_n_32768(name):
    run_at_n(name, BENCHMARKS[name])


@pytest.mark.slow
@pytest.mark.parametrize("name", sorted(COMPILED_ONLY))
def test_benchmark_compiles_at_n_32768(name):
    term, _ = BENCHMARKS[name]()[:2]
    kernel = LayoutAssignment(term, _args(name, N)).run()
    assert kernel.layout.n == N
    circuit = Lower(kernel).run()
    assert len(circuit[kernel].cts) == kernel.layout.num_ct()
