"""
Pytest configuration and fixtures for Rotom tests.

This module provides shared fixtures and utilities for running tests
with different backends (Toy and CKKS).
"""

import os

import pytest

from assignment.assignment import LayoutAssignment
from backends.openfhe_backend import CKKS
from backends.toy import Toy
from lower.lower import Lower


@pytest.fixture(autouse=True)
def _isolated_workdir(tmp_path, monkeypatch):
    """
    Run every test in an isolated working directory.

    The CKKS backend serializes artifacts into a relative `./modules/` path.
    Isolating the CWD per test prevents cross-test contamination and keeps
    each test's `modules/` output in its own directory.
    """
    old_cwd = os.getcwd()
    monkeypatch.chdir(tmp_path)
    try:
        yield
    finally:
        # monkeypatch will revert the cwd automatically, but keep a defensive
        # restore in case other code mutates CWD outside monkeypatch semantics.
        os.chdir(old_cwd)


@pytest.fixture(params=["toy", "ckks"])
def backend(request):
    """
    Fixture that parametrizes tests to run with both toy and ckks backends.

    Usage:
        def test_something(backend):
            # backend will be "toy" or "ckks"
            results = run_backend(backend, circuit_ir, inputs, args)
    """
    return request.param


def run_backend(backend_name, circuit_ir, inputs, args):
    """
    Run a backend and return results.

    Args:
        backend_name: "toy" or "ckks"
        circuit_ir: The circuit IR
        inputs: Input data dictionary
        args: Arguments object

    Returns:
        results: List of result arrays
    """
    # Ensure benchmark attribute exists (required by CKKS backend)
    if not hasattr(args, "benchmark"):
        args.benchmark = ""

    # For CKKS backend with small n, disable security requirements
    # CKKS requires minimum ring dimension of 16384 (n=8192) for security standards
    # For testing purposes, we allow smaller ring dimensions by setting not_secure
    if backend_name == "ckks" and hasattr(args, "n") and args.n < 8192:
        args.not_secure = True

    if backend_name == "toy":
        backend = Toy(circuit_ir, inputs, args)
        return backend.run()
    elif backend_name == "ckks":
        backend = CKKS(circuit_ir, inputs, args)
        _, results = backend.run()
        return results
    else:
        raise ValueError(f"Unknown backend: {backend_name}")


def assert_results_equal(expected_cts, results, backend_name):
    """
    Assert that expected and actual results are equal, using appropriate comparison.

    Args:
        expected_cts: Expected results (list of arrays)
        results: Actual results (list of arrays)
        backend_name: "toy" or "ckks" - determines comparison method
    """
    import numpy as np

    if backend_name == "ckks":
        # CKKS uses floating point, so use np.allclose
        for expected_vec, result_vec in zip(expected_cts, results):
            assert np.allclose(
                expected_vec, result_vec, rtol=1e-2, atol=1e-2
            ), f"Results not close enough. Expected: {expected_vec}, Got: {result_vec}"
    else:
        # Toy backend should have exact equality
        assert expected_cts == results


def run_compiler_and_backend(tensor_ir, inputs, args, backend_name):
    """
    Compile a TensorTerm computation and run it on the selected backend.

    This helper encapsulates the common pipeline:
    TensorTerm -> LayoutAssignment -> Lower -> backend.

    Returns:
        (results, kernel): backend outputs and the compiled kernel (for layout).
    """
    kernel = LayoutAssignment(tensor_ir, args).run()
    circuit_ir = Lower(kernel).run()
    results = run_backend(backend_name, circuit_ir, inputs, args)
    return results, kernel
