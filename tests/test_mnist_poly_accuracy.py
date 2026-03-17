from types import SimpleNamespace

import numpy as np

from e2e import main as e2e_main


def _build_args(fn: str, backend: str, label: int) -> SimpleNamespace:
    return SimpleNamespace(
        fn=fn,
        benchmark="main",
        backend=backend,
        n=32768,
        size=4,
        rolls=True,
        net="lan",
        cache=False,
        not_secure=False,
        label=label,
        serialize_inputs=False,
    )


def _predicted_label_from_results(results):
    # Mirror util.checker.check_label logic
    non_zero_results = [v for v in results[0] if v != 0]
    return non_zero_results.index(max(non_zero_results))


def _accuracy_for_config(fn: str, backend: str, indices) -> float:
    correct = 0
    total = 0
    for idx in indices:
        args = _build_args(fn=fn, backend=backend, label=int(idx))
        results, true_label = e2e_main(args)
        pred = _predicted_label_from_results(results)
        if pred == true_label:
            correct += 1
        total += 1
    return correct / float(total)


def _sample_indices():
    rng = np.random.RandomState(0)
    # MNIST test set has 10k samples; sampling from [0, 10000) is safe.
    return rng.choice(10000, size=4, replace=False)


def test_mnist_poly_toy_accuracy():
    """mnist_poly with Toy backend should reach > 75% accuracy on a few samples."""
    indices = _sample_indices()
    acc = _accuracy_for_config("mnist_poly", "toy", indices)
    assert acc >= 0.75, f"mnist_poly toy accuracy {acc:.3f} < 0.75"


def test_mnist_poly_heir_accuracy():
    """mnist_poly with HEIR backend should reach > 75% accuracy on a few samples."""
    indices = _sample_indices()
    acc = _accuracy_for_config("mnist_poly", "heir", indices)
    assert acc >= 0.75, f"mnist_poly heir accuracy {acc:.3f} < 0.75"


def test_mnist_poly_call_heir_accuracy():
    """mnist_poly_call with HEIR backend should reach > 75% accuracy on a few samples."""
    indices = _sample_indices()
    acc = _accuracy_for_config("mnist_poly_call", "heir", indices)
    assert acc >= 0.75, f"mnist_poly_call heir accuracy {acc:.3f} < 0.75"
