from argparse import Namespace


def get_default_args() -> Namespace:
    """Return a default args namespace used by tests."""
    return Namespace(
        backend="ckks",
        n=4096,
        rolls=False,
        strassens=False,
        net="lan",
        cache=False,
        serialize=False,
        mock=False,
        fuzz=False,
        fuzz_result=False,
        not_secure=False,
        conv_roll=False,
        fn="",
        benchmark="",
        layout_simplicity_weight=0.0,
        channel_gap_align_weight=0.0,
        # Toy.run(): skip per-kernel dense eval + apply_layout self-checks (see backends/toy.py).
        # End-to-end tests that call check_results() still validate the final output.
        skip_toy_eval_checks=False,
    )
