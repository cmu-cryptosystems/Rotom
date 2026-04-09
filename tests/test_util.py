import os
from argparse import Namespace


def _toy_ct_workers_from_env() -> int:
    """Optional ``ROTOM_TOY_CT_WORKERS`` for parallel Toy ciphertext roots in tests."""
    v = os.environ.get("ROTOM_TOY_CT_WORKERS", "").strip()
    if not v:
        return 1
    return int(v)


def get_default_args() -> Namespace:
    """Return a default args namespace used by tests.

    For Toy parallel ciphertext roots, set ``ROTOM_TOY_CT_WORKERS`` (e.g. ``16``).
    Slow ResNet e2e is skipped by default; add
    ``--override-ini="addopts=-v --tb=short --strict-markers --disable-warnings"``
    or ``-m slow`` with an addopts override per ``pytest.ini``.
    """
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
        toy_ct_workers=_toy_ct_workers_from_env(),
    )
