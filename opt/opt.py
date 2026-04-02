"""
Main optimizer class for applying optimization passes to FHE kernels.

This module provides the main Optimizer class that orchestrates the
application of various optimization passes to improve the efficiency
of homomorphic encryption computations. The optimizer applies passes
in sequence to transform kernels into more efficient forms.
"""

from copy import deepcopy as copy

from assignment.strategy_loader import get_strategy_module
from opt.bsgs_matmul import run_bsgs_matmul
from opt.ct_roll_bsgs import run_ct_roll_bsgs
from opt.roll_propagation import run_roll_propagation
from opt.roll_reordering import run_roll_reordering
from opt.rot_roll import run_rot_roll


class Optimizer:
    """
    Main optimizer for applying optimization passes to FHE kernels.

    The Optimizer class coordinates the application of various optimization
    passes to improve the efficiency of homomorphic encryption computations.
    It applies passes in sequence, with each pass potentially generating
    multiple optimized versions of the input kernels.

    Attributes:
        roll_flag: Boolean flag indicating whether roll optimizations should be applied
    """

    def __init__(self, roll_flag, network: str = "lan"):
        """
        Initialize the optimizer.

        Args:
            roll_flag: Boolean indicating whether to apply roll optimizations
            network: Cost network label (passed to layout_sort_key)
        """
        self.roll_flag = roll_flag
        self.network = network

    def run(self, kernels):
        """
        Apply optimization passes to a set of kernels.

        This method applies a sequence of optimization passes to improve
        the efficiency of the input kernels. The optimization sequence
        includes roll propagation, roll reordering, and rotation optimizations.

        Args:
            kernels: Set of kernels to optimize

        Returns:
            Set of optimized kernels

        Note:
            Currently includes commented-out BSGS optimizations that may
            be enabled in future versions.
        """
        strategy = get_strategy_module()
        cfg = strategy.optimizer_pass_config(self.roll_flag)

        optimized_kernels = []
        if self.roll_flag:
            for kernel in kernels:
                opt_kernel = copy(kernel)
                if cfg.get("roll_propagation", True):
                    opt_kernel = run_roll_propagation(opt_kernel)
                if cfg.get("roll_reordering", True):
                    opt_kernels = run_roll_reordering(opt_kernel)
                else:
                    opt_kernels = {opt_kernel}
                for ok in opt_kernels:
                    sub = ok
                    if cfg.get("rot_roll", True):
                        sub = run_rot_roll(sub)
                    if cfg.get("ct_roll_bsgs", True):
                        sub = run_ct_roll_bsgs(sub)
                    if cfg.get("bsgs_matmul", True):
                        sub = run_bsgs_matmul(sub)
                    optimized_kernels.append(sub)
        else:
            optimized_kernels = list(kernels)

        assert optimized_kernels
        sort_key = strategy.layout_sort_key
        optimized_kernels = sorted(
            optimized_kernels,
            key=lambda k: sort_key(k, self.network),
        )
        return optimized_kernels
