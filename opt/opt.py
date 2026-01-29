"""
Main optimizer class for applying optimization passes to FHE kernels.

This module provides the main Optimizer class that orchestrates the
application of various optimization passes to improve the efficiency
of homomorphic encryption computations. The optimizer applies passes
in sequence to transform kernels into more efficient forms.
"""

from copy import deepcopy as copy

from opt.bsgs_matmul import run_bsgs_matmul
from opt.ct_roll_bsgs import run_ct_roll_bsgs
from opt.roll_propagation import run_roll_propogation
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

    def __init__(self, roll_flag):
        """
        Initialize the optimizer.

        Args:
            roll_flag: Boolean indicating whether to apply roll optimizations
        """
        self.roll_flag = roll_flag

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
        # run optimization passes
        optimized_kernels = []
        if self.roll_flag:
            for kernel in kernels:
                opt_kernel = copy(kernel)
                opt_kernel = run_roll_propogation(opt_kernel)
                opt_kernels = run_roll_reordering(opt_kernel)
                for opt_kernel in opt_kernels:
                    opt_kernel = run_rot_roll(opt_kernel)
                    opt_kernel = run_ct_roll_bsgs(opt_kernel)
                    opt_kernel = run_bsgs_matmul(opt_kernel)
                    optimized_kernels.append(opt_kernel)
        else:
            optimized_kernels = list(kernels)

        # Make the result deterministic by sorting kernels by their layout string
        assert optimized_kernels
        optimized_kernels = sorted(
            optimized_kernels, key=lambda k: k.layout.layout_str()
        )
        return optimized_kernels
