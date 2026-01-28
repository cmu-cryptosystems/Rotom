"""Layout assignment module for optimizing tensor layouts in HE computations.

This module handles assigning optimal layouts to tensors in HE computation graphs.
It includes the core LayoutAssignment class which analyzes computation graphs and
determines efficient tensor layouts to minimize HE operation costs.

Key features:

- Layout optimization for matrix multiplication, convolution, and other tensor ops
- Roll propagation and reordering for efficient ciphertext rotations
- BSGS (baby-step giant-step) optimization for matrix multiplication
- Cost modeling and minimization of HE operations
- Shape tracking and validation

The module works closely with the kernel IR to generate optimized HE circuits
while maintaining correctness of the computation graph semantics.
"""

from copy import deepcopy as copy

# import frontend terms
from re import L

from assignment.gen.gen_binop import gen_binop
from assignment.gen.gen_block_matmul import gen_block_matmul
from assignment.gen.gen_conv2d import gen_conv2d, gen_conv2d_roll
from assignment.gen.gen_index import gen_index
from assignment.gen.gen_permute import gen_permute
from assignment.gen.gen_rescale import gen_rescale
from assignment.gen.gen_reshape import gen_reshape
from assignment.gen.gen_strassens import gen_strassens
from assignment.gen.gen_sum import gen_sum
from assignment.gen.gen_tensor import gen_tensor
from assignment.gen.gen_transpose import gen_transpose
from frontends.tensor import TensorOp, TensorTerm

# import layout assignment components
from ir.analysis.secret import Secret
from ir.analysis.shape import Shape

# import ir components
from ir.dim import DimType
from ir.kernel import Kernel, KernelDag, KernelOp
from ir.kernel_cost import KernelCost
from ir.layout import Layout
from ir.layout_utils import dimension_merging
from ir.roll import Roll

# import optimization components
from opt.opt import Optimizer

# import fuzzer
from util.fuzz import Fuzz

# import utility components
from util.kernel_util import get_cs_op_kernels


class LayoutAssignment:
    """Layout Assignment pass that assigns layouts to tensors in the computation graph.

    This pass takes a computation graph and determines the optimal layout for each tensor
    to minimize the cost of FHE operations. It uses various optimization techniques like:
    - Roll propagation and reordering
    - BSGS matrix multiplication
    - CT roll BSGS optimization
    - Replication hoisting
    - Shape checking and tile pruning

    The pass maintains mappings of kernels, costs, candidate layouts, and shapes for each
    tensor term in the computation graph.

    Args:
        comp: The computation graph to optimize layouts for
        args: Optional command line arguments containing optimization flags and parameters
            including network type, BSGS flags, backend selection etc. If None, default values
            will be used (n=4096, rolls=False, net="lan", strassens=False, backend="toy",
            fuzz=False, fuzz_result=False, fn="default").
    """

    def __init__(self, comp, args=None):
        self.comp = comp

        # Set default values inline
        self.n = args.n if args and hasattr(args, "n") else 4096
        self.kernels = {}
        self.kernel_costs = {}
        self.candidates = {}
        self.roll_flag = args.rolls if args and hasattr(args, "rolls") else False
        self.network = args.net if args and hasattr(args, "net") else "lan"
        self.strassens = (
            args.strassens if args and hasattr(args, "strassens") else False
        )
        self.backend = args.backend if args and hasattr(args, "backend") else "toy"
        self.fuzz = args.fuzz if args and hasattr(args, "fuzz") else False
        self.fuzz_result = (
            args.fuzz_result if args and hasattr(args, "fuzz_result") else False
        )
        self.conv_roll = (
            args.conv_roll if args and hasattr(args, "conv_roll") else False
        )
        self.fuzzer = Fuzz(self.n)
        self.fn = args.fn if args and hasattr(args, "fn") else "default"

        self.secret = Secret(self.comp)
        self.shape = Shape(self.comp)

    def generate_candidate_kernels(self, term):
        """Generates candidate kernels for a given tensor term.

        For each tensor term in the computation graph, this method generates possible kernels
        that could be used to implement the operation. The specific kernels generated depend on
        the operation type (e.g. tensor, matmul, add, etc) and optimization flags.

        Args:
            term: The tensor term to generate candidate kernels for

        Returns:
            kernels: A list of candidate kernels for the given term
        """

        kernels = []
        secret = self.secret.secret[term]
        shape = self.shape.padded_shapes[term]
        cs_kernels = self.get_cs_kernels(term)
        match term.op:
            case TensorOp.TENSOR:
                kernels = gen_tensor(term, secret, shape, self.n)
            case TensorOp.ADD | TensorOp.SUB | TensorOp.MUL | TensorOp.MATMUL:
                if self.strassens and term.op in [
                    TensorOp.MATMUL,
                ]:
                    kernel_map = gen_strassens(
                        term, cs_kernels, self.roll_flag, self.network
                    )
                    for term, kernels in kernel_map.items():
                        self.update_kernels(term, kernels)
                else:
                    cs_shapes = self.get_cs_shapes(term)
                    kernels = gen_binop(
                        term,
                        cs_kernels,
                        cs_shapes,
                        self.roll_flag,
                    )
            case TensorOp.SUM:
                kernels = gen_sum(term, cs_kernels[0])
            case TensorOp.TRANSPOSE:
                kernels = gen_transpose(term, cs_kernels[0])
            case TensorOp.CONV2D:
                assert self.roll_flag
                cs_shapes = self.get_unpadded_cs_shapes(term)
                if self.conv_roll:
                    kernels = gen_conv2d_roll(term, cs_kernels[0], cs_shapes)
                else:
                    kernels = gen_conv2d(term, cs_kernels[0], cs_shapes)
            case TensorOp.RESHAPE:
                kernels = gen_reshape(term, cs_kernels[0])
            case TensorOp.PERMUTE:
                kernels = gen_permute(term, cs_kernels[0])
            case TensorOp.RESCALE:
                kernels = gen_rescale(term, cs_kernels[0])
            case TensorOp.INDEX:
                kernels = gen_index(term, cs_kernels[0])
            case TensorOp.BLOCK_MATMUL:
                kernels = gen_block_matmul(term, cs_kernels)
            case TensorOp.POLY:
                raise NotImplementedError(term.op)
            case _:
                raise NotImplementedError(term.op)
        assert kernels
        return kernels

    def run(self):
        """Runs the layout assignment pass on the computation graph.

        This method traverses the computation graph in post-order and generates candidate kernel layouts
        for each tensor term. For each term, it:
        1. Gets the kernel layouts of child terms
        2. Generates new candidate kernel layouts based on the operation type
        3. Stores the kernels and their costs for later optimization

        The pass handles different operations like:
        - Tensor creation
        - Binary operations (add, mul, matmul etc)
        - Unary operations (sum, transpose etc)
        - Special operations (conv2d, reshape, permute)

        For each operation, it generates kernel layouts that are compatible with the input layouts
        while minimizing the cost of FHE operations. The pass tracks metrics like number of
        candidate layouts generated.

        For example, in Matrix-Vector Multiplication, if the inputs are (M: [1:4];[0:4], and V: [0:4]),
        then the aligned layouts would be:
        - example 1: (M: [1:4];[0:4], V: [0:4];[4])
        - example 2: (M: Roll(0,1) [1:4];[0:4], V: Roll(0,1) [0:4];[4])
        and the output layout for both would be [G:4];[0:4].
        """

        # run secret analysis and shape analysis
        self.secret.run()
        self.shape.run()

        # generate optimized candidate kernels
        for term in self.comp.post_order():
            candidate_kernels = self.generate_candidate_kernels(term)            
            kernels = Optimizer(self.roll_flag).run(candidate_kernels)

            for kernel in kernels:
                print("opt kernel: ", kernel)
                for k in kernel.post_order():
                    print(k)
                print()
            print()

            # prune the search space
            kernels = self.shape_check(kernels)
            # kernels = self.prune_tiles(kernels)
            kernels = self.add_equivalent_kernels(kernels)

            for kernel in kernels:
                print("final kernel: ", kernel)
                print( "cost: ", KernelCost(kernel, self.network).total_cost())
                for k in kernel.post_order():
                    print(k)
                print()
            print()

            # update kernel map
            self.update_kernels(term, kernels)

        # find cheapest layout assignment
        assignment = self.search(self.comp)

        # fuzz layout assignment
        if self.fuzz_result:
            cs_shapes = self.get_cs_shapes(self.comp)
            self.fuzzer.fuzz_kernel(assignment.kernel, cs_shapes)

        return self.combine_kernels(assignment)

    def canonicalize_kernels(self, kernels):
        """Canonicalizes kernel layouts by sorting ciphertext dimensions.

        This method ensures consistent ordering of ciphertext dimensions in kernel layouts
        by sorting them based on dimension index and stride. This helps with layout
        comparison and optimization by ensuring equivalent layouts have the same representation.

        The canonicalization process:
        1. Identifies kernels with multiple ciphertext dimensions
        2. Separates repeated and non-repeated ciphertext dimensions
        3. Sorts non-repeated dimensions by (dim, stride) tuple
        4. Creates new kernels with reordered dimensions if needed

        Args:
            kernels: List of Kernel objects to canonicalize

        Returns:
            List of canonicalized Kernel objects with consistent dimension ordering
        """
        # sort output cts
        sorted_kernels = []
        for kernel in kernels:
            if kernel.layout.num_ct() > 1 and len(kernel.layout.ct_dims) > 1:
                ct_dims = kernel.layout.ct_dims
                repeated_cts = [ct_dim for ct_dim in ct_dims if ct_dim.dim is None]
                non_repeated_cts = [
                    ct_dim for ct_dim in ct_dims if ct_dim.dim is not None
                ]
                non_repeated_cts = sorted(
                    non_repeated_cts, key=lambda x: (x.dim, x.stride)
                )
                aligned_dims = (
                    repeated_cts + non_repeated_cts + copy(kernel.layout.slot_dims)
                )
                if ct_dims != kernel.layout.ct_dims:
                    new_layout = Layout(
                        kernel.layout.term,
                        kernel.layout.rolls,
                        aligned_dims,
                        kernel.layout.n,
                        kernel.layout.secret,
                    )
                    new_kernel = Kernel(
                        KernelOp.REORDER,
                        [kernel],
                        new_layout,
                    )
                    sorted_kernels.append(new_kernel)
                else:
                    sorted_kernels.append(kernel)
            else:
                sorted_kernels.append(kernel)
        kernels = sorted_kernels
        assert kernels
        return kernels

    def get_last_kernels(self, kernel_dags):
        """Extracts the final kernel from each kernel DAG.

        This method takes a collection of kernel DAGs and returns the terminal
        kernel from each DAG. This is used to get the output kernels from
        child tensor terms for use in generating new kernels.

        Args:
            kernel_dags: Collection of KernelDag objects

        Returns:
            List of Kernel objects representing the final kernels from each DAG
        """
        last_kernels = []
        for kernel_dag in kernel_dags:
            last_kernels.append(kernel_dag.kernel)
        return last_kernels

    def get_cs_shapes(self, term):
        """Gets the padded shapes of child tensor terms.

        This method extracts the padded shapes of all child tensor terms
        from a given tensor term. Padded shapes are used for layout generation
        to ensure compatibility with HE vector sizes.

        Args:
            term: TensorTerm to get child shapes for

        Returns:
            List of shape tuples for child tensor terms
        """
        cs_shapes = []
        for cs_term in term.cs:
            if isinstance(cs_term, TensorTerm):
                cs_shapes.append(self.shape.padded_shapes[cs_term])
        return cs_shapes

    def get_unpadded_cs_shapes(self, term):
        """Gets the original (unpadded) shapes of child tensor terms.

        This method extracts the original shapes of all child tensor terms
        from a given tensor term, without any padding applied. This is used
        for operations that need the actual data dimensions rather than
        HE-compatible padded dimensions.

        Args:
            term: TensorTerm to get child shapes for

        Returns:
            List of original shape tuples for child tensor terms
        """
        cs_shapes = []
        for cs_term in term.cs:
            if isinstance(cs_term, TensorTerm):
                cs_shapes.append(self.shape.shapes[cs_term])
        return cs_shapes

    def get_cs_kernels(self, term):
        """Gets the child kernels for a tensor term based on its operation type.

        This method retrieves the appropriate child kernels for a tensor term
        based on its operation type. For binary operations, it returns kernels
        from both operands; for unary operations, it returns kernels from the
        single operand. The kernels are sorted by cost to prioritize cheaper
        options during layout generation.

        Args:
            term: TensorTerm to get child kernels for

        Returns:
            List of kernel lists for each child operand

        Raises:
            NotImplementedError: If the operation type is not supported
        """
        match term.op:
            case TensorOp.TENSOR:
                return []
            case (
                TensorOp.ADD
                | TensorOp.SUB
                | TensorOp.MUL
                | TensorOp.MATMUL
                | TensorOp.BLOCK_MATMUL
                | TensorOp.CONV2D
            ):
                a = self.get_last_kernels(self.kernels[term.cs[0]].values())
                b = self.get_last_kernels(self.kernels[term.cs[1]].values())

                # sort by cost
                a = sorted(
                    a,
                    key=lambda x: self.kernel_costs[term.cs[0]][
                        dimension_merging(x.layout)
                    ],
                )
                b = sorted(
                    b,
                    key=lambda x: self.kernel_costs[term.cs[1]][
                        dimension_merging(x.layout)
                    ],
                )

                return [a, b]
            case (
                TensorOp.TRANSPOSE
                | TensorOp.POLY
                | TensorOp.SUM
                | TensorOp.RESHAPE
                | TensorOp.PERMUTE
                | TensorOp.INDEX
                | TensorOp.RESCALE
            ):
                return [self.get_last_kernels(self.kernels[term.cs[0]].values())]
            case _:
                raise NotImplementedError(term.op)

    def get_cs_ops(self, kernel):
        """Extracts all CS (ciphertext slot) operations from a kernel.

        This method traverses a kernel in post-order and collects all
        operations that are of type CS (ciphertext slot operations).
        These operations represent the input/output layouts of the kernel.

        Args:
            kernel: Kernel to extract CS operations from

        Returns:
            List of CS operation nodes from the kernel
        """
        cs_ops = []
        for k in kernel.post_order():
            if k.op == KernelOp.CS:
                cs_ops.append(k)
        return cs_ops

    def add_equivalent_kernels(self, kernels):
        """Adds equivalent kernels by swapping roll dimensions.

        This method generates equivalent kernel layouts by swapping the dimensions
        involved in roll operations. This is useful for exploring different but
        functionally equivalent layout options during optimization.

        The method identifies kernels with exactly one roll operation where the
        roll_by dimension is empty (FILL type), then creates equivalent kernels
        by swapping the roll dimensions.

        Args:
            kernels: List of Kernel objects to generate equivalents for

        Returns:
            Set of Kernel objects including original and equivalent kernels
        """
        # check that the kernel only has 1 roll
        # and the roll_by dimension is empty
        eq_kernels = set(copy(kernels))
        for kernel in kernels:
            if len(kernel.layout.rolls) == 1:
                roll = kernel.layout.rolls[0]
                if (
                    roll.dim_to_roll_by.dim is None
                    and roll.dim_to_roll_by.dim_type == DimType.FILL
                ):
                    new_dims = copy(kernel.layout.get_dims())
                    roll_index = roll.roll_index(new_dims)
                    new_dims[roll_index[0]], new_dims[roll_index[1]] = (
                        new_dims[roll_index[1]],
                        new_dims[roll_index[0]],
                    )
                    new_roll = Roll(roll.dim_to_roll_by, roll.dim_to_roll)
                    new_layout = Layout(
                        kernel.layout.term,
                        [new_roll],
                        new_dims,
                        kernel.layout.n,
                        kernel.layout.secret,
                    )
                    new_kernel = copy(kernel)
                    new_kernel.layout = new_layout
                    eq_kernels.add(new_kernel)

        # Return a deterministic, sorted list of kernels
        assert eq_kernels
        return sorted(eq_kernels, key=lambda k: k.layout.layout_str())

    def prune(self, kernels):
        layouts = {}
        for kernel in kernels:
            # get input/output layouts as a key
            cs_ops = self.get_cs_ops(kernel)
            cs_ops.append(kernel.layout)
            key = tuple(cs_ops)

            # get cost of kernel
            kernel_cost = KernelCost(kernel, self.network).total_cost()

            if key not in layouts:
                layouts[key] = (kernel, kernel_cost)
            elif kernel_cost < layouts[key][1]:
                layouts[key] = (kernel, kernel_cost)
        return [cost_pair[0] for cost_pair in layouts.values()]

    def prune_tiles(self, kernels):
        new_kernels = []
        for kernel in kernels:
            if kernel.layout.rolls:
                new_kernels.append(kernel)

            # check if dims are tiled
            dim_order = []
            for dim in kernel.layout.slot_dims:
                if not dim_order:
                    dim_order.append(dim.dim)
                elif dim_order[-1] == dim.dim:
                    continue
                else:
                    dim_order.append(dim.dim)

            seen = set()
            prune = False
            for dim in dim_order:
                if dim not in seen:
                    seen.add(dim)
                else:
                    prune = True
                    continue
            if not prune:
                new_kernels.append(kernel)
        assert new_kernels
        return new_kernels

    def shape_check(self, kernels):
        """Checks if the kernel layouts match the expected shape of the tensor.

        This method filters kernels based on whether their layout dimensions match the
        expected shape of the tensor term. It compares the flattened shape of each kernel's
        layout against the target shape.

        Args:
            shape: List of integers representing the expected shape of the tensor
            kernels: List of Kernel objects to check

        Returns:
            List of Kernel objects whose layouts match the expected shape
        """

        assert kernels

        new_kernels = []
        for kernel in kernels:
            shape = self.shape.padded_shapes[kernel.layout.term]
            kernel_shape_map = {}
            for dim in kernel.layout.get_dims():
                # Skip EMPTY dimensions - they should have dim=None and shouldn't contribute to shape
                if dim.dim_type == DimType.EMPTY:
                    continue
                elif dim.dim is not None and dim.dim not in kernel_shape_map:
                    kernel_shape_map[dim.dim] = dim.extent
                elif dim.dim is not None:
                    kernel_shape_map[dim.dim] *= dim.extent

            # flatten kernel_shape
            kernel_shape = []
            if kernel_shape_map:
                for i in range(max(kernel_shape_map.keys()) + 1):
                    if i not in kernel_shape_map:
                        kernel_shape.append(1)
                    else:
                        kernel_shape.append(kernel_shape_map[i])
            else:
                # No dimensions with actual dim indices (e.g., 1x1 convolution with only replicated dims)
                # In this case, the kernel represents a broadcast/elementwise operation
                kernel_shape = [1] * len(shape)

            for i, k in enumerate(kernel_shape):
                if k != shape[i]:
                    raise ValueError(
                        f"kernel shape {kernel_shape} does not match expected shape {shape}"
                    )
            new_kernels.append(kernel)
        assert new_kernels
        return new_kernels

    def update_kernels(self, term, kernels):
        for kernel in kernels:
            # get cs kernels
            cs_kernels = get_cs_op_kernels(kernel)

            # get cs costs
            cs_costs = 0
            for cs_kernel in cs_kernels:
                cs_term = term.cs[cs_kernel.cs[0]]

                if cs_kernel.layout in self.kernel_costs[cs_term]:
                    cs_costs += self.kernel_costs[cs_term][
                        dimension_merging(cs_kernel.layout)
                    ]
                else:
                    cost = KernelCost(cs_kernel, self.network).total_cost()
                    self.kernel_costs[cs_term][
                        dimension_merging(cs_kernel.layout)
                    ] = cost
                    cs_costs += cost

            cs_kernel_list = []
            for cs_kernel in cs_kernels:
                cs_kernel_list.append(
                    self.kernels[cs_kernel.layout.term][
                        dimension_merging(cs_kernel.layout)
                    ]
                )

            # get total kernel cost
            kernel_cost = KernelCost(kernel, self.network).total_cost() + cs_costs
            kernel_layout = dimension_merging(kernel.layout)

            # initialize
            if term not in self.kernels:
                self.kernels[term] = {}
                self.kernel_costs[term] = {}

            # optimize
            if kernel_layout not in self.kernels[term]:
                self.kernels[term][kernel_layout] = KernelDag(kernel, cs_kernel_list)
                self.kernel_costs[term][kernel_layout] = kernel_cost
            elif kernel_cost < self.kernel_costs[term][kernel_layout]:
                self.kernels[term][kernel_layout] = KernelDag(kernel, cs_kernel_list)
                self.kernel_costs[term][kernel_layout] = kernel_cost

    def search(self, term):
        """Pick the minimum-cost kernel DAG for `term`.

        During `update_kernels` we already compute and cache the total cost of each
        candidate (including child kernel DAG costs) in `self.kernel_costs[term]`.
        Using that cache avoids recomputing costs and, critically, ensures we're
        comparing like-for-like totals (not just the root kernel op cost).
        """
        assert term in self.kernels and term in self.kernel_costs
        assert self.kernels[term] and self.kernel_costs[term]

        best_layout, _best_cost = min(self.kernel_costs[term].items(), key=lambda kv: kv[1])
        return self.kernels[term][best_layout]

    def remove_duplicate_kernels(self, kernels):
        new_kernels = []
        for kernel in kernels:
            if kernel not in new_kernels:
                new_kernels.append(kernel)
        return new_kernels

    def combine_kernels(self, kernel_dag):
        """Combines kernels in a kernel DAG by connecting input kernels to output kernels.

        This function takes a kernel DAG and creates a mapping between kernel layouts and their
        corresponding kernel objects. For each kernel in the DAG, it processes its child kernels
        (CS kernels) and updates their references to point to the actual kernel objects rather
        than just layout references.

        Args:
            kernel_dag: A KernelDag object containing the kernels to be combined

        Returns:
            The final combined kernel with all CS kernel references resolved
        """
        # Environment mapping from kernel layouts to kernel objects
        env = {}
        for kernel_dag_term in kernel_dag.post_order():
            kernel = kernel_dag_term.kernel
            for k in kernel.post_order():
                for i, k_cs in enumerate(k.cs):
                    if isinstance(k_cs, Kernel) and k_cs.op == KernelOp.CS:
                        k.cs[i] = env[dimension_merging(k_cs.layout)]

                if k.op != KernelOp.CS:
                    env[dimension_merging(k.layout)] = k

        kernel_layout = kernel_dag.kernel.layout
        return env[dimension_merging(kernel_layout)]
