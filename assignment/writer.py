"""
Writer utilities for outputting FHE circuit representations.

This module provides functionality for writing FHE circuits and kernels
to files in various formats. It handles the conversion from kernel
representations to lower-level FHE terms and outputs them in a
human-readable format.

Key classes:
- Writer: Main class for writing circuit metadata and implementations
"""

from lower.lower import Lower
from ir.kernel import KernelOp


class Writer:
    """Writer class for outputting FHE circuit representations.
    
    This class provides methods to write FHE circuits and kernels to files
    in a structured format. It handles the conversion from high-level kernel
    representations to lower-level FHE terms and outputs them with metadata.
    
    The writer can output:
    - Circuit metadata files with kernel information
    - Circuit implementation files with FHE terms
    - Structured output for debugging and analysis
    
    Args:
        fn: Filename prefix for output files
    """
    def __init__(self, fn):
        self.fn = fn

    def write_metadata(self, f, kernel):
        """Writes kernel metadata to a file.
        
        This method writes metadata about a kernel including the tensor term
        and all operations in the kernel in post-order traversal.
        
        Args:
            f: File handle to write to
            kernel: Kernel to write metadata for
        """
        # write each kernel to its own file
        f.write("# metadata\n")
        f.write(f"# {kernel.layout.term}\n")
        for k in kernel.post_order():
            f.write(f"{k}\n")
        f.write("\n")
        f.write("\n")

    def write_to_file(self, term, kernel_dag):
        """Writes a complete circuit to files.
        
        This method writes both metadata and implementation files for a
        kernel DAG representing a complete circuit. It creates separate
        files for metadata and the actual circuit implementation.
        
        Args:
            term: Root tensor term of the circuit
            kernel_dag: KernelDag representing the complete circuit
        """
        # write metadata to file
        metadata_file = f"output/metadata_{self.fn}.txt"
        with open(metadata_file, "w") as f:
            for kernel_dag_term in kernel_dag.post_order():
                f.write(f"# {kernel_dag_term.kernel.layout.term}\n")
                for k in kernel_dag_term.kernel.post_order():
                    f.write(f"{k}\n")
                f.write("\n")

        # write kernels to file
        file_name = f"output/circuit_{self.fn}.txt"
        with open(file_name, "w") as f:
            # combine all the kernels into a single circuit file
            # map cs from output kernels to input kernels

            env = {}  # fhe_term to index
            kernel_env = {}  # kernel_term to index
            for kernel_dag_term in kernel_dag.post_order():
                # lower each kernel to fhe_terms
                cts = Lower(kernel_dag_term.kernel).run()

                if kernel_dag_term.kernel.op == KernelOp.MATMUL:
                    for k in kernel_dag_term.kernel.post_order():
                        print(k)
                        print(k.cs)

                kernel_term = list(cts)[-1]
                fhe_term = cts[kernel_term]
                layout_term = kernel_term.layout
                f.write("=" * 60 + "\n")

                # write tensor term
                f.write(f"# {kernel_term.layout}\n")

                # write circuit
                for _, ct in fhe_term.items():
                    instrs, env = ct.instrs(env=env, kernel_env=kernel_env)
                    if not instrs:
                        continue
                    for instr in instrs:
                        f.write(f"{instr}\n")
                    f.write("\n")
                    if layout_term not in kernel_env:
                        kernel_env[layout_term] = []
                    kernel_env[layout_term].append(env[ct])