"""
Secret analysis for tracking encrypted vs plaintext tensors.

This module provides analysis to determine which tensors in a computation
graph are secret (encrypted) versus public (plaintext). This information
is crucial for proper layout assignment and optimization in homomorphic
encryption contexts.

Key Concepts:

- Secret tensors: Encrypted data that must be kept private
- Public tensors: Plaintext data that can be revealed
- Secrecy propagation: How secrecy flows through operations
"""

from frontends.tensor import TensorOp


class Secret:
    """Secret class for tracking which tensors are encrypted vs plaintext.

    This class maintains a mapping of which tensors in the computation graph are secret
    (encrypted) vs public (plaintext). It traverses the graph and determines the secrecy
    of each tensor based on the operation type and secrecy of input tensors.

    For example:
    - Constants are always public
    - Input tensors have explicitly specified secrecy
    - Binary operations (add, mul etc) produce secret output if either input is secret
    - Unary operations (transpose, sum etc) preserve the secrecy of their input
    - Conv2d requires encrypted image and public filter weights

    The secrecy information is used during layout assignment to ensure proper handling
    of encrypted vs plaintext tensors.
    """

    def __init__(self, comp):
        self.comp = comp
        self.secret = {}

    def get_secret(self, term, secrets):
        """find if each term is secret or public"""
        match term.op:
            case TensorOp.TENSOR:
                return term.cs[2]
            case TensorOp.CONST:
                return False
            case (
                TensorOp.TRANSPOSE
                | TensorOp.SUM
                | TensorOp.PRODUCT
                | TensorOp.POLY
                | TensorOp.RESHAPE
                | TensorOp.PERMUTE
                | TensorOp.INDEX
                | TensorOp.RESCALE
            ):
                return secrets[0]
            case (
                TensorOp.ADD
                | TensorOp.SUB
                | TensorOp.MUL
                | TensorOp.MATMUL
                | TensorOp.BLOCK_MATMUL
            ):
                a_secret = secrets[0]
                b_secret = secrets[1]
                return a_secret or b_secret
            case TensorOp.CONV2D:
                a_secret = secrets[0]
                b_secret = secrets[1]
                assert a_secret
                assert not b_secret
                return a_secret or b_secret
            case _:
                raise NotImplementedError(term.op)

    def get_term_secret(self, term):
        match term.op:
            case TensorOp.CONST:
                kernel_secret = False
                self.secret[term] = kernel_secret
            case TensorOp.TENSOR:
                kernel_secret = self.get_secret(term, [])
                self.secret[term] = kernel_secret
            case (
                TensorOp.ADD
                | TensorOp.SUB
                | TensorOp.MUL
                | TensorOp.MATMUL
                | TensorOp.BLOCK_MATMUL
            ):
                cs_secrets = [self.secret[term.cs[0]], self.secret[term.cs[1]]]
                kernel_secret = self.get_secret(term, cs_secrets)
                self.secret[term] = kernel_secret
            case TensorOp.CONV2D:
                assert self.secret[term.cs[0]]
                assert not self.secret[term.cs[1]]
                kernel_secret = self.secret[term.cs[0]]
                self.secret[term] = kernel_secret
            case (
                TensorOp.TRANSPOSE
                | TensorOp.SUM
                | TensorOp.RESHAPE
                | TensorOp.PERMUTE
                | TensorOp.INDEX
                | TensorOp.RESCALE
            ):
                cs_secrets = [self.secret[term.cs[0]]]
                kernel_secret = self.get_secret(term, cs_secrets)
                self.secret[term] = kernel_secret
            case _:
                raise NotImplementedError(term.op)

    def run(self):
        for term in self.comp.post_order():
            self.get_term_secret(term)
