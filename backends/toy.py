from frontends.tensor import TensorOp
from ir.he import HEOp
from ir.kernel import KernelOp
from util.layout_util import *


class Toy:
    """
    Toy Backend for plaintext simulation of FHE circuits.
    
    This backend executes FHE circuits in plaintext, providing a fast
    simulation environment for development, testing, and debugging. It
    implements the same interface as other backends but performs all
    operations on plaintext data.
    
    Attributes:
        circuit_ir: The FHE circuit intermediate representation
        inputs: Dictionary of input tensors
        n: HE vector size
        env: Environment for storing intermediate results
        input_cache: Cache for packed input tensors
    """

    def __init__(self, circuit_ir, inputs, args):
        """
        Initialize the Toy backend.
        
        Args:
            circuit_ir: The FHE circuit intermediate representation
            inputs: Dictionary mapping tensor names to input data
            args: Command line arguments containing configuration
        """
        self.circuit_ir = circuit_ir
        self.inputs = inputs
        self.n = args.n
        self.env = {}
        self.input_cache = {}

    def eval_mask(self, term):
        """
        Evaluate a mask operation.
        
        Args:
            term: The mask term to evaluate
            
        Returns:
            The mask value
        """
        return term.cs[0]

    def eval_pack(self, term):
        """
        Evaluate a pack operation for tensor data.
        
        Packs tensor data according to the specified layout and returns
        the packed vector at the given packing index.
        
        Args:
            term: The pack term containing layout and metadata
            
        Returns:
            The packed vector at the specified index
        """
        layout = term.cs[0]
        if (layout.term, layout) not in self.input_cache:
            tensor = layout.term.eval(self.inputs)
            # apply layout to tensor
            packed_tensor = apply_layout(tensor, layout)
            self.input_cache[(layout.term, layout)] = packed_tensor

        # get packing index and return packed vector
        packing_idx = int(term.metadata.split()[0])
        return self.input_cache[(layout.term, layout)][packing_idx]

    def eval_cs_pack(self, term):
        layout = term.cs[1]
        if (layout.term, layout) not in self.input_cache:
            tensor = self.inputs[term.cs[0]]
            # apply layout to tensor
            packed_tensor = apply_layout(tensor, layout)
            self.input_cache[(layout.term, layout)] = packed_tensor

        # get packing index and return packed vector
        packing_idx = int(term.metadata.split()[0])
        return self.input_cache[(layout.term, layout)][packing_idx]

    def eval_indices(self, term):
        tensor = self.inputs[term.cs[0].cs[0]]
        vector = [0] * self.n
        for filter_index, position, _ in term.cs[1]:
            vector[position[0]] = tensor[
                filter_index[0], filter_index[1], filter_index[2]
            ]
        return vector

    def eval_rot(self, term):
        """positive == left rotate"""
        vector = self.env[term.cs[0]].copy()
        return [vector[(term.cs[1] + i) % len(vector)] for i in range(len(vector))]

    def eval_add(self, term):
        # print("add:", term)
        # print(self.env[term.cs[0]])
        # print(self.env[term.cs[1]])
        # print("result:", [a * b for a, b in zip(self.env[term.cs[0]], self.env[term.cs[1]])])
        # print()
        return [a + b for a, b in zip(self.env[term.cs[0]], self.env[term.cs[1]])]

    def eval_sub(self, term):
        return [a - b for a, b in zip(self.env[term.cs[0]], self.env[term.cs[1]])]

    def eval_mul(self, term):
        print("mul:", term)
        print(self.env[term.cs[0]])
        print(self.env[term.cs[1]])
        print("result:", [a * b for a, b in zip(self.env[term.cs[0]], self.env[term.cs[1]])])
        print()
        return [a * b for a, b in zip(self.env[term.cs[0]], self.env[term.cs[1]])]

    def eval_poly(self, term):
        return self.env[term.cs[0]]

    def eval(self, term):
        match term.op:
            case HEOp.CS:
                return self.env[term.cs[0]]
            case HEOp.MASK:
                return self.eval_mask(term)
            case HEOp.PACK:
                return self.eval_pack(term)
            case HEOp.CS_PACK:
                return self.eval_cs_pack(term)
            case HEOp.INDICES:
                return self.eval_indices(term)
            case HEOp.ROT:
                return self.eval_rot(term)
            case HEOp.ADD:
                return self.eval_add(term)
            case HEOp.SUB:
                return self.eval_sub(term)
            case HEOp.MUL:
                return self.eval_mul(term)
            case HEOp.POLY:
                return self.eval_poly(term)
            case HEOp.ZERO_MASK:
                return [0] * self.n
            case _:
                raise NotImplementedError(term.op)

    def run(self):
        results = []
        for term, cts in self.circuit_ir.items():
            results = []
            for _, ct in cts.items():
                if isinstance(ct, list):
                    for c in ct:
                        for ct_term in c.post_order():
                            self.env[ct_term] = self.eval(ct_term)
                        results.append(self.env[ct_term])
                else:
                    for ct_term in ct.post_order():
                        self.env[ct_term] = self.eval(ct_term)
                    results.append(self.env[ct_term])


            print("expected layout:", term.layout)
            print("expected value:", term.layout.term.eval(self.inputs))
            expected = apply_layout(term.layout.term.eval(self.inputs), term.layout)
            for e in expected:
                print("e:", e)
            print()

            # skip checks for split rolls
            if term.op in [KernelOp.SPLIT_ROLL, KernelOp.REPLICATE, KernelOp.INDEX]:
                continue 

            print("kernel:", term)
            for k in term.post_order():
                print("k:", k)
            print()

            for r in results:
                print("r:", r)
            print()

            for e in expected:
                print("e:", e)
            print()

            assert results == expected
        return results


    def fuzz(self):
        results = []
        for term, cts in self.circuit_ir.items():
            # print("term:", term)
            results = []
            for ct_idx, ct in cts.items():
                # print("ct_idx:", ct_idx)
                # print("ct:", ct)
                if isinstance(ct, list):
                    for c in ct:
                        for ct_term in c.post_order():
                            self.env[ct_term] = self.eval(ct_term)
                        results.append(self.env[ct_term])
                else:
                    for ct_term in ct.post_order():
                        self.env[ct_term] = self.eval(ct_term)
                    results.append(self.env[ct_term])

            expected = apply_layout(
                term.layout.term.eval(self.inputs), term.layout)
            # if results != expected:
            #     print("results:", results)
            #     print("expected:", expected)
            #     print("diff:")
            #     for expected, result in zip(expected, results):
            #         print([e - r for e, r in zip(expected, result)])
            #     print()
            #     print(term)
            assert results == expected

            # check that results match up
            if term.layout.term.op == TensorOp.TENSOR:
                expected = apply_layout(
                    self.inputs[term.layout.term.cs[0]], term.layout
                )
                # print("term:", term)
                # print("layout:", term.layout)
                # print("layout offset:", term.layout.offset)
                # print(self.inputs[term.layout.term.cs[0]])
                # print("expected:", expected)
                # print("results:", results)
                # print()
                assert results[: len(expected)] == expected
            else:
                # print(term.layout.term)
                expected = apply_layout(
                    term.layout.term.eval(self.inputs), term.layout)
                # print("expected:")
                # for e in expected:
                #     print(e)
                # print()
                # print("results:")
                # for result in results:
                #     print(result)
                # print()

                # if results != expected:
                #     print("diff:")
                #     for expected, result in zip(expected, results):
                #         print([e - r for e, r in zip(expected, result)])
                #     print()
                #     print(term)
                assert results == expected
            print("check passed:", term)
            print()

        return results

    def fuzz(self):
        results = []
        for term, cts in self.circuit_ir.items():
            print("term:", term)
            results = []
            for _, ct in cts.items():
                if isinstance(ct, list):
                    for c in ct:
                        for ct_term in c.post_order():
                            self.env[ct_term] = self.eval(ct_term)
                        results.append(self.env[ct_term])
                else:
                    for ct_term in ct.post_order():
                        self.env[ct_term] = self.eval(ct_term)
                    results.append(self.env[ct_term])
        return results
