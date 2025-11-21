import os
import os.path
import shutil

import numpy as np

from ir.he import HEOp, HETerm
from util.layout_util import apply_layout


class HEIR:
    """
    target HEIR MLIR
    """

    def __init__(self, circuit_ir, inputs, args):
        self.circuit_ir = circuit_ir
        self.inputs = inputs
        self.n = args.n
        self.env = {}   
        self.pt_env = {}
        self.fn = args.fn

        if os.path.exists(f"heir/{self.fn}"):
            shutil.rmtree(f"heir/{self.fn}")
        os.mkdir(f"heir/{self.fn}")
        os.mkdir(f"heir/{self.fn}/base")
        os.mkdir(f"heir/{self.fn}/tensor")
        os.mkdir(f"heir/{self.fn}/inputs")

        self.input_cache = {}
        self.lines = []
        self.constants = []
        self.term_to_vector = {}
        self.term_to_id = {}
        self.term_to_type = {}
        self.next_id = 1
        self.returns = []



    def write_vector(self, fn, data):
        # Convert to numpy array for easier handling
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        
        with open(fn, 'w') as f:
            if data.ndim == 1:
                # 1D vector
                f.write(f"{len(data)}\n")
                for value in data:
                    f.write(f"{value}\n")
            elif data.ndim == 2:
                # 2D vector (matrix)
                rows, cols = data.shape
                f.write(f"{rows} {cols}\n")
                for row in data:
                    f.write(' '.join(map(str, row)) + '\n')
            else:
                raise ValueError(f"Only 1D and 2D vectors supported, got {data.ndim}D")


    def eval_mask(self, term):
        out_id = self.term_to_id[term]
        out_type = self.term_to_type[term]
        line = f"{out_id} = arith.constant dense<{term.cs[0]}> : {out_type}"
        self.lines.append(line)

    def eval_pack(self, term):
        layout = term.cs[0]
        if layout not in self.input_cache:
            fn = f"heir/{self.fn}/base/{layout.term}_{layout}.txt"
            if os.path.exists(fn):
                self.input_cache[layout] = self.read_vector(fn)
            else:
                tensor = layout.term.eval(self.inputs)
                tensor_fn = f"heir/{self.fn}/tensor/{layout.term}.txt"
                self.write_vector(tensor_fn, tensor)

                # apply layout to tensor
                packed_tensor = apply_layout(tensor, layout)
                self.input_cache[layout] = packed_tensor

                # save to cache
                self.write_vector(fn, packed_tensor)

        # get packing index and return packed vector
        packing_idx = int(term.metadata.split()[0])
        vector = self.input_cache[layout][packing_idx]
        return vector

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

        rot_amt = term.cs[1]
        if rot_amt not in self.term_to_id:
            self.term_to_id[rot_amt] = f"%c{self.next_id}"
            self.term_to_type[rot_amt] = "index"
            self.next_id += 1
            line = f"{self.term_to_id[rot_amt]} = arith.constant {rot_amt} : {self.term_to_type[rot_amt]}"
            self.constants.append(line)

        a_term_id = self.term_to_id[term.cs[0]]
        a_term_type = self.term_to_type[term.cs[0]]
        b_term_id = self.term_to_id[term.cs[1]]
        b_term_type = self.term_to_type[term.cs[1]]
        out_term_id = self.term_to_id[term]
        line = f"{out_term_id} = tensor_ext.rotate {a_term_id}, {b_term_id} : {a_term_type}, {b_term_type}"
        self.lines.append(line)

    def eval_add(self, term):
        a_term_id = self.term_to_id[term.cs[0]]
        b_term_id = self.term_to_id[term.cs[1]]
        out_term_id = self.term_to_id[term]
        out_term_type = self.term_to_type[term]
        line = f"{out_term_id} = arith.addf {a_term_id}, {b_term_id} : {out_term_type}"
        self.lines.append(line)

    def eval_sub(self, term):
        a_term_id = self.term_to_id[term.cs[0]]
        b_term_id = self.term_to_id[term.cs[1]]
        out_term_id = self.term_to_id[term]
        out_term_type = self.term_to_type[term]
        line = f"{out_term_id} = arith.subf {a_term_id}, {b_term_id} : {out_term_type}"
        self.lines.append(line)

    def eval_mul(self, term):
        if term.cs[0] not in self.term_to_id:
            vector = self.env[term.cs[0]
                              ] if term.cs[0].secret else self.pt_env[term.cs[0]]
            self.term_to_vector[term.cs[0]] = vector
            self.term_to_id[term.cs[0]] = f"%{self.next_id}"
            self.term_to_type[term.cs[0]] = self.term_type(term.cs[0])
            self.next_id += 1

        if term.cs[1] not in self.term_to_id:
            vector = self.env[term.cs[1]
                              ] if term.cs[1].secret else self.pt_env[term.cs[1]]
            self.term_to_vector[term.cs[1]] = vector
            self.term_to_id[term.cs[1]] = f"%{self.next_id}"
            self.term_to_type[term.cs[1]] = self.term_type(term.cs[1])
            self.next_id += 1

        a_term_id = self.term_to_id[term.cs[0]]
        b_term_id = self.term_to_id[term.cs[1]]
        out_term_id = self.term_to_id[term]
        out_term_type = self.term_to_type[term]
        line = f"{out_term_id} = arith.mulf {a_term_id}, {b_term_id} : {out_term_type}"
        self.lines.append(line)

    def term_input_type(self, term):
        if term.secret:
            return f"tensor<{self.n}xf64> " + "{secret.secret}"
        else:
            return f"tensor<{self.n}xf64>"

    def term_type(self, term):
        return f"tensor<{self.n}xf64>"

    def eval(self, term):
        if term in self.term_to_id:
            return

        if term.op not in [HEOp.CS] and term not in self.term_to_id:
            self.term_to_id[term] = f"%{self.next_id}"
            self.term_to_type[term] = self.term_type(term)
            self.next_id += 1

        match term.op:
            case HEOp.PACK:
                self.eval_pack(term)
            case HEOp.MASK:
                self.eval_mask(term)
            case HEOp.ADD:
                self.eval_add(term)
            case HEOp.SUB:
                self.eval_sub(term)
            case HEOp.MUL:
                self.eval_mul(term)
            case HEOp.ROT:
                self.eval_rot(term)
            case HEOp.CS:
                self.term_to_id[term] = self.term_to_id[term.cs[0]]
                self.term_to_type[term] = self.term_to_type[term.cs[0]]
            case _:
                raise NotImplementedError(term.op)

    def write_to_file(self):
        print("writing to file...")
        # create header
        parameters = []
        for term, vector in self.term_to_vector.items():
            term_id = self.term_to_id[term]
            parameters.append(
                f"{term_id} : {self.term_input_type(term)}")

            var_name = term_id.lstrip('%')
            fn = f"heir/{self.fn}/inputs/{var_name}.txt"
            self.write_vector(fn, vector)

        parameter_str = ", ".join(parameters)

        return_types = []
        for term in self.returns:
            return_types.append(self.term_to_type[term])
        return_types_str = ", ".join(return_types)

        lines = []
        header = f"func.func @{self.fn}({parameter_str}) -> {return_types_str} " + "{\n"
        lines.append(header)
        for c in self.constants:
            lines.append("  " + c + "\n")
        for line in self.lines:
            lines.append("  " + line + "\n")
        for ret in self.returns:
            lines.append(
                "  return "
                + self.term_to_id[ret]
                + " : "
                + self.term_to_type[ret]
                + "\n"
            )
        closer = "}\n"
        lines.append(closer)

        with open(f"heir/{self.fn}/{self.fn}.mlir", "w") as f:
            f.writelines(lines)

    def dagify_fhe_circuit(self):
        ct_env = {}
        results = []
        for _, cts in self.circuit_ir.items():
            results = []
            for term, ct in cts.items():
                if isinstance(ct, list):
                    for term in ct:
                        for ct_term in term.post_order():
                            for i, cs_ct_term in enumerate(ct_term.cs):
                                if isinstance(cs_ct_term, HETerm):
                                    ct_term.cs[i] = ct_env[cs_ct_term]
                            if ct_term.op == HEOp.CS:
                                ct_env[ct_term] = ct_term.cs[0]
                            else:
                                ct_env[ct_term] = ct_term
                else:
                    for ct_term in ct.post_order():
                        for i, cs_ct_term in enumerate(ct_term.cs):
                            if isinstance(cs_ct_term, HETerm):
                                ct_term.cs[i] = ct_env[cs_ct_term]
                        if ct_term.op == HEOp.CS:
                            ct_env[ct_term] = ct_term.cs[0]
                        else:
                            ct_env[ct_term] = ct_term
                results.append(ct_term)
        return results

    def preprocess_packing(self, cts):
        print("preprocessing packing...")
        for ct in cts:
            for ct_term in ct.post_order():
                if ct_term in self.env:
                    continue
                match ct_term.op:
                    case HEOp.PACK:
                        if not ct_term.secret:
                            self.pt_env[ct_term] = self.eval_pack(ct_term)
                        else:
                            self.env[ct_term] = self.eval_pack(ct_term)

    def preprocess_pt_compute(self, cts):
        print("preprocessing pt compute...")
        for ct in cts:
            for ct_term in ct.post_order():
                if ct_term.secret:
                    continue
                match ct_term.op:
                    case HEOp.PACK:
                        continue
                    case HEOp.MASK:
                        self.pt_env[ct_term] = ct_term.cs[0]
                    case HEOp.ADD:
                        if not ct_term.cs[0].secret and not ct_term.cs[1].secret:
                            assert ct_term.cs[0] in self.pt_env
                            assert ct_term.cs[1] in self.pt_env
                            self.pt_env[ct_term] = [
                                a + b
                                for a, b in zip(
                                    self.pt_env[ct_term.cs[0]],
                                    self.pt_env[ct_term.cs[1]],
                                )
                            ]
                    case HEOp.SUB:
                        if not ct_term.cs[0].secret and not ct_term.cs[1].secret:
                            assert ct_term.cs[0] in self.pt_env
                            assert ct_term.cs[1] in self.pt_env
                            self.pt_env[ct_term] = [
                                a - b
                                for a, b in zip(
                                    self.pt_env[ct_term.cs[0]],
                                    self.pt_env[ct_term.cs[1]],
                                )
                            ]
                    case HEOp.MUL:
                        if not ct_term.cs[0].secret and not ct_term.cs[1].secret:
                            assert ct_term.cs[0] in self.pt_env
                            assert ct_term.cs[1] in self.pt_env
                            self.pt_env[ct_term] = [
                                a * b
                                for a, b in zip(
                                    self.pt_env[ct_term.cs[0]],
                                    self.pt_env[ct_term.cs[1]],
                                )
                            ]
                    case HEOp.ROT:
                        if not ct_term.cs[0].secret:
                            assert ct_term.cs[0] in self.pt_env
                            self.pt_env[ct_term] = [
                                self.pt_env[ct_term.cs[0]][(
                                    i + ct_term.cs[1]) % self.n]
                                for i in range(len(self.pt_env[ct_term.cs[0]]))
                            ]
                    case HEOp.INDICES:
                        if not ct_term.secret:
                            tensor = self.inputs[ct_term.cs[0].cs[0]]
                            vector = [0] * self.n
                            for filter_index, position, _ in ct_term.cs[1]:
                                vector[position[0]] = tensor[
                                    filter_index[0], filter_index[1], filter_index[2]
                                ]
                            self.pt_env[ct_term] = vector
                    case HEOp.ZERO_MASK:
                        self.pt_env[ct_term] = [0] * self.n
                    case _:
                        raise NotImplementedError(ct_term.op)

    def run(self):
        print("evaluating terms...")

        print("dagifying...")
        cts = self.dagify_fhe_circuit()
        
        self.preprocess_packing(cts)
        self.preprocess_pt_compute(cts)

        for ct in cts:
            self.returns = []
            for ct_term in ct.post_order():
                if not ct_term.secret:
                    continue
                if ct_term in self.env:
                    continue
                self.eval(ct_term)
            self.returns.append(ct)

        self.write_to_file()
        print("done!")

    
    def serialize_results(self, results):
        os.makedirs(f"heir/{self.fn}/results", exist_ok=True)
        for i, result in enumerate(results):
            self.write_vector(f"heir/{self.fn}/results/result_{i}.txt", result)
    