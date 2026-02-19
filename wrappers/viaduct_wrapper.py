import argparse
import os.path
import re
from types import SimpleNamespace

from backends.openfhe_backend import CKKS
from frontends.tensor import TensorOp, TensorTerm
from ir.dim import Dim
from ir.he import HEOp, HETerm
from ir.layout import Layout
from ir.roll import Roll


class ViaductWrapper:
    def __init__(self, args):
        self.args = args
        self.env = {}
        self.outputs = []

    def parse_layout(self, line):
        pattern = r"\{([^}]*)\}"
        matches = re.findall(pattern, line)
        assert matches
        return matches

    def parse_roll(self, line):
        pattern = r"\d+"
        matches = re.findall(pattern, line)
        assert matches
        return [int(match) for match in matches]

    def parse_vector(self, line):
        pattern = r"val (\w+): (\w+) = vector\((\w+)(\(\w+\(\d,\d\)\))?(\[[^\]]+\])(<[^>]+>)\)"
        match = re.match(pattern, line)
        if match:
            key = match.group(1)
            secret = match.group(2) == "C"
            input_name = match.group(3)
            offset = match.group(5)
            dims = self.parse_layout(match.group(6))

            # clean dims:
            dims = [dim.replace("::", ":") for dim in dims]

            clean_dims = []
            for dim in dims:
                if "oob" in dim:
                    dim = dim.split("[")[0]
                clean_dims.append(str(dim))
            dims = [Dim.parse(dim) for dim in clean_dims]
            rolls = []
            if match.group(4):
                roll = self.parse_roll(match.group(4))
                rolls.append(Roll(dims[roll[0]], dims[roll[1]]))
            packing_index = int(key.split("_")[-1]) - 1

            # clean dims:
            layout = Layout(
                TensorTerm(TensorOp.TENSOR, [input_name]),
                rolls,
                dims,
                self.args.n,
                secret,
            )

            # create HETerm
            self.env[key] = HETerm(
                HEOp.PACK, [layout, offset], layout.secret, f"{packing_index} "
            )
            return self.env[key]
        else:
            raise ValueError(f"couldn't properly process vector: {line}")

    def parse_const(self, line):
        pattern = r"val (\w+): \w+ = const\(([-\d]+)\)"
        match = re.match(pattern, line)
        if match:
            key = match.group(1)
            value = int(match.group(2))
            self.env[key] = value
            return self.env[key]
        else:
            raise ValueError(f"couldn't properly process const: {line}")

    def parse_mask(self, line):
        key = line.split(":")[0].split()[-1]
        self.env[key] = HETerm(HEOp.MASK, [[1] * self.args.n], False, f"mask {key}")
        return self.env[key]

    def parse_cc(self, line):
        elems = line.split()
        key = elems[1]
        op1 = elems[3]
        op2 = elems[4]
        op3 = elems[5]

        if op1 == "rot":
            if "rot_amt" in op2:
                self.env[key] = self.env[op3] << self.env[op2]
            else:
                self.env[key] = self.env[op3] << int(op2)
        elif op2 == "*":
            self.env[key] = self.env[op1] * self.env[op3]
        elif op2 == "+":
            self.env[key] = self.env[op1] + self.env[op3]
        elif op2 == "-":
            self.env[key] = self.env[op1] - self.env[op3]
        return self.env[key]

    def parse_cp(self, line):
        elems = line.split()
        key = elems[1]
        op1 = elems[3]
        op2 = elems[4]
        op3 = elems[5]

        if op1 == "rot":
            self.env[key] = self.env[op3] << int(op2)
        elif op2 == "*":
            self.env[key] = self.env[op1] * self.env[op3]
        elif op2 == "+":
            self.env[key] = self.env[op1] + self.env[op3]
        elif op2 == "-":
            self.env[key] = self.env[op1] - self.env[op3]
        return self.env[key]

    def parse_pp(self, line):
        elems = line.split()
        key = elems[1]
        op1 = elems[3]
        op2 = elems[4]
        op3 = elems[5]

        if op2 == "*":
            self.env[key] = int(op1) * int(op3.replace("[", "").replace("]", ""))
        elif op2 == "-":
            self.env[key] = self.env[op1] - int(op3)
        else:
            print(line)
            raise NotImplementedError
        return self.env[key]

    def parse_assign(self, line):
        elems = line.split()
        self.env[elems[0]] = self.env[elems[2]]
        return self.env[elems[0]]

    def parse_for_block(self, lines):
        for_loop_var = "[" + lines[0].split()[1].replace(":", "") + "]"
        for_loop_extent = int(lines[0].split()[2])
        for i in range(for_loop_extent):
            for line in lines[1:]:
                if not line.strip():
                    continue

                line = line.replace(for_loop_var, f"[{i}]").strip()
                if line.startswith("CC:"):
                    self.parse_cc(line)
                elif line.startswith("CP:") or line.startswith("N:"):
                    self.parse_cp(line)
                elif line.startswith("PP:"):
                    self.parse_pp(line)
                elif "out" in line:
                    output = self.parse_assign(line)
                    self.outputs.append(output)
                elif len(line.split()) == 3 and "=" in line:
                    self.parse_assign(line)
                elif "encode" in line:
                    continue
                elif line.startswith("var"):
                    continue
                else:
                    raise NotImplementedError(f"not implemented: {line}")

    def parse_line(self, line):
        if "encode" in line:
            # skip
            return

        prefix = line.split()[0]
        if prefix.startswith("val"):
            if "vector" in line:
                return self.parse_vector(line)
            elif "const" in line:
                return self.parse_const(line)
            elif "mask" in line:
                return self.parse_mask(line)
            else:
                raise NotImplementedError(f"not implemented: val: {line}")
        elif prefix.startswith("CC:"):
            return self.parse_cc(line)
        elif prefix.startswith("CP:"):
            return self.parse_cp(line)
        elif prefix.startswith("var"):
            # skip
            return
        elif "out" in line:
            output = self.parse_assign(line)
            self.outputs.append(output)
        elif len(line.split()) == 3 and "=" in line:
            return self.parse_assign(line)
        else:
            raise NotImplementedError(f"not implemented: {line}")

    def create_comp(self):
        with open(self.args.path, "r") as file:
            for_block = False
            lines = []
            for line in file:
                if line.startswith("}"):
                    # parse for block
                    for_block = False
                    self.parse_for_block(lines)
                    lines = []
                    continue
                if line.startswith("for"):
                    for_block = True
                if for_block:
                    lines.append(line)
                    continue
                self.parse_line(line)
            return self.outputs

    def run(self, comp, inputs, path, args=None):
        benchmark = path.split("/")[-1]
        # Create args object for CKKS backend
        # Use provided args if available, otherwise use self.args
        source_args = args if args is not None else self.args
        ckks_args = SimpleNamespace(
            n=self.args.n,
            benchmark=f"viaduct_{benchmark}",
            mock=True,
            serialize=True,
            cache=False,
            net="lan",
            not_secure=getattr(source_args, "not_secure", False),
        )
        return CKKS(comp, inputs, ckks_args).run_wrapper()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="--path <filename>", type=str)
    parser.add_argument(
        "--n", help="--n <slots>, default=4096", nargs="?", type=int, default=4096
    )
    args = parser.parse_args()
    assert os.path.exists(args.path)

    w = ViaductWrapper(args)
    comp = w.create_comp()
    w.run(comp, {}, args.path)


if __name__ == "__main__":
    main()
