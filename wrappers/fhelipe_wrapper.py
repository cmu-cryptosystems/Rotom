import argparse
import os.path
import numpy as np
from types import SimpleNamespace


from frontends.tensor import TensorTerm, TensorOp
from ir.dim import Dim
from ir.layout import Layout
from ir.he import HETerm, HEOp
from backends.openfhe_backend import CKKS
from util.layout_util import convert_layout_to_mask


class FhelipeWrapper:
    def __init__(self, args):
        self.args = args
        self.env = {}
        self.seen = set()

    def parse_layout(self, name, layout_file):
        with open(layout_file, "r") as file:
            for line in file:
                layout = line.split("  ")[1].split()[1:]
        dim_index = None
        extent = 1
        dims = []
        for i in range(0, len(layout), 2):
            next_dim_index = layout[i]
            if dim_index is None:
                dim_index = next_dim_index
                extent *= 2
            elif dim_index != next_dim_index:
                if dim_index == "#":
                    dim = None
                else:
                    dim = int(dim_index)
                dims.append(Dim(dim, extent))

                dim_index = next_dim_index
                extent = 2
            else:
                extent *= 2
        if dim_index == "#":
            dim = None
        else:
            dim = int(dim_index)
        dims.append(Dim(dim, extent))
        layout = Layout(
            TensorTerm(TensorOp.TENSOR, [name]), [], dims, {}, self.args.n, True
        )
        return layout

    def parse_pt(self, name, pt_file):
        with open(pt_file, "r") as file:
            for line in file:
                elems = line.split(" ")

        if elems[0] == "INDIRECTION":
            num_dims = 1
            shape = int(elems[2])
            dims = []
            for i in range(num_dims):
                dims.append(Dim(i, int(shape ** (1 / num_dims))))

            layout = Layout(
                TensorTerm(TensorOp.TENSOR, [name]), [], dims, {}, self.args.n, False
            )
            return HETerm(HEOp.PACK, [layout], False, "0 ")
        elif elems[0] == "MASK":
            mask = [0] * self.args.n
            for elem in elems[3:]:
                if elem:
                    mask[int(elem)] = 1
            return HETerm(HEOp.MASK, [mask], False)

    def parse_input_c(self, line):
        elems = line.split()
        var = elems[0]
        name = elems[4]
        layout_file = self.args.path + f"/enc.cfg/{name}"
        layout = self.parse_layout(name, layout_file)
        
        if var in self.env:
            raise KeyError
        self.env[var] = HETerm(HEOp.PACK, [layout], True, f"0 {var}")
        return self.env[var]

    def parse_zero_c(self, line):
        # create an all 0 layout
        elems = line.split()
        var = elems[0]
        layout = Layout(
            TensorTerm(TensorOp.TENSOR, [f"zero_{var}"]),
            [],
            [Dim(0, self.args.n)],
            {},
            True,
            "0 ",
        )
        if var in self.env:
            raise KeyError
        self.env[var] = HETerm(HEOp.PACK, [layout], True, "0 ")
        return self.env[var]

    def parse_rotate_c(self, line):
        elems = line.split()
        var = elems[0]
        rot_amt = int(elems[4])
        src = elems[8]
        if var in self.env:
            raise KeyError
        self.env[var] = self.env[src] << rot_amt
        return self.env[var]

    def parse_add_cc(self, line):
        elems = line.split()
        var = elems[0]
        l = elems[7]
        r = elems[8]
        if var in self.env:
            raise KeyError
        self.env[var] = self.env[l] + self.env[r]
        return self.env[var]

    def parse_mul_cc(self, line):
        elems = line.split()
        var = elems[0]
        l = elems[7]
        r = elems[8]
        if var in self.env:
            raise KeyError
        self.env[var] = self.env[l] * self.env[r]
        return self.env[var]

    def parse_add_cp(self, line):
        elems = line.split()
        var = elems[0]
        pt = elems[4]
        r = elems[9]

        # pack pt:
        pt_file = self.args.path + f"/ch_ir/{pt}"
        pt_term = self.parse_pt(pt, pt_file)
        self.env[pt] = pt_term
        if var in self.env:
            raise KeyError
        self.env[var] = self.env[r] + pt_term
        return self.env[var]

    def parse_mul_cp(self, line):
        elems = line.split()
        var = elems[0]
        pt = elems[4]
        r = elems[9]

        # pack pt:
        pt_file = self.args.path + f"/ch_ir/{pt}"
        pt_term = self.parse_pt(pt, pt_file)
        self.env[pt] = pt_term
        if var in self.env:
            raise KeyError
        self.env[var] = self.env[r] * pt_term
        return self.env[var]

    def parse_rescale_c(self, line):
        elems = line.split()
        var = elems[0]
        # print(var)
        if var in self.env:
            raise KeyError
        self.env[var] = HETerm(
            HEOp.RESCALE, [self.env[elems[7]], var], self.env[elems[7]].secret
        )
        # self.env[var] = self.env[elems[7]]
        return self.env[var]

    def parse_output(self, line):
        elems = line.split()
        var = elems[0]
        if var in self.env:
            raise KeyError
        self.env[var] = self.env[elems[9]]
        return self.env[var]

    def create_comp(self):
        rt_file = self.args.path + "/rt.df"
        with open(rt_file, "r") as file:
            output = []
            for line in list(file)[2:]:
                op = line.split()[3]
                match op:
                    case "InputC":
                        term = self.parse_input_c(line)
                    case "ZeroC":
                        term = self.parse_zero_c(line)
                    case "RotateC":
                        term = self.parse_rotate_c(line)
                    case "AddCC":
                        term = self.parse_add_cc(line)
                    case "MulCC":
                        term = self.parse_mul_cc(line)
                    case "AddCP":
                        term = self.parse_add_cp(line)
                    case "MulCP":
                        term = self.parse_mul_cp(line)
                    case "RescaleC":
                        term = self.parse_rescale_c(line)
                    case "OutputC":
                        output.append(self.parse_output(line))
                    case _:
                        raise NotImplementedError(f"line: {line}")
                # if term in self.seen and "RescaleC" not in line:
                #     print("seen:", line)
                #     exit(0)
                # self.seen.add(term)
        return output

    def run(self, comp, inputs, path, args=None):
        benchmark = path.split("/")[-1]
        # Create args object for CKKS backend
        # Use provided args if available, otherwise use self.args
        source_args = args if args is not None else self.args
        ckks_args = SimpleNamespace(
            n=self.args.n,
            benchmark=f"fhelipe_{benchmark}",
            mock=True,
            serialize=True,
            cache=False,
            net="lan",
            not_secure=getattr(source_args, "not_secure", False)
        )
        return CKKS(comp, inputs, ckks_args).run_wrapper()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="--path <filename>", type=str)
    parser.add_argument(
        "-n", help="-n <slots>, default=4096", nargs="?", type=int, default=4096
    )
    args = parser.parse_args()
    assert os.path.exists(args.path)

    w = FhelipeWrapper(args)
    comp = w.create_comp()
    results = w.run(comp, {}, args.path)


if __name__ == "__main__":
    main()
