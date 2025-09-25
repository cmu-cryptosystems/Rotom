from enum import Enum


class FHEOp(Enum):
    CS = "CS"
    CS_PACK = "CS_PACK"
    PACK = "PACK"
    INDICES = "INDICES"
    ADD = "ADD"
    SUB = "SUB"
    MUL = "MUL"
    ROT = "ROT"
    MASK = "MASK"
    POLY = "POLY"
    RESCALE = "RESCALE"
    ZERO_MASK = "ZERO_MASK"


class FHETerm:
    def __init__(self, op, cs, secret, metadata=""):
        self.op = op
        self.cs = cs
        self.metadata = metadata

        assert isinstance(secret, bool)
        self.secret = secret

        if isinstance(self.cs, tuple):
            raise ValueError("?")

        cs_hashes = [c.hash if isinstance(c, FHETerm) else c for c in self.cs]
        if self.op == FHEOp.PACK or self.op == FHEOp.CS_PACK or self.op == FHEOp.CS:
            self.hash = hash(f"{self.op}:{cs_hashes}:{self.metadata}")
        else:
            self.hash = hash(f"{self.op}:{cs_hashes}")


    def ops(self):
        count = {}
        for term in self.post_order():
            match term.op:
                case FHEOp.ADD | FHEOp.MUL | FHEOp.ROT:
                    if term.secret:
                        if term.op not in count:
                            count[term.op] = 0
                        count[term.op] += 1
        return count

    def __hash__(self):
        return self.hash

    def __eq__(self, other):
        return self.hash == other.hash

    def __add__(self, other):
        return FHETerm(FHEOp.ADD, [self, other], self.secret or other.secret)

    def __sub__(self, other):
        return FHETerm(FHEOp.SUB, [self, other], self.secret or other.secret)

    def __mul__(self, other):
        return FHETerm(FHEOp.MUL, [self, other], self.secret or other.secret)

    def __lshift__(self, other):
        return FHETerm(FHEOp.ROT, [self, other], self.secret)

    def pack(layout, metadata):
        return FHETerm(FHEOp.PACK, [layout], layout.secret, metadata)

    def mask(mask):
        return FHETerm(FHEOp.MASK, mask, False)

    def format_metadata(self, instr_str):
        if self.metadata:
            return f"{instr_str} # {self.metadata}"
        return instr_str

    def instrs(self):
        env = {}
        instructions = []
        index = 0
        for term in self.post_order():
            if term not in env:
                env[term] = index
                instructions.append(term)
                index += 1

        instruction_strs = []
        for i, term in enumerate(instructions):
            match term.op:
                case FHEOp.PACK | FHEOp.CS_PACK:
                    instruction_strs.append(
                        term.format_metadata(f"{i} {term.secret}: ({term.cs[0].layout_str()})")
                    )
                case FHEOp.MASK:
                    instruction_strs.append(
                        term.format_metadata(f"{i} {term.secret}: mask")
                    )
                case FHEOp.ADD:
                    a = env[term.cs[0]]
                    b = env[term.cs[1]]
                    instruction_strs.append(term.format_metadata(f"{i} {term.secret}: (+ {a} {b})"))
                case FHEOp.SUB:
                    a = env[term.cs[0]]
                    b = env[term.cs[1]]
                    instruction_strs.append(term.format_metadata(f"{i} {term.secret}: (- {a} {b})"))
                case FHEOp.MUL:
                    a = env[term.cs[0]]
                    b = env[term.cs[1]]
                    instruction_strs.append(term.format_metadata(f"{i} {term.secret}: (* {a} {b}), ({term.cs[0].secret}, {term.cs[1].secret})"))
                case FHEOp.ROT:
                    a = env[term.cs[0]]
                    b = str(term.cs[1])
                    instruction_strs.append(term.format_metadata(f"{i} {term.secret}: (<< {a} {b})"))
                case FHEOp.POLY:
                    a = env[term.cs[0]]
                    instruction_strs.append(term.format_metadata(f"{i} {term.secret}: (poly {a})"))
        return instruction_strs

    def __repr__(self):
        match self.op:
            case FHEOp.MASK:
                return f"{self.op} {self.cs[0]} {self.metadata}"
            case FHEOp.ROT:
                return f"{self.op} {self.cs[1]} {self.metadata}"
            case _:
                return f"{self.op} {self.metadata}"

    def helper_post_order(self, seen):
        if self in seen:
            return []
        seen.add(self)
        match self.op:
            case FHEOp.PACK | FHEOp.INDICES | FHEOp.CS | FHEOp.CS_PACK:
                return [self]
            case FHEOp.ADD | FHEOp.SUB | FHEOp.MUL:
                a = self.cs[0].helper_post_order(seen)
                b = self.cs[1].helper_post_order(seen)
                return a + b + [self]
            case FHEOp.ROT | FHEOp.POLY | FHEOp.RESCALE:
                a = self.cs[0].helper_post_order(seen)
                return a + [self]
            case FHEOp.MASK | FHEOp.ZERO_MASK:
                return [self]
            case _:
                raise NotImplementedError(self.op)

    def post_order(self):
        return self.helper_post_order(set())
