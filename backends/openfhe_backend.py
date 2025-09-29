import time
import random
import numpy as np
import shutil
import os
import os.path
import pickle as pkl
from openfhe import *


from ir.he import HETerm, HEOp
from lower.lower_util import total_ops
from lower.circuit_opts.mask_opts import mask_identity_opt, zero_mask_identity_opt, zero_mask_opt
from lower.circuit_opts.rot_opts import rot_zero_opt, join_rot
from lower.circuit_opts.associativity import mul_associativity
from util.layout_util import *


class CKKS:
    """
    CKKS Backend using OpenFHE to run FHE in plaintext
    """

    def __init__(self, circuit_ir, inputs, args):
        self.circuit_ir = circuit_ir
        self.inputs = inputs
        self.n = args.n
        self.benchmark = args.benchmark
        self.cache = args.cache
        self.mock_inputs = args.mock
        self.serialize = args.serialize
        self.net = args.net
        self.not_secure = getattr(args, 'not_secure', False)  # Flag to disable 128-bit security
        self.env = {}
        self.pt_env = {}
        self.layout_map = {}
        self.input_cache = {}
        self.dependencies = {}
        self.rots = set()

        # clean data path
        if os.path.exists("./data"):
            shutil.rmtree("./data")
        os.mkdir("./data")

    def get_directory_size(self):
        path = "./data"
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(path):
            for file in filenames:
                file_path = os.path.join(dirpath, file)
                if os.path.isfile(file_path):  # Ensure it's a file
                    total_size += os.path.getsize(file_path)
        return total_size

    def convert_size(self, size):
        for unit in ['B', 'KB']:
            if size < 1024:
                return f"{size:.2f} {unit}"
            size /= 1024
        return f"{size:.2f} MB"

    def convert_comm_cost(self):
        data_size = self.get_directory_size()  # in bytes
        data_size *= 8  # in bits
        if self.net == "lan":
            # lan case, 1 Gbps
            return data_size / (10**9)
        else:
            # wan case, 100Mbps
            return data_size / (10**8)

    def find_unique_rots(self, cts):
        unique_rots = set()
        for ct in cts:
            for term in ct.post_order():
                if term.op == HEOp.ROT:
                    unique_rots.add(term.cs[1])
        return unique_rots

    def sum_rots(self, amt):
        powers = []
        power = 1

        pos_amt = abs(amt)
        while pos_amt > 0:
            if pos_amt & 1:  # If the current bit is 1, include the current power of 2
                powers.append(power)
            pos_amt >>= 1  # Shift n to the right
            power *= 2  # Move to the next power of 2

        if amt < 0:
            powers = [-p for p in powers]
        return powers

    def create_context(self, depth, rots):
        print("creating context...")
        parameters = CCParamsCKKSRNS()
        parameters.SetMultiplicativeDepth(depth)
        parameters.SetScalingModSize(45)
        parameters.SetScalingTechnique(ScalingTechnique.FLEXIBLEAUTO)
        
        # Set security level based on flag (inverted logic)
        if self.not_secure:
            parameters.SetSecurityLevel(HEStd_NotSet)
            print("Using default security level (not guaranteed)")
        else:
            parameters.SetSecurityLevel(HEStd_128_classic)
            print("Using 128-bit security level")
            
        parameters.SetRingDim(self.n * 2)

        self.cc = GenCryptoContext(parameters)

        print(
            f"CKKS scheme is using ring dimension {self.cc.GetRingDimension()}\n")

        self.cc.Enable(PKESchemeFeature.PKE)
        self.cc.Enable(PKESchemeFeature.KEYSWITCH)
        self.cc.Enable(PKESchemeFeature.LEVELEDSHE)
        self.keys = self.cc.KeyGen()
        self.cc.EvalMultKeyGen(self.keys.secretKey)

        print("generating rot keys...")
        print("len of rot keys:", len(rots))
        print(sorted(list(rots)))
        self.cc.EvalRotateKeyGen(self.keys.secretKey, list(rots))

        # if self.serialize:
        #     self.serialize_context_and_keys()

    def serialize_context_and_keys(self):
        serType = BINARY
        if not SerializeToFile("data/cryptocontext.txt", self.cc, serType):
            raise Exception(
                "Error writing serialization of the crypto context to cryptocontext.txt"
            )

        # Serialize the relinearization key
        if not self.cc.SerializeEvalMultKey(
            "data/key-eval-mult.txt", serType
        ):
            raise Exception(
                'Error writing serialization of the eval mult keys to "key-eval-mult.txt"'
            )

        # Serialize the rotation evaluation keys
        if not self.cc.SerializeEvalAutomorphismKey(
            "data/key-eval-rot.txt", serType
        ):
            raise Exception(
                'Error writing serialization of the eval rotate keys to "key-eval-rot.txt"'
            )

    def encode(self, vector):
        return self.cc.MakeCKKSPackedPlaintext(vector)

    def encrypt(self, pt):
        return self.cc.Encrypt(self.keys.publicKey, pt)

    def decrypt(self, ct):
        return self.cc.Decrypt(ct, self.keys.secretKey)

    def eval_mask(self, term):
        return term.cs[0]

    def serialize_ct(self, term, ct):
        if not SerializeToFile(f"data/{term}.txt", ct, BINARY):
            raise Exception(
                f"Error writing serialization of {term}"
            )

    def serialize_result(self, i, ct):
        if not SerializeToFile(f"data/result_{i}.txt", ct, BINARY):
            raise Exception(
                f"Error writing serialization of {i}"
            )

    def eval_pack(self, term, encrypt=False, cache=False):
        layout = term.cs[0]
        if self.mock_inputs:
            vector = [random.randint(0, 2)] * layout.n
            if encrypt:
                ct = self.encrypt(self.encode(vector))
                self.serialize_ct(term, ct)
                return ct
            return vector

        elif layout not in self.input_cache:
            fn = f"layout_cache/{self.benchmark}_{layout}.pkl"
            if cache and os.path.exists(fn):
                self.input_cache[layout] = pkl.load(open(fn, "rb"))
            else:
                tensor = layout.term.eval(self.inputs)

                # apply layout to tensor
                packed_tensor = apply_layout(tensor, layout)
                self.input_cache[layout] = packed_tensor

                # save to cache
                if cache:
                    pkl.dump(packed_tensor, open(fn, "wb"))

        # get packing index and return packed vector
        packing_idx = int(term.metadata.split()[0])
        vector = self.input_cache[layout][packing_idx]

        if encrypt:
            ct = self.encrypt(self.encode(vector))
            self.serialize_ct(term, ct)
            return ct
        return vector

    def eval_rot(self, term):
        """positive == left rotate"""
        base = self.env[term.cs[0]]
        try:
            return self.cc.EvalRotate(base, term.cs[1])
        except:
            return base

    def eval_add(self, term):
        return self.cc.EvalAdd(self.env[term.cs[0]], self.env[term.cs[1]])

    def eval_sub(self, term):
        return self.cc.EvalSub(self.env[term.cs[0]], self.env[term.cs[1]])

    def eval_mul(self, term):
        return self.cc.EvalMult(self.env[term.cs[0]], self.env[term.cs[1]])

    def eval_poly(self, term):
        return self.env[term.cs[0]]

    def eval(self, term):
        match term.op:
            case HEOp.PACK | HEOp.MASK:
                return self.env[term]
            case HEOp.ROT:
                return self.eval_rot(term)
            case HEOp.ADD:
                return self.eval_add(term)
            case HEOp.SUB:
                return self.eval_sub(term)
            case HEOp.MUL:
                return self.eval_mul(term)
            case HEOp.RESCALE:
                return self.env[term.cs[0]]
            case _:
                raise NotImplementedError(term.op)

    def track_dependencies(self, cts):
        print("tracking dependencies...")
        for ct in cts:
            for ct_term in ct.post_order():
                if not ct_term.secret:
                    continue
                for cs_ct_term in ct_term.cs:
                    if cs_ct_term not in self.dependencies:
                        self.dependencies[cs_ct_term] = []
                    self.dependencies[cs_ct_term].append(ct_term)

    def update_dependencies(self, term):
        match term.op:
            case HEOp.PACK | HEOp.MASK:
                pass
            case HEOp.ROT | HEOp.RESCALE:
                self.dependencies[term.cs[0]].remove(term)
                if not self.dependencies[term.cs[0]]:
                    del self.dependencies[term.cs[0]]
                    del self.env[term.cs[0]]
            case HEOp.ADD | HEOp.SUB | HEOp.MUL:
                self.dependencies[term.cs[0]].remove(term)
                self.dependencies[term.cs[1]].remove(term)
                if not self.dependencies[term.cs[0]]:
                    del self.dependencies[term.cs[0]]
                    del self.env[term.cs[0]]
                if not self.dependencies[term.cs[1]]:
                    del self.dependencies[term.cs[1]]
                    del self.env[term.cs[1]]
            case _:
                raise NotImplementedError(term.op)

    def preprocess_packing(self, cts):
        print("preprocessing packing...")
        for ct in cts:
            for ct_term in ct.post_order():
                if ct_term in self.env:
                    continue
                match ct_term.op:
                    case HEOp.PACK:
                        if not ct_term.secret:
                            self.pt_env[ct_term] = self.eval_pack(
                                ct_term, ct_term.secret, self.cache
                            )
                        else:
                            self.env[ct_term] = self.eval_pack(
                                ct_term, ct_term.secret, self.cache
                            )

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

    def preprocess_encoding(self, cts):
        print("preprocessing encoding...")
        for ct in cts:
            for ct_term in ct.post_order():
                match ct_term.op:
                    case HEOp.PACK | HEOp.ROT | HEOp.INDICES | HEOp.ZERO_MASK | HEOp.RESCALE:
                        continue
                    case HEOp.ADD | HEOp.SUB | HEOp.MUL:
                        if not ct_term.cs[0].secret and ct_term.cs[0] not in self.env:
                            self.env[ct_term.cs[0]] = self.encode(
                                self.pt_env[ct_term.cs[0]]
                            )
                        if not ct_term.cs[1].secret and ct_term.cs[1] not in self.env:
                            self.env[ct_term.cs[1]] = self.encode(
                                self.pt_env[ct_term.cs[1]]
                            )
                    case HEOp.CS:
                        if ct_term.cs[0] not in self.env:
                            self.env[ct_term.cs[0]] = self.encode(
                                self.pt_env[ct_term.cs[0]]
                            )
                        self.env[ct_term] = self.env[ct_term.cs[0]]
                    case HEOp.MASK:
                        self.env[ct_term] = self.encode(ct_term.cs[0])
                    case _:
                        raise NotImplementedError(ct_term.op)

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

    def run_fhe_circuit(self):
        print()
        ops = total_ops(self.circuit_ir)
        print(ops)
        print("== running circuit")
        start = time.time()
        results = []
        for ct in self.circuit_ir:
            for ct_term in ct.post_order():
                if not ct_term.secret:
                    continue
                if ct_term in self.env:
                    continue
                self.env[ct_term] = self.eval(ct_term)
                self.update_dependencies(ct_term)
            results.append(self.env[ct_term])
        runtime = time.time() - start
        print("circuit runtime:", runtime)
        return runtime, results

    def run_dagified_fhe_circuit(self, cts):
        print("== running circuit")
        start = time.time()
        results = []
        for ct in cts:
            if not ct.secret:
                continue
            for ct_term in ct.post_order():
                if not ct_term.secret:
                    continue
                if ct_term in self.env:
                    continue
                self.env[ct_term] = self.eval(ct_term)
                self.update_dependencies(ct_term)
            results.append(self.env[ct])
        runtime = time.time() - start
        print("circuit runtime:", runtime)

        ops = total_ops(cts)
        print(ops)
        return runtime, results

    def decrypt_to_plaintext(self, ct):
        pt = self.decrypt(ct)
        decrypted = [d.real for d in pt.GetCKKSPackedValue()]
        return decrypted

    def find_depth_rescale(self, ct):
        depth = {}
        for c in ct.post_order():
            match c.op:
                case HEOp.PACK | HEOp.MASK | HEOp.ZERO_MASK:
                    depth[c] = 0
                case HEOp.MUL | HEOp.ADD | HEOp.SUB:
                    depth[c] = max(depth[c.cs[0]], depth[c.cs[1]])
                case HEOp.ROT:
                    depth[c] = depth[c.cs[0]]
                case HEOp.RESCALE:
                    depth[c] = depth[c.cs[0]] + 1
                case _:
                    raise NotImplementedError(c.op)
        print("depth:", max(depth.values()))
        return max(depth.values())

    def find_depth(self, ct):
        depth = {}
        for c in ct.post_order():
            match c.op:
                case HEOp.PACK | HEOp.MASK | HEOp.ZERO_MASK:
                    depth[c] = 0
                case HEOp.MUL:
                    if c.secret:
                        depth[c] = max(depth[c.cs[0]], depth[c.cs[1]]) + 1
                    else:
                        depth[c] = max(depth[c.cs[0]], depth[c.cs[1]])
                case HEOp.ADD | HEOp.SUB:
                    depth[c] = max(depth[c.cs[0]], depth[c.cs[1]])
                case HEOp.ROT:
                    depth[c] = depth[c.cs[0]]
                case _:
                    raise NotImplementedError(c.op)
        print("depth:", max(depth.values()))
        return max(depth.values())

    def run_and_check(self, term, cts):
        opt_cts = []
        for ct_idx, ct in cts.items():
            opt_ct = rot_zero_opt(ct)
            opt_ct = zero_mask_opt(ct)
            opt_ct = mask_identity_opt(ct)
            opt_ct = zero_mask_identity_opt(ct)
            opt_cts.append(opt_ct)
        cts = opt_cts

        self.preprocess_packing(cts)
        self.preprocess_pt_compute(cts)
        self.preprocess_encoding(cts)
        self.track_dependencies(cts)

        runtime, results = self.run_dagified_fhe_circuit(cts)
        decrypted = [self.decrypt_to_plaintext(result) for result in results]
        expected_cts = apply_layout(
            term.layout.term.eval(self.inputs), term.layout)

        for a, b in zip(expected_cts, decrypted):
            diff = np.array(a) - np.array(b)
            if not np.allclose(a, b, rtol=1e-2, atol=1e-2):
                print(f"Expected: {a}")
                print(f"Actual: {b}")
                print(f"Max absolute difference: {np.max(np.abs(diff))}")
            assert np.allclose(a, b, rtol=1e-2, atol=1e-2), f"Values not close enough. Max diff: {np.max(np.abs(diff))}"
        print("pass!")
        print()

    def run(self):
        print("dagifying...")
        cts = self.dagify_fhe_circuit()
        print("running circuit opt...")
        opt_cts = []
        for ct in cts:
            opt_ct = rot_zero_opt(ct)
            opt_ct = join_rot(ct)
            opt_ct = zero_mask_opt(ct)
            opt_ct = mask_identity_opt(ct)
            opt_ct = zero_mask_identity_opt(ct)
            opt_ct = mul_associativity(ct)
            opt_cts.append(opt_ct)
        cts = opt_cts

        rots = self.find_unique_rots(cts)
        try:
            depth = self.find_depth(cts[0])
        except:
            depth = self.find_depth_rescale(cts[0])

        self.create_context(depth, rots)
        self.preprocess_packing(cts)
        self.preprocess_pt_compute(cts)
        self.preprocess_encoding(cts)
        self.track_dependencies(cts)

        # # HACK: reset secrets, bug in ^ opts
        # for ct in cts:
        #     for ct_term in ct.post_order():
        #         match ct_term.op:
        #             case HEOp.ADD | HEOp.MUL | HEOp.SUB:
        #                 ct_term.secret = ct_term.cs[0].secret or ct_term.cs[1].secret

        for ct in cts:
            for ct_term in ct.post_order():
                if ct_term.op == HEOp.ADD:
                    if ct_term.secret and not ct_term.cs[0].secret and not ct_term.cs[1].secret:
                        print("STILL BUGGED WTF")
                        exit(0)
                

        runtime, results = self.run_dagified_fhe_circuit(cts)
        for i, result in enumerate(results):
            self.serialize_result(i, result)

        print("runtime:", runtime)

        # get data size
        print("data size:", self.convert_size(self.get_directory_size()))
        print("comm cost:", self.convert_comm_cost())

        return runtime, [self.decrypt_to_plaintext(result) for result in results]

    def run_wrapper(self):
        cts = self.circuit_ir
        rots = self.find_unique_rots(cts)
        try:
            depth = self.find_depth(cts[0])
        except:
            depth = self.find_depth_rescale(cts[0])

        self.create_context(depth, rots)
        self.preprocess_packing(cts)
        self.preprocess_pt_compute(cts)
        self.preprocess_encoding(cts)
        self.track_dependencies(cts)
        runtime, results = self.run_fhe_circuit()
        for i, result in enumerate(results):
            self.serialize_result(i, result)

        print("runtime:", runtime)

        # get data size
        print("data size:", self.convert_size(self.get_directory_size()))
        print("comm cost:", self.convert_comm_cost())

        return runtime, [self.decrypt_to_plaintext(result) for result in results]
