from ir.he import HEOp, HETerm
from lower.layout_cts import LayoutCiphertexts


def lower_add(env, kernel):
    a_cts = env[kernel.cs[0]]
    b_cts = env[kernel.cs[1]]

    a_keys = sorted(a_cts.keys())
    b_keys = sorted(b_cts.keys())

    # Broadcast when ct counts differ: replicate the smaller operand.
    # Use block mapping: when len(a)=128 and len(b)=8, each b maps to 16 a's.
    if len(a_keys) >= len(b_keys):
        # broadcast b to match a
        ratio = len(a_keys) // len(b_keys)
        cts = {}
        for i, a_key in enumerate(a_keys):
            b_idx = i // ratio
            b_key = b_keys[b_idx]
            a_term = HETerm(HEOp.CS, [a_cts[a_key]], a_cts[a_key].secret)
            b_term = HETerm(HEOp.CS, [b_cts[b_key]], b_cts[b_key].secret)
            cts[a_key] = a_term + b_term
    else:
        # broadcast a to match b
        ratio = len(b_keys) // len(a_keys)
        cts = {}
        for i, b_key in enumerate(b_keys):
            a_idx = i // ratio
            a_key = a_keys[a_idx]
            a_term = HETerm(HEOp.CS, [a_cts[a_key]], a_cts[a_key].secret)
            b_term = HETerm(HEOp.CS, [b_cts[b_key]], b_cts[b_key].secret)
            cts[b_key] = a_term + b_term

    return LayoutCiphertexts(layout=kernel.layout, cts=cts)
