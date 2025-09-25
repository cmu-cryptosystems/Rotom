from ir.he import HEOp


def mul_associativity(he_term):
    """Optimize multiplication associativity in HE terms.

    This optimization handles several cases:
    1. For ROT(MUL(ct, pt)), moves rotation before multiplication:
       ROT(ct * pt) -> ROT(ct) * ROT(pt)

    2. For ROT(SUB(ct, MUL(ct, pt))), moves rotation before subtraction:
       ROT(ct - ct*pt) -> ROT(ct) - ROT(ct)*ROT(pt)

    3. For MUL(MUL(ct, pt1), pt2), combines plaintext multiplications:
       (ct * pt1) * pt2 -> ct * (pt1 * pt2)

    4. For MUL(SUB(ct1, MUL(ct2, pt1)), pt2), distributes multiplication:
       (ct1 - ct2*pt1) * pt2 -> ct1*pt2 - ct2*(pt1*pt2)

    Args:
        he_term: The FHE term to optimize

    Returns:
        The optimized FHE term with multiplication associativity applied
    """

    update_map = {}
    for term in he_term.post_order():
        # update term cs
        for i, cs in enumerate(term.cs):
            if term.op in [HEOp.PACK, HEOp.MASK, HEOp.ZERO_MASK]:
                continue
            if cs in update_map:
                term.cs[i] = update_map[cs]

        # update kernel with rewrites
        match term.op:
            case HEOp.ROT:
                if term.cs[0].op == HEOp.MUL and not term.cs[0].cs[1].secret:
                    # move rot before mask
                    rot_term = term.cs[0].cs[0] << term.cs[1]
                    pt_term = term.cs[0].cs[1] << term.cs[1]
                    mul_term = rot_term * pt_term
                    update_map[term] = mul_term
                elif term.cs[0].op == HEOp.SUB and term.cs[0].cs[1].op == HEOp.MUL and not term.cs[0].cs[1].cs[1].secret and term.cs[0].cs[0] == term.cs[0].cs[1].cs[0]:
                    rot_base = term.cs[0].cs[0] << term.cs[1]
                    rot_pt = term.cs[0].cs[1].cs[1] << term.cs[1]
                    sub_term = rot_base - (rot_base * rot_pt)
                    update_map[term] = sub_term
                else:
                    update_map[term] = term
            case HEOp.MUL:
                if term.cs[0].op == HEOp.MUL and not term.cs[1].secret and not term.cs[0].cs[1].secret:
                    pt_mul_term = term.cs[1] * term.cs[0].cs[1]
                    mul_term = term.cs[0].cs[0] * pt_mul_term
                    update_map[term] = mul_term
                elif term.cs[0].op == HEOp.SUB and term.cs[0].cs[1].op == HEOp.MUL and not term.cs[0].cs[1].cs[1].secret:
                    mul_base = term.cs[0].cs[0] * term.cs[1]
                    pt_mul = term.cs[0].cs[1].cs[1] * term.cs[1]
                    ct_mul = term.cs[0].cs[1].cs[0] * pt_mul
                    update_map[term] = mul_base - ct_mul
                else:
                    update_map[term] = term
            case _:
                update_map[term] = term

    # update term cs
    for i, cs in enumerate(he_term.cs):
        if he_term.op in [HEOp.PACK, HEOp.MASK, HEOp.ZERO_MASK]:
            continue
        if cs in update_map:
            he_term.cs[i] = update_map[cs]
    return update_map[he_term]
