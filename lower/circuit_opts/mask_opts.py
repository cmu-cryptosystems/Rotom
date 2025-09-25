from ir.he import HEOp


def zero_mask_opt(he_term):
    """
    set zero_mask
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
            case HEOp.MASK:
                if not any(term.cs[0]):
                    # mask is completely 0
                    term.op = HEOp.ZERO_MASK
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


def mask_identity_opt(he_term):
    """
    Mul by mask of all 1 can be optimized
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
            case HEOp.MUL:
                if term.cs[0].op == HEOp.MASK:
                    ones = all(term.cs[0].cs[0])
                    if ones:
                        # set output to term
                        update_map[term] = term.cs[1]
                    else:
                        update_map[term] = term
                elif term.cs[1].op == HEOp.MASK:
                    ones = all(term.cs[1].cs[0])
                    if ones:
                        # set output to term
                        update_map[term] = term.cs[0]
                    else:
                        update_map[term] = term

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


def zero_mask_identity_opt(he_term):
    """
    Mul by mask of all 0 can be optimized
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
            case HEOp.ADD:
                if term.cs[0].op == HEOp.ZERO_MASK:
                    update_map[term] = term.cs[1]
                elif term.cs[1].op == HEOp.ZERO_MASK:
                    update_map[term] = term.cs[0]
                else:
                    update_map[term] = term
            case HEOp.MUL:
                if term.cs[0].op == HEOp.ZERO_MASK:
                    update_map[term] = term.cs[0]
                elif term.cs[1].op == HEOp.ZERO_MASK:
                    update_map[term] = term.cs[1]
                else:
                    update_map[term] = term
            case HEOp.ROT:
                if term.cs[0].op == HEOp.ZERO_MASK:
                    update_map[term] = term.cs[0]
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
