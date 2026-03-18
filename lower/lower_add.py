from lower.lower_elementwise import lower_elementwise_binop


def lower_add(env, kernel):
    """Lower an elementwise ADD kernel to ciphertext additions."""
    return lower_elementwise_binop(env, kernel, lambda a, b: a + b)
