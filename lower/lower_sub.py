from lower.lower_elementwise import lower_elementwise_binop


def lower_sub(env, kernel):
    """Lower an elementwise SUB kernel to ciphertext subtractions."""
    return lower_elementwise_binop(env, kernel, lambda a, b: a - b)
