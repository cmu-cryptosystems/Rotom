from lower.lower_elementwise import lower_elementwise_binop


def lower_mul(env, kernel):
    """Lower an elementwise MUL kernel to ciphertext multiplications."""
    return lower_elementwise_binop(env, kernel, lambda a, b: a * b)
