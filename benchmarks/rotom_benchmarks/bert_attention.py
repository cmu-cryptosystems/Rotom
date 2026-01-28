import numpy as np

from frontends.tensor import TensorTerm


def bert_attention():
    """
    Full-size BERT-style attention benchmark matching the structure of
    ``tests/test_bert_attention_small.py``, but with large dimensions.

    Shapes:
        - seq_len    = 128
        - hidden_dim = 768
        - num_heads  = 12, head_dim = 64
    """

    seq_len = 128
    hidden_dim = 1024
    num_heads = 16
    head_dim = hidden_dim // num_heads

    inputs = {}
    inputs["h"] = np.array(
        [
            [np.random.choice(range(2)) for _ in range(hidden_dim)]
            for _ in range(seq_len)
        ]
    )
    inputs["wq"] = np.array(
        [
            [np.random.choice(range(2)) for _ in range(hidden_dim)]
            for _ in range(hidden_dim)
        ]
    )
    inputs["bq"] = np.array([np.random.choice(range(2)) for _ in range(hidden_dim)])
    inputs["wk"] = np.array(
        [
            [np.random.choice(range(2)) for _ in range(hidden_dim)]
            for _ in range(hidden_dim)
        ]
    )
    inputs["bk"] = np.array([np.random.choice(range(2)) for _ in range(hidden_dim)])
    inputs["wv"] = np.array(
        [
            [np.random.choice(range(2)) for _ in range(hidden_dim)]
            for _ in range(hidden_dim)
        ]
    )
    inputs["bv"] = np.array([np.random.choice(range(2)) for _ in range(hidden_dim)])

    # Define TensorTerms matching the numpy input shapes.
    h = TensorTerm.Tensor("h", [seq_len, hidden_dim], True)
    wq = TensorTerm.Tensor("wq", [hidden_dim, hidden_dim], False)
    bq = TensorTerm.Tensor("bq", [hidden_dim], False)
    wk = TensorTerm.Tensor("wk", [hidden_dim, hidden_dim], False)
    bk = TensorTerm.Tensor("bk", [hidden_dim], False)
    wv = TensorTerm.Tensor("wv", [hidden_dim, hidden_dim], False)
    bv = TensorTerm.Tensor("bv", [hidden_dim], False)

    # Linear projections
    q = h @ wq + bq
    k = h @ wk + bk
    v = h @ wv + bv

    # Reshape / permute into [num_heads, seq_len, head_dim] as in the small unit test.
    blocked_q = q.reshape(1, {1: num_heads, 2: head_dim}).permute({0: 1, 1: 0, 2: 2})
    blocked_kt = k.reshape(1, {1: num_heads, 2: head_dim}).permute({0: 2, 1: 0, 2: 1})
    blocked_v = v.reshape(1, {1: num_heads, 2: head_dim}).permute({0: 1, 1: 0, 2: 2})

    head_results = None
    for h_idx in range(num_heads):
        # After permute, each slice is [seq_len, head_dim].
        q_h = blocked_q[h_idx, :, :]
        k_h = blocked_kt[h_idx, :, :]
        v_h = blocked_v[h_idx, :, :]

        # Simple (unnormalized) attention: (Q K^T) V
        qk_h = q_h @ k_h
        out_h = qk_h @ v_h

        if head_results is None:
            head_results = out_h
        else:
            head_results = head_results + out_h

    # Return the TensorTerm computation, inputs, and a recommended ring dimension.
    return head_results, inputs, 8192
