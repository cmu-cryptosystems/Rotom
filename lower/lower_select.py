from ir.he import HETerm
from lower.layout_cts import LayoutCiphertexts
from util.layout_util import get_dim_indices_by_dim
from util.util import split_lists


def lower_select(env, kernel):
    input_cts = env[kernel.cs[0]]
    selected_cts = {}
    selected_cts[0] = input_cts[kernel.cs[1]]
    return LayoutCiphertexts(layout=kernel.layout, cts=selected_cts)
