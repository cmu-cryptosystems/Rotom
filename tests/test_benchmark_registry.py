"""Benchmarks main.py runs by name build a term the evaluator accepts."""

import numpy as np

from benchmarks.rotom_benchmarks.bert_attention import bert_attention
from benchmarks.rotom_benchmarks.convolution.convolution import convolution
from benchmarks.rotom_benchmarks.convolution.convolution_32768 import (
    convolution_32768,
)


def test_convolution_benchmarks_have_a_filter_slice_per_input_channel():
    for build, size in ((convolution, 32), (convolution_32768, 64)):
        term, inputs, _ = build()
        assert inputs["b"].shape == (1, 8, 3, 3)
        assert np.asarray(term.eval(inputs)).shape == (1, size, size)


def test_bert_attention_returns_a_term_and_its_inputs():
    term, inputs = bert_attention()
    assert {"h", "wq", "wk", "wv"} <= set(inputs)
    assert term is not None
