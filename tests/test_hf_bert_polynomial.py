from argparse import Namespace
from pathlib import Path

from assignment.assignment import LayoutAssignment
from assignment.gen.gen_transpose import gen_transpose
from benchmarks.rotom_benchmarks.hf_bert_direct_manifest import write_hf_bert_direct_manifest
from benchmarks.rotom_benchmarks import hf_bert_polynomial as bert_poly
from frontends.tensor import TensorTerm
from ir.dim import Dim
from ir.kernel import Kernel, KernelOp
from ir.layout import Layout
from lower.circuit_serializer import serialize_circuit
from lower.lower import Lower


def test_hf_bert_polynomial_preserves_scope_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(
        bert_poly,
        "load_bert_polynomial_spec",
        lambda model_name, seq_len, num_layers, **_kwargs: bert_poly.BertPolynomialSpec(
            model_name=model_name,
            seq_len=2,
            hidden_size=4,
            intermediate_size=8,
            num_hidden_layers=1,
            num_labels=2,
        ),
    )
    args = Namespace(
        hf_model="test-bert",
        seq_len=2,
        num_layers=1,
        n=16,
        rolls=True,
        net="lan",
        strassens=False,
        backend="toy",
        fuzz=False,
        fuzz_result=False,
        conv_roll=False,
        diagonal_first=True,
        fn="test",
    )

    tensor_ir, _inputs, _n = bert_poly.hf_bert_polynomial(args)
    kernel = LayoutAssignment(tensor_ir, args).run()
    circuit_ir = Lower(kernel).run()
    paths = serialize_circuit(circuit_ir, tmp_path, "tiny_bert")

    text = "\n".join(Path(path).read_text() for key, path in paths.items() if key != "manifest")
    assert "scope=bert.encoder.layer.0.attention.self.query" in text
    assert "scope=bert.encoder.layer.0.attention.self;op=attention_scores" in text
    assert "scope=bert.encoder.layer.0.output.LayerNorm" in text
    assert "scope=classifier" in text


def test_transpose_generator_uses_current_layout_constructor():
    term = TensorTerm.Tensor("a", [2, 4], True)
    transpose = term.T
    layout = Layout(term, [], [Dim(0, 2), Dim(1, 4)], 8, True)
    kernel = Kernel(KernelOp.TENSOR, [], layout)

    [transposed] = list(gen_transpose(transpose, [kernel]))

    assert transposed.op == KernelOp.TRANSPOSE
    assert transposed.layout.term is transpose
    assert transposed.layout.n == 8
    assert {dim.dim: dim.extent for dim in transposed.layout.get_dims() if dim.dim is not None} == {
        0: 4,
        1: 2,
    }


def test_direct_bert_manifest_emits_scope_comments(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "benchmarks.rotom_benchmarks.hf_bert_direct_manifest.load_bert_polynomial_spec",
        lambda model_name, seq_len, num_layers, **_kwargs: bert_poly.BertPolynomialSpec(
            model_name=model_name,
            seq_len=2,
            hidden_size=4,
            intermediate_size=8,
            num_hidden_layers=2,
            num_labels=2,
        ),
    )
    args = Namespace(
        hf_model="test-bert",
        seq_len=2,
        num_layers=2,
        hidden_size=None,
        intermediate_size=None,
        num_labels=None,
        n=16,
    )

    manifest = write_hf_bert_direct_manifest(args, tmp_path, "direct_bert")
    text = "\n".join(path.read_text() for path in tmp_path.glob("*.txt"))

    assert manifest.exists()
    assert "scope=bert.encoder.layer.0.attention.self.query;op=linear" in text
    assert "scope=bert.encoder.layer.1.output.LayerNorm;op=polynorm_proxy" in text
    assert "scope=classifier;op=linear" in text
