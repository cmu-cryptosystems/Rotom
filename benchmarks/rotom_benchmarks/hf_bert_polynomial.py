from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from frontends.tensor import TensorTerm


@dataclass(frozen=True)
class BertPolynomialSpec:
    model_name: str
    seq_len: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_labels: int


def hf_bert_polynomial(args) -> tuple[TensorTerm, dict[str, np.ndarray], int]:
    """Build a polynomial BERT-shaped graph from the embedding-output boundary.

    The real Hugging Face model remains the profiling source of truth. This
    Rotom graph is a compiler surrogate with matching module scopes so Orbit can
    place bootstraps using constraints exported by the profiler.
    """

    spec = load_bert_polynomial_spec(
        args.hf_model,
        seq_len=args.seq_len,
        num_layers=args.num_layers,
        hidden_size=getattr(args, "hidden_size", None),
        intermediate_size=getattr(args, "intermediate_size", None),
        num_labels=getattr(args, "num_labels", None),
    )
    hidden = spec.hidden_size
    intermediate = spec.intermediate_size
    h = _tag(
        TensorTerm.Tensor("bert.embeddings.output", [spec.seq_len, hidden], True),
        "bert.embeddings.output",
        "embedding_output",
    )

    inputs: dict[str, np.ndarray] = {
        "bert.embeddings.output": np.zeros((spec.seq_len, hidden), dtype=float)
    }

    for layer in range(spec.num_hidden_layers):
        prefix = f"bert.encoder.layer.{layer}"
        query = _linear(h, hidden, spec.seq_len, f"{prefix}.attention.self.query", layer)
        key = _linear(h, hidden, spec.seq_len, f"{prefix}.attention.self.key", layer)
        value = _linear(h, hidden, hidden, f"{prefix}.attention.self.value", layer)
        key_t = _tag(key.T, f"{prefix}.attention.self.key_transpose", "transpose", layer)
        scores = _tag(query @ key_t, f"{prefix}.attention.self", "attention_scores", layer)
        context = _tag(scores @ value, f"{prefix}.attention.self", "linear_attention", layer)
        attention = _linear(context, hidden, hidden, f"{prefix}.attention.output.dense", layer)
        h = _poly_norm(attention + h, f"{prefix}.attention.output.LayerNorm", layer)

        intermediate_hidden = _linear(h, hidden, intermediate, f"{prefix}.intermediate.dense", layer)
        activated = _poly_gelu(intermediate_hidden, f"{prefix}.intermediate", layer)
        output = _linear(activated, intermediate, hidden, f"{prefix}.output.dense", layer)
        h = _poly_norm(output + h, f"{prefix}.output.LayerNorm", layer)

    pooled = _linear(h, hidden, hidden, "bert.pooler.dense", None)
    pooled = _tag(pooled * pooled, "bert.pooler.activation", "degree2_activation", None)
    logits = _linear(pooled, hidden, spec.num_labels, "classifier", None)
    return logits, inputs, args.n


def load_bert_polynomial_spec(
    model_name: str,
    *,
    seq_len: int,
    num_layers: int | None,
    hidden_size: int | None = None,
    intermediate_size: int | None = None,
    num_labels: int | None = None,
) -> BertPolynomialSpec:
    if hidden_size is not None:
        configured_layers = num_layers or 12
        return BertPolynomialSpec(
            model_name=model_name,
            seq_len=seq_len,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size or hidden_size * 4,
            num_hidden_layers=configured_layers,
            num_labels=num_labels or 2,
        )

    try:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(model_name)
        hidden_size = int(config.hidden_size)
        intermediate_size = int(config.intermediate_size)
        configured_layers = int(config.num_hidden_layers)
        num_labels = int(getattr(config, "num_labels", 2))
    except Exception:
        hidden_size = 768
        intermediate_size = 3072
        configured_layers = 12
        num_labels = 2

    return BertPolynomialSpec(
        model_name=model_name,
        seq_len=seq_len,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_layers or configured_layers,
        num_labels=num_labels,
    )


def _linear(
    value: TensorTerm,
    in_features: int,
    out_features: int,
    scope: str,
    layer: int | None,
) -> TensorTerm:
    weight = _tag(
        TensorTerm.Tensor(f"{scope}.weight", [in_features, out_features], False),
        f"{scope}.weight",
        "weight",
        layer,
    )
    result = _tag(value @ weight, scope, "linear", layer)
    bias = _tag(
        TensorTerm.Tensor(f"{scope}.bias", [out_features], False),
        f"{scope}.bias",
        "bias",
        layer,
    )
    return _tag(result + bias, scope, "linear_bias", layer)


def _poly_gelu(value: TensorTerm, scope: str, layer: int) -> TensorTerm:
    square = _tag(value * value, scope, "gelu_degree2", layer)
    return _tag(value + square, scope, "polynomial_gelu", layer)


def _poly_norm(value: TensorTerm, scope: str, layer: int) -> TensorTerm:
    square = _tag(value * value, scope, "polynorm_square", layer)
    return _tag(value + square, scope, "polynorm_proxy", layer)


def _tag(
    term: TensorTerm,
    scope: str,
    op: str,
    layer: int | None = None,
) -> TensorTerm:
    term.orbit_scope = scope
    term.orbit_op = op
    term.orbit_layer = layer
    return term
