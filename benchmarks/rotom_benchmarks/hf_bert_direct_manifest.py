from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from benchmarks.rotom_benchmarks.hf_bert_polynomial import load_bert_polynomial_spec


class _InstructionWriter:
    def __init__(self) -> None:
        self.next_index = 0
        self.lines: list[str] = []

    def add(self, token: str, op: str, *, scope: str, kind: str, layer: int | None) -> int:
        index = self.next_index
        self.next_index += 1
        metadata = f"{index} scope={scope};op={kind}"
        if layer is not None:
            metadata += f";layer={layer}"
        self.lines.append(f"{index} {token}: {op} # {metadata}")
        return index


def write_hf_bert_direct_manifest(args: Any, output_dir: Path, circuit_name: str) -> Path:
    """Write an Orbit-readable polynomial BERT surrogate without layout search.

    The full Rotom layout assignment path can be too large for the 12-layer
    surrogate. Orbit's compiler input needs a TDAG-shaped instruction stream
    with stable scope comments, so this direct writer emits compact parseable
    Rotom-format ops while keeping the same Hugging Face module names.
    """

    spec = load_bert_polynomial_spec(
        args.hf_model,
        seq_len=args.seq_len,
        num_layers=args.num_layers,
        hidden_size=getattr(args, "hidden_size", None),
        intermediate_size=getattr(args, "intermediate_size", None),
        num_labels=getattr(args, "num_labels", None),
    )
    writer = _InstructionWriter()
    hidden = writer.add(
        "ci",
        f"pack ([0:{spec.seq_len}:1][1:{spec.hidden_size}:1])",
        scope="bert.embeddings.output",
        kind="embedding_output",
        layer=None,
    )

    for layer in range(spec.num_hidden_layers):
        prefix = f"bert.encoder.layer.{layer}"
        residual = hidden
        query = _linear(writer, hidden, f"{prefix}.attention.self.query", layer)
        key = _linear(writer, hidden, f"{prefix}.attention.self.key", layer)
        value = _linear(writer, hidden, f"{prefix}.attention.self.value", layer)
        key_t = writer.add(
            "ci",
            f"(<< {key} 1)",
            scope=f"{prefix}.attention.self.key_transpose",
            kind="transpose",
            layer=layer,
        )
        scores = writer.add(
            "ci",
            f"(* {query} {key_t})",
            scope=f"{prefix}.attention.self",
            kind="attention_scores",
            layer=layer,
        )
        scores = _rescale(writer, scores, f"{prefix}.attention.self", "attention_rescale", layer)
        context = writer.add(
            "ci",
            f"(* {scores} {value})",
            scope=f"{prefix}.attention.self",
            kind="linear_attention",
            layer=layer,
        )
        context = _rescale(writer, context, f"{prefix}.attention.self", "context_rescale", layer)
        attention = _linear(writer, context, f"{prefix}.attention.output.dense", layer)
        attention_residual = writer.add(
            "ci",
            f"(+ {attention} {residual})",
            scope=f"{prefix}.attention.output",
            kind="residual_add",
            layer=layer,
        )
        hidden = _polynorm(writer, attention_residual, f"{prefix}.attention.output.LayerNorm", layer)
        intermediate = _linear(writer, hidden, f"{prefix}.intermediate.dense", layer)
        activated = _poly_gelu(writer, intermediate, f"{prefix}.intermediate", layer)
        output = _linear(writer, activated, f"{prefix}.output.dense", layer)
        output_residual = writer.add(
            "ci",
            f"(+ {output} {hidden})",
            scope=f"{prefix}.output",
            kind="residual_add",
            layer=layer,
        )
        hidden = _polynorm(writer, output_residual, f"{prefix}.output.LayerNorm", layer)

    pooled = _linear(writer, hidden, "bert.pooler.dense", None)
    pooled = _poly_square(writer, pooled, "bert.pooler.activation", "degree2_activation", None)
    logits = _linear(writer, pooled, "classifier", None)

    output_dir.mkdir(parents=True, exist_ok=True)
    kernel_path = output_dir / f"{circuit_name}_kernel_0.txt"
    kernel_path.write_text(
        "\n".join(
            [
                "# Rotom-Orbit HE Kernel Instruction File  (format v2)",
                "# Kernel Index: 0",
                "# Operation: DirectBertPolynomial",
                "# Layout: direct:hf_bert_polynomial",
                "# Format: {index} {ci|pl}: {operation}  [# metadata]",
                "#======================================================================",
                "",
                *writer.lines,
                "",
                f"# Total instructions: {writer.next_index}",
                f"# Output indices: [{logits}]",
                "",
            ]
        ),
        encoding="utf-8",
    )
    manifest = {
        "format_version": 2,
        "circuit_name": circuit_name,
        "params": {},
        "kernels": [
            {
                "kernel_idx": 0,
                "operation": "DirectBertPolynomial",
                "layout": "direct:hf_bert_polynomial",
                "num_ciphertexts": 1,
                "instructions": list(range(writer.next_index)),
                "dependencies": [],
                "outputs": [logits],
            }
        ],
    }
    manifest_path = output_dir / f"{circuit_name}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def _linear(writer: _InstructionWriter, value: int, scope: str, layer: int | None) -> int:
    weight = writer.add("pl", "pack ([W])", scope=f"{scope}.weight", kind="weight", layer=layer)
    product = writer.add("ci", f"(* {value} {weight})", scope=scope, kind="linear", layer=layer)
    bias = writer.add("pl", "pack ([B])", scope=f"{scope}.bias", kind="bias", layer=layer)
    result = writer.add("ci", f"(+ {product} {bias})", scope=scope, kind="linear_bias", layer=layer)
    return _rescale(writer, result, scope, "linear_rescale", layer)


def _poly_square(
    writer: _InstructionWriter,
    value: int,
    scope: str,
    kind: str,
    layer: int | None,
) -> int:
    square = writer.add("ci", f"(* {value} {value})", scope=scope, kind=kind, layer=layer)
    return _rescale(writer, square, scope, f"{kind}_rescale", layer)


def _poly_gelu(writer: _InstructionWriter, value: int, scope: str, layer: int) -> int:
    square = _poly_square(writer, value, scope, "gelu_degree2", layer)
    return writer.add("ci", f"(+ {value} {square})", scope=scope, kind="polynomial_gelu", layer=layer)


def _polynorm(writer: _InstructionWriter, value: int, scope: str, layer: int) -> int:
    square = _poly_square(writer, value, scope, "polynorm_square", layer)
    return writer.add("ci", f"(+ {value} {square})", scope=scope, kind="polynorm_proxy", layer=layer)


def _rescale(writer: _InstructionWriter, value: int, scope: str, kind: str, layer: int | None) -> int:
    return writer.add("ci", f"(rescale {value} / 2^40)", scope=scope, kind=kind, layer=layer)
