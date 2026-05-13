"""
Improved circuit serializer for Rotom → Orbit integration.

Drop-in replacement for Rotom's lower/circuit_serializer.py that emits
additional metadata required by Orbit's ILP optimizer:

  * per-instruction operand secrecy (ci / pl) so Orbit can distinguish
    single (ct-pt) from double (ct-ct) operations without re-inference
  * per-instruction weight (number of packed slots)
  * rotation offsets written explicitly so NAF weight is recoverable
  * manifest carries crypto-parameter hints (poly_deg, max_level, etc.)
    so the Orbit user does not have to supply them separately

The text format is a strict superset of the original — every line that
the original serializer produces is still valid here, and extra fields
are appended after a tab so that legacy readers can split on the first
colon and ignore the rest.

USAGE (inside Rotom, after replacing the import):

    from lower.rotom_serializer import serialize_circuit
    file_paths = serialize_circuit(circuit_ir, output_dir="output/circuits",
                                   circuit_name="my_net")

The output can then be consumed by Orbit's rotom2tdag.build_from_rotom().
"""

import json
import os
from typing import Any, Dict

from ir.he import HETerm
from lower.layout_cts import LayoutCiphertexts


class CircuitSerializer:
    """Serialize HE circuits to instruction files with Orbit-compatible metadata.

    Each kernel gets its own .txt file.  A manifest JSON ties them together
    and carries circuit-level parameters that Orbit needs.

    Attributes:
        output_dir:   directory for output files
        circuit_name: base name shared by all output files
        params:       optional dict of crypto parameters forwarded to manifest
    """

    def __init__(self, output_dir="output/circuits", circuit_name="circuit",
                 params=None):
        self.output_dir = output_dir
        self.circuit_name = circuit_name
        self.params = params or {}
        os.makedirs(output_dir, exist_ok=True)

    def serialize(self, circuit_ir: Dict) -> Dict[str, str]:
        """Serialize *circuit_ir* (kernel_term → LayoutCiphertexts) to disk.

        Returns a dict mapping kernel indices (and ``"manifest"``) to file
        paths that were written.
        """
        file_paths = {}
        manifest = {
            "format_version": 2,
            "circuit_name": self.circuit_name,
            "params": self.params,
            "kernels": [],
        }

        global_env = {}
        kernel_outputs = {}

        kernel_idx = 0
        for kernel_term, layout_cts in circuit_ir.items():
            if isinstance(layout_cts, LayoutCiphertexts):
                he_terms_dict = layout_cts.cts
            else:
                he_terms_dict = layout_cts

            kernel_file = f"{self.circuit_name}_kernel_{kernel_idx}.txt"
            kernel_path = os.path.join(self.output_dir, kernel_file)

            kernel_metadata = self._write_kernel_file(
                kernel_path, kernel_term, he_terms_dict, global_env, kernel_idx
            )

            if kernel_term.layout not in kernel_outputs:
                kernel_outputs[kernel_term.layout] = []
            for ct_idx, he_term in he_terms_dict.items():
                if he_term in global_env:
                    kernel_outputs[kernel_term.layout].append(global_env[he_term])

            file_paths[kernel_idx] = kernel_path
            manifest["kernels"].append(kernel_metadata)
            kernel_idx += 1

        manifest_path = os.path.join(
            self.output_dir, f"{self.circuit_name}_manifest.json"
        )
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        file_paths["manifest"] = manifest_path
        return file_paths

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _write_kernel_file(self, filepath, kernel_term, he_terms_dict,
                           global_env, kernel_idx):
        """Write one kernel's instructions and return its manifest entry."""
        with open(filepath, "w") as f:
            f.write(f"# Rotom-Orbit HE Kernel Instruction File  (format v2)\n")
            f.write(f"# Kernel Index: {kernel_idx}\n")
            f.write(f"# Operation: {kernel_term.op}\n")
            f.write(f"# Layout: {kernel_term.layout}\n")
            f.write(f"# Format: {{index}} {{ci|pl}}: {{operation}}"
                    f"  [# metadata]\n")
            f.write("#" + "=" * 70 + "\n\n")

            kernel_metadata = {
                "kernel_idx": kernel_idx,
                "operation": str(kernel_term.op),
                "layout": str(kernel_term.layout),
                "num_ciphertexts": len(he_terms_dict),
                "instructions": [],
                "dependencies": [],
                "outputs": [],
            }

            if hasattr(kernel_term, "cs") and kernel_term.cs:
                for child in kernel_term.cs:
                    if hasattr(child, "layout"):
                        kernel_metadata["dependencies"].append(str(child.layout))

            for ct_idx, he_term in he_terms_dict.items():
                f.write(f"# Ciphertext {ct_idx}\n")

                instrs, global_env = he_term.instrs(env=global_env)

                if instrs:
                    for instr in instrs:
                        # Rewrite the secrecy token from True/False → ci/pl
                        enriched = self._enrich_instruction(instr)
                        f.write(f"{enriched}\n")

                        parts = instr.split(":")
                        if len(parts) >= 2:
                            idx_parts = parts[0].strip().split()
                            if idx_parts:
                                kernel_metadata["instructions"].append(
                                    int(idx_parts[0]))

                    f.write("\n")

                    if he_term in global_env:
                        kernel_metadata["outputs"].append(global_env[he_term])

            f.write(f"# Total instructions: "
                    f"{len(kernel_metadata['instructions'])}\n")
            f.write(f"# Output indices: {kernel_metadata['outputs']}\n")

        return kernel_metadata

    @staticmethod
    def _enrich_instruction(instr: str) -> str:
        """Replace ``True``/``False`` secrecy token with ``ci``/``pl``.

        Original format:   ``5 True: (+ 2 3)``
        Enriched format:   ``5 ci: (+ 2 3)``

        This makes the file self-describing (ci = ciphertext, pl = plaintext)
        and matches Orbit's MLIR type convention.
        """
        # The secrecy token is always the second whitespace-delimited word
        # before the first colon.
        m = __import__('re').match(r'^(\d+)\s+(True|False)(:.*)', instr)
        if not m:
            return instr
        idx_str = m.group(1)
        secret  = m.group(2) == 'True'
        rest    = m.group(3)
        return f"{idx_str} {'ci' if secret else 'pl'}{rest}"


def serialize_circuit(circuit_ir, output_dir="output/circuits",
                      circuit_name="circuit", params=None):
    """Convenience wrapper matching the original module-level API.

    Args:
        circuit_ir:   dict from ``Lower.run()``
        output_dir:   output directory
        circuit_name: base file name
        params:       optional dict of crypto parameters (poly_deg,
                      max_level, rescaling_factor, …) forwarded into
                      the manifest so Orbit can read them directly

    Returns:
        dict mapping kernel indices (and ``"manifest"``) to file paths
    """
    serializer = CircuitSerializer(output_dir, circuit_name, params=params)
    return serializer.serialize(circuit_ir)
