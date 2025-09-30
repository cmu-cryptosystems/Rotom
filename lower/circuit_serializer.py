"""
Circuit serialization module for outputting HE circuits to instruction files.

This module provides functionality to serialize finalized HE circuits into
modular instruction files - one file per kernel. This allows for better
organization, debugging, and potential external execution of circuits.
"""

import os
import json
from typing import Dict, Any
from ir.he import HETerm


class CircuitSerializer:
    """Serializes HE circuits to modular instruction files.
    
    Each kernel in the circuit gets its own instruction file, allowing
    for modular loading and execution. Files are written in a human-readable
    text format with optional JSON metadata.
    
    Attributes:
        output_dir: Directory where instruction files will be written
        circuit_name: Base name for the circuit files
    """
    
    def __init__(self, output_dir="output/circuits", circuit_name="circuit"):
        """Initialize the circuit serializer.
        
        Args:
            output_dir: Directory to write instruction files
            circuit_name: Base name for circuit files
        """
        self.output_dir = output_dir
        self.circuit_name = circuit_name
        os.makedirs(output_dir, exist_ok=True)
    
    def serialize(self, circuit_ir: Dict) -> Dict[str, str]:
        """Serialize a circuit to modular instruction files.
        
        Writes one instruction file per kernel term in the circuit.
        Also creates a manifest file describing all kernels.
        
        Args:
            circuit_ir: Circuit IR from Lower.run() - dict mapping kernel_terms to {ct_idx: HETerm}
            
        Returns:
            Dictionary mapping kernel indices to file paths
        """
        file_paths = {}
        manifest = {
            "circuit_name": self.circuit_name,
            "kernels": []
        }
        
        # Track global instruction environment
        global_env = {}
        kernel_outputs = {}  # Maps kernel terms to their output indices
        
        # Process each kernel term from the circuit_ir
        kernel_idx = 0
        for kernel_term, he_terms_dict in circuit_ir.items():
            # Generate filename for this kernel
            kernel_file = f"{self.circuit_name}_kernel_{kernel_idx}.txt"
            kernel_path = os.path.join(self.output_dir, kernel_file)
            
            # Write kernel instructions
            kernel_metadata = self._write_kernel_file(
                kernel_path, 
                kernel_term, 
                he_terms_dict,
                global_env,
                kernel_idx
            )
            
            # Track this kernel's outputs for dependency resolution
            if kernel_term.layout not in kernel_outputs:
                kernel_outputs[kernel_term.layout] = []
            
            # Store output instruction indices for this kernel
            for ct_idx, he_term in he_terms_dict.items():
                if he_term in global_env:
                    kernel_outputs[kernel_term.layout].append(global_env[he_term])
            
            file_paths[kernel_idx] = kernel_path
            manifest["kernels"].append(kernel_metadata)
            kernel_idx += 1
        
        # Write manifest file
        manifest_path = os.path.join(self.output_dir, f"{self.circuit_name}_manifest.json")
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        file_paths['manifest'] = manifest_path
        
        return file_paths
    
    def _write_kernel_file(self, filepath, kernel_term, he_terms_dict, global_env, kernel_idx):
        """Write a single kernel to an instruction file.
        
        Args:
            filepath: Path to write the kernel file
            kernel_term: The kernel term being written
            he_terms_dict: Dictionary of ciphertext index to HETerm
            global_env: Global environment for instruction numbering
            kernel_idx: Index of this kernel in the circuit
            
        Returns:
            Metadata dictionary for this kernel
        """
        with open(filepath, 'w') as f:
            # Write header
            f.write(f"# HE Kernel Instruction File\n")
            f.write(f"# Kernel Index: {kernel_idx}\n")
            f.write(f"# Operation: {kernel_term.op}\n")
            f.write(f"# Layout: {kernel_term.layout}\n")
            f.write(f"# Format: {{index}} {{is_secret}}: {{operation}} {{operands}}\n")
            f.write("#" + "=" * 70 + "\n\n")
            
            # Track kernel-specific metadata
            kernel_metadata = {
                "kernel_idx": kernel_idx,
                "operation": str(kernel_term.op),
                "layout": str(kernel_term.layout),
                "num_ciphertexts": len(he_terms_dict),
                "instructions": [],
                "dependencies": [],
                "outputs": []
            }
            
            # Get dependencies (input kernel references)
            if hasattr(kernel_term, 'cs') and kernel_term.cs:
                for child in kernel_term.cs:
                    if hasattr(child, 'layout'):
                        kernel_metadata["dependencies"].append(str(child.layout))
            
            # Process each ciphertext in this kernel
            for ct_idx, he_term in he_terms_dict.items():
                f.write(f"# Ciphertext {ct_idx}\n")
                
                # Generate instructions
                instrs, global_env = he_term.instrs(env=global_env)
                
                if instrs:
                    for instr in instrs:
                        f.write(f"{instr}\n")
                        
                        # Parse instruction for metadata
                        # Format: {index} {is_secret}: {operation} {operands}
                        parts = instr.split(':')
                        if len(parts) >= 2:
                            index_and_secret = parts[0].strip().split()
                            if len(index_and_secret) >= 1:
                                instr_idx = int(index_and_secret[0])
                                kernel_metadata["instructions"].append(instr_idx)
                    
                    f.write("\n")
                    
                    # Track output of this ciphertext
                    if he_term in global_env:
                        kernel_metadata["outputs"].append(global_env[he_term])
            
            # Write summary footer
            f.write(f"# Total instructions: {len(kernel_metadata['instructions'])}\n")
            f.write(f"# Output indices: {kernel_metadata['outputs']}\n")
        
        return kernel_metadata


def serialize_circuit(circuit_ir, output_dir="output/circuits", circuit_name="circuit"):
    """Convenience function to serialize circuit IR to instruction files.
    
    Args:
        circuit_ir: Circuit IR from Lower.run()
        output_dir: Directory for output files
        circuit_name: Base name for circuit files
        
    Returns:
        Dictionary mapping kernel indices to file paths
    """
    serializer = CircuitSerializer(output_dir, circuit_name)
    return serializer.serialize(circuit_ir)
