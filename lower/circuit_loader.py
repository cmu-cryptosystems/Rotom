"""
Circuit loader module for reading instruction files back into executable form.

This module provides functionality to load modular instruction files
and reconstruct HE circuits for execution or analysis.
"""

import os
import json
from typing import Dict, List, Tuple, Any
from ir.he import HEOp, HETerm


class InstructionParser:
    """Parser for HE instruction strings."""
    
    @staticmethod
    def parse_instruction(instr_line: str) -> Tuple[int, bool, str, List]:
        """Parse a single instruction line.
        
        Format: {index} {is_secret}: {operation} {operands}
        
        Args:
            instr_line: Instruction line to parse
            
        Returns:
            Tuple of (index, is_secret, operation, operands)
        """
        # Remove comments
        if '#' in instr_line:
            instr_line = instr_line.split('#')[0]
        
        instr_line = instr_line.strip()
        if not instr_line:
            return None
        
        # Parse index and secret flag
        parts = instr_line.split(':')
        if len(parts) < 2:
            return None
        
        index_secret = parts[0].strip().split()
        if len(index_secret) != 2:
            return None
        
        index = int(index_secret[0])
        is_secret = index_secret[1].lower() == 'true'
        
        # Parse operation and operands
        op_str = parts[1].strip()
        
        # Handle different operation types
        if op_str.startswith('pack'):
            # pack (layout_str)
            return (index, is_secret, 'pack', [op_str])
        elif op_str.startswith('mask'):
            # mask [values...]
            return (index, is_secret, 'mask', [op_str])
        elif op_str.startswith('zero mask'):
            return (index, is_secret, 'zero_mask', [])
        elif op_str.startswith('('):
            # Binary operation: (op arg1 arg2) or unary: (<< arg shift)
            op_str = op_str.strip('()')
            op_parts = op_str.split()
            if len(op_parts) >= 2:
                op_symbol = op_parts[0]
                operands = [int(x) if x.lstrip('-').isdigit() else x for x in op_parts[1:]]
                
                # Map symbols to operations
                op_map = {
                    '+': 'add',
                    '-': 'sub',
                    '*': 'mul',
                    '<<': 'rot'
                }
                operation = op_map.get(op_symbol, op_symbol)
                return (index, is_secret, operation, operands)
        
        return None


class CircuitLoader:
    """Loads HE circuits from modular instruction files.
    
    This class can read instruction files created by CircuitSerializer
    and reconstruct the circuit structure for execution or analysis.
    """
    
    def __init__(self, circuit_dir: str, circuit_name: str):
        """Initialize the circuit loader.
        
        Args:
            circuit_dir: Directory containing instruction files
            circuit_name: Base name of the circuit
        """
        self.circuit_dir = circuit_dir
        self.circuit_name = circuit_name
        self.manifest = None
        self.instructions = {}  # Maps instruction index to parsed instruction
        
    def load(self) -> Dict[str, Any]:
        """Load the circuit from instruction files.
        
        Returns:
            Dictionary containing:
                - manifest: Circuit manifest
                - instructions: All parsed instructions
                - kernels: List of kernel metadata
        """
        # Load manifest
        manifest_path = os.path.join(self.circuit_dir, f"{self.circuit_name}_manifest.json")
        with open(manifest_path, 'r') as f:
            self.manifest = json.load(f)
        
        # Load each kernel file
        all_instructions = {}
        kernel_data = []
        
        for kernel_meta in self.manifest['kernels']:
            kernel_idx = kernel_meta['kernel_idx']
            kernel_file = f"{self.circuit_name}_kernel_{kernel_idx}.txt"
            kernel_path = os.path.join(self.circuit_dir, kernel_file)
            
            kernel_instrs = self._load_kernel_file(kernel_path)
            all_instructions.update(kernel_instrs)
            
            kernel_data.append({
                'metadata': kernel_meta,
                'instructions': kernel_instrs
            })
        
        self.instructions = all_instructions
        
        return {
            'manifest': self.manifest,
            'instructions': all_instructions,
            'kernels': kernel_data
        }
    
    def _load_kernel_file(self, filepath: str) -> Dict[int, Tuple]:
        """Load instructions from a kernel file.
        
        Args:
            filepath: Path to kernel instruction file
            
        Returns:
            Dictionary mapping instruction index to parsed instruction
        """
        instructions = {}
        
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parsed = InstructionParser.parse_instruction(line)
                if parsed:
                    index, is_secret, operation, operands = parsed
                    instructions[index] = (is_secret, operation, operands)
        
        return instructions
    
    def get_execution_order(self) -> List[int]:
        """Get instructions in execution order.
        
        Returns:
            List of instruction indices in execution order
        """
        return sorted(self.instructions.keys())
    
    def get_kernel_outputs(self, kernel_idx: int) -> List[int]:
        """Get output instruction indices for a kernel.
        
        Args:
            kernel_idx: Index of the kernel
            
        Returns:
            List of output instruction indices
        """
        if not self.manifest:
            raise RuntimeError("Circuit not loaded. Call load() first.")
        
        for kernel in self.manifest['kernels']:
            if kernel['kernel_idx'] == kernel_idx:
                return kernel['outputs']
        
        return []


def load_circuit(circuit_dir: str, circuit_name: str = "circuit") -> Dict[str, Any]:
    """Convenience function to load a circuit from instruction files.
    
    Args:
        circuit_dir: Directory containing instruction files
        circuit_name: Base name of the circuit
        
    Returns:
        Dictionary containing loaded circuit data
    """
    loader = CircuitLoader(circuit_dir, circuit_name)
    return loader.load()
