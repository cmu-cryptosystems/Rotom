#!/usr/bin/env python3
"""
Sphinx-based documentation generation script for Rotom.

This script generates comprehensive documentation using Sphinx for all modules
in the FHE compiler project. It creates both HTML and PDF documentation
with professional formatting and cross-references.

Usage:
    python generate_docs_sphinx.py [--output-dir docs] [--format html,pdf] [--clean]
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from tqdm import tqdm


def run_sphinx_command(command, source_dir, output_dir=None):
    """Run a Sphinx command.

    Args:
        command: Sphinx command to run (e.g., 'html', 'pdf')
        source_dir: Source directory containing conf.py
        output_dir: Output directory (optional)

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        cmd = ["sphinx-build", "-b", command]
        if output_dir:
            cmd.extend(["-d", os.path.join(output_dir, "doctrees")])
            cmd.extend([source_dir, os.path.join(output_dir, command)])
        else:
            cmd.extend([source_dir, f"_{command}"])

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            return True
        else:
            return False
    except Exception as e:
        return False


def generate_api_docs(source_dir):
    """Generate API documentation files for all modules.

    Args:
        source_dir: Source directory for Sphinx documentation
    """
    modules = [
        # Frontend modules
        ("frontends.tensor", "Tensor Frontend"),
        # IR modules
        ("ir", "Intermediate Representations"),
        ("ir.dim", "Dimension Representation"),
        ("ir.roll", "Rolls"),
        ("ir.layout", "Layout IR"),
        ("ir.kernel", "Kernel IR"),
        ("ir.he", "HE IR"),
        # Assignment modules
        ("assignment.alignment", "Layout Alignment"),
        ("assignment.assignment", "Layout Assignment"),
    ]

    # Create individual module documentation files with progress bar
    progress_bar = tqdm(modules, desc="Generating API docs", unit="module")
    for module_name, title in progress_bar:
        progress_bar.set_postfix(module=module_name)
        create_module_doc(source_dir, module_name, title)
        time.sleep(0.01)  # Small delay to show progress
    progress_bar.close()


def create_module_doc(source_dir, module_name, title):
    """Create a Sphinx documentation file for a module.

    Args:
        source_dir: Source directory for Sphinx documentation
        module_name: Name of the module to document
        title: Title for the documentation page
    """
    # Convert module name to filename
    filename = module_name.replace(".", "_") + ".rst"
    filepath = os.path.join(source_dir, "api_reference", filename)

    content = f"""{title}
{'=' * len(title)}

Module Contents
---------------

.. automodule:: {module_name}
   :members:
   :undoc-members:
   :show-inheritance:
"""

    with open(filepath, "w") as f:
        f.write(content)


def clean_docs(output_dir="."):
    """Clean generated documentation files.

    Args:
        output_dir: Directory containing the docs (default: current directory)
    """
    # Clean common Sphinx output directories
    dirs_to_clean = ["html", "doctrees", "_build"]
    clean_progress = tqdm(dirs_to_clean, desc="Cleaning directories", unit="dir")
    for dir_name in clean_progress:
        clean_progress.set_postfix(dir=dir_name)
        dir_path = os.path.join(output_dir, dir_name)
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
            print(f"Cleaned {dir_path}")
        time.sleep(0.01)
    clean_progress.close()

    # Clean any .doctrees directories
    print("Scanning for additional cleanup...")
    doctrees_found = []
    for root, dirs, files in os.walk(output_dir):
        if ".doctrees" in dirs:
            doctrees_found.append(os.path.join(root, ".doctrees"))

    if doctrees_found:
        doctrees_progress = tqdm(doctrees_found, desc="Cleaning doctrees", unit="dir")
        for doctrees_path in doctrees_progress:
            doctrees_progress.set_postfix(
                dir=os.path.basename(os.path.dirname(doctrees_path))
            )
            shutil.rmtree(doctrees_path)
            print(f"Cleaned {doctrees_path}")
            time.sleep(0.01)
        doctrees_progress.close()


def generate_all_docs(output_dir=".", formats=None, clean=False):
    """Generate documentation for all modules using Sphinx.

    Args:
        output_dir: Directory to write documentation to (default: current directory)
        formats: List of output formats (default: ['html'])
        clean: Whether to clean output directory first
    """
    if formats is None:
        formats = ["html"]

    source_dir = os.path.join(output_dir, "source")

    # Ensure source directory exists
    if not os.path.exists(source_dir):
        return False

    print("Setting up documentation generation...")

    # Create necessary subdirectories
    os.makedirs(os.path.join(output_dir, "html"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "doctrees"), exist_ok=True)

    # Clean output directory if requested
    if clean:
        print("Cleaning output directory...")
        clean_docs(output_dir)

    # Generate API documentation files
    generate_api_docs(source_dir)

    successful = 0
    failed = 0

    # Generate documentation for each format with progress bar
    format_progress = tqdm(formats, desc="Building documentation", unit="format")
    for fmt in format_progress:
        format_progress.set_postfix(format=fmt)
        if run_sphinx_command(fmt, source_dir, output_dir):
            successful += 1
        else:
            failed += 1
        time.sleep(0.1)  # Small delay to show progress
    format_progress.close()

    return successful > 0


def main():
    """Main function to run the Sphinx documentation generation script."""
    parser = argparse.ArgumentParser(
        description="Generate Sphinx documentation for Rotom"
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Output directory for documentation (default: current directory)",
    )
    parser.add_argument(
        "--format",
        default="html",
        help="Output format: html, pdf, or both (default: html)",
    )
    parser.add_argument(
        "--clean", action="store_true", help="Clean output directory before generating"
    )
    parser.add_argument(
        "--clean-only",
        action="store_true",
        help="Only clean, don't generate documentation",
    )

    args = parser.parse_args()

    print("Starting Rotom documentation generation...")

    # If clean-only mode, just clean and exit
    if args.clean_only:
        clean_docs(args.output_dir)
        print("Cleanup completed!")
        return

    # Parse formats
    if args.format == "both":
        formats = ["html", "pdf"]
    else:
        formats = [args.format]

    # Generate documentation
    success = generate_all_docs(args.output_dir, formats, args.clean)

    if success:
        print("Documentation generation completed successfully!")
        print(f"Output directory: {args.output_dir}")
        print(f"Formats generated: {', '.join(formats)}")
    else:
        print("Documentation generation failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
