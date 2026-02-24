#!/usr/bin/env python3
"""
Script to run all tests in the ./scripts directory and its subdirectories.
This script discovers and executes all shell scripts found in the scripts folder.
"""

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Tuple


def find_test_scripts(scripts_dir: str = "./scripts") -> List[Path]:
    """Find all shell scripts in the scripts directory and subdirectories."""
    scripts_path = Path(scripts_dir)
    if not scripts_path.exists():
        print(f"Error: Scripts directory '{scripts_dir}' not found!")
        return []

    test_scripts = []
    for script_file in scripts_path.rglob("*.sh"):
        if script_file.is_file():
            test_scripts.append(script_file)

    return sorted(test_scripts)


def run_script(script_path: Path) -> Tuple[bool, str]:
    """Run a single shell script and return success status and output."""
    print(f"\n{'='*60}")
    print(f"Running: ./{script_path}")
    print(f"{'='*60}")

    try:
        # Run the script and capture output
        start_time = time.time()
        result = subprocess.run(
            ["bash", str(script_path)],
            capture_output=True,
            text=True,
            shell=False,
            cwd=".",
        )
        end_time = time.time()

        duration = end_time - start_time

        if result.returncode == 0:
            print(f"✅ SUCCESS - {script_path.name} completed in {duration:.2f}s")
            if result.stdout.strip():
                print("Output:")
                print(result.stdout)
        else:
            print(f"❌ FAILED - {script_path.name} failed after {duration:.2f}s")
            print(f"Return code: {result.returncode}")
            if result.stderr.strip():
                print("Error output:")
                print(result.stderr)
            if result.stdout.strip():
                print("Standard output:")
                print(result.stdout)

        return result.returncode == 0, result.stdout + result.stderr

    except Exception as e:
        print(f"❌ ERROR - Failed to run {script_path.name}: {e}")
        return False, str(e)


def main():
    """Main function to run all tests."""
    print("🚀 Starting test execution for all scripts in ./scripts directory")
    print(f"Current working directory: {os.getcwd()}")

    # Find all test scripts
    test_scripts = find_test_scripts()

    if not test_scripts:
        print("No test scripts found in ./scripts directory!")
        sys.exit(1)

    print(f"\nFound {len(test_scripts)} test scripts:")
    for script in test_scripts:
        print(f"  - {script}")

    # Run all tests
    print(f"\n{'='*60}")
    print("EXECUTING TESTS")
    print(f"{'='*60}")

    successful_tests = 0
    failed_tests = 0
    test_results = []

    start_time = time.time()

    for script in test_scripts:
        success, output = run_script(script)
        test_results.append((script, success, output))

        if success:
            successful_tests += 1
        else:
            failed_tests += 1

    total_time = time.time() - start_time

    # Print summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Total tests run: {len(test_scripts)}")
    print(f"Successful: {successful_tests}")
    print(f"Failed: {failed_tests}")
    print(f"Total execution time: {total_time:.2f}s")

    if failed_tests > 0:
        print(f"\n❌ {failed_tests} test(s) failed:")
        for script, success, output in test_results:
            if not success:
                print(f"  - {script}")
        sys.exit(1)
    else:
        print(f"\n🎉 All tests passed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()
