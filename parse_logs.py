import os
import re
from argparse import ArgumentParser

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rc
from matplotlib.ticker import FuncFormatter

# Set up LaTeX rendering
rc("text", usetex=True)
rc("font", family="serif", serif=["Computer Modern"])

# Benchmark name mapping from log file names to display names
BENCHMARK_MAP = {
    "matmul": "MatMul",
    "double_matmul": "Double-MatMul",
    "ttm": "TTM",
    "convolution": "Convolution",
    "logreg": "LogReg-MatVecMul",
    "bert": "Bert Attention",
    "bert_attention": "Bert Attention",
}

# System name mapping
SYSTEM_MAP = {
    "rotom": "Rotom",
    "fhelipe": "Fhelipe",
    "viaduct-he-e1-o0": "Viaduct-HE-e1-o0",
    "viaduct-he-e2-o1": "Viaduct-HE-e2-o1",
    "viaduct": "Viaduct-HE-e1-o0",  # default viaduct
}

# Expected benchmarks and systems
BENCHMARKS = [
    "MatMul",
    "Double-MatMul",
    "TTM",
    "Convolution",
    "LogReg-MatVecMul",
    "Bert Attention",
]
SYSTEMS = ["Rotom", "Fhelipe", "Viaduct-HE-e1-o0", "Viaduct-HE-e2-o1"]


def walk_files_recursively(directory):
    """Walk through all files in the given directory and its subdirectories."""
    paths = []
    if not os.path.exists(directory):
        return paths
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            paths.append(file_path)
    return paths


def parse_filepath(filepath):
    """Parse the filepath to extract system, benchmark, n value, and run number."""
    parts = filepath.split("/")

    # Extract system name (e.g., logs/rotom/matmul/matmul_128_64_1.txt -> rotom)
    system_name = None
    for part in parts:
        if part in SYSTEM_MAP or part.lower() in SYSTEM_MAP:
            system_name = part.lower()
            break

    # Extract file name without extension
    file_name = parts[-1].split(".")[0]

    # Try to extract n value and run from filename
    # Patterns:
    # - matmul_128_64_1 (dimensions, n is in log)
    # - ttm_8192_1 (n is 8192)
    # - bert_8192_1 (n is 8192)
    # - logreg_8192_1 (n is 8192)
    n_value = None
    run = None

    # Extract run number (last digit)
    run_match = re.search(r"_(\d+)$", file_name)
    if run_match:
        run = run_match.group(1)
        file_name = file_name[: run_match.start()]

    # Extract n value - look for patterns like _8192, _32768
    # Common n values: 8192, 32768, 4096, 16384
    n_match = re.search(r"_(\d+)$", file_name)
    if n_match:
        potential_n = int(n_match.group(1))
        # Check if it's a common n value (not a dimension)
        if potential_n in [8192, 32768, 4096, 16384]:
            n_value = potential_n
            file_name = file_name[: n_match.start()]

    # For matmul with dimensions (e.g., matmul_128_64), n is not in filename
    # We'll need to extract it from the log file or use a mapping
    # For now, we'll handle this in the main function by reading the log

    # Extract benchmark name
    benchmark_name = file_name.lower()

    return system_name, benchmark_name, n_value, run


def parse_file_data(path):
    """Parse log file to extract runtime, compile time, data size, n value, etc."""
    data = []
    compile_time = None
    runtime = None
    data_size = None
    n_value = None

    try:
        with open(path, "r") as file:
            for line in file:
                line = line.strip()

                # Extract compile time
                if "compile time:" in line.lower():
                    try:
                        compile_time = float(line.split(":")[-1].strip())
                    except:
                        pass

                # Extract runtime (look for "runtime:" or "circuit runtime:")
                if "runtime:" in line.lower() and "circuit runtime" not in line.lower():
                    try:
                        runtime = float(line.split(":")[-1].strip())
                    except:
                        pass
                elif "circuit runtime:" in line.lower():
                    try:
                        runtime = float(line.split(":")[-1].strip())
                    except:
                        pass

                # Extract data size
                if "data size:" in line.lower():
                    data_size_str = line.split(":")[-1].strip()
                    # Extract number from "18.01 MB" -> 18.01
                    size_match = re.search(r"([\d.]+)", data_size_str)
                    if size_match:
                        data_size = float(size_match.group(1))

                # Try to extract n value from log (e.g., "ring dimension 65536" or "--n 8192")
                if "ring dimension" in line.lower():
                    dim_match = re.search(r"ring dimension (\d+)", line.lower())
                    if dim_match:
                        # Map ring dimension to n value
                        ring_dim = int(dim_match.group(1))
                        # Common mappings: 16384 -> 8192, 65536 -> 32768
                        if ring_dim == 16384:
                            n_value = 8192
                        elif ring_dim == 65536:
                            n_value = 32768

                # Extract FHEOp dictionary
                if "FHEOp" in line or "HEOp" in line:
                    data.append(line)
    except Exception as e:
        print(f"Warning: Error parsing {path}: {e}")

    result = {
        "runtime": runtime,
        "compile time": compile_time,
        "data size": data_size,
        "n_from_log": n_value,
    }

    return result


def parse_data_to_dictionary(data_lines):
    """Parse FHEOp dictionary lines into a dictionary."""
    result = {}
    for line in data_lines:
        line = line.strip()
        if line.startswith("{'FHEOp") or line.startswith("{'HEOp"):
            clean_line = line.strip("{}\n")
            parts = clean_line.split(", ")
            for part in parts:
                try:
                    enum_part, value = part.split(": ")
                    op_name = enum_part.split("'")[1].split("'")[0].split(".")[-1]
                    result[op_name] = int(value)
                except:
                    pass
    return result


def aggregate_results(results):
    """Aggregate results across multiple runs for each benchmark/system/n combination."""
    # Group by (system, benchmark, n)
    grouped = {}
    for result in results:
        key = (result.get("system"), result.get("benchmark"), result.get("n"))
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(result)

    # Aggregate (take mean)
    aggregated = []
    for key, group in grouped.items():
        system, benchmark, n = key

        # Filter out None values
        runtimes = [r["runtime"] for r in group if r.get("runtime") is not None]
        compile_times = [
            r["compile time"] for r in group if r.get("compile time") is not None
        ]
        data_sizes = [r["data size"] for r in group if r.get("data size") is not None]

        agg_result = {
            "system": system,
            "benchmark": benchmark,
            "n": n,
            "runtime": np.mean(runtimes) if runtimes else None,
            "compile time": np.mean(compile_times) if compile_times else None,
            "data size": np.mean(data_sizes) if data_sizes else None,
        }

        # Calculate LAN and WAN times
        if agg_result["runtime"] is not None and agg_result["data size"] is not None:
            data_bytes = (
                agg_result["data size"] * 8
            )  # MB to Mbits (assuming 8 bits per byte)
            agg_result["lan_runtime"] = agg_result["runtime"] + data_bytes * 1e-4
            agg_result["wan_runtime"] = agg_result["runtime"] + data_bytes * 1e-3
        else:
            agg_result["lan_runtime"] = None
            agg_result["wan_runtime"] = None

        aggregated.append(agg_result)

    return aggregated


def generate_compile_plot(data_df, output_path="plots/results_compile.pdf"):
    """Generate compile time plot."""
    benchmarks = BENCHMARKS
    systems = SYSTEMS

    # Create DataFrame structure
    plot_data = []
    for benchmark in benchmarks:
        for n in [8192, 32768]:
            row = {"Benchmark": benchmark, "n": n}
            for system in systems:
                # Find matching data
                matches = data_df[
                    (data_df["benchmark"] == benchmark)
                    & (data_df["n"] == n)
                    & (data_df["system"] == system)
                ]
                if not matches.empty and matches.iloc[0]["compile time"] is not None:
                    row[system] = matches.iloc[0]["compile time"]
                else:
                    row[system] = np.nan
            plot_data.append(row)

    df = pd.DataFrame(plot_data)

    # Replace NaN with placeholder
    timeout_placeholder = 0.00001
    vis_df = df.copy()
    for col in systems:
        vis_df[col] = vis_df[col].fillna(timeout_placeholder)

    # Create figure
    n_benchmarks = len(benchmarks)
    fig = plt.figure(figsize=(20, 6))
    width_ratios = [0.5 if benchmark == "Distance" else 1 for benchmark in benchmarks]
    gs = gridspec.GridSpec(1, n_benchmarks, width_ratios=width_ratios)

    ax0 = plt.subplot(gs[0])
    axs = [ax0] + [plt.subplot(gs[i], sharey=ax0) for i in range(1, n_benchmarks)]

    def power_notation(x, pos):
        if x <= 0:
            return "0"
        exponent = int(np.log10(x))
        if exponent == 0:
            return f"{x:.1f}"
        elif exponent == 1:
            return f"{x:.1f}"
        else:
            return f"$10^{{{exponent}}}$"

    bar_width = 0.2
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#006400"]

    y_min = 0.05
    y_max = 100000
    max_label_position = y_max * 0.9

    for i, benchmark in enumerate(benchmarks):
        ax = axs[i]
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(True)
        ax.spines["left"].set_visible(True if i == 0 else False)

        benchmark_data = vis_df[vis_df["Benchmark"] == benchmark]
        orig_data = df[df["Benchmark"] == benchmark]

        sizes = benchmark_data["n"].unique()
        x_positions = np.arange(len(sizes))

        for pos, size in enumerate(sizes):
            size_data = benchmark_data[benchmark_data["n"] == size]
            orig_size_data = orig_data[orig_data["n"] == size]

            for k, system in enumerate(systems):
                if not size_data.empty:
                    value = size_data[system].iloc[0]
                    orig_value = (
                        orig_size_data[system].iloc[0]
                        if not orig_size_data.empty
                        else np.nan
                    )

                    bar = ax.bar(
                        pos + (k - 1.5) * bar_width,
                        value,
                        bar_width,
                        color=colors[k],
                        alpha=0.8,
                    )

                    if np.isnan(orig_value):
                        ax.text(
                            pos + (k - 1.5) * bar_width,
                            0.1,
                            "t-out",
                            ha="center",
                            va="bottom",
                            fontsize=14,
                            color="red",
                            rotation=90,
                            bbox=dict(
                                facecolor="white",
                                edgecolor="red",
                                alpha=0.7,
                                pad=1,
                                boxstyle="round,pad=0.2",
                            ),
                        )
                        bar[0].set_hatch("xxxx")
                        bar[0].set_facecolor("white")
                        bar[0].set_edgecolor(colors[k])
                    elif not np.isnan(orig_value) and orig_value > 0.05:
                        if orig_value < 1:
                            label_text = f"{orig_value:.2f}"
                        elif orig_value < 10:
                            label_text = f"{orig_value:.1f}"
                        elif orig_value < 1000:
                            label_text = f"{int(orig_value)}"
                        else:
                            label_text = f"{int(orig_value/1000)}K"

                        if orig_value >= max_label_position * 0.85:
                            y_pos = max_label_position
                        else:
                            y_pos = min(orig_value * 1.05, max_label_position)

                        rotation = 90 if orig_value > 1000 else 0
                        ax.text(
                            pos + (k - 1.5) * bar_width,
                            y_pos,
                            label_text,
                            ha="center",
                            va="bottom",
                            fontsize=12,
                            rotation=rotation,
                        )

        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(FuncFormatter(power_notation))
        ax.set_xticks(x_positions)
        size_labels = []
        for size in sizes:
            if size == 8192:
                size_labels.append("n=8K")
            elif size == 32768:
                size_labels.append("n=32K")
            else:
                size_labels.append(f"n={size}")
        ax.set_xticklabels(size_labels, ha="center", fontsize=16)
        if i > 0:
            ax.tick_params(labelleft=False)
        ax.set_title(f"{benchmark}", fontsize=18, pad=20)

    axs[0].set_ylabel("Compile Time in Log-Scale [s]", fontsize=18)
    axs[0].set_ylim(y_min, y_max)

    handles, labels = [], []
    for k, system in enumerate(systems):
        handles.append(plt.Rectangle((0, 0), 1, 1, color=colors[k], alpha=0.8))
        labels.append(system)

    timeout_patch = plt.Rectangle(
        (0, 0), 1, 1, facecolor="white", edgecolor="red", hatch="xxxx"
    )
    handles.append(timeout_patch)
    labels.append("Timeout")

    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=5,
        fontsize=18,
        bbox_to_anchor=(0.5, 0.05),
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.66, bottom=0.2, wspace=0.05)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Generated compile plot: {output_path}")


def generate_exec_plot(data_df, output_path="plots/results_exec.pdf"):
    """Generate execution time plot (LAN and WAN)."""
    benchmarks = BENCHMARKS
    systems = SYSTEMS

    # Create DataFrame structure for LAN and WAN
    lan_data = []
    wan_data = []

    for benchmark in benchmarks:
        for n in [8192, 32768]:
            lan_row = {"Benchmark": benchmark, "n": n}
            wan_row = {"Benchmark": benchmark, "n": n}

            for system in systems:
                matches = data_df[
                    (data_df["benchmark"] == benchmark)
                    & (data_df["n"] == n)
                    & (data_df["system"] == system)
                ]
                if not matches.empty:
                    match = matches.iloc[0]
                    if match["lan_runtime"] is not None:
                        lan_row[system] = match["lan_runtime"]
                    else:
                        lan_row[system] = np.nan

                    if match["wan_runtime"] is not None:
                        wan_row[system] = match["wan_runtime"]
                    else:
                        wan_row[system] = np.nan
                else:
                    lan_row[system] = np.nan
                    wan_row[system] = np.nan

            lan_data.append(lan_row)
            wan_data.append(wan_row)

    lan_df = pd.DataFrame(lan_data)
    wan_df = pd.DataFrame(wan_data)

    # Calculate WAN overhead
    overhead_df = wan_df.copy()
    for system in systems:
        system_indices = ~wan_df[system].isna() & ~lan_df[system].isna()
        overhead_df.loc[system_indices, system] = (
            wan_df.loc[system_indices, system] - lan_df.loc[system_indices, system]
        )

    for system in systems:
        overhead_df.loc[overhead_df[system] < 0, system] = 0

    # Create figure
    n_benchmarks = len(benchmarks)
    fig = plt.figure(figsize=(20, 6))
    width_ratios = [0.5 if benchmark == "Distance" else 1 for benchmark in benchmarks]
    gs = gridspec.GridSpec(1, n_benchmarks, width_ratios=width_ratios)

    ax0 = plt.subplot(gs[0])
    axs = [ax0] + [plt.subplot(gs[i], sharey=ax0) for i in range(1, n_benchmarks)]

    def power_notation(x, pos):
        if x <= 0:
            return "0"
        exponent = int(np.log10(x))
        if exponent == 0:
            return f"{x:.1f}"
        elif exponent == 1:
            return f"{x:.1f}"
        else:
            return f"$10^{{{exponent}}}$"

    bar_width = 0.2
    colors = {
        "Rotom": "#1f77b4",
        "Fhelipe": "#ff7f0e",
        "Viaduct-HE-e1-o0": "#2ca02c",
        "Viaduct-HE-e2-o1": "#006400",
    }

    lan_alpha = 0.9
    wan_alpha = 0.6

    y_min = 0.01
    y_max = 10000
    max_label_position = y_max * 0.9

    for i, benchmark in enumerate(benchmarks):
        ax = axs[i]
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(True)
        ax.spines["left"].set_visible(True if i == 0 else False)

        lan_benchmark_data = lan_df[lan_df["Benchmark"] == benchmark]
        wan_benchmark_data = wan_df[wan_df["Benchmark"] == benchmark]
        overhead_benchmark_data = overhead_df[overhead_df["Benchmark"] == benchmark]

        sizes = lan_benchmark_data["n"].unique()
        x_positions = np.arange(len(sizes))

        for pos, size in enumerate(sizes):
            lan_size_data = lan_benchmark_data[lan_benchmark_data["n"] == size]
            wan_size_data = wan_benchmark_data[wan_benchmark_data["n"] == size]
            overhead_size_data = overhead_benchmark_data[
                overhead_benchmark_data["n"] == size
            ]

            for k, system in enumerate(systems):
                if not lan_size_data.empty and not wan_size_data.empty:
                    lan_value = (
                        lan_size_data[system].iloc[0]
                        if not np.isnan(lan_size_data[system].iloc[0])
                        else 0
                    )
                    wan_value = (
                        wan_size_data[system].iloc[0]
                        if not np.isnan(wan_size_data[system].iloc[0])
                        else 0
                    )
                    overhead_value = (
                        overhead_size_data[system].iloc[0]
                        if not np.isnan(overhead_size_data[system].iloc[0])
                        else 0
                    )

                    is_timeout = np.isnan(lan_size_data[system].iloc[0]) or np.isnan(
                        wan_size_data[system].iloc[0]
                    )

                    if not is_timeout:
                        lan_bar = ax.bar(
                            pos + (k - 1.5) * bar_width,
                            lan_value,
                            bar_width,
                            color=colors[system],
                            alpha=lan_alpha,
                        )

                        if overhead_value > 0:
                            wan_bar = ax.bar(
                                pos + (k - 1.5) * bar_width,
                                overhead_value,
                                bar_width,
                                bottom=lan_value,
                                color=colors[system],
                                alpha=wan_alpha,
                                hatch="////",
                            )

                        if wan_value < 1:
                            label_text = f"{wan_value:.2f}"
                        elif wan_value < 10:
                            label_text = f"{wan_value:.1f}"
                        elif wan_value < 1000:
                            label_text = f"{int(wan_value)}"
                        else:
                            label_text = f"{int(wan_value/1000)}K"

                        if wan_value >= max_label_position * 0.9:
                            y_pos = max_label_position
                        else:
                            y_pos = min(wan_value * 1.05, max_label_position)

                        rotation = 90 if wan_value > 1000 else 0
                        ax.text(
                            pos + (k - 1.5) * bar_width,
                            y_pos,
                            label_text,
                            ha="center",
                            va="bottom",
                            fontsize=12,
                            rotation=rotation,
                        )
                    else:
                        timeout_placeholder = 0.00001
                        ax.text(
                            pos + (k - 1.5) * bar_width,
                            0.02,
                            "t-out",
                            ha="center",
                            va="bottom",
                            fontsize=14,
                            color="red",
                            rotation=90,
                            bbox=dict(
                                facecolor="white",
                                edgecolor="red",
                                alpha=0.7,
                                pad=1,
                                boxstyle="round,pad=0.2",
                            ),
                        )
                        timeout_bar = ax.bar(
                            pos + (k - 1.5) * bar_width,
                            timeout_placeholder,
                            bar_width,
                            color="white",
                            edgecolor=colors[system],
                            hatch="xxxx",
                        )

        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(FuncFormatter(power_notation))
        ax.set_xticks(x_positions)
        size_labels = []
        for size in sizes:
            if size == 8192:
                size_labels.append("n=8K")
            elif size == 32768:
                size_labels.append("n=32K")
            else:
                size_labels.append(f"n={size}")
        ax.set_xticklabels(size_labels, ha="center", fontsize=16)
        if i > 0:
            ax.tick_params(labelleft=False)
        ax.set_title(f"{benchmark}", fontsize=18, pad=20)

    axs[0].set_ylabel("Runtime in Log-Scale [s]", fontsize=18)
    axs[0].set_ylim(y_min, y_max)

    handles = []
    labels = []

    for system in systems:
        handles.append(
            plt.Rectangle((0, 0), 1, 1, color=colors[system], alpha=lan_alpha)
        )
        labels.append(system)

    lan_patch = plt.Rectangle(
        (0, 0), 1, 1, facecolor="white", edgecolor="gray", alpha=lan_alpha
    )
    wan_patch = plt.Rectangle(
        (0, 0), 1, 1, facecolor="white", edgecolor="gray", alpha=wan_alpha, hatch="////"
    )
    handles.extend([lan_patch, wan_patch])
    labels.extend(["LAN", "WAN Overhead"])

    timeout_patch = plt.Rectangle(
        (0, 0), 1, 1, facecolor="white", edgecolor="red", hatch="xxxx"
    )
    handles.append(timeout_patch)
    labels.append("Timeout")

    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=7,
        fontsize=18,
        bbox_to_anchor=(0.5, 0.05),
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.66, bottom=0.2, wspace=0.05)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Generated exec plot: {output_path}")


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--logs-dir", default="./logs", help="Directory containing log files"
    )
    parser.add_argument(
        "--output-dir", default="./plots", help="Output directory for plots"
    )
    parser.add_argument("--sys", help="Specific system to process (optional)")
    parser.add_argument(
        "--fn", help="Specific function/benchmark to process (optional)"
    )
    args = parser.parse_args()

    # Find all log files
    if args.sys:
        directories = [os.path.join(args.logs_dir, args.sys)]
    else:
        directories = [
            os.path.join(args.logs_dir, d)
            for d in os.listdir(args.logs_dir)
            if os.path.isdir(os.path.join(args.logs_dir, d))
        ]

    all_paths = []
    for directory in directories:
        all_paths.extend(walk_files_recursively(directory))

    # Parse all log files
    results = []
    for path in all_paths:
        system_name, benchmark_name, n_value, run = parse_filepath(path)

        if not system_name:
            continue

        # Map system name
        system = SYSTEM_MAP.get(system_name, system_name.capitalize())

        # Map benchmark name
        benchmark = None
        # Sort by length (longest first) to match more specific names first
        for key in sorted(BENCHMARK_MAP.keys(), key=len, reverse=True):
            if key in benchmark_name:
                benchmark = BENCHMARK_MAP[key]
                break

        if not benchmark:
            # Skip unknown benchmarks
            continue

        # Filter by function if specified
        if args.fn and args.fn.lower() not in benchmark_name.lower():
            continue

        # Parse file data
        file_data = parse_file_data(path)

        # Use n from log if not in filename, or try to infer from benchmark name
        if n_value is None:
            n_value = file_data.get("n_from_log")

        # If still no n value, try to infer from benchmark name patterns
        if n_value is None:
            # For matmul_128_64, we need to map dimensions to n
            # Based on scripts: matmul_128_64 uses --n 8192, matmul_256_128 uses --n 32768
            if "matmul_128_64" in benchmark_name:
                n_value = 8192
            elif "matmul_256_128" in benchmark_name:
                n_value = 32768
            elif "double_matmul_128_64" in benchmark_name:
                n_value = 8192
            elif "double_matmul_256_128" in benchmark_name:
                n_value = 32768

        # Skip if we still can't determine n
        if n_value is None:
            continue

        result = {
            "system": system,
            "benchmark": benchmark,
            "n": n_value,
            "run": run,
            "runtime": file_data.get("runtime"),
            "compile time": file_data.get("compile time"),
            "data size": file_data.get("data size"),
        }

        results.append(result)

    if not results:
        print("No log files found or parsed. Check your logs directory.")
        return

    # Aggregate results
    aggregated = aggregate_results(results)
    data_df = pd.DataFrame(aggregated)

    # Print summary
    print("\n=== Parsed Results Summary ===")
    print(data_df.to_string())
    print()

    # Generate plots
    compile_path = os.path.join(args.output_dir, "results_compile.pdf")
    exec_path = os.path.join(args.output_dir, "results_exec.pdf")

    generate_compile_plot(data_df, compile_path)
    generate_exec_plot(data_df, exec_path)

    print("\n=== Done ===")
    print(f"Plots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
