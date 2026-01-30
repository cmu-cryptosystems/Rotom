import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rc
from matplotlib.ticker import FuncFormatter, LogFormatter, LogLocator

# Method 1: Using rc parameters
rc("text", usetex=True)
rc("font", family="serif", serif=["Computer Modern"])

# Parse the data from the table
benchmarks = [
    "MatMul",
    "Double-MatMul",
    "TTM",
    "Convolution",
    "LogReg-MatVecMul",
    "Bert Attention",
]
systems = ["Rotom", "Fhelipe", "Viaduct-HE-e1-o0", "Viaduct-HE-e2-o1"]

# Create DataFrame to hold the data
data = []

# Distance - only n=8192
# data.append({'Benchmark': 'Distance', 'n': 8192, 'Rotom': 0.28, 'Fhelipe': 1.30, 'Viaduct-HE-e1-o0': 0.110, 'Viaduct-HE-e2-o1': 77.86})

# MatMul - both sizes
data.append(
    {
        "Benchmark": "MatMul",
        "n": 8192,
        "Rotom": 0.10,
        "Fhelipe": 2.19,
        "Viaduct-HE-e1-o0": 72.02,
        "Viaduct-HE-e2-o1": 17603.89,
    }
)
data.append(
    {
        "Benchmark": "MatMul",
        "n": 32768,
        "Rotom": 0.11,
        "Fhelipe": 6.89,
        "Viaduct-HE-e1-o0": 997.45,
        "Viaduct-HE-e2-o1": np.nan,
    }
)

# Double-MatMul - both sizes
data.append(
    {
        "Benchmark": "Double-MatMul",
        "n": 8192,
        "Rotom": 0.28,
        "Fhelipe": 2.73,
        "Viaduct-HE-e1-o0": 6346.91,
        "Viaduct-HE-e2-o1": 50040.36,
    }
)
data.append(
    {
        "Benchmark": "Double-MatMul",
        "n": 32768,
        "Rotom": 0.32,
        "Fhelipe": 11.05,
        "Viaduct-HE-e1-o0": np.nan,
        "Viaduct-HE-e2-o1": np.nan,
    }
)

# TTM - both sizes
data.append(
    {
        "Benchmark": "TTM",
        "n": 8192,
        "Rotom": 1.58,
        "Fhelipe": 14.91,
        "Viaduct-HE-e1-o0": np.nan,
        "Viaduct-HE-e2-o1": np.nan,
    }
)
data.append(
    {
        "Benchmark": "TTM",
        "n": 32768,
        "Rotom": 1.34,
        "Fhelipe": 4.84,
        "Viaduct-HE-e1-o0": np.nan,
        "Viaduct-HE-e2-o1": np.nan,
    }
)

# Convolution - both sizes
data.append(
    {
        "Benchmark": "Convolution",
        "n": 8192,
        "Rotom": 1.64,
        "Fhelipe": 1.67,
        "Viaduct-HE-e1-o0": 81.60,
        "Viaduct-HE-e2-o1": 1259.03,
    }
)
data.append(
    {
        "Benchmark": "Convolution",
        "n": 32768,
        "Rotom": 21.97,
        "Fhelipe": 15.89,
        "Viaduct-HE-e1-o0": 1579.479,
        "Viaduct-HE-e2-o1": 22748.81,
    }
)

# LogReg-MatVecMul - both sizes
data.append(
    {
        "Benchmark": "LogReg-MatVecMul",
        "n": 8192,
        "Rotom": 3.00,
        "Fhelipe": 2.34,
        "Viaduct-HE-e1-o0": 58.76,
        "Viaduct-HE-e2-o1": 1052.58,
    }
)
data.append(
    {
        "Benchmark": "LogReg-MatVecMul",
        "n": 32768,
        "Rotom": 1.67,
        "Fhelipe": 2.40,
        "Viaduct-HE-e1-o0": 58.455,
        "Viaduct-HE-e2-o1": 535.105,
    }
)

# Bert Attention - both sizes
data.append(
    {
        "Benchmark": "Bert Attention",
        "n": 8192,
        "Rotom": 243.56,
        "Fhelipe": 113.63,
        "Viaduct-HE-e1-o0": np.nan,
        "Viaduct-HE-e2-o1": np.nan,
    }
)
data.append(
    {
        "Benchmark": "Bert Attention",
        "n": 32768,
        "Rotom": 5.42,
        "Fhelipe": 49.47,
        "Viaduct-HE-e1-o0": np.nan,
        "Viaduct-HE-e2-o1": np.nan,
    }
)

# Convert to DataFrame
df = pd.DataFrame(data)

# Replace NaN values with a small placeholder value for timeout visualization
timeout_placeholder = 0.00001  # Very small value, well below the visible range

# Create a modified dataframe with timeout placeholders for visualization
vis_df = df.copy()
for col in systems:
    vis_df[col] = vis_df[col].fillna(timeout_placeholder)

# Set up the figure with custom grid layout where Distance plot is half the width
n_benchmarks = len(benchmarks)

# Create a figure
fig = plt.figure(figsize=(20, 6))

# Create a grid with custom width ratios - Distance is half width
width_ratios = [0.5 if benchmark == "Distance" else 1 for benchmark in benchmarks]
gs = gridspec.GridSpec(1, n_benchmarks, width_ratios=width_ratios)

# Create subplots
ax0 = plt.subplot(gs[0])
axs = [ax0] + [plt.subplot(gs[i], sharey=ax0) for i in range(1, n_benchmarks)]


# Function to format y-axis ticks in 10^n notation
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


# Set bar width and colors
bar_width = 0.2
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#006400"]  # Colors for the systems

# Common y-axis limit - set it here to ensure text doesn't exceed it
y_min = 0.05
y_max = 100000  # 10^5 (100,000 seconds)
max_label_position = y_max * 0.9  # Maximum height for labels

# For each benchmark
for i, benchmark in enumerate(benchmarks):
    ax = axs[i]

    # Turn off top and right spines, keep left and bottom
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_visible(
        True if i == 0 else False
    )  # Only show left spine for first subplot

    # Filter data for this benchmark
    benchmark_data = vis_df[vis_df["Benchmark"] == benchmark]
    # Original data for reference (with NaN for timeouts)
    orig_data = df[df["Benchmark"] == benchmark]

    # Set up x-coordinates for bars
    sizes = benchmark_data["n"].unique()
    x_positions = np.arange(len(sizes))

    # For each matrix size
    for pos, size in enumerate(sizes):
        size_data = benchmark_data[benchmark_data["n"] == size]
        orig_size_data = orig_data[orig_data["n"] == size]

        # For each system
        for k, system in enumerate(systems):
            if not size_data.empty:
                value = size_data[system].iloc[0]
                orig_value = (
                    orig_size_data[system].iloc[0]
                    if not orig_size_data.empty
                    else np.nan
                )

                # Draw the bar (always draw a bar, even for timeouts)
                bar = ax.bar(
                    pos + (k - 1.5) * bar_width,
                    value,
                    bar_width,
                    color=colors[k],
                    alpha=0.8,
                    label=system if i == 0 and pos == 0 else "",
                )

                # If this is a timeout, mark it appropriately
                if np.isnan(orig_value):
                    # Add timeout indicator at a visible height
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

                    # Add hatch pattern to the bar to indicate timeout
                    bar[0].set_hatch("xxxx")
                    bar[0].set_facecolor("white")
                    bar[0].set_edgecolor(colors[k])

                # Add value label for non-timeout bars with real values
                elif not np.isnan(orig_value) and orig_value > 0.05:
                    if orig_value < 1:
                        label_text = f"{orig_value:.2f}"
                    elif orig_value < 10:
                        label_text = f"{orig_value:.1f}"
                    elif orig_value < 1000:
                        label_text = f"{int(orig_value)}"
                    else:
                        label_text = f"{int(orig_value/1000)}K"

                    # Calculate position for label, ensuring it's above the bar but within subplot limits
                    if orig_value >= max_label_position * 0.85:
                        # For very tall bars, place text at 90% of max height
                        y_pos = max_label_position
                    else:
                        # For normal bars, place text just above bar
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

    # Set log scale for y-axis
    ax.set_yscale("log")

    # Apply power notation formatter to y-axis
    ax.yaxis.set_major_formatter(FuncFormatter(power_notation))

    # Set x-axis properties
    ax.set_xticks(x_positions)
    # Convert size labels to K format for better readability
    size_labels = []
    for size in sizes:
        if size == 8192:
            size_labels.append("n=8K")
        elif size == 32768:
            size_labels.append("n=32K")
        else:
            size_labels.append(f"n={size}")
    ax.set_xticklabels(size_labels, ha="center", fontsize=16)

    # Remove y-axis labels for all but the first subplot
    if i > 0:
        ax.tick_params(labelleft=False)

    # Set title with higher positioning
    ax.set_title(f"{benchmark}", fontsize=18, pad=20)

    # # Add grid
    # ax.grid(True, which="major", ls="-", alpha=0.2, axis='y')
    # ax.grid(True, which="minor", ls=":", alpha=0.1, axis='y')

# Set y-axis label for the first subplot
axs[0].set_ylabel("Compile Time in Log-Scale [s]", fontsize=18)

# Set common y-axis limits
axs[0].set_ylim(y_min, y_max)  # From 0.05s to 100,000s

# Add a common legend at the bottom
handles, labels = [], []
for k, system in enumerate(systems):
    handles.append(plt.Rectangle((0, 0), 1, 1, color=colors[k], alpha=0.8))
    labels.append(system)

# Add timeout indicator to legend
timeout_patch = plt.Rectangle(
    (0, 0), 1, 1, facecolor="white", edgecolor="red", hatch="xxxx"
)
handles.append(timeout_patch)
labels.append("Timeout")

# Move legend to bottom of figure, closer to the graph
fig.legend(
    handles, labels, loc="lower center", ncol=5, fontsize=18, bbox_to_anchor=(0.5, 0.05)
)

# Add a main title for the entire figure
# fig.suptitle('Compile Time Comparison Across Benchmarks', fontsize=16)

plt.tight_layout()
plt.subplots_adjust(top=0.66, bottom=0.2, wspace=0.05)

plt.savefig("results_compile.pdf", bbox_inches="tight")
