import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.gridspec as gridspec
from matplotlib.ticker import LogFormatter, LogLocator, FuncFormatter
from matplotlib import rc

# Method 1: Using rc parameters
rc('text', usetex=True)
rc('font', family='serif', serif=['Computer Modern'])

# Parse the data from the tables (LAN and WAN Runtime data)
benchmarks = ['MatMul', 'Double-MatMul', 'TTM', 'Convolution', 'LogReg-MatVecMul', 'Bert Attention']
systems = ['Rotom', 'Fhelipe', 'Viaduct-HE-e1-o0', 'Viaduct-HE-e2-o1']

# Create DataFrame to hold the LAN runtime data
lan_data = []

# Distance - only n=8192
# lan_data.append({'Benchmark': 'Distance', 'n': 8192, 'Rotom': 0.05, 'Fhelipe': 0.13, 'Viaduct-HE-e1-o0': 0.06, 'Viaduct-HE-e2-o1': 0.06})

# MatMul - both sizes
lan_data.append({'Benchmark': 'MatMul', 'n': 8192, 'Rotom': 0.10, 'Fhelipe': 6.37, 'Viaduct-HE-e1-o0': 4.41, 'Viaduct-HE-e2-o1': 0.10})
lan_data.append({'Benchmark': 'MatMul', 'n': 32768, 'Rotom': 1.36, 'Fhelipe': 27.34, 'Viaduct-HE-e1-o0': 44.69, 'Viaduct-HE-e2-o1': np.nan})

# Double-MatMul - both sizes
lan_data.append({'Benchmark': 'Double-MatMul', 'n': 8192, 'Rotom': 2.97, 'Fhelipe': 33.38, 'Viaduct-HE-e1-o0': 12.05, 'Viaduct-HE-e2-o1': 6.32})
lan_data.append({'Benchmark': 'Double-MatMul', 'n': 32768, 'Rotom': 25.28, 'Fhelipe': 320.28, 'Viaduct-HE-e1-o0': np.nan, 'Viaduct-HE-e2-o1': np.nan})

# TTM - both sizes
lan_data.append({'Benchmark': 'TTM', 'n': 8192, 'Rotom': 91.46, 'Fhelipe': 166.74, 'Viaduct-HE-e1-o0': np.nan, 'Viaduct-HE-e2-o1': np.nan})
lan_data.append({'Benchmark': 'TTM', 'n': 32768, 'Rotom': 126.21, 'Fhelipe': 197.812, 'Viaduct-HE-e1-o0': np.nan, 'Viaduct-HE-e2-o1': np.nan})

# Convolution - both sizes
lan_data.append({'Benchmark': 'Convolution', 'n': 8192, 'Rotom': 0.09, 'Fhelipe': 0.10, 'Viaduct-HE-e1-o0': 0.228, 'Viaduct-HE-e2-o1': 0.251})
lan_data.append({'Benchmark': 'Convolution', 'n': 32768, 'Rotom': 0.25, 'Fhelipe': 0.63, 'Viaduct-HE-e1-o0': 0.99, 'Viaduct-HE-e2-o1': 0.25})

# LogReg-MatVecMul - both sizes
lan_data.append({'Benchmark': 'LogReg-MatVecMul', 'n': 8192, 'Rotom': 0.30, 'Fhelipe': 5.37, 'Viaduct-HE-e1-o0': 9.70, 'Viaduct-HE-e2-o1': 9.74})
lan_data.append({'Benchmark': 'LogReg-MatVecMul', 'n': 32768, 'Rotom': 3.64, 'Fhelipe': 7.38, 'Viaduct-HE-e1-o0': 42.71, 'Viaduct-HE-e2-o1': 42.89})

# Bert Attention - both sizes
lan_data.append({'Benchmark': 'Bert Attention', 'n': 8192, 'Rotom': 300.54, 'Fhelipe': 3926.59, 'Viaduct-HE-e1-o0': np.nan, 'Viaduct-HE-e2-o1': np.nan})
lan_data.append({'Benchmark': 'Bert Attention', 'n': 32768, 'Rotom': 59.17, 'Fhelipe': 4753.58, 'Viaduct-HE-e1-o0': np.nan, 'Viaduct-HE-e2-o1': np.nan})

# Create DataFrame to hold the WAN runtime data
wan_data = []

# Distance - only n=8192
# wan_data.append({'Benchmark': 'Distance', 'n': 8192, 'Rotom': 0.06, 'Fhelipe': 0.14, 'Viaduct-HE-e1-o0': 0.07, 'Viaduct-HE-e2-o1': 0.07})

# MatMul - both sizes
wan_data.append({'Benchmark': 'MatMul', 'n': 8192, 'Rotom': 0.20, 'Fhelipe': 7.77, 'Viaduct-HE-e1-o0': 4.88, 'Viaduct-HE-e2-o1': 0.33})
wan_data.append({'Benchmark': 'MatMul', 'n': 32768, 'Rotom': 1.41, 'Fhelipe': 28.10, 'Viaduct-HE-e1-o0': 48.39, 'Viaduct-HE-e2-o1': np.nan})

# Double-MatMul - both sizes
wan_data.append({'Benchmark': 'Double-MatMul', 'n': 8192, 'Rotom': 4.70, 'Fhelipe': 35.13, 'Viaduct-HE-e1-o0': 12.99, 'Viaduct-HE-e2-o1': 7.02})
wan_data.append({'Benchmark': 'Double-MatMul', 'n': 32768, 'Rotom': 39.12, 'Fhelipe': 334.16, 'Viaduct-HE-e1-o0': np.nan, 'Viaduct-HE-e2-o1': np.nan})

# TTM - both sizes
wan_data.append({'Benchmark': 'TTM', 'n': 8192, 'Rotom': 99.03, 'Fhelipe': 178.10, 'Viaduct-HE-e1-o0': np.nan, 'Viaduct-HE-e2-o1': np.nan})
wan_data.append({'Benchmark': 'TTM', 'n': 32768, 'Rotom': 133.78, 'Fhelipe': 209.16, 'Viaduct-HE-e1-o0': np.nan, 'Viaduct-HE-e2-o1': np.nan})

# Convolution - both sizes
wan_data.append({'Benchmark': 'Convolution', 'n': 8192, 'Rotom': 0.10, 'Fhelipe': 0.11, 'Viaduct-HE-e1-o0': 0.24, 'Viaduct-HE-e2-o1': 0.26})
wan_data.append({'Benchmark': 'Convolution', 'n': 32768, 'Rotom': 0.43, 'Fhelipe': 0.67, 'Viaduct-HE-e1-o0': 1.04, 'Viaduct-HE-e2-o1': 0.52})

# LogReg-MatVecMul - both sizes
wan_data.append({'Benchmark': 'LogReg-MatVecMul', 'n': 8192, 'Rotom': 0.30, 'Fhelipe': 5.39, 'Viaduct-HE-e1-o0': 11.48, 'Viaduct-HE-e2-o1': 11.52})
wan_data.append({'Benchmark': 'LogReg-MatVecMul', 'n': 32768, 'Rotom': 3.69, 'Fhelipe': 7.44, 'Viaduct-HE-e1-o0': 49.81, 'Viaduct-HE-e2-o1': 49.98})

# Bert Attention - both sizes
wan_data.append({'Benchmark': 'Bert Attention', 'n': 8192, 'Rotom': 300.69, 'Fhelipe': 3926.61, 'Viaduct-HE-e1-o0': np.nan, 'Viaduct-HE-e2-o1': np.nan})
wan_data.append({'Benchmark': 'Bert Attention', 'n': 32768, 'Rotom': 59.37, 'Fhelipe': 4753.67, 'Viaduct-HE-e1-o0': np.nan, 'Viaduct-HE-e2-o1': np.nan})

# Convert to DataFrames
lan_df = pd.DataFrame(lan_data)
wan_df = pd.DataFrame(wan_data)

# Calculate WAN overhead (difference between WAN and LAN)
# This will be the top segment of our stacked bar
overhead_df = wan_df.copy()
for system in systems:
    # For each system, calculate WAN - LAN
    system_indices = ~wan_df[system].isna() & ~lan_df[system].isna()
    overhead_df.loc[system_indices, system] = wan_df.loc[system_indices, system] - lan_df.loc[system_indices, system]

# Check for any negative differences (data inconsistency) and set to zero
for system in systems:
    overhead_df.loc[overhead_df[system] < 0, system] = 0

# Set up the figure with custom grid layout where Distance plot is half the width
n_benchmarks = len(benchmarks)

# Create a figure
fig = plt.figure(figsize=(20, 6))

# Create a grid with custom width ratios - Distance is half width
width_ratios = [0.5 if benchmark == 'Distance' else 1 for benchmark in benchmarks]
gs = gridspec.GridSpec(1, n_benchmarks, width_ratios=width_ratios)

# Create subplots
ax0 = plt.subplot(gs[0])
axs = [ax0] + [plt.subplot(gs[i], sharey=ax0) for i in range(1, n_benchmarks)]

# Function to format y-axis ticks in 10^n notation
def power_notation(x, pos):
    if x <= 0:
        return '0'
    exponent = int(np.log10(x))
    if exponent == 0:
        return f'{x:.1f}'
    elif exponent == 1:
        return f'{x:.1f}'
    else:
        return f'$10^{{{exponent}}}$'

# Set bar width and colors - Updated with both Viaduct systems in green shades
bar_width = 0.2
colors = {
    'Rotom': '#1f77b4',  # Blue
    'Fhelipe': '#ff7f0e',  # Orange
    'Viaduct-HE-e1-o0': '#2ca02c',  # Medium green
    'Viaduct-HE-e2-o1': '#006400'   # Dark green
}

color_list = [colors[system] for system in systems]
lan_alpha = 0.9  # Solid for LAN
wan_alpha = 0.6  # Lighter for WAN overhead

# Common y-axis limit - set it here to ensure text doesn't exceed it
y_min = 0.01
y_max = 10000  # 10^4 (10,000 seconds) - adjusted based on the runtime data range
max_label_position = y_max * 0.9  # Maximum height for labels

# For each benchmark
for i, benchmark in enumerate(benchmarks):
    ax = axs[i]
    
    # Turn off top and right spines, keep left and bottom
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True if i == 0 else False)  # Only show left spine for first subplot
    
    # Filter data for this benchmark
    lan_benchmark_data = lan_df[lan_df['Benchmark'] == benchmark]
    wan_benchmark_data = wan_df[wan_df['Benchmark'] == benchmark]
    overhead_benchmark_data = overhead_df[overhead_df['Benchmark'] == benchmark]
    
    # Set up x-coordinates for bars
    sizes = lan_benchmark_data['n'].unique()
    x_positions = np.arange(len(sizes))
    
    # For each matrix size
    for pos, size in enumerate(sizes):
        lan_size_data = lan_benchmark_data[lan_benchmark_data['n'] == size]
        wan_size_data = wan_benchmark_data[wan_benchmark_data['n'] == size]
        overhead_size_data = overhead_benchmark_data[overhead_benchmark_data['n'] == size]
        
        # For each system
        for k, system in enumerate(systems):
            if not lan_size_data.empty and not wan_size_data.empty:
                lan_value = lan_size_data[system].iloc[0] if not np.isnan(lan_size_data[system].iloc[0]) else 0
                wan_value = wan_size_data[system].iloc[0] if not np.isnan(wan_size_data[system].iloc[0]) else 0
                overhead_value = overhead_size_data[system].iloc[0] if not np.isnan(overhead_size_data[system].iloc[0]) else 0
                
                # Check if this is a timeout case
                is_timeout = np.isnan(lan_size_data[system].iloc[0]) or np.isnan(wan_size_data[system].iloc[0])
                
                if not is_timeout:
                    # Draw the LAN part first (bottom of stack)
                    lan_bar = ax.bar(pos + (k - 1.5) * bar_width, lan_value, bar_width, 
                                    color=colors[system], alpha=lan_alpha)
                    
                    # Draw the WAN overhead part (top of stack)
                    if overhead_value > 0:
                        wan_bar = ax.bar(pos + (k - 1.5) * bar_width, overhead_value, bar_width, 
                                        bottom=lan_value, color=colors[system], alpha=wan_alpha, 
                                        hatch='////')
                    
                    # Add label for total (WAN value)
                    if wan_value < 1:
                        label_text = f"{wan_value:.2f}"
                    elif wan_value < 10:
                        label_text = f"{wan_value:.1f}"
                    elif wan_value < 1000:
                        label_text = f"{int(wan_value)}"
                    else:
                        label_text = f"{int(wan_value/1000)}K"
                    
                    # Calculate position for label, ensuring it's above the bar but within subplot limits
                    if wan_value >= max_label_position * 0.9:
                        # For very tall bars, place text at 90% of max height
                        y_pos = max_label_position
                    else:
                        # For normal bars, place text just above bar
                        y_pos = min(wan_value * 1.05, max_label_position)
                    
                    rotation = 90 if wan_value > 1000 else 0
                    ax.text(pos + (k - 1.5) * bar_width, y_pos, label_text, 
                           ha='center', va='bottom', fontsize=12, rotation=rotation)
                else:
                    # For timeout cases
                    timeout_placeholder = 0.00001  # Very small value for visualization
                    ax.text(pos + (k - 1.5) * bar_width, 0.02, "t-out", 
                           ha='center', va='bottom', fontsize=14, color='red', rotation=90,
                           bbox=dict(facecolor='white', edgecolor='red', alpha=0.7, pad=1, boxstyle='round,pad=0.2'))
                    
                    # Add hatched bar to represent timeout
                    timeout_bar = ax.bar(pos + (k - 1.5) * bar_width, timeout_placeholder, bar_width, 
                                         color='white', edgecolor=colors[system], hatch='xxxx')
    
    # Set log scale for y-axis
    ax.set_yscale('log')
    
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
    ax.set_xticklabels(size_labels, ha='center', fontsize=16)
    
    # Remove y-axis labels for all but the first subplot
    if i > 0:
        ax.tick_params(labelleft=False)
    
    # Set title with higher positioning
    ax.set_title(f"{benchmark}", fontsize=18, pad=20)
    

# Set y-axis label for the first subplot
axs[0].set_ylabel('Runtime in Log-Scale [s]', fontsize=18)

# Set common y-axis limits
axs[0].set_ylim(y_min, y_max)  # From 0.01s to 10,000s

# Add a common legend at the bottom
handles = []
labels = []

# Add system colors to legend
for system in systems:
    handles.append(plt.Rectangle((0,0), 1, 1, color=colors[system], alpha=lan_alpha))
    labels.append(system)

# Add LAN and WAN to legend 
lan_patch = plt.Rectangle((0,0), 1, 1, facecolor='white', edgecolor='gray', alpha=lan_alpha)
wan_patch = plt.Rectangle((0,0), 1, 1, facecolor='white', edgecolor='gray', alpha=wan_alpha, hatch='////')
handles.extend([lan_patch, wan_patch])
labels.extend(['LAN', 'WAN Overhead'])

# Add timeout indicator to legend
timeout_patch = plt.Rectangle((0,0), 1, 1, facecolor='white', edgecolor='red', hatch='xxxx')
handles.append(timeout_patch)
labels.append('Timeout')
    
# Move legend to bottom of figure, closer to the graph
fig.legend(handles, labels, loc='lower center', ncol=7, fontsize=18, bbox_to_anchor=(0.5, 0.05))

# Add a main title for the entire figure
# fig.suptitle('LAN vs WAN Runtime Comparison Across Benchmarks', fontsize=14)

plt.tight_layout()
plt.subplots_adjust(top=0.66, bottom=0.2, wspace=0.05)

plt.savefig("results_exec.pdf", bbox_inches='tight')
