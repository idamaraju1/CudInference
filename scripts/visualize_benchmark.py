#!/usr/bin/env python3
"""
Visualize benchmark results from OnnxRunner CPU vs GPU comparison.

Usage:
    python3 visualize_benchmark.py benchmark_results.json [--output results.png] [--interactive]
"""

import json
import sys
import argparse
from pathlib import Path

# Optional imports for visualization
try:
    import matplotlib.pyplot as plt
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not found. Install with: pip3 install matplotlib numpy")
    print("Table output will still work.\n")

def load_benchmark_data(json_file):
    """Load benchmark data from JSON file."""
    with open(json_file, 'r') as f:
        return json.load(f)

def create_comparison_chart(data, output_file=None, interactive=False):
    """Create a bar chart comparing CPU vs GPU execution times."""
    operations = data['operations']

    # Extract data
    op_names = [f"{op['op_type']}\n({op['node_name']})" for op in operations]
    cpu_times = [op['cpu_time_ms'] for op in operations]
    gpu_times = [op['gpu_time_ms'] for op in operations]
    speedups = [op['speedup'] for op in operations]

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('OnnxRunner: CPU vs GPU Performance Benchmark',
                 fontsize=16, fontweight='bold')

    # 1. Side-by-side bar chart
    ax1 = axes[0, 0]
    x = np.arange(len(op_names))
    width = 0.35

    bars1 = ax1.bar(x - width/2, cpu_times, width, label='CPU', color='#FFA500', alpha=0.8)
    bars2 = ax1.bar(x + width/2, gpu_times, width, label='GPU', color='#00CED1', alpha=0.8)

    ax1.set_xlabel('Operation', fontweight='bold')
    ax1.set_ylabel('Execution Time (ms)', fontweight='bold')
    ax1.set_title('Execution Time: CPU vs GPU')
    ax1.set_xticks(x)
    ax1.set_xticklabels(op_names, rotation=45, ha='right', fontsize=8)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=7)

    # 2. Speedup chart
    ax2 = axes[0, 1]
    colors = ['#00FF00' if s > 1.0 else '#FF6B6B' for s in speedups]
    bars = ax2.barh(op_names, speedups, color=colors, alpha=0.7)
    ax2.axvline(x=1.0, color='black', linestyle='--', linewidth=2, label='No speedup')
    ax2.set_xlabel('Speedup Factor (CPU time / GPU time)', fontweight='bold')
    ax2.set_title('GPU Speedup per Operation')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='x')

    # Add value labels
    for i, (bar, speedup) in enumerate(zip(bars, speedups)):
        width = bar.get_width()
        label = f'{speedup:.2f}x' if speedup > 1 else f'{1/speedup:.2f}x slower'
        ax2.text(width, bar.get_y() + bar.get_height()/2., f'  {label}',
                ha='left', va='center', fontsize=9, fontweight='bold')

    # 3. Stacked time contribution
    ax3 = axes[1, 0]
    cpu_percentages = [t / data['total_cpu_time_ms'] * 100 for t in cpu_times]
    gpu_percentages = [t / data['total_gpu_time_ms'] * 100 for t in gpu_times]

    x_pos = [0, 1]
    bottom_cpu = 0
    bottom_gpu = 0

    colors_palette = plt.cm.Set3(np.linspace(0, 1, len(operations)))

    for i, op in enumerate(operations):
        ax3.bar(0, cpu_percentages[i], bottom=bottom_cpu,
               color=colors_palette[i], alpha=0.8, label=op['op_type'] if i < 10 else '')
        ax3.bar(1, gpu_percentages[i], bottom=bottom_gpu,
               color=colors_palette[i], alpha=0.8)
        bottom_cpu += cpu_percentages[i]
        bottom_gpu += gpu_percentages[i]

    ax3.set_xticks([0, 1])
    ax3.set_xticklabels(['CPU', 'GPU'])
    ax3.set_ylabel('Percentage of Total Time (%)', fontweight='bold')
    ax3.set_title('Time Distribution by Operation')
    if len(operations) <= 10:
        ax3.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=8)
    ax3.set_ylim(0, 100)

    # 4. Total comparison with speedup
    ax4 = axes[1, 1]
    total_cpu = data['total_cpu_time_ms']
    total_gpu = data['total_gpu_time_ms']
    overall_speedup = data['overall_speedup']

    bars = ax4.bar(['CPU', 'GPU'], [total_cpu, total_gpu],
                   color=['#FFA500', '#00CED1'], alpha=0.8, width=0.5)
    ax4.set_ylabel('Total Execution Time (ms)', fontweight='bold')
    ax4.set_title(f'Total Execution Time\n(Overall Speedup: {overall_speedup:.2f}x)',
                 fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f} ms', ha='center', va='bottom',
                fontsize=12, fontweight='bold')

    # Add speedup annotation
    ax4.annotate(f'{overall_speedup:.2f}x faster' if overall_speedup > 1 else f'{1/overall_speedup:.2f}x slower',
                xy=(0.5, max(total_cpu, total_gpu) * 0.5),
                xytext=(0.5, max(total_cpu, total_gpu) * 0.7),
                ha='center', fontsize=14, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', lw=2, color='red'))

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {output_file}")

    if interactive:
        plt.show()

    return fig

def create_detailed_table(data):
    """Print a detailed table of benchmark results."""
    operations = data['operations']

    print("\n" + "="*80)
    print("DETAILED BENCHMARK RESULTS")
    print("="*80)
    print(f"{'Operation':<25} {'CPU (ms)':<12} {'GPU (ms)':<12} {'Speedup':<15}")
    print("-"*80)

    for op in operations:
        speedup_str = f"{op['speedup']:.2f}x" if op['speedup'] > 1 else f"{1/op['speedup']:.2f}x slower"
        print(f"{op['op_type']:<25} {op['cpu_time_ms']:<12.3f} {op['gpu_time_ms']:<12.3f} {speedup_str:<15}")

    print("-"*80)
    print(f"{'TOTAL':<25} {data['total_cpu_time_ms']:<12.3f} {data['total_gpu_time_ms']:<12.3f} "
          f"{data['overall_speedup']:.2f}x")
    print("="*80 + "\n")

def main():
    parser = argparse.ArgumentParser(
        description='Visualize OnnxRunner benchmark results',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('json_file', type=str, help='Input JSON file with benchmark results')
    parser.add_argument('--output', '-o', type=str, help='Output image file (e.g., results.png)')
    parser.add_argument('--interactive', '-i', action='store_true',
                       help='Show interactive plot window')
    parser.add_argument('--table', '-t', action='store_true',
                       help='Print detailed table to console')

    args = parser.parse_args()

    # Check if file exists
    if not Path(args.json_file).exists():
        print(f"Error: File '{args.json_file}' not found", file=sys.stderr)
        return 1

    # Load data
    try:
        data = load_benchmark_data(args.json_file)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON file: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error loading file: {e}", file=sys.stderr)
        return 1

    # Print table if requested
    if args.table:
        create_detailed_table(data)

    # Create visualization if output file specified or interactive mode
    if args.output or args.interactive:
        if not MATPLOTLIB_AVAILABLE:
            print("Error: matplotlib is required for visualization", file=sys.stderr)
            print("Install with: pip3 install matplotlib numpy", file=sys.stderr)
            return 1
        try:
            create_comparison_chart(data, args.output, args.interactive)
        except Exception as e:
            print(f"Error creating visualization: {e}", file=sys.stderr)
            return 1

    if not args.output and not args.interactive and not args.table:
        print("No output specified. Use --output, --interactive, or --table")
        print("Example: python3 visualize_benchmark.py results.json --output chart.png --table")
        return 1

    return 0

if __name__ == '__main__':
    sys.exit(main())
