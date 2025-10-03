"""
Load Cell Data Visualization
=============================
Quick visualization of your raw load cell data.
Run this FIRST to understand your signal before filtering.

Author: Codeharshw
License: MIT
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ============================================
# CONFIGURATION - Edit this section
# ============================================

# Your CSV file name (must be in same directory)
CSV_FILENAME = 'local_Loadcell_data.csv'

# Column names in your CSV
TIMESTAMP_COLUMN = 'Timestamp (ms)'
WEIGHT_COLUMN = 'Weight (g)'

# ============================================
# Main Script - No need to edit below
# ============================================

def load_and_visualize(filename, timestamp_col, weight_col):
    """Load data and create comprehensive visualization."""
    
    try:
        # Load data
        df = pd.read_csv(filename)
        print(f"✓ Successfully loaded {filename}")
        print(f"  Rows: {len(df)}")
        print(f"  Columns: {list(df.columns)}")
        
    except FileNotFoundError:
        print(f"✗ Error: '{filename}' not found.")
        print(f"  Make sure the file is in the same directory as this script.")
        return
    except Exception as e:
        print(f"✗ Error loading file: {e}")
        return
    
    # Validate columns
    if timestamp_col not in df.columns or weight_col not in df.columns:
        print(f"✗ Error: Required columns not found")
        print(f"  Looking for: '{timestamp_col}' and '{weight_col}'")
        print(f"  Available columns: {list(df.columns)}")
        return
    
    # Prepare data
    time_s = df[timestamp_col].astype(float) / 1000.0
    weight = df[weight_col].astype(float).values
    
    # Calculate statistics
    duration = time_s.max() - time_s.min()
    dt_mean = np.mean(np.diff(time_s))
    fs = 1.0 / dt_mean if dt_mean > 0 else 0
    
    # Print summary
    print(f"\n=== Data Summary ===")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Data points: {len(df)}")
    print(f"Sampling rate: {fs:.2f} Hz (approx)")
    print(f"Weight range: {weight.min():.2f}g to {weight.max():.2f}g")
    print(f"Weight mean: {weight.mean():.2f}g")
    print(f"Weight std: {weight.std():.2f}g")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f'Load Cell Data Analysis: {filename}', 
                 fontsize=14, fontweight='bold')
    
    # 1. Full time series
    axes[0, 0].plot(time_s, weight, linewidth=1, alpha=0.8)
    axes[0, 0].set_xlabel('Time (seconds)')
    axes[0, 0].set_ylabel('Weight (g)')
    axes[0, 0].set_title('Full Signal')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Zoomed view (middle 25% of data)
    start = len(weight) // 4
    end = start + len(weight) // 4
    axes[0, 1].plot(time_s[start:end], weight[start:end], linewidth=1)
    axes[0, 1].set_xlabel('Time (seconds)')
    axes[0, 1].set_ylabel('Weight (g)')
    axes[0, 1].set_title('Zoomed View (middle section)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Histogram
    axes[1, 0].hist(weight, bins=50, edgecolor='black', alpha=0.7)
    axes[1, 0].axvline(weight.mean(), color='red', 
                       linestyle='--', label=f'Mean: {weight.mean():.2f}g')
    axes[1, 0].set_xlabel('Weight (g)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Distribution of Weight Values')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Noise characteristics (differences between consecutive points)
    noise = np.diff(weight)
    axes[1, 1].plot(time_s[1:], noise, linewidth=0.5, alpha=0.6)
    axes[1, 1].axhline(0, color='red', linestyle='--', linewidth=1)
    axes[1, 1].set_xlabel('Time (seconds)')
    axes[1, 1].set_ylabel('Weight Change (g)')
    axes[1, 1].set_title(f'Point-to-Point Noise (std={np.std(noise):.3f}g)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('loadcell_data_visualization.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved as 'loadcell_data_visualization.png'")
    plt.show()
    
    # Filtering recommendations
    print(f"\n=== Filtering Recommendations ===")
    noise_level = np.std(noise)
    if noise_level < 0.1:
        print("Low noise signal - Light filtering recommended")
        print("  Try: Moving Average with 5s window")
    elif noise_level < 1.0:
        print("Moderate noise - Medium filtering recommended")
        print("  Try: Savitzky-Golay with 10s window, order 3")
    else:
        print("High noise signal - Aggressive filtering recommended")
        print("  Try: Moving Average with 20s window")
        print("   or: Butterworth filter with 2 Hz cutoff")


if __name__ == "__main__":
    load_and_visualize(CSV_FILENAME, TIMESTAMP_COLUMN, WEIGHT_COLUMN)
