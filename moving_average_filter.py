"""
Moving Average Filter for Load Cell Data
=========================================
Simple and robust smoothing filter.
Best for: Quick noise reduction, real-time applications

Author: Codeharshw
License: MIT
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

# ============================================
# CONFIGURATION - Edit this section
# ============================================

# Your CSV file name
CSV_FILENAME = 'local_Loadcell_data.csv'

# Column names in your CSV
TIMESTAMP_COLUMN = 'Timestamp (ms)'
WEIGHT_COLUMN = 'Weight (g)'

# Filtering parameter
WINDOW_SECONDS = 10  # Averaging window in seconds
                     # Larger = more smoothing
                     # Try: 5 (light), 10 (medium), 20 (heavy)

# ============================================
# Main Script
# ============================================

def apply_moving_average(filename, timestamp_col, weight_col, window_sec):
    """Apply moving average filter to load cell data."""
    
    # Load data
    try:
        df = pd.read_csv(filename)
        print(f"✓ Loaded {filename} ({len(df)} rows)")
    except FileNotFoundError:
        print(f"✗ Error: '{filename}' not found.")
        return None
    except Exception as e:
        print(f"✗ Error: {e}")
        return None
    
    # Validate
    if timestamp_col not in df.columns or weight_col not in df.columns:
        print(f"✗ Error: Required columns not found")
        print(f"  Available: {list(df.columns)}")
        return None
    
    # Prepare data
    time_s = df[timestamp_col].astype(float) / 1000.0
    signal_raw = df[weight_col].astype(float).values
    
    if len(signal_raw) < 50:
        print(f"✗ Error: Signal too short ({len(signal_raw)} samples)")
        return None
    
    # Estimate sampling rate
    dt_mean = np.mean(np.diff(time_s))
    dt_std = np.std(np.diff(time_s))
    fs = 1.0 / dt_mean
    
    print(f"\n=== Sampling Rate ===")
    print(f"Estimated: {fs:.2f} Hz")
    print(f"Timestamp spacing: {dt_mean*1000:.2f}±{dt_std*1000:.2f}ms")
    
    if dt_std / dt_mean > 0.1:
        warnings.warn(f"Non-uniform sampling (CV={dt_std/dt_mean:.1%})")
    
    # Calculate window in samples
    window_samples = int(window_sec * fs)
    if window_samples < 1:
        window_samples = 1
        warnings.warn(f"Window too small, using 1 sample")
    
    print(f"\n=== Filter Parameters ===")
    print(f"Window: {window_sec}s = {window_samples} samples")
    print(f"Filter type: Moving Average")
    
    # Apply filter
    signal_filt = pd.Series(signal_raw).rolling(
        window=window_samples,
        center=True,
        min_periods=1
    ).mean().values
    
    # Add to dataframe
    df['Weight_filtered_g'] = signal_filt
    
    # Statistics
    noise_removed = signal_raw - signal_filt
    
    print(f"\n=== Results ===")
    print(f"Original  - min: {signal_raw.min():.2f}g, max: {signal_raw.max():.2f}g, std: {signal_raw.std():.2f}g")
    print(f"Filtered  - min: {signal_filt.min():.2f}g, max: {signal_filt.max():.2f}g, std: {signal_filt.std():.2f}g")
    print(f"Noise reduction: {100*(1-signal_filt.std()/signal_raw.std()):.1f}%")
    print(f"RMS noise removed: {np.sqrt(np.nanmean(noise_removed**2)):.3f}g")
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    
    # Full view
    ax1.plot(time_s, signal_raw, label='Original', alpha=0.6, linewidth=1)
    ax1.plot(time_s, signal_filt, label=f'Moving Average ({window_sec}s)', 
             linewidth=2, color='green')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Weight (g)')
    ax1.set_title('Load Cell Signal: Original vs Filtered')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Zoomed view
    zoom_start = len(signal_raw) // 3
    zoom_end = zoom_start + min(1000, len(signal_raw) // 4)
    ax2.plot(time_s[zoom_start:zoom_end], signal_raw[zoom_start:zoom_end],
             label='Original', alpha=0.6, linewidth=1)
    ax2.plot(time_s[zoom_start:zoom_end], signal_filt[zoom_start:zoom_end],
             label=f'Moving Average ({window_sec}s)', linewidth=2, color='green')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Weight (g)')
    ax2.set_title('Zoomed View')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('moving_average_filtered.png', dpi=150)
    print(f"\n✓ Plot saved as 'moving_average_filtered.png'")
    plt.show()
    
    # Save filtered data
    output_file = filename.replace('.csv', '_moving_average_filtered.csv')
    df.to_csv(output_file, index=False)
    print(f"✓ Filtered data saved as '{output_file}'")
    
    return df


if __name__ == "__main__":
    apply_moving_average(CSV_FILENAME, TIMESTAMP_COLUMN, WEIGHT_COLUMN, WINDOW_SECONDS)
