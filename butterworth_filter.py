"""
Butterworth Low-Pass Filter for Load Cell Data
===============================================
Precise frequency-domain filtering with flat passband.
Best for: Removing specific frequency noise, drift removal

Author: Load Cell Filtering Toolkit
License: MIT
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, sosfreqz
import warnings

# ============================================
# CONFIGURATION - Edit this section
# ============================================

# Your CSV file name
CSV_FILENAME = 'local_Loadcell_data.csv'

# Column names in your CSV
TIMESTAMP_COLUMN = 'Timestamp (ms)'
WEIGHT_COLUMN = 'Weight (g)'

# Filter parameters
CUTOFF_HZ = 2.0  # Cutoff frequency in Hz
                 # For drift removal: 0.001-0.01 Hz
                 # For noise smoothing: 1-3 Hz
ORDER = 4        # Filter order (2-8, typically 4)
                 # Higher = sharper cutoff but more artifacts

# ============================================
# Main Script
# ============================================

def apply_butterworth_filter(filename, timestamp_col, weight_col, cutoff_hz, order):
    """Apply Butterworth low-pass filter to load cell data."""
    
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
    nyq = fs / 2.0
    
    print(f"\n=== Sampling Rate ===")
    print(f"Estimated: {fs:.2f} Hz (Nyquist: {nyq:.2f} Hz)")
    print(f"Timestamp spacing: {dt_mean*1000:.2f}±{dt_std*1000:.2f}ms")
    
    if dt_std / dt_mean > 0.1:
        warnings.warn(f"Non-uniform sampling (CV={dt_std/dt_mean:.1%})")
    
    # Validate cutoff
    if cutoff_hz >= nyq:
        warnings.warn(f"Cutoff {cutoff_hz} Hz >= Nyquist {nyq:.2f} Hz. Reducing.")
        cutoff_hz = 0.45 * nyq
    
    if cutoff_hz < 0.001:
        warnings.warn(f"Cutoff {cutoff_hz} Hz is very low.")
    
    # Design filter
    Wn = cutoff_hz / nyq
    sos = butter(order, Wn, btype='low', analog=False, output='sos')
    
    print(f"\n=== Filter Parameters ===")
    print(f"Cutoff frequency: {cutoff_hz:.3f} Hz")
    print(f"Order: {order}")
    print(f"Filter type: Butterworth (zero-phase)")
    
    # Apply filter
    try:
        signal_filt = sosfiltfilt(sos, signal_raw)
    except ValueError as e:
        print(f"✗ Filter error: {e}")
        print(f"  Try reducing ORDER or increasing signal length")
        return None
    
    # Add to dataframe
    df['Weight_filtered_g'] = signal_filt
    
    # Statistics
    noise_removed = signal_raw - signal_filt
    
    print(f"\n=== Results ===")
    print(f"Original  - min: {signal_raw.min():.2f}g, max: {signal_raw.max():.2f}g, std: {signal_raw.std():.2f}g")
    print(f"Filtered  - min: {signal_filt.min():.2f}g, max: {signal_filt.max():.2f}g, std: {signal_filt.std():.2f}g")
    print(f"Noise reduction: {100*(1-signal_filt.std()/signal_raw.std()):.1f}%")
    print(f"RMS noise removed: {np.sqrt(np.mean(noise_removed**2)):.3f}g")
    
    # Frequency response
    w, h = sosfreqz(sos, worN=2048, fs=fs)
    cutoff_3db = w[np.argmax(np.abs(h) < 1/np.sqrt(2))] if len(w[np.abs(h) < 1/np.sqrt(2)]) > 0 else cutoff_hz
    print(f"Actual -3dB point: {cutoff_3db:.3f} Hz")
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    
    # Time domain
    ax1.plot(time_s, signal_raw, label='Original', alpha=0.6, linewidth=1)
    ax1.plot(time_s, signal_filt, label=f'Butterworth ({cutoff_hz} Hz, order={order})', 
             linewidth=2, color='red')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Weight (g)')
    ax1.set_title('Load Cell Signal: Original vs Butterworth Filtered')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Frequency response
    ax2.semilogx(w, 20 * np.log10(np.abs(h)), linewidth=2)
    ax2.axvline(cutoff_hz, color='r', linestyle='--', label=f'Cutoff: {cutoff_hz} Hz')
    ax2.axhline(-3, color='gray', linestyle=':', label='-3 dB')
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Magnitude (dB)')
    ax2.set_title('Filter Frequency Response')
    ax2.grid(True, which='both', alpha=0.3)
    ax2.legend()
    ax2.set_xlim([0.01 if cutoff_hz < 0.1 else 0.1, nyq])
    
    plt.tight_layout()
    plt.savefig('butterworth_filtered.png', dpi=150)
    print(f"\n✓ Plot saved as 'butterworth_filtered.png'")
    plt.show()
    
    # Save filtered data
    output_file = filename.replace('.csv', '_butterworth_filtered.csv')
    df.to_csv(output_file, index=False)
    print(f"✓ Filtered data saved as '{output_file}'")
    
    return df


if __name__ == "__main__":
    apply_butterworth_filter(CSV_FILENAME, TIMESTAMP_COLUMN, WEIGHT_COLUMN, 
                            CUTOFF_HZ, ORDER)
