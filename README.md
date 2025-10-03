# Load Cell Signal Filtering Toolkit

A comprehensive Python toolkit for filtering and analyzing load cell data with three powerful filtering techniques.

## 📊 Overview

This toolkit provides ready-to-use scripts for cleaning noisy load cell measurements using:
- **Moving Average Filter** - Simple, robust smoothing
- **Butterworth Filter** - Precise frequency-domain filtering
- **Savitzky-Golay Filter** - Preserves peaks and features

## 🚀 Quick Start

### Requirements
```bash
pip install numpy pandas matplotlib scipy
```

### Data Format
Your CSV file should have these columns:
- `Timestamp (ms)` - Time in milliseconds
- `Weight (g)` - Load cell readings in grams

Example:
```
Timestamp (ms),Weight (g)
0,245.3
10,246.1
20,244.8
```

### Usage
1. Place your CSV file in the same directory as the scripts
2. Update the filename in the script:
   ```python
   df = pd.read_csv('your_data.csv')
   ```
3. Run the desired filter:
   ```bash
   python moving_average_filter.py
   python butterworth_filter.py
   python savitzky_golay_filter.py
   ```

## 📈 Filter Comparison

| Filter | Best For | Pros | Cons |
|--------|----------|------|------|
| **Moving Average** | Quick smoothing, real-time | Simple, fast, no ringing | Blunts sharp features |
| **Butterworth** | Frequency-specific noise | Precise cutoff, flat passband | Can overshoot |
| **Savitzky-Golay** | Preserving peaks/valleys | Keeps features intact | Computationally intensive |

## 🔧 Tuning Parameters

### Moving Average
- `WINDOW_SECONDS`: Averaging window (5-20s typical)
  - Smaller = less smoothing, preserves details
  - Larger = more smoothing, removes noise

### Butterworth
- `CUTOFF_HZ`: Frequency cutoff
  - 0.001-0.01 Hz: Remove drift only
  - 1-3 Hz: Smooth noise, keep signal shape
- `ORDER`: Filter steepness (2-8, typically 4)

### Savitzky-Golay
- `WINDOW_SECONDS`: Fitting window (5-20s typical)
- `POLYORDER`: Polynomial order (2-5, typically 3)
  - Higher = better peak preservation

## 📁 Output

Each script generates:
- Filtered signal added to DataFrame as `Weight_filtered_g`
- Analysis plot saved as PNG
- Statistical summary printed to console

## 💡 Tips

1. **Start with visualization**: Run `visualize_data.py` first
2. **Compare filters**: Try all three on your data
3. **Iterate parameters**: Adjust based on your noise characteristics
4. **Check sampling rate**: Scripts auto-detect and warn if issues

## 🐛 Troubleshooting

**"Signal too short" error**
- Need at least 50 data points

**"Non-uniform sampling" warning**
- Your timestamps aren't evenly spaced
- Filters still work but may be less accurate

**Filter artifacts at edges**
- Normal behavior, especially for Butterworth
- Data edges are padded/extrapolated

## 📝 License

MIT License - Free to use and modify

## 🤝 Contributing

Issues and pull requests welcome!

## ✨ Example Results

After filtering, you'll see:
- Original vs filtered signal plots
- Noise reduction statistics
- Filter frequency response (Butterworth)
- Peak preservation analysis (Savitzky-Golay)
