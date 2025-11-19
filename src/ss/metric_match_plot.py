import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from pathlib import Path
import numpy as np

# Configuration: Update this path to where your output files were saved
WORKING_DIR = Path("C:/Users/usuario/Documents/TrebleSDK/v2/configs/results")
DATA_FILE = WORKING_DIR / "acoustic_metrics_data.csv"

def generate_metric_match_plot():
    """Generates a log-frequency plot comparing Predicted vs. Target T20 for the calibrated model."""
    
    if not DATA_FILE.exists():
        print(f"Error: Data file not found at {DATA_FILE}. Please run analyze_ga_results.py first.")
        return

    df = pd.read_csv(DATA_FILE)

    # --- Data Filtering and Aggregation ---
    # 1. Filter for Calibrated GA results and T20 metric
    df_filtered = df[
        (df['Source'] == 'Calibrated_GA') & 
        (df['Metric'] == 't20')
    ].copy()

    if df_filtered.empty:
        print("Error: No 'Calibrated_GA' T20 data found in acoustic_metrics_data.csv.")
        return

    # 2. Group by Band_Hz and calculate the mean for Predicted and Target
    df_summary = df_filtered.groupby('Band_Hz')[['Predicted', 'Target']].mean().reset_index()

    # --- Plotting Setup ---
    fig, ax = plt.subplots(figsize=(10, 6))

    # --- Plotting ---
    # Target (Measured)
    ax.plot(df_summary['Band_Hz'], df_summary['Target'], 
            label='Target (Measured)', 
            color='black', 
            linewidth=2, 
            marker='s')
            
    # Predicted (Calibrated GA)
    ax.plot(df_summary['Band_Hz'], df_summary['Predicted'], 
            label='Predicted (Calibrated GA)', 
            color='red', 
            linewidth=2, 
            linestyle='--', 
            marker='o')

    # --- Axes and Labels ---
    ax.set_xscale('log')
    ax.set_xticks(df_summary['Band_Hz']) # Set ticks to the actual band frequencies
    # Use ScalarFormatter to display log ticks as standard numbers (e.g., 1000 instead of 10^3)
    ax.get_xaxis().set_major_formatter(ScalarFormatter()) 
    
    ax.set_xlabel('Octave Band Center Frequency (Hz)', fontsize=12)
    ax.set_ylabel('Reverberation Time, T20 (s)', fontsize=12)
    ax.set_title('T20 Match: Target vs. Calibrated Prediction (Averaged Across Receivers)', fontsize=14, pad=20)
    
    # Ensure y-axis starts at 0 or a sensible minimum
    y_min = max(0, df_summary[['Target', 'Predicted']].min().min() * 0.9)
    ax.set_ylim(bottom=y_min)

    plt.grid(True, which='both', linestyle=':', linewidth=0.5, alpha=0.7)
    plt.legend(loc='best')
    fig.tight_layout()
    plt.savefig(WORKING_DIR / "t20_metric_match_plot.png", dpi=300)
    print(f"\nSUCCESS: Metric match plot saved to {WORKING_DIR / 't20_metric_match_plot.png'}")
    plt.show()

if __name__ == "__main__":
    generate_metric_match_plot()