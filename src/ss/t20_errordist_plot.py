import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

# Configuration: Update this path to where your output files were saved
WORKING_DIR = Path("C:/Users/usuario/Documents/TrebleSDK/v2/configs/results")
DATA_FILE = WORKING_DIR / "acoustic_metrics_data.csv"

def generate_error_distribution_plot():
    """Generates a box-and-whisker plot for the T20 error distribution by band."""
    
    if not DATA_FILE.exists():
        print(f"Error: Data file not found at {DATA_FILE}. Please run analyze_ga_results.py first.")
        return

    df = pd.read_csv(DATA_FILE)

    # --- Data Filtering ---
    # Filter for the T20 metric (Error is already calculated in the CSV)
    df_filtered = df[
        (df['Source'] == 'Calibrated_GA') & 
        (df['Metric'] == 't20')
    ].copy()
    
    if df_filtered.empty:
        print("Error: No 'Calibrated_GA' T20 data found in acoustic_metrics_data.csv.")
        return

    # Prepare data for plotting: list of error arrays, ordered by Band_Hz
    # Ensure correct band order
    band_order = sorted(df_filtered['Band_Hz'].unique())
    plot_data = [df_filtered[df_filtered['Band_Hz'] == band]['Error'] for band in band_order]

    # --- Plotting Setup ---
    fig, ax = plt.subplots(figsize=(10, 6))

    # --- Plotting Box Plot ---
    bp = ax.boxplot(plot_data, 
                    labels=[f"{b} Hz" for b in band_order],
                    patch_artist=True,  # Allows color filling
                    medianprops={'color': 'red', 'linewidth': 2}, # Highlight the median
                    boxprops={'facecolor': 'lightblue', 'alpha': 0.7}) 

    # --- Annotations and Labels ---
    ax.axhline(0, color='gray', linestyle='--', linewidth=1) # Line for zero error
    
    ax.set_xlabel('Octave Band Center Frequency', fontsize=12)
    ax.set_ylabel('Error ($\Delta T20$) = Predicted - Target (s)', fontsize=12)
    ax.set_title('Distribution of T20 Error Across Receiver Positions (Calibrated GA)', fontsize=14, pad=20)
    
    # Add a minimal buffer to the y-limits
    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    ax.set_ylim(ax.get_ylim()[0] - y_range * 0.05, ax.get_ylim()[1] + y_range * 0.05)

    plt.grid(True, axis='y', linestyle=':', linewidth=0.5, alpha=0.7)
    fig.tight_layout()
    plt.savefig(WORKING_DIR / "t20_error_distribution_plot.png", dpi=300)
    print(f"\nSUCCESS: Error distribution plot saved to {WORKING_DIR / 't20_error_distribution_plot.png'}")
    plt.show()

if __name__ == "__main__":
    generate_error_distribution_plot()