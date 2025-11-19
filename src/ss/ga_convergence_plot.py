import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Configuration: Update this path to where your output files were saved
WORKING_DIR = Path("C:/Users/usuario/Documents/TrebleSDK/v2/configs/results")
DATA_FILE = WORKING_DIR / "ga_summary_data.csv"

def generate_convergence_plot():
    """Generates a dual-axis plot of Best Loss and Mean Loss vs. Generation."""
    
    if not DATA_FILE.exists():
        print(f"Error: Data file not found at {DATA_FILE}. Please run analyze_ga_results.py first.")
        return

    df = pd.read_csv(DATA_FILE)

    if df.empty:
        print("Error: ga_summary_data.csv is empty.")
        return
        
    # --- Plotting Setup ---
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Custom colors and style for academic paper look
    COLOR_BEST = '#1f77b4'  # Blue
    COLOR_MEAN = '#ff7f0e'  # Orange
    
    # --- Left Y-Axis (Best Loss) ---
    ax1.plot(df['Generation'], df['Best_Loss_H'], 
             color=COLOR_BEST, 
             linestyle='-', 
             marker='o', 
             markersize=5, 
             label='Best Individual Loss ($J_{best}$)')
    
    ax1.set_xlabel('Generation', fontsize=12)
    ax1.set_ylabel('Best Individual Loss (Huber-Loss $J$)', color=COLOR_BEST, fontsize=12)
    ax1.tick_params(axis='y', labelcolor=COLOR_BEST)
    
    # Ensure Generation axis has only integers (crucial for GA plots)
    ax1.set_xticks(df['Generation'].unique())
    
    # --- Right Y-Axis (Mean Loss) ---
    ax2 = ax1.twinx()  # Instantiate a second axes that shares the same x-axis
    ax2.plot(df['Generation'], df['Mean_Loss_H'], 
             color=COLOR_MEAN, 
             linestyle='--', 
             marker='^', 
             markersize=5, 
             label='Population Mean Loss ($\overline{J}$)')
             
    ax2.set_ylabel('Population Mean Loss (Huber-Loss $\overline{J}$)', color=COLOR_MEAN, fontsize=12)
    ax2.tick_params(axis='y', labelcolor=COLOR_MEAN)

    # --- Final Touches ---
    plt.title('Genetic Algorithm Calibration Convergence', fontsize=14, pad=20)
    fig.tight_layout() # Adjust layout to prevent labels from overlapping
    
    # Add a combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.savefig(WORKING_DIR / "convergence_plot.png", dpi=300)
    print(f"\nSUCCESS: Convergence plot saved to {WORKING_DIR / 'convergence_plot.png'}")
    plt.show()

if __name__ == "__main__":
    generate_convergence_plot()