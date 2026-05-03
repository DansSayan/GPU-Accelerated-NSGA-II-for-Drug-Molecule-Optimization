import pandas as pd
import matplotlib.pyplot as plt
import os

# --------------------------------
# MAIN VISUALIZATION FUNCTION
# --------------------------------
def create_plots(pareto_csv="pareto_front.csv", original_csv="processed_molecules.csv", output_dir="."):
    """
    Create visualization plots comparing original and Pareto populations
    
    Parameters:
    -----------
    pareto_csv : str
        Path to Pareto front CSV file
    original_csv : str
        Path to original population CSV file
    output_dir : str
        Directory to save plot images
        
    Returns:
    --------
    plot_paths : list
        List of paths to generated plot images
    """
    # Load data
    pareto = pd.read_csv(pareto_csv)
    original = pd.read_csv(original_csv)
    
    print(f"Pareto shape: {pareto.shape}")
    print(f"Original shape: {original.shape}")
    
    plot_paths = []
    
    # --------------------------------
    # 1. MOL WT vs LOGP
    # --------------------------------
    plt.figure(figsize=(10, 6))
    plt.scatter(original['mol_wt'], original['logp'], alpha=0.3, s=20, label="Original")
    plt.scatter(pareto['mol_wt'], pareto['logp'], color='red', s=50, label="Pareto Front")
    plt.xlabel("Molecular Weight")
    plt.ylabel("LogP")
    plt.title("MolWt vs LogP (Optimization)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plot1_path = os.path.join(output_dir, "mw_vs_logp.png")
    plt.savefig(plot1_path, dpi=100, bbox_inches='tight')
    plt.close()
    plot_paths.append(plot1_path)
    
    # --------------------------------
    # 2. LOGP vs TPSA
    # --------------------------------
    plt.figure(figsize=(10, 6))
    plt.scatter(original['logp'], original['tpsa'], alpha=0.3, s=20, label="Original")
    plt.scatter(pareto['logp'], pareto['tpsa'], color='red', s=50, label="Pareto Front")
    plt.xlabel("LogP")
    plt.ylabel("TPSA")
    plt.title("LogP vs TPSA (Optimization)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plot2_path = os.path.join(output_dir, "logp_vs_tpsa.png")
    plt.savefig(plot2_path, dpi=100, bbox_inches='tight')
    plt.close()
    plot_paths.append(plot2_path)
    
    # --------------------------------
    # 3. MOL WT vs TPSA
    # --------------------------------
    plt.figure(figsize=(10, 6))
    plt.scatter(original['mol_wt'], original['tpsa'], alpha=0.3, s=20, label="Original")
    plt.scatter(pareto['mol_wt'], pareto['tpsa'], color='red', s=50, label="Pareto Front")
    plt.xlabel("Molecular Weight")
    plt.ylabel("TPSA")
    plt.title("MolWt vs TPSA (Optimization)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plot3_path = os.path.join(output_dir, "mw_vs_tpsa.png")
    plt.savefig(plot3_path, dpi=100, bbox_inches='tight')
    plt.close()
    plot_paths.append(plot3_path)
    
    # --------------------------------
    # 4. DISTRIBUTION COMPARISON
    # --------------------------------
    features = ['mol_wt', 'logp', 'tpsa']
    
    for col in features:
        plt.figure(figsize=(10, 6))
        
        plt.hist(original[col], bins=30, alpha=0.5, label="Original")
        plt.hist(pareto[col], bins=30, alpha=0.7, label="Pareto")
        
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.title(f"{col} Distribution")
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        
        plot_path = os.path.join(output_dir, f"{col}_distribution.png")
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')
        plt.close()
        plot_paths.append(plot_path)
    
    print(f"All plots saved successfully. Total plots: {len(plot_paths)}")
    return plot_paths


# --------------------------------
# STANDALONE MODE (for CLI execution)
# --------------------------------
if __name__ == "__main__":
    create_plots()