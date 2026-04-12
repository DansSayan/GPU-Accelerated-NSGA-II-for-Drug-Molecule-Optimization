import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------
# LOAD DATA
# --------------------------------
pareto = pd.read_csv("pareto_front.csv")
original = pd.read_csv("processed_molecules.csv")

print("Pareto shape:", pareto.shape)
print("Original shape:", original.shape)

# --------------------------------
# 1. MOL WT vs LOGP
# --------------------------------
plt.figure()
plt.scatter(original['mol_wt'], original['logp'], alpha=0.3, label="Original")
plt.scatter(pareto['mol_wt'], pareto['logp'], label="Pareto Front")

plt.xlabel("Molecular Weight")
plt.ylabel("LogP")
plt.title("MolWt vs LogP (Optimization)")
plt.legend()

plt.savefig("mw_vs_logp.png")
plt.close()

# --------------------------------
# 2. LOGP vs TPSA
# --------------------------------
plt.figure()
plt.scatter(original['logp'], original['tpsa'], alpha=0.3, label="Original")
plt.scatter(pareto['logp'], pareto['tpsa'], label="Pareto Front")

plt.xlabel("LogP")
plt.ylabel("TPSA")
plt.title("LogP vs TPSA (Optimization)")
plt.legend()

plt.savefig("logp_vs_tpsa.png")
plt.close()

# --------------------------------
# 3. MOL WT vs TPSA
# --------------------------------
plt.figure()
plt.scatter(original['mol_wt'], original['tpsa'], alpha=0.3, label="Original")
plt.scatter(pareto['mol_wt'], pareto['tpsa'], label="Pareto Front")

plt.xlabel("Molecular Weight")
plt.ylabel("TPSA")
plt.title("MolWt vs TPSA")
plt.legend()

plt.savefig("mw_vs_tpsa.png")
plt.close()

# --------------------------------
# 4. DISTRIBUTION COMPARISON
# --------------------------------
features = ['mol_wt', 'logp', 'tpsa']

for col in features:
    plt.figure()
    
    plt.hist(original[col], bins=30, alpha=0.5, label="Original")
    plt.hist(pareto[col], bins=30, alpha=0.7, label="Pareto")
    
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.title(f"{col} Distribution")
    plt.legend()
    
    plt.savefig(f"{col}_distribution.png")
    plt.close()

print("All plots saved successfully.")