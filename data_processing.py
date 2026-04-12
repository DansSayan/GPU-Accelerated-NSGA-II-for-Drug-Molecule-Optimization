import pandas as pd
import numpy as np
import gzip
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem import Descriptors

sample_n  = int(input("Number of molecules to sample : ") or 10000)
seed      = int(input("Random seed                   : ") or 42)
mw_max    = float(input("Max molecular weight filter   : ") or 1000)
logp_min  = float(input("Min LogP filter               : ") or -5)
logp_max  = float(input("Max LogP filter               : ") or 10)

# --------------------------------
# STEP 1: LOAD DATA
# --------------------------------
FILE_PATH = "chembl_33_chemreps.txt.gz"

print("Loading dataset...")

with gzip.open(FILE_PATH, "rt") as f:
    df = pd.read_csv(f, sep="\t")

print("Columns:", df.columns)

# Keep only SMILES
df = df[['canonical_smiles']].dropna()

# Sample (optional)
df = df.sample(n=sample_n, random_state=seed)

print("Loaded:", df.shape)


# --------------------------------
# STEP 2: FILTER VALID MOLECULES
# --------------------------------
def is_valid(smiles):
    return Chem.MolFromSmiles(smiles) is not None

df['valid'] = df['canonical_smiles'].apply(is_valid)
df = df[df['valid'] == True].drop(columns=['valid'])

print("Valid molecules:", len(df))


# --------------------------------
# STEP 3: FEATURE EXTRACTION
# --------------------------------
def compute_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    
    return {
        "mol_wt": Descriptors.MolWt(mol),
        "logp": Descriptors.MolLogP(mol),
        "h_donors": Descriptors.NumHDonors(mol),
        "h_acceptors": Descriptors.NumHAcceptors(mol),
        "tpsa": Descriptors.TPSA(mol)
    }

features = []
valid_smiles = []

for sm in df['canonical_smiles']:
    try:
        feat = compute_features(sm)
        features.append(feat)
        valid_smiles.append(sm)
    except:
        continue

features_df = pd.DataFrame(features)
features_df['smiles'] = valid_smiles

print("Feature shape:", features_df.shape)


# --------------------------------
# STEP 4: PREPROCESSING
# --------------------------------

# Remove extreme outliers
features_df = features_df[
    (features_df['mol_wt'] < mw_max) &
    (features_df['logp'] > logp_min) &
    (features_df['logp'] < logp_max)
]

# Normalize features
cols = ['mol_wt', 'logp', 'h_donors', 'h_acceptors', 'tpsa']

features_df[cols] = (features_df[cols] - features_df[cols].mean()) / features_df[cols].std()

print("After preprocessing:", features_df.shape)


# --------------------------------
# STEP 5: VISUALIZATION
# --------------------------------

print("Generating plots...")

# Histograms
features_df[cols].hist(figsize=(10, 6))
plt.suptitle("Feature Distributions (Normalized)")
plt.tight_layout()
plt.show()

# Correlation heatmap
corr = features_df[cols].corr()

plt.figure()
plt.imshow(corr)
plt.title("Feature Correlation")
plt.colorbar()
plt.xticks(range(len(cols)), cols, rotation=45)
plt.yticks(range(len(cols)), cols)
plt.show()


# --------------------------------
# STEP 6: SAVE PROCESSED DATA
# --------------------------------

features_df.to_csv("processed_molecules.csv", index=False)

print("Saved: processed_molecules.csv")