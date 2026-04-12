import os
import requests
import gzip
import random
import pandas as pd

from rdkit import Chem
from rdkit.Chem import Descriptors

import cudf
import cupy as cp

# -------------------------------
# STEP 1: Download dataset
# -------------------------------
def download_dataset(url, save_path):
    if not os.path.exists(save_path):
        print("Downloading dataset...")
        r = requests.get(url)
        
        # Check if it's actually gzip
        if "text/html" in r.headers.get("Content-Type", ""):
            raise Exception("Downloaded file is HTML, not dataset. Check URL.")
        
        with open(save_path, "wb") as f:
            f.write(r.content)
        
        print("Download complete.")
    else:
        print("Dataset already exists.")

# Example: ChEMBL SMILES file
DATA_URL = "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_33/chembl_33_chemreps.txt.gz"
FILE_PATH = "chembl_33_chemreps.txt.gz"

download_dataset(DATA_URL, FILE_PATH)

# -------------------------------
# STEP 2: Load + sample dataset
# -------------------------------
def load_smiles(file_path, sample_size=10000):
    print("Loading dataset...")
    
    with gzip.open(file_path, "rt") as f:
        df = pd.read_csv(f, sep="\t")
    
    df = df[['canonical_smiles']].dropna()
    
    # Random sampling
    df = df.sample(n=sample_size, random_state=42)
    
    print(f"Loaded {len(df)} molecules")
    return df

df = load_smiles(FILE_PATH, sample_size=10000)

# -------------------------------
# STEP 3: Filter valid molecules
# -------------------------------
def filter_valid_smiles(df):
    valid_smiles = []
    
    for sm in df['canonical_smiles']:
        mol = Chem.MolFromSmiles(sm)
        if mol is not None:
            valid_smiles.append(sm)
    
    print(f"Valid molecules: {len(valid_smiles)}")
    return pd.DataFrame(valid_smiles, columns=["smiles"])

df = filter_valid_smiles(df)

# -------------------------------
# STEP 4: Feature extraction (RDKit)
# -------------------------------
def compute_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    
    return {
        "mol_wt": Descriptors.MolWt(mol),
        "logp": Descriptors.MolLogP(mol),
        "num_h_donors": Descriptors.NumHDonors(mol),
        "num_h_acceptors": Descriptors.NumHAcceptors(mol),
        "tpsa": Descriptors.TPSA(mol)
    }

def featurize(df):
    features = []
    
    for sm in df['smiles']:
        try:
            features.append(compute_features(sm))
        except:
            continue
    
    features_df = pd.DataFrame(features)
    print("Feature extraction complete.")
    return features_df

features_df = featurize(df)

# -------------------------------
# STEP 5: Convert to GPU (cuDF)
# -------------------------------
def to_gpu(df):
    print("Converting to cuDF...")
    gdf = cudf.DataFrame.from_pandas(df)
    return gdf

gdf = to_gpu(features_df)

# -------------------------------
# STEP 6: Convert to CuPy arrays
# -------------------------------
def to_cupy(gdf):
    print("Converting to CuPy...")
    cp_array = cp.asarray(gdf.values)
    return cp_array

cp_data = to_cupy(gdf)

# -------------------------------
# DONE
# -------------------------------
print("Pipeline complete.")
print("GPU Data shape:", cp_data.shape)