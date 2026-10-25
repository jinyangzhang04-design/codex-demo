import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

DESCRIPTOR_FUNCS = [Descriptors.MolWt, Descriptors.MolLogP, Descriptors.NumHDonors, Descriptors.NumHAcceptors, Descriptors.TPSA]

def featurize(smiles: str, n_bits: int = 2048) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
    bits = np.asarray(fp, dtype=np.float32)
    desc = np.asarray([f(mol) for f in DESCRIPTOR_FUNCS], dtype=np.float32)
    return np.concatenate([bits, desc])
