import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator

_MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
RDLogger.DisableLog("rdApp.warning")

DESCRIPTOR_FUNCS = [Descriptors.MolWt, Descriptors.MolLogP, Descriptors.NumHDonors, Descriptors.NumHAcceptors, Descriptors.TPSA]

def featurize(smiles: str, n_bits: int = 2048) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    generator = _MORGAN_GENERATOR if n_bits == 2048 else rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=n_bits)
    bits = generator.GetFingerprintAsNumPy(mol).astype(np.float32)
    desc = np.asarray([f(mol) for f in DESCRIPTOR_FUNCS], dtype=np.float32)
    return np.concatenate([bits, desc])
