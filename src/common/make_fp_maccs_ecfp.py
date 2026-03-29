from pathlib import Path
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys

ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = ROOT / "data" / "processed" / "chemberta_train.csv"
TEST_PATH = ROOT / "data" / "processed" / "chemberta_test.csv"
OUT_TRAIN = ROOT / "data" / "processed" / "fp1190_train.csv"
OUT_TEST = ROOT / "data" / "processed" / "fp1190_test.csv"

RADIUS = 2
N_BITS = 1024


def smiles_to_fp(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    maccs = MACCSkeys.GenMACCSKeys(mol)
    maccs_arr = np.array(list(maccs.ToBitString()), dtype=np.int8)

    ecfp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=RADIUS, nBits=N_BITS)
    ecfp_arr = np.array(list(ecfp.ToBitString()), dtype=np.int8)

    # RDKit MACCS는 167비트라서 보통 첫 비트 버리고 166개 사용
    maccs_166 = maccs_arr[1:]

    fp = np.concatenate([maccs_166, ecfp_arr], axis=0)
    return fp


def build_fp_csv(in_path: Path, out_path: Path):
    df = pd.read_csv(in_path)

    rows = []
    invalid = 0

    for i, row in df.iterrows():
        smiles = row["SMILES"]
        label = row["Label"]
        ref = row["ref"]

        fp = smiles_to_fp(smiles)
        if fp is None:
            invalid += 1
            continue

        record = {
            "SMILES": smiles,
            "Label": label,
            "ref": ref,
        }

        for j in range(166):
            record[f"MACCS{j}"] = int(fp[j])

        for j in range(1024):
            record[f"ECFP{j}"] = int(fp[166 + j])

        rows.append(record)

        if (i + 1) % 200 == 0 or (i + 1) == len(df):
            print(f"{in_path.name}: processed {i+1} / {len(df)}")

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_path, index=False)

    print(f"\nSaved: {out_path}")
    print("shape:", out_df.shape)
    print("invalid:", invalid)


def main():
    build_fp_csv(TRAIN_PATH, OUT_TRAIN)
    build_fp_csv(TEST_PATH, OUT_TEST)


if __name__ == "__main__":
    main()