from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]

import pandas as pd

# 원본 통합 데이터셋 읽기
df = pd.read_csv(str(ROOT / "data" / "raw" / "Dataset.csv"))

print("컬럼명:", df.columns.tolist())
print("전체 shape:", df.shape)

# 원본 StackDILI 기준 컬럼명
smiles_col = "SMILES"
label_col = "Label"
ref_col = "ref"

# 논문/깃 기준 split
train_df = df[df[ref_col] != "DILIrank"][[smiles_col, label_col, ref_col]].copy()
test_df  = df[df[ref_col] == "DILIrank"][[smiles_col, label_col, ref_col]].copy()

# 저장
train_df.to_csv(str(ROOT / "data" / "processed" / "chemberta_train.csv"), index=False)
test_df.to_csv(str(ROOT / "data" / "processed" / "chemberta_test.csv"), index=False)

print("\n[train]")
print(train_df.shape)
print(train_df[label_col].value_counts())

print("\n[test]")
print(test_df.shape)
print(test_df[label_col].value_counts())

print("\n저장 완료:")
print(str(ROOT / "data" / "processed" / "chemberta_train.csv"))
print(str(ROOT / "data" / "processed" / "chemberta_test.csv"))