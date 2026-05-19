import pandas as pd

START_DATE = "2018-01-01"
END_DATE = "2025-01-01"

path = "../data/gpr_index.csv"

df = pd.read_csv(path)

gpr_index = df[["month", "GPR"]].copy()
gpr_index["month"] = pd.to_datetime(gpr_index["month"], format="%d/%m/%Y")
gpr_index = gpr_index[(gpr_index["month"] >= START_DATE) & (gpr_index["month"] <= END_DATE)]

gpr_index.to_csv("../data/gpr_index_ready.csv", index=False)
