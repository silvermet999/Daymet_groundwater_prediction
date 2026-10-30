import pandas as pd
from matplotlib import pyplot as plt

gwl = pd.read_excel("src/001_RSESQ_niveaux_eau_2026-08-04.xlsx", sheet_name="Niveaux_journaliers")
gw_long = gwl.melt(
            id_vars=['DATE'],
            var_name='ID',
            value_name='gwl'
        )

wells = pd.read_csv("../src/70ids.csv")
gw_long["ID"] = gw_long["ID"].astype(str)
wells["ID_sel"] = wells["ID_sel"].astype(str)
gw_long["ID"] = gw_long["ID"].map(lambda x: x.lstrip("0"))
ids = wells["ID_sel"].unique()
gwl = gw_long[gw_long["ID"].isin(ids)]
# confirm
print(gwl["ID"].nunique())
gwl = gwl[gwl["DATE"] >= pd.Timestamp("2011-01-01")]

def plot():
    for f in gwl.columns:
        plt.figure(figsize=(14, 4))
        plt.plot(gwl["DATE"], gwl[f])
        plt.title(f"feature: {f}")
        plt.xlabel("time")
        plt.ylabel(f"{f}")
        plt.savefig(f"plots/gwl plots/plot_{f}")
