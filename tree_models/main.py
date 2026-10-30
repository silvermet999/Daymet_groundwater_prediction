import argparse
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from statsmodels.tools import add_constant


# After prep
# df = pd.read_csv("datasets/70daymet.csv")
# df = df.drop("Unnamed: 0", axis=1)
# df["DATE"] = pd.to_datetime(df["year"], format="%Y") \
#                           + pd.to_timedelta(df["yday"] - 1, unit="D")
# df = df.drop(["year", "yday"], axis=1)
# gwl = pd.read_excel("src/001_RSESQ_niveaux_eau_2026-08-04.xlsx", sheet_name="Niveaux_journaliers")
# ids_sel = pd.read_csv("src/70ids.csv")
# #gwl = gwl[["DATE", "03040001"]]
# gw_long = gwl.melt(
#             id_vars=['DATE'],
#             var_name='ID',
#             value_name='gwl'
#         )
# gw_long['ID'] = gw_long['ID'].map(lambda x: x.lstrip("0"))
# gw_long['ID'] = gw_long['ID'].astype(str)
# ids_sel["ID_sel"] = ids_sel["ID_sel"].astype(str)
# gw_long["DATE"] = pd.to_datetime(gw_long["DATE"], errors="coerce")
# gwl = gw_long[gw_long["ID"].isin(ids_sel["ID_sel"])].copy()
# gwl = gwl[gwl["DATE"]>=pd.Timestamp("2011-01-01")]
# gwl = gwl[gwl["DATE"]<pd.Timestamp("2026-01-01")]
# #gwl = gwl.drop("variable", axis=1)
# gwl=gwl[~gwl["DATE"].isin(["2012-02-29", "2016-02-29", "2020-02-29", "2024-02-29"])]
# gwl["DATE"] = pd.to_datetime(gwl["DATE"])
# df["DATE"] = pd.to_datetime(df["DATE"])
# df = df[df["DATE"]>=pd.Timestamp("2011-01-01")]
# df = df.merge(gwl, on="DATE", how='outer')
# df.to_csv("datasets/3040001.csv", index=False)
# Missing from daymet
# dayl (s)  prcp (mm/day)  srad (W/m^2)  ...  vp (Pa)        DATE       gwl
# 12045       NaN            NaN           NaN  ...      NaN 2012-12-31  7.405500
# 13506       NaN            NaN           NaN  ...      NaN 2016-12-31  7.335994
# 14967       NaN            NaN           NaN  ...      NaN 2020-12-31  7.660333
# 16428       NaN            NaN           NaN  ...      NaN 2024-12-31  7.540400


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--daymet_ds", default="datasets/daymet.csv", type=str)
    parser.add_argument("--first_run", help="This action concatenate the groundwater "
                                                                  "levels in the excel sheet and saves the new dataset. "
                                                                  "If your daymet does not contain the ground water levels, "
                                                                  "set this action to true")
    parser.add_argument("--gwl", default="src/001_RSESQ_niveaux_eau_2026-08-04.xlsx", type=str)
    parser.add_argument("--sheet", default="Niveaux_journaliers", type=str)
    parser.add_argument("--correlation_heatmap")
    parser.add_argument("--vif")
    return parser.parse_args(args)

def correlation_heatmap():
    df_num = df.drop(["DATE"], axis=1)
    corr = df_num.corr()
    f, ax = plt.subplots(figsize=(11, 9))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr, annot=True, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.savefig("corr-after.png")

def feature_plot():
    for site_id in df["ID"].unique():
        subset = df[df["ID"] == site_id]
        plt.figure(figsize=(14, 4))
        plt.plot(subset["yday"], subset["swe (kg/m^2)"])
        plt.title(f"snow level at site {site_id}")
        plt.xlabel("day of year")
        plt.ylabel("snow level")
        plt.savefig(f"plot_{site_id}.png")

def vif_funct():
    X = df.drop(["ID", "gwl"], axis=1)
    Xc = add_constant(X)
    vif = pd.Series(
        [variance_inflation_factor(Xc.values, i) for i in range(1, Xc.shape[1])],
        index=X.columns
    ).sort_values(ascending=False)
    return vif


def minmaxsc(val, min_val, max_val):
    scaled = (val - min_val) / (max_val - min_val)
    return scaled


def add_features(g):
    g = g.sort_values("DATE").copy()
    g["tmean"] = (g["tmax (deg c)"] + g["tmin (deg c)"]) / 2
    g["trange"] = g["tmax (deg c)"] - g["tmin (deg c)"]

    for w in (7, 30, 90, 180, 365):
        g[f"prcp_sum_{w}"] = g["prcp (mm/day)"].rolling(w).sum()
        g[f"tmean_avg_{w}"] = g["tmean"].rolling(w).mean()
        g[f"srad_avg_{w}"] = g["srad (W/m^2)"].rolling(w).mean()

    g["swe_max_30"] = g["swe (kg/m^2)"].rolling(30).max()
    g["swe_diff_7"] = g["swe (kg/m^2)"].diff(7)
    g["vp_avg_30"] = g["vp (Pa)"].rolling(30).mean()
    return g

def scale_site():
    indiv_results = {}

    for site_id, subset in df.groupby("ID"):
        subset = subset.sort_values("DATE")
        subset["doy"] = subset["DATE"].dt.dayofyear
        subset["doy_sin"] = np.sin(2 * np.pi * subset["doy"] / 365.25)
        subset["doy_cos"] = np.cos(2 * np.pi * subset["doy"] / 365.25)
        subset['year'] = subset['DATE'].dt.year

        # Based on SHAP
        X = subset.drop(columns=["lat", "lon", "ID", "DATE", "tmean_avg_90", "gwl"])
        y = subset[["gwl"]]
        static_cols = ["doy", "doy_sin", "doy_cos", "year"]
        feat_cols = [c for c in X.columns if c not in static_cols]

        n = len(subset)
        i_train = int(n * 0.6)
        i_val = int(n * (0.6 + 0.2))
        X_train, X_val, X_test = X.iloc[:i_train], X.iloc[i_train:i_val], X.iloc[i_val:]
        y_train, y_val, y_test = y.iloc[:i_train], y.iloc[i_train:i_val], y.iloc[i_val:]

        mask_train = y_train["gwl"].notna().values
        mask_val = y_val["gwl"].notna().values
        mask_test = y_test["gwl"].notna().values

        X_min, X_max = X_train[feat_cols].min(), X_train[feat_cols].max()
        y_min, y_max = y_train.min(), y_train.max()

        def scale_X(Xs):
            scaled = minmaxsc(Xs[feat_cols], X_min, X_max)
            return pd.concat([scaled, Xs[static_cols]], axis=1)[X.columns]

        X_train_sc = scale_X(X_train)
        X_val_sc = scale_X(X_val)
        X_test_sc = scale_X(X_test)


        y_train_sc = minmaxsc(y_train, y_min, y_max)
        y_val_sc = minmaxsc(y_val, y_min, y_max)
        y_test_sc = minmaxsc(y_test, y_min, y_max)

        indiv_results[site_id] = {
            "X_train": X_train_sc, "X_val": X_val_sc, "X_test": X_test_sc,
            "y_train": y_train_sc, "y_val": y_val_sc, "y_test": y_test_sc,
            "mask_train": mask_train, "mask_val": mask_val, "mask_test": mask_test
        }
    X_train_all = pd.concat([d["X_train"] for d in indiv_results.values()])
    y_train_all = pd.concat([d["y_train"] for d in indiv_results.values()])
    mask_train_all = np.concatenate([d["mask_train"] for d in indiv_results.values()])

    X_val_all = pd.concat([d["X_val"] for d in indiv_results.values()])
    y_val_all = pd.concat([d["y_val"] for d in indiv_results.values()])
    mask_val_all = np.concatenate([d["mask_val"] for d in indiv_results.values()])

    X_test_all = pd.concat([d["X_test"] for d in indiv_results.values()])
    y_test_all = pd.concat([d["y_test"] for d in indiv_results.values()])
    mask_test_all = np.concatenate([d["mask_test"] for d in indiv_results.values()])

    concat_results = {"X_train": X_train_all, "X_val": X_val_all, "X_test": X_test_all,
               "y_train": y_train_all, "y_val": y_val_all, "y_test": y_test_all,
               "mask_train": mask_train_all, "mask_val": mask_val_all, "mask_test": mask_test_all,
                "X_min": X_min, "X_max": X_max, "y_min": y_min, "y_max": y_max
               }
    return concat_results, indiv_results


if __name__ == "__main__":
    args = sys.argv[1:]
    args = parse_args(args)
    df = pd.read_csv(args.daymet_ds)
    print("number of wells: ", df["lat"].nunique())
    if args.first_run:
        gw_df = pd.read_excel(args.gwl, sheet_name=args.sheet)
        gw_long = gw_df.melt(
            id_vars=['DATE'],
            var_name='ID',
            value_name='gwl'
        )

        df['ID'] = df['ID'].astype(str)
        gw_long['ID'] = gw_long['ID'].astype(str)
        gw_long["DATE"] = pd.to_datetime(gw_long["DATE"], errors="coerce")
        gw_long['year'] = gw_long['DATE'].dt.year
        gw_long['yday'] = gw_long['DATE'].dt.dayofyear
        gw_long = gw_long.drop("DATE", axis=1)
        df = df.merge(
            gw_long,
            on=['ID', 'year', 'yday'],
            how='inner'
        )
        df.to_csv(args.daymet_ds, index=False)
    df["DATE"] = pd.to_datetime(df["DATE"])
    df = df.groupby("ID", group_keys=False).apply(add_features)
    if args.correlation_heatmap:
        correlation_heatmap()
    if args.vif:
        vif_funct()

# df = pd.read_csv("../datasets/70ID_daymet_gwl.csv")
# df["DATE"] = pd.to_datetime(df["DATE"])
# df["tavg (deg c)"] = (df["tmax (deg c)"] + df["tmin (deg c)"]) / 2
# df = df.drop(["tmax (deg c)", "tmin (deg c)"], axis=1)
# #
# alpha = 1.26
# es = 0.6108 * np.exp((17.27 * df["tavg (deg c)"]) / (df["tavg (deg c)"] + 237.3))
# ea = df["vp (Pa)"] / 1000
# # vapor pressure deficit
# df["vpd"] = es - ea
# df = df.drop(["vp (Pa)"], axis=1)
# delta = (4098 * es)/((df["tavg (deg c)"] + 237.3)**2)
# gamma = 0.066
# rn = df["srad (W/m^2)"] * df["dayl (s)"] / 1e6
# df["pet"] = alpha * (delta / (delta + gamma)) * rn
# vp (Pa)  9.451984e+00  // dayl (s)  8.053110e+00



