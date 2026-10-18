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
    plt.savefig("corr.png")

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

# if __name__ == "__main__":
#     args = sys.argv[1:]
#     args = parse_args(args)
#     df = pd.read_csv(args.daymet_ds)
#     print("number of wells: ", df["lat"].nunique())
#     if args.first_run:
#         gw_df = pd.read_excel(args.gwl, sheet_name=args.sheet)
#         gw_long = gw_df.melt(
#             id_vars=['DATE'],
#             var_name='ID',
#             value_name='gwl'
#         )
#
#         df['ID'] = df['ID'].astype(str)
#         gw_long['ID'] = gw_long['ID'].astype(str)
#         gw_long["DATE"] = pd.to_datetime(gw_long["DATE"], errors="coerce")
#         gw_long['year'] = gw_long['DATE'].dt.year
#         gw_long['yday'] = gw_long['DATE'].dt.dayofyear
#         gw_long = gw_long.drop("DATE", axis=1)
#         df = df.merge(
#             gw_long,
#             on=['ID', 'year', 'yday'],
#             how='inner'
#         )
#         df.to_csv(args.daymet_ds, index=False)

df = pd.read_csv("datasets/70ID_daymet_gwl.csv")
df["DATE"] = pd.to_datetime(df["DATE"])
df["tavg (deg c)"] = (df["tmax (deg c)"] + df["tmin (deg c)"]) / 2
df = df.drop(["tmax (deg c)", "tmin (deg c)"], axis=1)

alpha = 1.26
es = 0.6108 * np.exp((17.27 * df["tavg (deg c)"]) / (df["tavg (deg c)"] + 237.3))
ea = df["vp (Pa)"] / 1000
# vapor pressure deficit
df["vpd"] = es - ea
df = df.drop(["vp (Pa)"], axis=1)
# delta = (4098 * es)/((df["tavg (deg c)"] + 237.3)**2)
# gamma = 0.066
# rn = df["srad (W/m^2)"] * df["dayl (s)"] / 1e6
# df["pet"] = alpha * (delta / (delta + gamma)) * rn
# vp (Pa)  9.451984e+00  // dayl (s)  8.053110e+00
# if args.correlation_heatmap:
#     correlation_heatmap()
# if args.vif:
#     vif_funct()


X = df.drop(["gwl"], axis=1)
y = df["gwl"]

def scale_site(subset):
    subset = subset.sort_values("DATE")
    X = subset.drop(columns=["DATE", "ID", "gwl"])
    y = subset[["gwl"]]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, shuffle=False
    )

    mask_train = y_train["gwl"].notna().astype(int).values
    mask_val = y_val["gwl"].notna().astype(int).values
    mask_test = y_test["gwl"].notna().astype(int).values

    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    X_train_s = x_scaler.fit(X_train)
    x_params = (X_train_s.data_min_, X_train_s.data_max_)
    X_train_sc = x_scaler.transform(X_train)
    X_test_sc = x_scaler.transform(X_test)
    X_val_sc = x_scaler.transform(X_val)
    y_train_s = y_scaler.fit(y_train)
    y_params = (y_train_s.data_min_, y_train_s.data_max_)
    y_train_sc = y_scaler.transform(y_train)
    y_test_sc = y_scaler.transform(y_test)
    y_val_sc = y_scaler.transform(y_val)


    data = {
        "X_train": X_train_sc, "X_test": X_test_sc, "X_val": X_val_sc,
        "y_train": y_train_sc, "y_test": y_test_sc, "y_val": y_val_sc,
        "mask_train": mask_train, "mask_test": mask_test, "mask_val": mask_val
    }
    return data, x_params, y_params

def minmaxsc(val, min_val, max_val):
    std = (val - min_val) / (max_val - min_val)
    scaled = std * (1.0 - 0.0) + 0.0
    return scaled

data_by_id = {}
x_params = {}
y_params = {}
sec_transform = {}

for site_id, subset in df.groupby("ID"):
    data_by_id[site_id], x_params[site_id], y_params[site_id] = scale_site(subset)
    for i in data_by_id[site_id]["X_train"]:
        for j in i:
            sec_transform[site_id]["X_train"] = minmaxsc(data_by_id[site_id]["X_train"], x_params[site_id][0],
                                                         x_params[site_id][1])
            sec_transform[site_id]["X_test"] = minmaxsc(data_by_id[site_id]["X_test"], x_params[site_id][0],
                                                        x_params[site_id][1])
            sec_transform[site_id]["X_val"] = minmaxsc(data_by_id[site_id]["X_val"], x_params[site_id][0],
                                                       x_params[site_id][1])
            sec_transform[site_id]["y_train"] = minmaxsc(data_by_id[site_id]["y_train"], y_params[site_id][0],
                                                         y_params[site_id][1])
            sec_transform[site_id]["y_test"] = minmaxsc(data_by_id[site_id]["y_test"], y_params[site_id][0],
                                                        y_params[site_id][1])
            sec_transform[site_id]["y_val"] = minmaxsc(data_by_id[site_id]["y_val"], y_params[site_id][0],
                                                       y_params[site_id][1])
    # sec_transform[site_id]["X_train"] = minmaxsc(data_by_id[site_id]["X_train"], x_params[site_id][0], x_params[site_id][1])
    # sec_transform[site_id]["X_test"] = minmaxsc(data_by_id[site_id]["X_test"], x_params[site_id][0], x_params[site_id][1])
    # sec_transform[site_id]["X_val"] = minmaxsc(data_by_id[site_id]["X_val"], x_params[site_id][0], x_params[site_id][1])
    # sec_transform[site_id]["y_train"] = minmaxsc(data_by_id[site_id]["y_train"], y_params[site_id][0], y_params[site_id][1])
    # sec_transform[site_id]["y_test"] = minmaxsc(data_by_id[site_id]["y_test"], y_params[site_id][0], y_params[site_id][1])
    # sec_transform[site_id]["y_val"] = minmaxsc(data_by_id[site_id]["y_val"], y_params[site_id][0], y_params[site_id][1])

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False, random_state=0)
# X_train_sel = X_train.drop(["DATE", "ID"], axis=1)
# X_test_sel = X_test.drop(["DATE", "ID"], axis=1)
#
# mask_train = y_train.notna().astype(int)
# mask_test = y_test.notna().astype(int)
# scaler = MinMaxScaler()
# X_train_sc = scaler.fit_transform(X_train_sel)
# print("X_train_min", scaler.data_min_)
# print("X_train_max", scaler.data_max_)
# X_test_sc = scaler.transform(X_test_sel)
# print("X_test_min", scaler.data_min_)
# print("X_test_max", scaler.data_max_)
# y_train = pd.DataFrame(y_train)
# y_test = pd.DataFrame(y_test)
# y_train_sc = scaler.fit_transform(y_train)
# print("y_train_min", scaler.data_min_)
# print("y_train_max", scaler.data_max_)
# y_test_sc = scaler.transform(y_test)
# print("y_test_min", scaler.data_min_)
# print("y_test_max", scaler.data_max_)


# X_train_min [ 3.04555400e+04  0.00000000e+00  2.76200000e+01  0.00000000e+00
#  -3.00950000e+01  1.46113932e-02]
# X_train_max [5.59444500e+04 1.04500000e+02 6.72070000e+02 3.04650000e+02
#  2.93600000e+01 2.00570292e+00]
# X_test_min [ 3.04555400e+04  0.00000000e+00  2.76200000e+01  0.00000000e+00
#  -3.00950000e+01  1.46113932e-02]
# X_test_max [5.59444500e+04 1.04500000e+02 6.72070000e+02 3.04650000e+02
#  2.93600000e+01 2.00570292e+00]
# y_train_min [7.408]
# y_train_max [9.188]
# y_test_min [7.408]
# y_test_max [9.188]

# 2nd Transformation

#




# def minmaxsc(val, min_val, max_val, col_name):
#     std = (val - min_val) / (max_val - min_val)
#     scaled = std * (1.0 - 0.0) + 0.0
#     df = pd.DataFrame(scaled)
#     df.columns= [col_name]
#     return df

# 3040001
# X_train_dayl = minmaxsc(X_train_sc[:, 0], 3.04555400e+04, 5.59444500e+04, "dayl")
# X_train_prcp = minmaxsc(X_train_sc[:, 1], 0.00000000e+00, 7.27200000e+01, "prcp")
# X_train_srad = minmaxsc(X_train_sc[:, 2], 2.57200000e+01, 6.48160000e+02, "srad")
# X_train_swe = minmaxsc(X_train_sc[:, 3], 0.00000000e+00, 2.02640000e+02, "swe")
# X_train_tavg = minmaxsc(X_train_sc[:, 4], -2.66300000e+01, 2.86250000e+01, "tavg")
# X_train_vpd = minmaxsc(X_train_sc[:, 5], 2.02779958e-02, 1.78190295e+00, "vpd")
#
# y_train_sec = minmaxsc(y_train_sc, 6.878, 9.188, "gwl")
#
# X_test_dayl = minmaxsc(X_test_sc[:, 0], 3.04555400e+04, 5.59444500e+04, "dayl")
# X_test_prcp = minmaxsc(X_test_sc[:, 1], 0.00000000e+00, 7.27200000e+01, "prcp")
# X_test_srad = minmaxsc(X_test_sc[:, 2], 2.57200000e+01, 6.48160000e+02, "srad")
# X_test_swe = minmaxsc(X_test_sc[:, 3], 0.00000000e+00, 2.02640000e+02, "swe")
# X_test_tavg = minmaxsc(X_test_sc[:, 4], -2.66300000e+01, 2.86250000e+01, "tavg")
# X_test_vpd = minmaxsc(X_test_sc[:, 5], 2.02779958e-02, 1.78190295e+00, "vpd")
#
# y_test_sec = minmaxsc(y_test_sc, 6.878, 9.188, "gwl")

# 2150001
# X_train_dayl = minmaxsc(X_train_sc[:, 0], 2.87934600e+04, 5.76065300e+04, "dayl")
# X_train_prcp = minmaxsc(X_train_sc[:, 1], 0.00000000e+00, 4.69200000e+01, "prcp")
# X_train_srad = minmaxsc(X_train_sc[:, 2], 2.42500000e+01, 6.16680000e+02, "srad")
# X_train_swe = minmaxsc(X_train_sc[:, 3], 0.00000000e+00, 2.19770000e+02, "swe")
# X_train_tavg = minmaxsc(X_train_sc[:, 4], -2.83700000e+01, 2.40550000e+01, "tavg")
# X_train_vpd = minmaxsc(X_train_sc[:, 5], 1.61709353e-02, 1.31841858e+00, "vpd")
#
# y_train_sec = minmaxsc(y_train_sc, 3.72965, 5.381201, "gwl")
#
# X_test_dayl = minmaxsc(X_test_sc[:, 0], 2.87934600e+04, 5.76065300e+04, "dayl")
# X_test_prcp = minmaxsc(X_test_sc[:, 1], 0.00000000e+00, 4.69200000e+01, "prcp")
# X_test_srad = minmaxsc(X_test_sc[:, 2], 2.42500000e+01, 6.16680000e+02, "srad")
# X_test_swe = minmaxsc(X_test_sc[:, 3], 0.00000000e+00, 2.19770000e+02, "swe")
# X_test_tavg = minmaxsc(X_test_sc[:, 4], -2.83700000e+01, 2.40550000e+01, "tavg")
# X_test_vpd = minmaxsc(X_test_sc[:, 5], 1.61709353e-02, 1.31841858e+00, "vpd")
#
# y_test_sec = minmaxsc(y_test_sc, 3.72965, 5.381201, "gwl")




# X_train_all = pd.concat([X_train_dayl, X_train_prcp, X_train_srad, X_train_swe, X_train_tavg, X_train_vpd], axis=1)
# X_test_all = pd.concat([X_test_dayl, X_test_prcp, X_test_srad, X_test_swe, X_test_tavg, X_test_vpd], axis=1)
#
# X_train_all_res = X_train_all.reset_index(drop=True)
# X_test_all_res = X_test_all.reset_index(drop=True)
# X_train_res = X_train.reset_index(drop=True)
# X_test_res = X_test.reset_index(drop=True)
# X_train_temp = pd.concat([X_train_res["DATE"], X_train_all_res], axis=1)
# X_test_temp = pd.concat([X_test_res["DATE"], X_test_all_res], axis=1)




# tmax (deg c)     29.377323
# tmin (deg c)     24.133560
# vp (Pa)           9.366750
# srad (W/m^2)      7.431760
# dayl (s)          6.560792
# swe (kg/m^2)      2.012111
# prcp (mm/day)     1.373646

# tavg
# tavg (deg c)     10.316909
# vp (Pa)           9.282189
# dayl (s)          5.488508
# srad (W/m^2)      3.282004
# swe (kg/m^2)      1.810828
# prcp (mm/day)     1.261499

# vaporation
# tavg (deg c)     6.718244
# vpd              5.268158
# dayl (s)         4.805470
# srad (W/m^2)     3.783207
# swe (kg/m^2)     1.924782
# prcp (mm/day)    1.252992
