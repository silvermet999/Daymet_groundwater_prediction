import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from statsmodels.tools import add_constant



# df0 = pd.read_csv("datasets/daymet0.csv")
# df1 = pd.read_csv("datasets/daymet1.csv")
# df2 = pd.read_csv("datasets/daymet2.csv")

# def load_csv(file_num):
#     df = pd.read_csv(f"datasets/daymet{file_num}.csv")
#     df = df.drop("Unnamed: 0", axis=1)
#     return df

df = pd.read_csv("datasets/daymet.csv")
df["lat"].nunique()

# gw_df = pd.read_csv("datasets/GWL0.csv")
# gw_long = gw_df.melt(
#     id_vars=['DATE'],
#     var_name='ID',
#     value_name='gwl'
# )
#
# daymet['ID'] = daymet['ID'].astype(str)
# gw_long['ID'] = gw_long['ID'].astype(str)
# gw_long["DATE"] = pd.to_datetime(gw_long["DATE"], errors="coerce")
# gw_long['year'] = gw_long['DATE'].dt.year
# gw_long['yday'] = gw_long['DATE'].dt.dayofyear
# gw_long = gw_long.drop("DATE", axis=1)
# df = daymet.merge(
#     gw_long,
#     on=['ID', 'year', 'yday'],
#     how='inner'
# )

df["tavg (deg c)"] = (df["tmax (deg c)"] + df["tmin (deg c)"]) / 2
# df = df.drop(["tmax (deg c)", "tmin (deg c)"], axis=1)

alpha = 1.26
es = 0.6108 * np.exp((17.27 * df["tavg (deg c)"])/ (df["tavg (deg c)"] + 237.3))
ea = df["vp (Pa)"] / 1000
# vapor pressure deficit
df["vpd"] = es - ea
# delta = (4098 * es)/((df["tavg (deg c)"] + 237.3)**2)
# gamma = 0.066
# rn = df["srad (W/m^2)"] * df["dayl (s)"] / 1e6
# df["pet"] = alpha * (delta / (delta + gamma)) * rn



def correlation_heatmap(): # VP is + correlated with tmax and tmin // tmax is + with tmin and dayl // lat + with long
    df_num = df.drop(["ID"], axis=1)
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
# vp (Pa)  9.451984e+00  // dayl (s)  8.053110e+00

df_vif = df.drop(["vp (Pa)", "tmax (deg c)", "tmin (deg c)"], axis=1)

X = df_vif.drop(["year", "yday", "gwl", "ID"], axis=1)
y = df_vif["gwl"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=0)

scaler = MinMaxScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)
X_train_scaled = pd.DataFrame(X_train, columns=X.columns)
X_test_scaled = pd.DataFrame(X_test, columns=X.columns)



# StandardScaler
# Scale per-site (per well/station), not globally, if sites differ a lot climatically
