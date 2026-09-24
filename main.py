import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split


# df0 = pd.read_csv("datasets/daymet0.csv")
# df1 = pd.read_csv("datasets/daymet1.csv")
# df2 = pd.read_csv("datasets/daymet2.csv")

# def load_csv(file_num):
#     df = pd.read_csv(f"datasets/daymet{file_num}.csv")
#     df = df.drop("Unnamed: 0", axis=1)
#     return df

df = pd.read_csv("datasets/daymet.csv")
df["lat"].nunique() # 93 puits

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
delta = (4098 * es)/((df["tavg (deg c)"] + 237.3)**2)
gamma = 0.066
rn = df["srad (W/m^2)"] * df["dayl (s)"] / 1e6
df["pet"] = alpha * (delta / (delta + gamma)) * rn
df["wat_bal"] = df["prcp (mm/day)"] - df["pet"]
df["sum_wat_bal"] = df.groupby("ID")["wat_bal"].cumsum()



def correlation_heatmap(): # VP is + correlated with tmax and tmin // tmax is + with tmin and dayl // lat + with long
    df_num = df.drop(["ID"], axis=1)
    corr = df_num.corr()
    f, ax = plt.subplots(figsize=(11, 9))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr, annot=True, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.savefig("corr_feat.png")



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
    df_num = df.drop(["ID", "gwl"], axis=1)
    vif = pd.DataFrame({"feature": df_num.columns, "VIF": [variance_inflation_factor(df_num.values, i) for i in range(df_num.shape[1])]})
    return vif
# vp (Pa)  9.451984e+00  // dayl (s)  8.053110e+00

df = df.drop(["vp (Pa)", "sum_wat_bal"], axis=1)

X = df.drop(["year", "yday", "gwl", "ID"], axis=1)
y = df["gwl"]




















X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=0)

scaler = MinMaxScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)
X_train_scaled = pd.DataFrame(X_train, columns=X.columns)
X_test_scaled = pd.DataFrame(X_test, columns=X.columns)

# [I 2026-09-23 19:14:12,685] Trial 2 finished with value: 0.18545076292065454 and parameters: {'rf_max_depth': 171, 'rf_n_estimators': 60}. Best is trial 0 with value: 0.192179429588696.
# [I 2026-09-23 19:36:02,967] Trial 4 finished with value: 0.019178117222788883 and parameters: {'gb_n_estimators': 28, 'gb_learning_rate': 0.041214172103686865, 'gb_max_depth': 3}. Best is trial 0 with value: 0.2588294856911901.
# [I 2026-09-23 20:01:30,516] Trial 3 finished with value: 0.06998034166623315 and parameters: {'n_estimators': 685, 'learning_rate': 0.11722262954430264, 'max_depth': 4, 'min_child_weight': 9, 'subsample': 0.7830166168752186, 'colsample_bytree': 0.9118279529827669, 'reg_alpha': 0.0001824977499260441, 'reg_lambda': 0.01144346763699817}. Best is trial 2 with value: 0.21327336117921045.


# StandardScaler
# Scale per-site (per well/station), not globally, if sites differ a lot climatically
