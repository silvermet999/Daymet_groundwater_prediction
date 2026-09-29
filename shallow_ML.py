import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_score
from xgboost import XGBRegressor
import main


best_xgb_param = {'n_estimators': 39, 'learning_rate': 0.15221024036816133, 'max_depth': 8, 'min_child_weight': 4, 'subsample': 0.8338818781124975, 'colsample_bytree': 0.814782173388384, 'reg_alpha': 0.9857289954384345, 'reg_lambda': 6.137611450745331e-07}
best_rf_param = {'max_depth': 171, 'n_estimators': 60}
best_gb_param = {'n_estimators': 28, 'learning_rate': 0.025133109440305577, 'max_depth': 7}

start_time = time.time()
model = XGBRegressor(**best_xgb_param)

model.fit(main.X_train_sc, main.y_train)
model = XGBRegressor(**best_xgb_param)

model.fit(main.X_train_sc, main.y_train)
y_pred = model.predict(main.X_test_sc)
score = r2_score(main.y_test, y_pred)
print(score)
print("--- %s seconds ---" % (time.time() - start_time))

# df = pd.read_csv("datasets/null.csv")
# df = df.drop(["Unnamed: 0"], axis=1)
# df["tavg (deg c)"] = (df["tmax (deg c)"] + df["tmin (deg c)"]) / 2
# # df = df.drop(["tmax (deg c)", "tmin (deg c)"], axis=1)
#
# alpha = 1.26
# es = 0.6108 * np.exp((17.27 * df["tavg (deg c)"])/ (df["tavg (deg c)"] + 237.3))
# ea = df["vp (Pa)"] / 1000
# delta = (4098 * es)/((df["tavg (deg c)"] + 237.3)**2)
# gamma = 0.066
# rn = df["srad (W/m^2)"] * df["dayl (s)"] / 1e6
# df["pet"] = alpha * (delta / (delta + gamma)) * rn
# df["wat_bal"] = df["prcp (mm/day)"] - df["pet"]
#
# df = df.drop(["vp (Pa)"], axis=1)
# X = df.drop(columns=["year", "yday", "ID", "gwl"])
# preds = model.predict(X)
# df.loc[df["gwl"].isna(), "gwl"] = preds
#
# df.to_csv("data_filled.csv", index=False)
