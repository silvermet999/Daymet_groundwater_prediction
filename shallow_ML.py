import argparse
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import RobustScaler
from xgboost import XGBRegressor
import main
import matplotlib.pyplot as plt

def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--rf", default=False)
    parser.add_argument("--gb", default=False)
    parser.add_argument("--null_ds", default="datasets/null.csv", type=str)
    parser.add_argument("--synth_ds", default="ds_filled.csv", type=str)
    parser.add_argument("--gwl", default="src/001_RSESQ_niveaux_eau_2026-08-04.xlsx", type=str)
    parser.add_argument("--sheet", default="Niveaux_journaliers", type=str)
    parser.add_argument("--gen_samples")
    parser.add_argument("--plot_prediction")
    parser.add_argument("--run_metrics", help="NSE, pBias, KGE, FMS, FHV, FLV")
    parser.add_argument("--pbias")
    parser.add_argument("--kge")
    parser.add_argument("--fhv")
    parser.add_argument("--fms")
    parser.add_argument("--flv")
    return parser.parse_args(args)


def nse(obs, sim):
    return 1 - ((obs - sim) ** 2).sum() / ((obs - obs.mean()) ** 2).sum()
def pbias(obs, sim):
    return 100 * np.sum(sim - obs) / np.sum(obs)

def kge(obs, sim):
    r = np.corrcoef(obs, sim)[0, 1]
    alpha = sim.std() / obs.std()
    beta = sim.mean() / obs.mean()
    return 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)

def fhv(obs, sim, h=0.02):
    o = np.sort(obs)[::-1]
    s = np.sort(sim)[::-1]
    n = max(1, int(h * len(o)))
    return 100 * np.sum(s[:n] - o[:n]) / np.sum(o[:n])

def fms(obs, sim, lo=0.2, hi=0.7, eps=1e-6):
    def slope(x):
        return np.log(np.quantile(x, 1 - lo) + eps) - np.log(np.quantile(x, 1 - hi) + eps)
    return 100 * (slope(sim) - slope(obs)) / slope(obs)

def flv(obs, sim, l=0.3, eps=1e-6):
    def low(x):
        x = np.sort(x)[: max(1, int(l * len(x)))]
        return np.sum(np.log(x + eps) - np.log(x.min() + eps))
    return -100 * (low(sim) - low(obs)) / (low(obs) + 1e-6)


if __name__ == "__main__":
    args = sys.argv[1:]
    args = parse_args(args)
    if args.rf:
        best_rf_param = {'max_depth': 38, 'n_estimators': 49}
        model = RandomForestRegressor(**best_rf_param)
    elif args.gb:
        best_gb_param = {'n_estimators': 47, 'learning_rate': 0.061573566036953464, 'max_depth': 11}
        model = RandomForestRegressor(**best_gb_param)
    else:
        best_xgb_param = {"colsample_bytree": 0.7875429935811782, "learning_rate": 0.07850245127227708, "max_depth": 8,
                          "min_child_weight": 8, "n_estimators": 85, "reg_alpha": 8.678053133506951e-05,
                          "reg_lambda": 1.9948778966859684e-06, "subsample": 0.7595133180855639}
        model= XGBRegressor(**best_xgb_param)
    start_time = time.time()
    model.fit(main.X_train_sc, main.y_train)
    y_pred = model.predict(main.X_test_sc)
    score = r2_score(main.y_test, y_pred)
    print(score)
    print("--- %s seconds ---" % (time.time() - start_time))
    if args.gen_samples:
        df = pd.read_csv(args.null_ds)
        df = df.drop(['Unnamed: 0', "Unnamed: 0.1"], axis=1)
        df["tavg (deg c)"] = (df["tmax (deg c)"] + df["tmin (deg c)"]) / 2
        es = 0.6108 * np.exp((17.27 * df["tavg (deg c)"]) / (df["tavg (deg c)"] + 237.3))
        ea = df["vp (Pa)"] / 1000
        # vapor pressure deficit
        df["vpd"] = es - ea

        df = df.drop(["vp (Pa)", "tmax (deg c)", "tmin (deg c)"], axis=1)
        X = df.drop(columns=["year", "yday", "gwl", "ID"])
        X_scaled = main.scaler.transform(X)
        preds = model.predict(X_scaled)
        df.loc[df["gwl"].isna(), "gwl"] = preds

        df.to_csv(args.synth_ds, index=False)

    if args.plot_prediction:
        gw_df = pd.read_excel(args.gwl, sheet_name=args.sheet)
        df_true = gw_df.melt(
            id_vars=['DATE'],
            var_name='ID',
            value_name='gwl'
        )
        df_pred = pd.read_csv(args.synth_ds)
        df_pred["DATE"] = pd.to_datetime(df_pred["year"].astype(str), format="%Y") \
                          + pd.to_timedelta(df_pred["yday"] - 1, unit="D")
        df_true['ID'] = df_true['ID'].map(lambda x: x.lstrip("0"))
        df_pred['ID'] = df_pred['ID'].astype(str)
        df_true["ID"] = df_true["ID"].astype(str)
        # all_ids = sorted(set(df_true["ID"]) | set(df_pred["ID"]))
        for site_id in df_pred["ID"].unique():
            true_sub = df_true[df_true["ID"] == site_id].sort_values(["DATE"])
            pred_sub = df_pred[df_pred["ID"] == site_id].sort_values(["DATE"])

            plt.figure(figsize=(14, 4))
            plt.plot(true_sub["DATE"], true_sub["gwl"], label="Ground truth")
            plt.plot(pred_sub["DATE"], pred_sub["gwl"], label="regressor", linestyle="--")
            plt.title(f"groundwater level at site {site_id}")
            plt.xlabel("year")
            plt.ylabel("groundwater level")
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"plot_{site_id}.png")

    if args.run_metrics:
        start_time = time.time()
        df = pd.read_csv(args.synth_ds)
        df["DATE"] = pd.to_datetime(df["year"].astype(str), format="%Y") \
                     + pd.to_timedelta(df["yday"] - 1, unit="D")
        df = df.sort_values(["ID", "DATE"])

        cutoff = pd.Timestamp("2025-01-01")
        train = df[df["DATE"] < cutoff]
        test = df[df["DATE"] >= cutoff]
        features = df.drop(["ID", "gwl", "DATE"], axis=1).columns

        model.fit(train[features], train["gwl"])
        test = test.assign(pred=model.predict(test[features]))
        if args.pbias:
            per_well = test.groupby("ID").apply(lambda g: pbias(g["gwl"], g["pred"]))
        elif args.kge:
            per_well = test.groupby("ID").apply(lambda g: kge(g["gwl"], g["pred"]))
        elif args.fhv:
            per_well = test.groupby("ID").apply(lambda g: fhv(g["gwl"], g["pred"]))
        elif args.fms:
            per_well = test.groupby("ID").apply(lambda g: fms(g["gwl"], g["pred"]))
        elif args.flv:
            per_well = test.groupby("ID").apply(lambda g: flv(g["gwl"], g["pred"]))
        else:
            per_well = test.groupby("ID").apply(lambda g: nse(g["gwl"], g["pred"]))
        print(per_well)
        print("--- %s seconds ---" % (time.time() - start_time))




