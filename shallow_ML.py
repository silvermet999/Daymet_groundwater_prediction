import argparse
import sys
import time

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import RobustScaler
from xgboost import XGBRegressor
from main import scale_site
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
        best_rf_param = {'max_depth': 47, 'n_estimators': 38}
        model = RandomForestRegressor(**best_rf_param)
    elif args.gb:
        best_gb_param = {'learning_rate': 0.09648622384311142, 'max_depth': 10}
        model = HistGradientBoostingRegressor(**best_gb_param)
    else:
        best_xgb_param = {'n_estimators': 98, 'learning_rate': 0.26402543073672413, 'max_depth': 10, 'min_child_weight': 6, 'subsample': 0.9538400037435389, 'colsample_bytree': 0.735840741595466, 'reg_alpha': 0.00039226402768744443, 'reg_lambda': 0.3963835374683215}
        model= XGBRegressor(**best_xgb_param)
    start_time = time.time()
    df, _ = scale_site()

    mask_train = df["mask_train"]
    mask_val = df["mask_val"]
    mask_test = df["mask_test"]

    X_train = df["X_train"][mask_train]
    X_val = df["X_val"][mask_val]
    X_test = df["X_test"][mask_test]

    y_train = df["y_train"][mask_train]
    y_val = df["y_val"][mask_val]
    y_test = df["y_test"][mask_test]
    model.fit(X_train.to_numpy(), y_train.to_numpy().ravel())
    y_pred = model.predict(X_test.to_numpy())
    score = r2_score(y_test.to_numpy().ravel(), y_pred.ravel())
    print(score)
    print("--- %s seconds ---" % (time.time() - start_time))

    if args.plot_prediction:
        plot_df = X_test.copy()
        plot_df["obs"] = np.asarray(y_test).ravel()
        plot_df["pred"] = np.asarray(y_pred).ravel()
        plot_df["DATE"] = pd.to_datetime(
            X_test["year"].astype(str) + X_test["doy"].astype(str).str.zfill(3), format="%Y%j"
        )

        for site_id, sub in plot_df.groupby("ID"):
            sub = sub.sort_values("DATE")

            plt.figure(figsize=(14, 4))
            plt.plot(sub["DATE"], sub["obs"], label="Ground truth")
            plt.plot(sub["DATE"], sub["pred"], label="regressor", linestyle="--")
            plt.title(f"groundwater level at site {site_id}")
            plt.xlabel("date")
            plt.ylabel("groundwater level")
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"plot_{site_id}.png")

    if args.run_metrics:
        start_time = time.time()
        model.fit(X_train.to_numpy(), y_train.to_numpy().ravel())
        test = X_test.copy()
        test["pred"] = model.predict(X_test)
        test["obs"] = y_test.to_numpy().ravel()
        if args.pbias:
            per_well = (
                test.dropna(subset=["obs"])
                .groupby("ID")
                .apply(lambda g: pbias(g["obs"].to_numpy(), g["pred"].to_numpy()))
            )
        elif args.kge:
            per_well = (
                test.dropna(subset=["obs"])
                .groupby("ID")
                .apply(lambda g: kge(g["obs"].to_numpy(), g["pred"].to_numpy()))
            )
        elif args.fhv:
            per_well = (
                test.dropna(subset=["obs"])
                .groupby("ID")
                .apply(lambda g: fhv(g["obs"].to_numpy(), g["pred"].to_numpy()))
            )
        elif args.fms:
            per_well = (
                test.dropna(subset=["obs"])
                .groupby("ID")
                .apply(lambda g: fms(g["obs"].to_numpy(), g["pred"].to_numpy()))
            )
        elif args.flv:
            per_well = (
                test.dropna(subset=["obs"])
                .groupby("ID")
                .apply(lambda g: flv(g["obs"].to_numpy(), g["pred"].to_numpy()))
            )
        else:
            per_well = (
                test.dropna(subset=["obs"])
                .groupby("ID")
                .apply(lambda g: nse(g["obs"].to_numpy(), g["pred"].to_numpy()))
            )
        print(per_well)
        print("--- %s seconds ---" % (time.time() - start_time))
    if args.shap:
        explainer = shap.Explainer(model)
        shap_values = explainer(X_train)
        shap.plots.waterfall(shap_values[0], show=False)
        plt.gcf().set_size_inches(20, 20)
        plt.savefig("shapely.png", dpi=300, bbox_inches="tight")
        shap.save_html(
            "force_first_pred.html",
            shap.plots.force(shap_values[0])
        )
        shap.save_html(
            "force_500.html",
            shap.plots.force(shap_values[:500])
        )






