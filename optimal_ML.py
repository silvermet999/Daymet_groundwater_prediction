import time

import numpy as np
import optuna
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, TimeSeriesSplit, KFold
from xgboost import XGBRegressor as xgb

from main import scale_site
from sklearn.metrics import make_scorer, r2_score


def nse(obs, sim):
    return 1 - ((obs - sim) ** 2).sum() / ((obs - obs.mean()) ** 2).sum()
def objective(trial):
    # gb_n_estimators = trial.suggest_int("gb_n_estimators", 5, 50)
    # gb_learning_rate = trial.suggest_float("gb_learning_rate", 1e-3, 1e-1, log=True)
    # gb_max_depth = trial.suggest_int("gb_max_depth", 3, 12)
    # obj = GradientBoostingRegressor(
    #     n_estimators=gb_n_estimators,
    #     learning_rate=gb_learning_rate,
    #     max_depth=gb_max_depth
    # )

    # rf_max_depth = trial.suggest_int("max_depth", 5, 50)
    # rf_n_estimators = trial.suggest_int("n_estimators", 5, 50)
    # obj = RandomForestRegressor(
    #     max_depth=rf_max_depth,
    #     n_estimators=rf_n_estimators,
    # )
    param = {
        "n_estimators": trial.suggest_int("n_estimators", 5, 100),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }
    obj = xgb(objective="reg:squarederror",
              tree_method="hist",
              random_state=42,
              n_jobs=-1,
              **param)
    # cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # score = cross_val_score(obj, main.X_train_all, y_train, cv=5,
    #                         scoring="r2").mean()

    #
    # y_test = main.y_test_sec.fillna(0)
    # obj.fit(main.X_train_all, y_train)
    # y_pred = obj.predict(main.X_test_all)
    # r2...

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

    obj.fit(X_train.to_numpy(), y_train.to_numpy().ravel())
    pred = obj.predict(X_test.to_numpy())
    score = nse(y_test.to_numpy().ravel(), pred.ravel())

    return score


start_time = time.time()
study = optuna.create_study(direction="maximize")
study.optimize(objective,
               n_trials=100)
print(study.best_trial)
print("Number of finished trials: ", len(study.trials))
print("Best trial:")
trial = study.best_trial

print("  Value: {}".format(trial.value))
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

best_params = study.best_params
print("--- %s seconds ---" % (time.time() - start_time))
