import time

import numpy as np
import optuna
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, TimeSeriesSplit, KFold
from xgboost import XGBRegressor as xgb

import main
from sklearn.metrics import make_scorer, r2_score


def objective(trial):
    # gb_n_estimators = trial.suggest_int("gb_n_estimators", 10, 30)
    # gb_learning_rate = trial.suggest_float("gb_learning_rate", 1e-3, 1e-1, log=True)
    # gb_max_depth = trial.suggest_int("gb_max_depth", 3, 12)
    # classifier_obj = GradientBoostingRegressor(
    #     n_estimators=gb_n_estimators,
    #     learning_rate=gb_learning_rate,
    #     max_depth=gb_max_depth
    # )

    rf_max_depth = trial.suggest_int("max_depth", 10, 200)
    rf_n_estimators = trial.suggest_int("n_estimators", 10, 200)
    classifier_obj = RandomForestRegressor(
        max_depth=rf_max_depth,
        n_estimators=rf_n_estimators,
    )
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    score = cross_val_score(classifier_obj, main.X_train_sc, main.y_train, cv=cv,
                            scoring="r2").mean()
    return score


# XGB
def objectivexgb(trial):
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
    classifier_obj = xgb(objective = "reg:squarederror",
    tree_method = "hist",
    random_state = 42,
    n_jobs = -1,
    ** param)
    cv = KFold(n_splits=5, shuffle = True, random_state=42)
    score = cross_val_score(classifier_obj, main.X_train_sc, main.y_train, cv=cv,
                                                    scoring="r2").mean()
    return score


# dtrain = xgb.DMatrix(main.X_train_sc, label=main.y_train)  # ONLY FOR XGB
# dvalid = xgb.DMatrix(main.X_test_sc, label=main.y_test)  # ONLY FOR XGB
start_time = time.time()
study = optuna.create_study(direction="maximize")
study.optimize(objective,
               n_trials=5)  # 200 for XGB, if the results stagnate or do not improve by much we stop the trials
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
