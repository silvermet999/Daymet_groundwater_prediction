### 1. Daymet dataset
You need the 001_RSESQ_niveaux_eau_2026-08-04.xlsm file for longitude, latitude and well ID
```python prep.py```

### 2. Data processing
You need the 001_RSESQ_niveaux_eau_2026-08-04.xlsm file for the groundwater level
```python main.py```

### 3. Find the best hyperparameter
For now, the code is not automated. To run RF, replace "objectivexgb" with "objective" in line 64 and to run gb, 
you also need to comment the RF objective and uncomment GB's inside the objective function in line 15. It runs XGB by default
```python optimal_ML.py```
This part is optional, if you want to test the best hyperparameters in your run, copy and paste the best results in 
shallow_ML.py 's dictionary (best_xgb_param, best_rf_param or/and best_gb_param)

### 4. Run the regressors
For now, the code is not automated. To run RF, replace "XGBRegressor(**best_xgb_param)" with 
"RandomForestRegressor(**best_rf_param)". To run GB, replace "XGBRegressor(**best_xgb_param)" with 
"GradientBoostingRegressor(**best_gb_param)"
```python shallow_ML.py```
This code runs R2 and NSE only because it is not automated. To run pbias, kge, fhv, fms, flv, replace "nse" in line 68 
with "pbias", "kge", "fhv", "fms", "flv".