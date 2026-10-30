### 1. Daymet dataset
You need the 001_RSESQ_niveaux_eau_2026-08-04.xlsm file for longitude, latitude and well ID
```python first_preparations/prep.py --path <path to excel file> --sheet <name of the excel sheet> --features <features from the excel file to add to Daymet> --save <name of the final file>```

### 2. Data processing
You need the 001_RSESQ_niveaux_eau_2026-08-04.xlsm file for the groundwater level.
For tree models run:
```python tree_models/main.py --daymet_ds <path to the daymet dataset>```
If first run (gwl dataset is separate from the daymet dataset), run:
```python tree_models/main.py --daymet_ds <path to the daymet dataset> --first_run --gwl <path to gwl excel file> --sheet <name of the excel file>```
To run correlation heatmap
```python tree_models/main.py --daymet_ds <path to the daymet dataset> --correlation_heatmap```
To run VIF
```python tree_models/main.py --daymet_ds <path to the daymet dataset> --vif```

To run LSTM replace the directory "tree_models" with LSTM

### 3. Find the best hyperparameter
For now, the code is not automated, please run it in the python console. To run RF, replace "objectivexgb" with "objective" in line 64 and to run gb, 
you also need to comment the RF objective and uncomment GB's inside the objective function in line 15. It runs XGB by default
```python optimal_ML.py```
This part is optional, if you want to test the best hyperparameters in your run, copy and paste the best results in 
shallow_ML.py 's dictionary (best_xgb_param, best_rf_param or/and best_gb_param)

### 4. Run the regressors
To run RF:
```python tree_models/shallow_ML.py --rf```
To run GB:
```python tree_models/shallow_ML.py --gb```
To run XGB:
```python tree_models/shallow_ML.py```
This code runs R2 and NSE only because it is not automated. 
To run NSE:
```python tree_models/shallow_ML.py --run_metrics```
To run pbias: 
```python tree_models/shallow_ML.py --run_metrics --pbias```
To run kge:
```python tree_models/shallow_ML.py --run_metrics --kge```
To run fhv:
```python tree_models/shallow_ML.py --run_metrics --fhv```
To run fms: 
```python tree_models/shallow_ML.py --run_metrics --fms```
To run flv: 
```python tree_models/shallow_ML.py --run_metrics --flv```
To generate samples for the null values (data imputation):
```python tree_models/shallow_ML.py --gen_samples```
To plot predictions of the simulation against the observation:
```python tree_models/shallow_ML.py --plot_prediction```
To run SHAP:
```python tree_models/shallow_ML.py --shap```

### 4. Run the regressors
Run LSTM model:
```python LSTM/LSTM.py```