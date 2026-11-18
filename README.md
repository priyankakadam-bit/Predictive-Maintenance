# Predictive Maintenance - Machine Failure Prediction

**Live app:** https://predictive-maintenance-01.streamlit.app/

## Business Problem
Unplanned machine downtime is expensive in manufacturing. This project predicts **machine failures from sensor
readings before they happen**, so maintenance can be planned instead of reacting to breakdowns.

## Dataset
- **Source:** AI4I 2020 Predictive Maintenance Dataset (UCI Machine Learning Repository)
- **Size:** 10,000 machine readings: product type (L / M / H), air and process temperature, rotational speed, torque, tool wear
- **Target:** `Machine failure` (0 = healthy, 1 = failure)
- **Challenge:** only **3.4%** of readings are failures (339 of 10,000)

## Methodology
1. **EDA:** sensor distributions, failure rate by product type, correlations
2. **Cleaning:** drop IDs and the failure sub-type columns (TWF, HDF, PWF, OSF, RNF). They are only known after a failure, so keeping them would be data leakage.
3. **Encoding:** product type H = 0, L = 1, M = 2. The mapping is saved in `model_config.json` and the app reads it from there.
4. **Feature engineering**, matching how failures happen physically:
   - `temp_diff` = process − air temperature (heat dissipation)
   - `power` = torque × rotational speed (power failures)
   - `wear_torque` = tool wear × torque (overstrain)
   - `wear_rate` = tool wear ÷ rotational speed
5. **Stratified 80 / 20 split**, then **SMOTE on the training data only**
6. **StandardScaler**, fitted on training data
7. **Models:** Logistic Regression, Random Forest, XGBoost
8. **Streamlit app** for live predictions

## Results

Test set: 2,000 readings, 68 failures.

| Model | Precision (failure) | Recall (failure) | F1 (failure) | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.18 | 0.87 | 0.30 | 0.927 |
| **Random Forest** (deployed) | **0.67** | **0.85** | **0.75** | **0.985** |
| XGBoost | 0.66 | 0.85 | 0.74 | 0.976 |

- **Random Forest** gives the best balance: it catches **85% of failures** with far fewer false alarms than Logistic Regression.
- Logistic Regression has high recall, but 82% of its alarms are false, because failures depend on combinations and extremes (e.g. low speed + small temperature gap) that a linear model cannot separate.
- In maintenance, **recall matters most**: a missed failure costs far more than a false alarm.

## Streamlit App
Choose the machine type and sensor values to get the failure probability and a risk level:
- **High risk:** 50% or more ("Failure predicted")
- **Medium risk:** 25–50%
- **Low risk:** below 25%

Run locally:
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Project Structure
```
predictive-maintenance/
├── 01.Exploratory Data Analysis(EDA).ipynb   ← EDA
├── 02.modeling.ipynb                         ← features, SMOTE, model training and comparison
├── streamlit_app.py                          ← web app
├── ai4i2020.csv                              ← dataset
├── best_model.pkl                            ← trained Random Forest
├── scaler.pkl                                ← fitted StandardScaler
├── model_config.json                         ← type encoding and feature order used by the app
├── requirements.txt                          ← scikit-learn pinned to the version the model was saved with
└── README.md
```

## Limitations
- AI4I 2020 is a **synthetic** dataset built from failure rules, so real factory data would be noisier.
- Random failures (RNF, 0.1%) cannot be predicted by design, and tool-wear failures happen at a random point between 200 and 240 minutes.

## Tech Stack
Python · pandas · scikit-learn · XGBoost · imbalanced-learn (SMOTE) · Streamlit · Matplotlib · Seaborn
