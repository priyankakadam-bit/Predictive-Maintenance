import json
import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Predictive Maintenance", page_icon="⚙️", layout="wide")

HERE = os.path.dirname(os.path.abspath(__file__))

# Must match 02.modeling.ipynb. LabelEncoder sorts alphabetically: H=0, L=1, M=2.
DEFAULT_TYPE_MAP = {"H": 0, "L": 1, "M": 2}
FEATURES = [
    "Type", "Air temperature [K]", "Process temperature [K]",
    "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]",
    "temp_diff", "power", "wear_torque", "wear_rate",
]


def find_file(name):
    """Look in models/ first (where the notebook saves), then next to this script."""
    for folder in ("models", ""):
        path = os.path.join(HERE, folder, name)
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(f"{name} not found in models/ or the app folder")


@st.cache_resource
def load_artifacts():
    model  = joblib.load(find_file("best_model.pkl"))
    scaler = joblib.load(find_file("scaler.pkl"))
    try:
        with open(find_file("model_config.json")) as f:
            type_map = json.load(f)["type_map"]
    except FileNotFoundError:
        type_map = DEFAULT_TYPE_MAP
    return model, scaler, type_map


model, scaler, TYPE_MAP = load_artifacts()

st.title("Predictive Maintenance Dashboard")
st.markdown("Enter machine sensor readings in the sidebar to predict failure risk.")
st.divider()

st.sidebar.header("Machine Sensor Inputs")

TYPE_LABELS = {"L (Low quality)": "L", "M (Medium quality)": "M", "H (High quality)": "H"}
machine_type = st.sidebar.selectbox("Machine Type", list(TYPE_LABELS))

# Ranges cover the full training data (ai4i2020.csv)
air_temp  = st.sidebar.slider("Air Temperature (K)",     295.0, 305.0, 300.0, 0.1)
proc_temp = st.sidebar.slider("Process Temperature (K)", 305.0, 315.0, 310.0, 0.1)
rpm       = st.sidebar.slider("Rotational Speed (rpm)",  1100,  2900,  1500)
torque    = st.sidebar.slider("Torque (Nm)",             3.0,   80.0,  40.0, 0.5)
tool_wear = st.sidebar.slider("Tool Wear (minutes)",     0,     260,   100)

# Feature engineering - MUST match the training notebook exactly
temp_diff   = proc_temp - air_temp
power       = torque * rpm * (2 * np.pi / 60)
wear_torque = tool_wear * torque
wear_rate   = tool_wear / (rpm + 1)

input_df = pd.DataFrame([[
    TYPE_MAP[TYPE_LABELS[machine_type]], air_temp, proc_temp,
    rpm, torque, tool_wear,
    temp_diff, power, wear_torque, wear_rate,
]], columns=FEATURES)

input_scaled = scaler.transform(input_df)
probability  = model.predict_proba(input_scaled)[0][1] * 100

# One set of thresholds for everything on the page, so the messages never disagree.
# 50% is the same cut-off model.predict() uses.
if probability >= 50:
    status, risk = "FAILURE PREDICTED", "high"
elif probability >= 25:
    status, risk = "OPERATING NORMALLY", "medium"
else:
    status, risk = "OPERATING NORMALLY", "low"

col1, col2, col3 = st.columns([2, 2, 3])
with col1:
    if risk == "high":
        st.error(status)
    else:
        st.success(status)
with col2:
    st.metric("Failure Probability", f"{probability:.1f}%")
with col3:
    st.progress(int(probability))

st.divider()

if risk == "high":
    st.error(f"HIGH RISK ({probability:.1f}%) - Schedule maintenance immediately!")
elif risk == "medium":
    st.warning(f"MEDIUM RISK ({probability:.1f}%) - Monitor closely.")
else:
    st.success(f"LOW RISK ({probability:.1f}%) - Machine is healthy.")

st.subheader("Input Features Sent to Model")
st.dataframe(input_df.T.rename(columns={0: "Value"}))
