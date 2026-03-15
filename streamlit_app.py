import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Predictive Maintenance", page_icon="machine", layout="wide")

@st.cache_resource
def load_artifacts():
    model  = joblib.load("models/best_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    return model, scaler

model, scaler = load_artifacts()

st.title("Predictive Maintenance Dashboard")
st.markdown("Enter machine sensor readings in the sidebar to predict failure risk.")
st.divider()

st.sidebar.header("Machine Sensor Inputs")

machine_type = st.sidebar.selectbox("Machine Type", ["L (Low quality)", "M (Medium quality)", "H (High quality)"])
type_map = {"L (Low quality)": 0, "M (Medium quality)": 1, "H (High quality)": 2}

air_temp  = st.sidebar.slider("Air Temperature (K)",       295.0, 305.0, 300.0, 0.1)
proc_temp = st.sidebar.slider("Process Temperature (K)",   305.0, 315.0, 310.0, 0.1)
rpm       = st.sidebar.slider("Rotational Speed (rpm)",    1000,  2500,  1500)
torque    = st.sidebar.slider("Torque (Nm)",               10.0,  70.0,  40.0, 0.5)
tool_wear = st.sidebar.slider("Tool Wear (minutes)",       0,     250,   100)

# Feature engineering - MUST match the training notebook exactly
temp_diff   = proc_temp - air_temp
power       = torque * rpm * (2 * np.pi / 60)
wear_torque = tool_wear * torque
wear_rate   = tool_wear / (rpm + 1)

input_df = pd.DataFrame([[
    type_map[machine_type], air_temp, proc_temp,
    rpm, torque, tool_wear,
    temp_diff, power, wear_torque, wear_rate
]], columns=[
    "Type", "Air temperature [K]", "Process temperature [K]",
    "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]",
    "temp_diff", "power", "wear_torque", "wear_rate"
])

input_scaled = scaler.transform(input_df)
prediction   = model.predict(input_scaled)[0]
probability  = model.predict_proba(input_scaled)[0][1] * 100

col1, col2, col3 = st.columns([2, 2, 3])
with col1:
    if prediction == 1:
        st.error("FAILURE PREDICTED")
    else:
        st.success("OPERATING NORMALLY")
with col2:
    st.metric("Failure Probability", f"{probability:.1f}%")
with col3:
    st.progress(int(probability))

st.divider()

if probability >= 70:
    st.error(f"HIGH RISK ({probability:.1f}%) - Schedule maintenance immediately!")
elif probability >= 40:
    st.warning(f"MEDIUM RISK ({probability:.1f}%) - Monitor closely.")
else:
    st.success(f"LOW RISK ({probability:.1f}%) - Machine is healthy.")

st.subheader("Input Features Sent to Model")
st.dataframe(input_df.T.rename(columns={0: "Value"}), use_container_width=True)

