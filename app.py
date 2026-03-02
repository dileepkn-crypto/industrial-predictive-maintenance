import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Safe matplotlib import
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except:
    MATPLOTLIB_AVAILABLE = False

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("ai4i2020.csv")

df = df.drop(['UDI', 'Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'], axis=1)
df['Type'] = df['Type'].map({'L': 0, 'M': 1, 'H': 2})

features = [
    'Type',
    'Air temperature [K]',
    'Process temperature [K]',
    'Rotational speed [rpm]',
    'Torque [Nm]',
    'Tool wear [min]'
]

X = df[features]
y = df['Machine failure']

# -----------------------------
# Model
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestClassifier(n_estimators=150, max_depth=10)
model.fit(X_scaled, y)

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Industrial AI Dashboard", layout="wide")
st.title("🏭 Industrial Predictive Maintenance & Recommendation System")

st.sidebar.header("Enter Machine Parameters")

machine_type = st.sidebar.selectbox("Machine Type", ["L", "M", "H"])
machine_map = {"L": 0, "M": 1, "H": 2}

air_temp = st.sidebar.number_input("Air Temperature (K)", value=300.0)
process_temp = st.sidebar.number_input("Process Temperature (K)", value=310.0)
speed = st.sidebar.number_input("Rotational Speed (rpm)", value=1500.0)
torque = st.sidebar.number_input("Torque (Nm)", value=40.0)
wear = st.sidebar.number_input("Tool Wear (min)", value=100.0)

# -----------------------------
# Button
# -----------------------------
if st.button("🔍 Analyze Machine"):

    sample = np.array([
        machine_map[machine_type],
        air_temp, process_temp, speed, torque, wear
    ]).reshape(1, -1)

    sample_scaled = scaler.transform(sample)

    prediction = model.predict(sample_scaled)[0]
    prob = model.predict_proba(sample_scaled)[0][1] * 100

    # -----------------------------
    # STATUS LOGIC
    # -----------------------------
    if prediction == 1 or (air_temp > 320 and process_temp > 330):
        status = "CRITICAL"
        priority = "HIGH"
    elif (air_temp > 310 or process_temp > 320 or wear > 150 or torque > 50):
        status = "WARNING"
        priority = "MEDIUM"
    else:
        status = "NORMAL"
        priority = "LOW"

    # -----------------------------
    # ✅ IMPROVED RECOMMENDATION LOGIC
    # -----------------------------
    actions = []

    if status == "CRITICAL":

        if wear > 200:
            actions.append("Replace tool immediately")

        if torque > 60:
            actions.append("Check bearing and motor load")

        if air_temp > 320:
            actions.append("Inspect cooling system")

        if speed < 1200:
            actions.append("Check motor efficiency")

        if not actions:
            actions.append("General emergency maintenance required")

    elif status == "WARNING":

        if wear > 150:
            actions.append("Schedule tool replacement")

        if air_temp > 310:
            actions.append("Monitor temperature regularly")

        if torque > 50:
            actions.append("Check load conditions")

        if not actions:
            actions.append("Preventive maintenance required")

    else:
        actions.append("Machine operating normally")

    # -----------------------------
    #  RANKING SYSTEM
    # -----------------------------
    scores = {}

    for act in actions:
        score = 0

        if "tool" in act:
            score += wear / 300

        if "bearing" in act:
            score += torque / 100

        if "cooling" in act:
            score += air_temp / 350

        if "motor" in act:
            score += (1500 - speed) / 1500

        scores[act] = score

    ranked = sorted(actions, key=lambda x: scores[x], reverse=True)

    # -----------------------------
    # OUTPUT
    # -----------------------------
    st.subheader("📊 Machine Health Analysis Report")

    col1, col2, col3 = st.columns(3)
    col1.metric("Failure Probability", f"{prob:.2f}%")
    col2.metric("Maintenance Priority", priority)
    col3.metric("Machine Status", status)

    if status == "CRITICAL":
        st.error("🚨 FAILURE DETECTED – IMMEDIATE ACTION REQUIRED")
    elif status == "WARNING":
        st.warning("⚠️ MAINTENANCE REQUIRED SOON")
    else:
        st.success("✅ SYSTEM NORMAL")

    # -----------------------------
    # TEMPERATURE VISUALIZATION
    # -----------------------------
    st.subheader("🌡️ Temperature Monitoring")

    st.write("Air Temperature Level")
    st.progress(int(min((air_temp / 350) * 100, 100)))

    st.write("Process Temperature Level")
    st.progress(int(min((process_temp / 350) * 100, 100)))

    # -----------------------------
    # RECOMMENDATIONS
    # -----------------------------
    st.subheader("🔧 Recommended Maintenance Actions")

    report_text = f"""
Machine Status: {status}
Failure Probability: {prob:.2f}%
Priority: {priority}

Recommendations:
"""

    for i, act in enumerate(ranked, 1):
        st.write(f"{i}. {act} (Score: {scores[act]:.2f})")
        report_text += f"{i}. {act}\n"

    # -----------------------------
    # FEATURE IMPORTANCE
    # -----------------------------
    if MATPLOTLIB_AVAILABLE:
        st.subheader("📈 Key Factors Influencing Machine Condition")
        importance = model.feature_importances_
        fig, ax = plt.subplots()
        ax.barh(features, importance)
        st.pyplot(fig)

    # -----------------------------
    # DOWNLOAD REPORT
    # -----------------------------
    st.download_button(
        "📥 Download Report",
        data=report_text,
        file_name="maintenance_report.txt"
    )