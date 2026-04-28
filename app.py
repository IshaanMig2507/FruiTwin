import streamlit as st
import pandas as pd
import joblib
# import shap
# import matplotlib.pyplot as plt

st.markdown("""
<style>
.card {
    background-color: #1c1f26;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.3);
}
.big-font {
    font-size: 28px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# LOAD MODELS
# -----------------------------
xgb_model = joblib.load("xgb_model.pkl")
rf_reg = joblib.load("rf_reg.pkl")

# -----------------------------
# PAGE CONFIG + DARK MODE
# -----------------------------
st.set_page_config(page_title="Fruit Spoilage Predictor", layout="centered")

st.markdown("""
<style>
.stApp {background-color: #0e1117; color: white;}
.stMetric {background-color: #1c1f26; padding: 10px; border-radius: 10px;}
.stButton>button {background-color: #ff4b4b; color: white; border-radius: 8px;}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# HEADER
# -----------------------------
st.markdown("""
<h1 style='text-align: center; color: #ff4b4b; margin-bottom: 0;'>
FruiTwin
</h1>

<p style='text-align: center; font-size:18px; margin-top: 5px;'>
Digital twin intelligence for cold chain optimization
</p>

<p style='text-align: center; font-size:16px; color:#cccccc; margin-top:10px;'>
Predict, simulate, and optimize perishable supply chains with data-driven insights.
</p>
""", unsafe_allow_html=True)

st.markdown("---")
# -----------------------------
# INPUTS (2 COLUMN UI)
# -----------------------------
st.subheader("▶ Shipment Inputs")

col1, col2 = st.columns(2)

with col1:
    
    avg_temp = st.slider("🌡 Avg Temperature (°C)", 0, 40, 20)
    max_temp = st.slider("🌡 Max Temperature (°C)", 0, 45, 25)
    min_temp = st.slider("🌡 Min Temperature (°C)", 0, 35, 15)

    humidity = st.slider("Humidity (%)", 20, 100, 60)
    transit_stress = st.slider("Transit Stress", 0.0, 1.5, 0.3)

with col2:
    delay = st.slider("Delay Hours", 0.0, 10.0, 2.0)
    delay_ratio = st.slider("Delay Ratio", 0.0, 2.0, 0.5)

    refrigeration = st.selectbox("Refrigeration", [0, 1])
    failures = st.slider("Refrigeration Failures", 0, 3, 0)

    damage = st.slider("Damage Incidents", 0, 4, 1)

st.markdown("---")

# -----------------------------
# CREATE INPUT DATA
# -----------------------------
input_data = pd.DataFrame({
    "avg_temperature": [avg_temp],
    "max_temperature": [max_temp],
    "min_temperature": [min_temp],
    "delay_hours": [delay],
    "delay_ratio": [delay_ratio],
    "refrigeration_on": [refrigeration],
    "refrigeration_failures": [failures],
    "damage_incidents": [damage],
    "humidity_avg": [humidity],
    "transit_stress_ratio": [transit_stress]
})

# -----------------------------
# FEATURE HANDLING
# -----------------------------
xgb_cols = xgb_model.named_steps["preprocessing"].feature_names_in_
rf_cols = rf_reg.named_steps["preprocessor"].feature_names_in_

all_cols = list(set(xgb_cols).union(set(rf_cols)))

for col in all_cols:
    if col not in input_data.columns:
        input_data[col] = 0

# derived feature
input_data["cold_chain_effectiveness"] = (
    input_data["refrigeration_on"]
    - 0.3 * input_data["refrigeration_failures"]
).clip(0, 1)

input_xgb = input_data[xgb_cols]
input_rf = input_data[rf_cols]

# -----------------------------
# RECOMMENDATION FUNCTION
# -----------------------------
def generate_recommendations(df):
    recs = []

    if df["avg_temperature"].iloc[0] > 25:
        recs.append("Reduce temperature (strongest impact)")

    if df["refrigeration_on"].iloc[0] == 0:
        recs.append("Enable refrigeration")

    if df["refrigeration_failures"].iloc[0] > 0:
        recs.append("Fix refrigeration failures")

    if df["delay_hours"].iloc[0] > 4:
        recs.append("Reduce delays")

    if df["damage_incidents"].iloc[0] > 1:
        recs.append("Improve packaging")

    return recs

# -----------------------------
# PREDICTION
# -----------------------------
if st.button("Run Prediction"):

    spoilage_prob = xgb_model.predict_proba(input_xgb)[0][1]
    shelf_life = rf_reg.predict(input_rf)[0]

    st.markdown("## ▶ Prediction Results")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        <div class="card">
            <p>Spoilage Risk</p>
            <p class="big-font">{spoilage_prob:.2f}</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="card">
            <p>Shelf Life (days)</p>
            <p class="big-font">{shelf_life:.2f}</p>
        </div>
        """, unsafe_allow_html=True)

    # -----------------------------

    st.progress(float(spoilage_prob))

    if spoilage_prob > 0.7:
        st.error("✖ High Risk — Immediate action required")
    elif spoilage_prob > 0.4:
        st.warning("⚠ Moderate Risk — Needs monitoring")
    else:
        st.success("✔ Low Risk — Safe shipment")
    # RECOMMENDATIONS
    # -----------------------------
    st.markdown("## ▶ AI Recommendations")

    recs = generate_recommendations(input_data)

    for r in recs:
        st.markdown(f"""
        <div style='background:#262730;padding:10px;border-radius:8px;margin:5px 0;'>
            ▸{r}
        </div>
        """, unsafe_allow_html=True)

    # -----------------------------
    # SHAP EXPLANATION
    # # -----------------------------
    # st.markdown("### 🧠 Model Explanation")

    # xgb = xgb_model.named_steps["classifier"]
    # preprocessor = xgb_model.named_steps["preprocessing"]

    # transformed = preprocessor.transform(input_xgb)

    # explainer = shap.Explainer(xgb)
    # shap_values = explainer(transformed)

    # fig = plt.figure()
    # shap.plots.waterfall(shap_values[0], show=False)
    # st.pyplot(fig)

# -----------------------------
# SCENARIO SIMULATION
# -----------------------------
st.markdown("---")
st.subheader("▶ Scenario Simulation")

if st.button("Improve Cold Chain"):

    sim = input_data.copy()

    sim["avg_temperature"] -= 5
    sim["max_temperature"] -= 5
    sim["min_temperature"] -= 5

    sim["refrigeration_on"] = 1
    sim["refrigeration_failures"] = 0

    sim["cold_chain_effectiveness"] = (
        sim["refrigeration_on"]
        - 0.3 * sim["refrigeration_failures"]
    ).clip(0, 1)

    sim_xgb = sim[xgb_cols]
    sim_rf = sim[rf_cols]

    new_prob = xgb_model.predict_proba(sim_xgb)[0][1]
    new_shelf = rf_reg.predict(sim_rf)[0]

    st.markdown("## ▶ After Improvement")

    colC, colD = st.columns(2)

    with colC:
        st.metric("New Spoilage Risk", f"{new_prob:.2f}")

    with colD:
        st.metric("New Shelf Life", f"{new_shelf:.2f}")

    # -----------------------------
    # CHART
    # -----------------------------
    # -----------------------------
    # CHART
    # -----------------------------
    st.markdown("### ▶ Impact Comparison")

    # recompute ORIGINAL values (important fix)
    orig_prob = xgb_model.predict_proba(input_xgb)[0][1]
    orig_shelf = rf_reg.predict(input_rf)[0]

    chart_df = pd.DataFrame({
        "Scenario": ["Original", "Improved"],
        "Spoilage Risk": [orig_prob, new_prob],
        "Shelf Life": [orig_shelf, new_shelf]
    }).set_index("Scenario")

    st.line_chart(chart_df)

    ##side by side comparision

    st.markdown("## ▶ Before vs After")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Original Risk", f"{orig_prob:.2f}")
        st.metric("Original Shelf Life", f"{orig_shelf:.2f}")

    with col2:
        st.metric("Improved Risk", f"{new_prob:.2f}")
        st.metric("Improved Shelf Life", f"{new_shelf:.2f}")

    st.success("Cold-chain optimization improves outcomes!")

    ##top 3 rec
    def get_top_actions(input_data):
        actions = []

        temp = input_data["avg_temperature"].iloc[0]
        ref = input_data["refrigeration_on"].iloc[0]
        fail = input_data["refrigeration_failures"].iloc[0]
        delay = input_data["delay_hours"].iloc[0]
        damage = input_data["damage_incidents"].iloc[0]

        # assign impact scores
        if temp > 25:
            actions.append(("Reduce temperature", 5))

        if ref == 0:
            actions.append(("Enable refrigeration", 4))

        if fail > 0:
            actions.append(("Fix refrigeration failures", 4))

        if delay > 4:
            actions.append(("Reduce delays", 3))

        if damage > 1:
            actions.append(("Improve packaging", 2))

        # sort by importance
        actions = sorted(actions, key=lambda x: x[1], reverse=True)

        return [a[0] for a in actions[:3]]

    st.markdown("## ▶ Top Actions to Reduce Spoilage")

    top_actions = get_top_actions(input_data)

    for i, action in enumerate(top_actions, 1):
        st.markdown(f"""
        <div style='background:#262730;padding:12px;border-radius:10px;margin:6px 0;'>
            <b>{i}.</b> {action}
        </div>
        """, unsafe_allow_html=True)



    ##FOOTER
    st.markdown("---")
    st.markdown(
        "<p style='text-align:center; color:#aaaaaa;'>FruiTwin ❤️ From data to decisions</p>",
        unsafe_allow_html=True
    )
    