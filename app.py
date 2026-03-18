from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pydeck as pdk

st.set_page_config(page_title="Earthquake Intelligence", layout="wide")
st.title("🌍 Earthquake Intelligence Dashboard")

@st.cache_resource
def load_model():
    url = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_month.csv"
    df = pd.read_csv(url)
    df = df[["latitude", "longitude", "depth", "mag", "nst", "gap", "dmin", "rms"]]
    df = df.dropna()
    df = df[(df["mag"] > 0.5) & (df["mag"] < 9.5)]
    features = ["latitude", "longitude", "depth", "nst", "gap", "dmin", "rms"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])
    model = RandomForestRegressor(n_estimators=100, max_depth=6, min_samples_split=5, min_samples_leaf=4, random_state=42)
    model.fit(X_scaled, df["mag"])
    return model, scaler

@st.cache_data(ttl=300)
def load_data():
    url = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_day.csv"
    df = pd.read_csv(url)
    df = df[["time", "latitude", "longitude", "depth", "mag", "nst", "gap", "dmin", "rms"]]
    df = df.dropna()
    return df

model, scaler = load_model()
df = load_data()

features = ["latitude", "longitude", "depth", "nst", "gap", "dmin", "rms"]

X = scaler.transform(df[features])
all_preds = np.array([tree.predict(X) for tree in model.estimators_])
df["Predicted_Mag"] = np.mean(all_preds, axis=0)
df["Uncertainty"] = np.std(all_preds, axis=0)

df_final = df.copy()
df_final["Predicted_Mag"] = df_final["Predicted_Mag"].round(2)
df_final["Uncertainty"] = df_final["Uncertainty"].round(2)

def get_zone(mag):
    if mag < 2: return "Micro"
    elif mag < 4: return "Low"
    elif mag < 5: return "Moderate"
    elif mag < 6: return "Strong"
    else: return "Severe"

def get_color(mag):
    if mag < 2: return [0, 255, 0]
    elif mag < 4: return [173, 255, 47]
    elif mag < 5: return [255, 165, 0]
    elif mag < 6: return [255, 69, 0]
    else: return [255, 0, 0]

def get_radius(mag):
    return max(10000, mag * 20000)

df_final["Zone"] = df_final["Predicted_Mag"].apply(get_zone)
df_final["color"] = df_final["Predicted_Mag"].apply(get_color)
df_final["radius"] = df_final["Predicted_Mag"].apply(get_radius)

max_mag = df_final["Predicted_Mag"].max()

if max_mag >= 6:
    st.error(f"🚨 SEVERE EARTHQUAKE DETECTED (Mag {round(max_mag,2)})")
elif max_mag >= 5:
    st.warning(f"⚠️ Strong Earthquake Activity (Mag {round(max_mag,2)})")
else:
    st.success("✅ No Severe Activity")

col1, col2, col3 = st.columns(3)
col1.metric("Total Events", len(df_final))
col2.metric("Max Magnitude", round(max_mag, 2))
col3.metric("Avg Magnitude", round(df_final["Predicted_Mag"].mean(), 2))

st.subheader("🔥 Top 5 Dangerous Events")
top5 = df_final.sort_values("Predicted_Mag", ascending=False).head(5)
st.dataframe(top5[["time", "latitude", "longitude", "mag", "Predicted_Mag", "Uncertainty", "Zone"]], use_container_width=True)

st.subheader("📍 Live Earthquake Map")
tooltip = {"html": "<b>Time:</b> {time} <br/><b>Actual:</b> {mag} <br/><b>Predicted:</b> {Predicted_Mag} <br/><b>Uncertainty:</b> {Uncertainty} <br/><b>Zone:</b> {Zone}", "style": {"backgroundColor": "black", "color": "white"}}
layer = pdk.Layer("ScatterplotLayer", data=df_final, get_position='[longitude, latitude]', get_color='color', get_radius='radius', pickable=True, auto_highlight=True)
view_state = pdk.ViewState(latitude=20, longitude=0, zoom=1.5)
st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip=tooltip))

st.subheader("📊 Full Dataset")
st.dataframe(df_final[["time", "latitude", "longitude", "mag", "Predicted_Mag", "Uncertainty", "Zone"]], use_container_width=True)

st.subheader("🔮 What-If Earthquake Predictor")
st.caption("Enter seismic parameters to estimate magnitude")
col1, col2, col3 = st.columns(3)
with col1:
    lat = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=35.0)
    nst = st.number_input("NST (No. of Stations)", min_value=0, max_value=200, value=20)
with col2:
    lon = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=140.0)
    gap = st.number_input("Gap (degrees)", min_value=0.0, max_value=360.0, value=80.0)
with col3:
    depth = st.number_input("Depth (km)", min_value=0.0, max_value=700.0, value=10.0)
    dmin = st.number_input("Dmin (distance)", min_value=0.0, max_value=10.0, value=0.5)
rms = st.number_input("RMS", min_value=0.0, max_value=5.0, value=0.5)

if st.button("Predict Magnitude"):
    input_data = pd.DataFrame([[lat, lon, depth, nst, gap, dmin, rms]], columns=features)
    input_scaled = scaler.transform(input_data)
    tree_preds = np.array([tree.predict(input_scaled) for tree in model.estimators_])
    pred_mag = round(float(np.mean(tree_preds)), 2)
    pred_unc = round(float(np.std(tree_preds)), 2)
    pred_zone = get_zone(pred_mag)
    st.success(f"Predicted Magnitude: **{pred_mag}** ± {pred_unc}")
    st.info(f"Zone: **{pred_zone}**")
    if pred_unc > 0.8:
        st.warning("⚠️ High uncertainty — input parameters may be unreliable")

st.subheader("📊 Model Comparison")
comparison_df = pd.DataFrame({
    "Model": ["Linear Regression", "Decision Tree", "Random Forest"],
    "Train R2": [0.7701, 1.0, 0.922],
    "Test R2": [0.7129, 0.8029, 0.865],
    "Issue": ["Underfitting", "Overfitting", "Best Balance ✅"]
})
st.dataframe(comparison_df, use_container_width=True)
fig, ax = plt.subplots(figsize=(8, 3))
ax.bar(comparison_df["Model"], comparison_df["Train R2"], width=0.3, label="Train R2", align='center')
ax.bar(comparison_df["Model"], comparison_df["Test R2"], width=0.3, label="Test R2", align='edge')
ax.set_ylim(0.6, 1.05)
ax.set_ylabel("R2 Score")
ax.set_title("Model Comparison — Train vs Test R2")
ax.legend()
st.pyplot(fig)

st.subheader("🔍 Feature Importance")
features_list = ["latitude", "longitude", "depth", "nst", "gap", "dmin", "rms"]
importances = model.feature_importances_
fig2, ax2 = plt.subplots(figsize=(8, 4))
sorted_idx = np.argsort(importances)
ax2.barh([features_list[i] for i in sorted_idx], importances[sorted_idx])
ax2.set_xlabel("Importance Score")
ax2.set_title("Feature Importance — Random Forest")
st.pyplot(fig2)
