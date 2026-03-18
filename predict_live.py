import pandas as pd
import numpy as np
import joblib

# ------------------ LOAD ------------------
model = joblib.load("final_model_no_region.pkl")
scaler = joblib.load("scaler.pkl")

# ------------------ FETCH DATA ------------------
url = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_day.csv"
df = pd.read_csv(url)

# ------------------ SELECT ------------------
df = df[["time", "latitude", "longitude", "depth", "mag", "nst", "gap", "dmin", "rms"]]
df = df.dropna()

# ------------------ MODEL ------------------
features = ["latitude", "longitude", "depth", "nst", "gap", "dmin", "rms"]

X_scaled = scaler.transform(df[features])
all_tree_preds = np.array([tree.predict(X_scaled) for tree in model.estimators_])

df["Predicted_Mag"] = np.mean(all_tree_preds, axis=0).round(2)
df["Uncertainty"] = np.std(all_tree_preds, axis=0).round(2)

df_final = df.copy()

# ------------------ ZONE ------------------
def get_zone(mag):
    if mag < 2:
        return "Micro (Very Low)"
    elif mag < 4:
        return "Zone 1 (Low)"
    elif mag < 5:
        return "Zone 2 (Moderate)"
    elif mag < 6:
        return "Zone 3 (Strong)"
    else:
        return "Zone 4 (Severe)"

df_final["Zone"] = df_final["Predicted_Mag"].apply(get_zone)
df_final = df_final.sort_index()

# ------------------ OUTPUT ------------------
print(df_final[[
    "time", "latitude", "longitude",
    "mag", "Predicted_Mag", "Uncertainty", "Zone"
]].head(10))