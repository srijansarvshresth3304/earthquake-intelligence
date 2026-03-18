# import pandas as pd
# import numpy as np

# from sklearn.model_selection import RandomizedSearchCV
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import r2_score

# from sklearn.ensemble import RandomForestRegressor

# # ------------------ LOAD ------------------
# df = pd.read_csv("all_month.csv")

# # ------------------ SELECT ------------------

# # ------------------ CLEAN ------------------
# df = df[["time", "latitude", "longitude", "depth", "mag", "nst", "gap", "dmin", "rms"]]
# df = df.dropna()
# # Nayi line
# df = df[(df["mag"] > 0.5) & (df["mag"] < 9.5)]

# # ------------------ SORT BY TIME ------------------
# df["time"] = pd.to_datetime(df["time"])
# df = df.sort_values("time")

# # ------------------ SPLIT ------------------
# split_index = int(len(df) * 0.8)

# train_df = df.iloc[:split_index].copy()
# test_df = df.iloc[split_index:].copy()

# features = ["latitude", "longitude", "depth", "nst", "gap", "dmin", "rms"]

# X_train = train_df[features]
# y_train = train_df["mag"]

# X_test = test_df[features]
# y_test = test_df["mag"]

# # ------------------ SCALING ------------------
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# # ------------------ MODEL ------------------
# param_dist = {
#     "n_estimators": [50, 100, 150],
#     "max_depth": [5, 6, 8],
#     "min_samples_split": [5, 10],
#     "min_samples_leaf": [4, 6]
# }

# rf = RandomForestRegressor(random_state=42)

# search = RandomizedSearchCV(
#     rf,
#     param_distributions=param_dist,
#     n_iter=10,
#     scoring="r2",
#     cv=3,
#     random_state=42,
#     n_jobs=-1
# )

# # ================== NORMAL TRAIN ==================
# search.fit(X_train_scaled, y_train)
# model = search.best_estimator_

# train_preds = model.predict(X_train_scaled)
# test_preds = model.predict(X_test_scaled)

# print("\n===== NORMAL MODEL =====")
# print("Train R2:", round(r2_score(y_train, train_preds), 4))
# print("Test R2:", round(r2_score(y_test, test_preds), 4))

# # ================== SHUFFLE TEST ==================
# y_train_shuffled = y_train.sample(frac=1, random_state=42).reset_index(drop=True)

# search.fit(X_train_scaled, y_train_shuffled)
# model_fake = search.best_estimator_

# fake_preds = model_fake.predict(X_test_scaled)

# print("\n===== SHUFFLE TEST =====")
# print("Test R2 (SHUFFLED):", round(r2_score(y_test, fake_preds), 4))

# import joblib
# joblib.dump(model, "models/final_model_no_region.pkl")
# joblib.dump(scaler, "models/scaler.pkl")

import sys
print(sys.version)