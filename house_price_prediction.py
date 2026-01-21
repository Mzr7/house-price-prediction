import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -----------------------------
# 1. Load Dataset
# -----------------------------
df = pd.read_csv("data/housing.csv")

print("Dataset Loaded Successfully")
print(df.head())
print(df.info())

# -----------------------------
# 2. Handle Missing Values
# -----------------------------
df.fillna(df.median(numeric_only=True), inplace=True)

# -----------------------------
# 3. Feature Engineering
# -----------------------------
df["rooms_per_household"] = df["total_rooms"] / df["households"]
df["bedrooms_per_room"] = df["total_bedrooms"] / df["total_rooms"]
df["population_per_household"] = df["population"] / df["households"]

# -----------------------------
# 4. Encode Categorical Feature
# -----------------------------
df = pd.get_dummies(df, drop_first=True)

# -----------------------------
# 5. Split Features & Target
# -----------------------------
X = df.drop("median_house_value", axis=1)
y = df["median_house_value"]

# -----------------------------
# 6. Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 7. Feature Scaling
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# 8. Train Ridge Regression Model
# -----------------------------
model = Ridge(alpha=1.0)
model.fit(X_train_scaled, y_train)

# -----------------------------
# 9. Predictions
# -----------------------------
y_pred = model.predict(X_test_scaled)

# -----------------------------
# 10. Model Evaluation
# -----------------------------
print("\nModel Performance:")
print("MAE :", mean_absolute_error(y_test, y_pred))
print("MSE :", mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2 Score:", r2_score(y_test, y_pred))

# -----------------------------
# 11. Visualization
# -----------------------------
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.xlabel("Actual House Price")
plt.ylabel("Predicted House Price")
plt.title("Actual vs Predicted House Prices")
plt.grid(True)
plt.show()

