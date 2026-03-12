import pandas as pd
import numpy as np
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("data/food_waste.csv")

X = df.drop("Food_Wasted", axis=1)
y = df["Food_Wasted"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("📊 Model Performance")
print("MAE :", mae)
print("RMSE:", rmse)
print("R2  :", r2)

os.makedirs("model", exist_ok=True)
with open("model/rf_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("✅ Random Forest model saved successfully!")