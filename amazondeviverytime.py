# amazon_delivery_training.py
# Mitali Shirdhakar
#Data Science
#Problem Statement:
""""This project aims to predict delivery times for e-commerce orders based on a variety of factors such as product size, distance, traffic conditions, and shipping method. Using the provided dataset, learners will preprocess, analyze, and build regression models to accurately estimate delivery times. The final application will allow users to input relevant details and receive estimated delivery times via a user-friendly interface."""
import pandas as pd
import numpy as np
import math
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
import mlflow
import mlflow.sklearn


# -----------------------
# 1. Load Data
# -----------------------
df = pd.read_csv("C:/Users/Mitali/OneDrive/Desktop/amazon_delivery.csv")

# Fill missing categorical values
df["Agent_Rating"] = df["Agent_Rating"].ffill()
df["Weather"] = df["Weather"].ffill()

# Fill missing numeric values with mean
df = df.fillna(df.mean(numeric_only=True))

# -----------------------
# 2. Haversine Distance
# -----------------------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

df["Known_Distance_km"] = df.apply(
    lambda row: haversine(row["Store_Latitude"], row["Store_Longitude"],
                          row["Drop_Latitude"], row["Drop_Longitude"]), axis=1
)

# -----------------------
# 3. Convert Times
# -----------------------
df['Order_Date'] = pd.to_datetime(df['Order_Date'], errors='coerce', format='%Y-%m-%d')
df['Order_Time'] = pd.to_datetime(df['Order_Time'], errors='coerce', format='%H:%M:%S').dt.hour + \
                   pd.to_datetime(df['Order_Time'], errors='coerce', format='%H:%M:%S').dt.minute/60
df['Pickup_Time'] = pd.to_datetime(df['Pickup_Time'], errors='coerce', format='%H:%M:%S').dt.hour + \
                    pd.to_datetime(df['Pickup_Time'], errors='coerce', format='%H:%M:%S').dt.minute/60

# -----------------------
# 4. Encode Categorical Columns
# -----------------------
encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    if col != 'Order_ID':  # skip ID
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

# -----------------------
# 5. Features & Target
# -----------------------
X = df.drop(columns=['Delivery_Time', 'Order_ID', 'Order_Date','Known_Distance_km' ,'Store_Latitude', 'Store_Longitude', 'Drop_Latitude', 'Drop_Longitude'])
y = df['Delivery_Time']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Handle missing values
imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# -----------------------
# 6. Train Models
# -----------------------
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(random_state=42, n_estimators=100),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42)
}

results = {}
best_model = None
best_score = -np.inf

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results[name] = {"RMSE": rmse, "MAE": mae, "R²": r2}

    if r2 > best_score:
        best_score = r2
        best_model = model

results_df = pd.DataFrame(results).T
print("\nModel Performance Comparison:\n")
print(results_df)

print(f"\n✅ Best Model: {best_model.__class__.__name__} with R² = {best_score:.3f}")

# -----------------------
# 7. Save Best Model & Encoders
# -----------------------
joblib.dump(best_model, "trained_model.pkl")
joblib.dump(encoders, "label_encoders.pkl")
print("\n✅ Model and encoders saved successfully!")
mlflow.set_experiment("Amazon_Delivery_Time_Prediction")

for name, model in models.items():
    with mlflow.start_run(run_name=name):
        # Train model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Log metrics
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("R2", r2)

        # Log parameters (example for Random Forest)
        if hasattr(model, "n_estimators"):
            mlflow.log_param("n_estimators", model.n_estimators)
        if hasattr(model, "max_depth"):
            mlflow.log_param("max_depth", model.max_depth)

        # Log model
        mlflow.sklearn.log_model(model, artifact_path="model")

        # Update best model tracking
        if r2 > best_score:
            best_score = r2
            best_model = model
mlflow.log_artifact("label_encoders.pkl")

