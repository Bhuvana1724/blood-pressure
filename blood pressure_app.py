import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load your dataset (replace with actual path or source)
# Example assumes CSV with columns like: 'age', 'weight', 'height', etc.
data = pd.read_csv('blood_pressure_data.csv')

# Step 2: Define features and target
features = ['age', 'weight', 'height', 'bmi', 'cholesterol', 'glucose']  # adjust as needed
target_sys = 'systolic_bp'
target_dia = 'diastolic_bp'

X = data[features]
y_sys = data[target_sys]
y_dia = data[target_dia]

# Step 3: Split into train and test sets
X_train, X_test, y_train_sys, y_test_sys = train_test_split(X, y_sys, test_size=0.2, random_state=42)
_, _, y_train_dia, y_test_dia = train_test_split(X, y_dia, test_size=0.2, random_state=42)

# Step 4: Train model
model_sys = RandomForestRegressor(n_estimators=100, random_state=42)
model_dia = RandomForestRegressor(n_estimators=100, random_state=42)

model_sys.fit(X_train, y_train_sys)
model_dia.fit(X_train, y_train_dia)

# Step 5: Predict and evaluate
y_pred_sys = model_sys.predict(X_test)
y_pred_dia = model_dia.predict(X_test)

print("Systolic BP - R2 Score:", r2_score(y_test_sys, y_pred_sys))
print("Diastolic BP - R2 Score:", r2_score(y_test_dia, y_pred_dia))
print("Systolic BP - RMSE:", mean_squared_error(y_test_sys, y_pred_sys, squared=False))
print("Diastolic BP - RMSE:", mean_squared_error(y_test_dia, y_pred_dia, squared=False))
