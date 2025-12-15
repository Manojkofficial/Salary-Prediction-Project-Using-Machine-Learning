import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

data = pd.read_csv(r"D:\SEM-5\in-234\CA-2\Salary Data.csv")
print("Top 4 rows of the dataset:")
print(data.head(4))
print(f"\nDataset shape -> Rows: {data.shape[0]}, Columns: {data.shape[1]}")
print("\nColumn data types:")
print(data.dtypes)

print("\nSum of missing values with corresponding columns:")
print(data.isna().sum())

print("\nBasic statistics for numeric columns:")
print(data.describe())
data = data.dropna(axis=1, how="all")
data.columns = data.columns.str.strip().str.replace(" ", "_")

numeric_cols = data.select_dtypes(include=["int64", "float64"]).columns
categorical_cols = data.select_dtypes(include=["object"]).columns

data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())
data[categorical_cols] = data[categorical_cols].fillna("Unknown")
print("\nMissing values after cleaning:")
print(data.isna().sum())
X = data.drop("Salary", axis=1)
y = data["Salary"]
print("\nFeature columns:")
print(X.columns.tolist())
print("\nTarget column:")
print("Salary")
preprocess = ColumnTransformer(transformers=[("numeric", StandardScaler(), X.select_dtypes(include=["int64", "float64"]).columns),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), X.select_dtypes(include=["object"]).columns),])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
linear = Pipeline([("preprocess", preprocess),("model", LinearRegression())])
linear.fit(X_train, y_train)
y_pred_linear = linear.predict(X_test)
linear_mae = mean_absolute_error(y_test, y_pred_linear)
linear_mse = mean_squared_error(y_test, y_pred_linear)
linear_rmse = np.sqrt(linear_mse)
linear_r2 = r2_score(y_test, y_pred_linear)


plt.figure(figsize=(7, 5))
plt.scatter(y_test, y_test, color="blue", alpha=0.6, label="Actual Salary")
plt.scatter(y_test, y_pred_linear, color="orange", alpha=0.6, label="Predicted Salary")
m, b = np.polyfit(y_test, y_pred_linear, 1)
plt.plot(y_test, m * y_test + b, color="red", label="Best-fit line")
plt.xlabel("Actual Salary")
plt.ylabel("Predicted Salary")
plt.title("Linear Regression: Actual vs Predicted")
plt.legend()
plt.tight_layout()
plt.show()


poly = Pipeline([("preprocess", preprocess),("poly_features", PolynomialFeatures(degree=2, include_bias=False)),("model", LinearRegression())
])

poly.fit(X_train, y_train)
y_pred_poly = poly.predict(X_test)
poly_mae = mean_absolute_error(y_test, y_pred_poly)
poly_mse = mean_squared_error(y_test, y_pred_poly)
poly_rmse = np.sqrt(poly_mse)
poly_r2 = r2_score(y_test, y_pred_poly)

plt.figure(figsize=(7, 5))
plt.scatter(y_test, y_test, color="blue", alpha=0.6, label="Actual Salary")
plt.scatter(y_test, y_pred_poly, color="orange", alpha=0.6, label="Predicted Salary")
poly_coeffs = np.polyfit(y_test, y_pred_poly, 2)
poly_curve = np.poly1d(poly_coeffs)
x_range = np.linspace(min(y_test), max(y_test), 200)
plt.plot(x_range, poly_curve(x_range), color="green", label="Polynomial fit")
plt.xlabel("Actual Salary")
plt.ylabel("Predicted Salary")
plt.title("Polynomial Regression: Actual vs Predicted")
plt.legend()
plt.tight_layout()
plt.show()

rf_pipe = Pipeline([("preprocess", preprocess),("model", RandomForestRegressor(n_estimators=200,random_state=42,n_jobs=-1
    ))
])

rf_pipe.fit(X_train, y_train)
y_pred_rf = rf_pipe.predict(X_test)
rf_mae = mean_absolute_error(y_test, y_pred_rf)
rf_mse = mean_squared_error(y_test, y_pred_rf)
rf_rmse = np.sqrt(rf_mse)
rf_r2 = r2_score(y_test, y_pred_rf)

plt.figure(figsize=(7, 5))
plt.scatter(y_test, y_test, color="blue", alpha=0.6, label="Actual Salary")
plt.scatter(y_test, y_pred_rf, color="orange", alpha=0.6, label="Predicted Salary")
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color="red",
    linestyle="--",
    label="Perfect prediction"
)
plt.xlabel("Actual Salary")
plt.ylabel("Predicted Salary")
plt.title("Random Forest: Actual vs Predicted")
plt.legend()
plt.tight_layout()
plt.show()

metrics_df = pd.DataFrame({
    "Model": ["Linear Regression", "Polynomial Regression", "Random Forest"],
    "MAE":  [linear_mae,  poly_mae,  rf_mae],
    "R2":   [linear_r2,   poly_r2,   rf_r2],
    "MSE":  [linear_mse,  poly_mse,  rf_mse],
    "RMSE": [linear_rmse, poly_rmse, rf_rmse]
}).set_index("Model")

plt.style.use("seaborn-v0_8")
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

metrics_df[["MAE", "MSE", "RMSE"]].plot(kind="bar", ax=axes[0])
axes[0].set_title("Error Metrics by Model")
axes[0].set_ylabel("Error (log scale)")
axes[0].set_yscale("log")
axes[0].set_xlabel("")
axes[0].grid(axis="y", linestyle="--", alpha=0.4)
axes[0].legend(title="Metric", fontsize=9)

for bar in axes[0].patches:
    height = bar.get_height()
    axes[0].annotate(
        f"{height:.2e}",
        (bar.get_x() + bar.get_width() / 2., height),
        ha="center",
        va="bottom",
        fontsize=8,
        rotation=90
    )

metrics_df[["R2"]].plot(kind="bar", ax=axes[1], color=["tab:green"])
axes[1].set_title("R² by Model")
axes[1].set_ylabel("R²")
axes[1].set_ylim(-1.1, 1.05)
axes[1].set_xlabel("")
axes[1].grid(axis="y", linestyle="--", alpha=0.4)
axes[1].legend_.remove()

for bar in axes[1].patches:
    height = bar.get_height()
    axes[1].annotate(
        f"{height:.2f}",
        (bar.get_x() + bar.get_width() / 2., height),
        ha="center",
        va="bottom",
        fontsize=10
    )

plt.tight_layout()
plt.show()

print("\nMODEL COMPARISON SUMMARY\n")
print(metrics_df)

models_summary = {
    "Linear Regression": {"MAE": linear_mae, "MSE": linear_mse, "RMSE": linear_rmse, "R2": linear_r2},
    "Polynomial Regression": {"MAE": poly_mae, "MSE": poly_mse, "RMSE": poly_rmse, "R2": poly_r2},
    "Random Forest": {"MAE": rf_mae, "MSE": rf_mse, "RMSE": rf_rmse, "R2": rf_r2}
}

best_mae_model = min(models_summary, key=lambda name: models_summary[name]["MAE"])
best_rmse_model = min(models_summary, key=lambda name: models_summary[name]["RMSE"])
best_r2_model = max(models_summary, key=lambda name: models_summary[name]["R2"])

count = {name: 0 for name in models_summary}
count[best_mae_model] += 1
count[best_rmse_model] += 1
count[best_r2_model] += 1

overall_best_model = max(count, key=count.get)

print(f"\nLowest MAE  : {best_mae_model}  (MAE = {models_summary[best_mae_model]['MAE']:.2f})")
print(f"Lowest RMSE : {best_rmse_model} (RMSE = {models_summary[best_rmse_model]['RMSE']:.2f})")
print(f"Highest R²  : {best_r2_model}   (R² = {models_summary[best_r2_model]['R2']:.4f})")

print(f"\nOverall best model for this data (based on MAE, RMSE, and R²): {overall_best_model}")
