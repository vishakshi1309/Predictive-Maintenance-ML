import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from pathlib import Path

CHARTS_DIR = Path(
    r"C:\Users\Vishakshi\Documents\predictive_maintenance\charts"
)

CHARTS_DIR.mkdir(parents=True, exist_ok=True)

os.chdir(r"C:\Users\Vishakshi\Documents\predictive_maintenance\data")
print(os.getcwd())
print(os.path.exists("data/train_FD001.txt"))
#print(os.listdir("data"))
print("Current folder:", os.getcwd())
print("Files and folders here:", os.listdir())

# Load the three dataset files
train = pd.read_csv("train_FD001.txt", sep=r"\s+", header=None)
test = pd.read_csv("test_FD001.txt", sep=r"\s+", header=None)
rul = pd.read_csv("RUL_FD001.txt", sep=r"\s+", header=None)
print("Training data shape:", train.shape)
print("Testing data shape:", test.shape)
print("RUL data shape:", rul.shape)
print("\nTraining data preview:")
print(train.head())
#adding column name
columns = [
    "engine_id",
    "cycle",
    "setting_1",
    "setting_2",
    "setting_3",
    "sensor_1",
    "sensor_2",
    "sensor_3",
    "sensor_4",
    "sensor_5",
    "sensor_6",
    "sensor_7",
    "sensor_8",
    "sensor_9",
    "sensor_10",
    "sensor_11",
    "sensor_12",
    "sensor_13",
    "sensor_14",
    "sensor_15",
    "sensor_16",
    "sensor_17",
    "sensor_18",
    "sensor_19",
    "sensor_20",
    "sensor_21"
]

# Assign column names
train.columns = columns
test.columns = columns
print(train.head())
print("\nColumn names:")
print(train.columns.tolist())

#EDA

print("Dataset shape:")
print(train.shape)

print("\nFirst five rows:")
print(train.head())

print("\nColumn data types:")
print(train.dtypes)
print("\nMissing values in each column:")
print(train.isnull().sum())

print("\nStatistical summary:")
print(train.describe())

print("Number of engines:", train["engine_id"].nunique())

print("\nCycle statistics:")
print(train["cycle"].describe())

print("\nDuplicate rows:", train.duplicated().sum())

print("\nSensor statistics:")
print(train.iloc[:, 5:26].describe().round(2))

sensors = train.iloc[:, 5:26]
# Find sensors with no variation
constant_sensors = sensors.columns[sensors.nunique() == 1]
print("Constant sensors:")
print(constant_sensors.tolist())
print("\nNumber of constant sensors:", len(constant_sensors))

# Find the last recorded cycle for each engine
last_cycle = train.groupby("engine_id")["cycle"].max()

# Add the last cycle to every row
train["last_cycle"] = train["engine_id"].map(last_cycle)

# Calculate Remaining Useful Life (RUL)
train["RUL"] = train["last_cycle"] - train["cycle"]

# Display the results
print(train[["engine_id", "cycle", "last_cycle", "RUL"]].head(10))

# Check the RUL column
print("Total rows:", len(train))
print("Missing RUL values:", train["RUL"].isnull().sum())
print("Minimum RUL:", train["RUL"].min())
print("Maximum RUL:", train["RUL"].max())

# Display the last 5 records of engine 1
print(train[train["engine_id"] == 1][
    ["engine_id", "cycle", "last_cycle", "RUL"]
].tail())

# Select the input features
X = train.drop(columns=["engine_id", "last_cycle", "RUL"])

# Select the target
y = train["RUL"]

# Check the shapes
print("Features shape:", X.shape)
print("Target shape:", y.shape)

print(X.head())
print(y.head())

engine_ids = train["engine_id"].unique()

# Split the engine IDs, not individual rows
train_engines, val_engines = train_test_split(
    engine_ids,
    test_size=0.2,
    random_state=42
)

# Select rows belonging to each group
X_train = X[train["engine_id"].isin(train_engines)]
y_train = y.loc[X_train.index]

X_val = X[train["engine_id"].isin(val_engines)]
y_val = y.loc[X_val.index]

# Check the results
print("Training engines:", len(train_engines))
print("Validation engines:", len(val_engines))
print("Training rows:", len(X_train))
print("Validation rows:", len(X_val))

# Confirm that the engine groups do not overlap
print("Overlapping engines:",
      len(set(train_engines) & set(val_engines)))

#model inplementation    ****RANDON FOREST******
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
# Train the model
model.fit(X_train, y_train)
# Predict RUL for validation engines
y_pred = model.predict(X_val)

# Evaluate the model
mae = mean_absolute_error(y_val, y_pred)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))

print("Mean Absolute Error (MAE):", round(mae, 2))
print("Root Mean Squared Error (RMSE):", round(rmse, 2))

#Decision Tree Regression

# Create the Decision Tree model
dt_model = DecisionTreeRegressor(random_state=42)

# Train the model
dt_model.fit(X_train, y_train)

# Predict RUL for validation engines
dt_pred = dt_model.predict(X_val)

# Evaluate the model
dt_mae = mean_absolute_error(y_val, dt_pred)
dt_mse = mean_squared_error(y_val, dt_pred)
dt_rmse = np.sqrt(dt_mse)
dt_r2 = r2_score(y_val, dt_pred)

print("Decision Tree Results")
print("MAE:", round(dt_mae, 2))
print("MSE:", round(dt_mse, 2))
print("RMSE:", round(dt_rmse, 2))
print("R2 Score:", round(dt_r2, 3))

#Linear regression model

lr_model = LinearRegression()

# Train the model
lr_model.fit(X_train, y_train)

# Predict RUL
lr_pred = lr_model.predict(X_val)

# Evaluate the model
lr_mae = mean_absolute_error(y_val, lr_pred)
lr_mse = mean_squared_error(y_val, lr_pred)
lr_rmse = np.sqrt(lr_mse)
lr_r2 = r2_score(y_val, lr_pred)

print("Linear Regression Results")
print("MAE:", round(lr_mae, 2))
print("MSE:", round(lr_mse, 2))
print("RMSE:", round(lr_rmse, 2))
print("R2 Score:", round(lr_r2, 3))

# GradientBoostingRegressor

gb_model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    random_state=42
)

# Train the model
gb_model.fit(X_train, y_train)

# Predict RUL
gb_pred = gb_model.predict(X_val)

# Evaluate the model
gb_mae = mean_absolute_error(y_val, gb_pred)
gb_mse = mean_squared_error(y_val, gb_pred)
gb_rmse = np.sqrt(gb_mse)
gb_r2 = r2_score(y_val, gb_pred)

print("Gradient Boosting Results")
print("MAE:", round(gb_mae, 2))
print("MSE:", round(gb_mse, 2))
print("RMSE:", round(gb_rmse, 2))
print("R2 Score:", round(gb_r2, 3))

#polynomial regression

# Create Polynomial Regression model
poly_model = make_pipeline(
    StandardScaler(),
    PolynomialFeatures(degree=2, include_bias=False),
    Ridge(alpha=1.0)
)

# Train the model
poly_model.fit(X_train, y_train)

# Predict RUL
poly_pred = poly_model.predict(X_val)

# Evaluate the model
poly_mae = mean_absolute_error(y_val, poly_pred)
poly_mse = mean_squared_error(y_val, poly_pred)
poly_rmse = np.sqrt(poly_mse)
poly_r2 = r2_score(y_val, poly_pred)

print("Polynomial Regression Results")
print("MAE:", round(poly_mae, 2))
print("MSE:", round(poly_mse, 2))
print("RMSE:", round(poly_rmse, 2))
print("R2 Score:", round(poly_r2, 3))

# continued random forest
rf_mse = mean_squared_error(y_val, y_pred)
rf_rmse = np.sqrt(rf_mse)
rf_r2 = r2_score(y_val, y_pred)

print("Random Forest MSE:", round(rf_mse, 2))
print("Random Forest RMSE:", round(rf_rmse, 2))
print("Random Forest R2:", round(rf_r2, 3))

#visualization 
models = [
    "Linear Regression",
    "Polynomial Regression",
    "Decision Tree",
    "Random Forest",
    "Gradient Boosting"
]

# Actual model results
mae = [25.16, 22.56, 35.66, 24.01, 22.70]
rmse = [31.66, 29.03, 49.15, 31.70, 29.76]
r2 = [0.768, 0.804, 0.439, 0.767, 0.794]

# Chart 1: MAE Comparison
plt.figure(figsize=(11, 6))
bars = plt.barh(models, mae)
plt.gca().invert_yaxis()

plt.title(
    "Predictive Maintenance | MAE Comparison",
    fontsize=16, fontweight="bold", pad=18
)
plt.xlabel("Mean Absolute Error (cycles)", fontsize=12)
plt.ylabel("Machine Learning Model", fontsize=12)

plt.xlim(0, max(mae) * 1.18)
plt.grid(axis="x", linestyle="--", alpha=0.4)
plt.gca().set_axisbelow(True)

for bar, value in zip(bars, mae):
    plt.text(
        value + 0.3,
        bar.get_y() + bar.get_height() / 2,
        f"{value:.2f}",
        va="center",
        fontsize=10,
        fontweight="bold"
    )

plt.tight_layout()
plt.savefig(CHARTS_DIR / "MAE_Comparison.png", dpi=300, bbox_inches="tight")
print("MAE saved:", (CHARTS_DIR / "MAE_Comparison.png").stat().st_size, "bytes")
plt.show()

# Chart 2: RMSE Comparison
plt.figure(figsize=(11, 6))

bars = plt.barh(models, rmse)
plt.gca().invert_yaxis()
plt.title(
    "Predictive Maintenance | RMSE Comparison",
    fontsize=16, fontweight="bold", pad=18)
plt.xlabel("Root Mean Squared Error (cycles)", fontsize=12)
plt.ylabel("Machine Learning Model", fontsize=12)
plt.xlim(0, max(rmse) * 1.18)
plt.grid(axis="x", linestyle="--", alpha=0.4)
plt.gca().set_axisbelow(True)
for bar, value in zip(bars, rmse):
    plt.text(
        value + 0.5,
        bar.get_y() + bar.get_height() / 2,
        f"{value:.2f}",
        va="center",
        fontsize=10,
        fontweight="bold"    )
plt.tight_layout()
plt.savefig(CHARTS_DIR / "RMSE_Comparison.png", dpi=300, bbox_inches="tight")
print("RMSE saved:", (CHARTS_DIR / "RMSE_Comparison.png").stat().st_size, "bytes")
plt.show()

# Chart 3: R2 Score Comparison
plt.figure(figsize=(11, 6))
bars = plt.barh(models, r2)
plt.gca().invert_yaxis()
plt.title(
    "Predictive Maintenance | R² Score Comparison",
    fontsize=16, fontweight="bold", pad=18)
plt.xlabel("R² Score (higher is better)", fontsize=12)
plt.ylabel("Machine Learning Model", fontsize=12)
plt.xlim(0, 1.0)
plt.grid(axis="x", linestyle="--", alpha=0.4)
plt.gca().set_axisbelow(True)

for bar, value in zip(bars, r2):
    plt.text(
        value + 0.015,
        bar.get_y() + bar.get_height() / 2,
        f"{value:.3f}",
        va="center",
        fontsize=10,
        fontweight="bold"  )
plt.tight_layout()
plt.savefig(CHARTS_DIR / "R2_Score_Comparison.png", dpi=300, bbox_inches="tight")
plt.show()

engine_lifespan = train.groupby("engine_id")["cycle"].max()

sns.set_theme(style="whitegrid")

plt.figure(figsize=(11, 6))
ax = sns.histplot(
    engine_lifespan,
    bins=15,
    color="#4776E6",
    edgecolor="white",
    linewidth=1.2)
median_life = engine_lifespan.median()
plt.axvline(
    median_life,
    color="#E76F51",
    linestyle="--",
    linewidth=2.5,
    label=f"Median: {median_life:.0f} cycles"
)
plt.title(
    "Engine Lifespan Distribution",
    fontsize=17,
    fontweight="bold",
    pad=18
)
plt.xlabel("Final Recorded Operating Cycle", fontsize=12)
plt.ylabel("Number of Engines", fontsize=12)
plt.legend(frameon=True)
sns.despine()
plt.tight_layout()
plt.savefig(CHARTS_DIR / "Engine_Lifespan_Distribution.png", dpi=300, bbox_inches="tight")
plt.show()

sns.set_theme(style="whitegrid")
plt.figure(figsize=(11, 6))
sns.histplot(
    data=train,
    x="RUL",
    bins=30,
    color="#18A999",
    edgecolor="white",
    linewidth=0.8)
plt.axvline(
    train["RUL"].median(),
    color="#8B5CF6",
    linestyle="--",
    linewidth=2.5,
    label=f'Median RUL: {train["RUL"].median():.0f} cycles')
plt.title(
    "Remaining Useful Life (RUL) Distribution",
    fontsize=17,
    fontweight="bold",
    pad=18)
plt.xlabel("Remaining Useful Life (cycles)", fontsize=12)
plt.ylabel("Number of Observations", fontsize=12)
plt.legend()
sns.despine()
plt.tight_layout()
plt.savefig(CHARTS_DIR / "RUL_Distribution.png", dpi=300, bbox_inches="tight")
plt.show()

sensor_columns = [
    col for col in train.columns
    if col.startswith("sensor_")
]
# Remove constant sensors because their correlation is undefined
sensor_data = train[sensor_columns].loc[
    :, train[sensor_columns].nunique() > 1]
# Calculate correlation
correlation = sensor_data.corr()
# Create the heatmap
plt.figure(figsize=(13, 9))

sns.heatmap(
    correlation,
    cmap="magma",
    annot=True,
    fmt=".2f",
    linewidths=0.4,
    square=True,
    vmin=-1,
    vmax=1,
    center=0,
    annot_kws={"size": 8},
    cbar_kws={"label": "Correlation coefficient"}
)
plt.title(
    "NASA C-MAPSS | Sensor Correlation Heatmap",
    fontsize=17,
    fontweight="bold",
    pad=18)
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(CHARTS_DIR / "Sensor_Correlation_Heatmap.png", dpi=300, bbox_inches="tight")
plt.show()

plot_data = train.copy()
plot_data["life_progress"] = (
    plot_data["cycle"] /
    plot_data.groupby("engine_id")["cycle"].transform("max")
) * 100
plot_data["life_stage"] = pd.cut(
    plot_data["life_progress"],
    bins=20,
    labels=False,
    include_lowest=True
)
selected_sensors = ["sensor_2", "sensor_3", "sensor_4"]
trend_data = (
    plot_data.groupby("life_stage", observed=True)[selected_sensors]
    .mean()
)
# Create three separate charts
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(
    1, 3,
    figsize=(16, 5),
    sharex=True)
colors = ["#4776E6", "#18A999", "#E76F51"]
for ax, sensor, color in zip(axes, selected_sensors, colors):

    values = trend_data[sensor]

    # Normalize each sensor independently
    if values.max() != values.min():
        values = (
            (values - values.min()) /
            (values.max() - values.min())
        )
    ax.plot(
        trend_data.index + 1,
        values,
        color=color,
        linewidth=2.5,
        marker="o",
        markersize=4
    )
    ax.set_title(
        sensor.replace("_", " ").title(),
        fontsize=13,
        fontweight="bold"
    )
    ax.set_xlabel("Engine Life Stage (%)")
    ax.set_ylabel("Normalized Reading")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.25)
fig.suptitle(
    "Sensor Behaviour Across Engine Life",
    fontsize=17,
    fontweight="bold",
    y=1.04)
plt.tight_layout()
plt.savefig(CHARTS_DIR / "Normalized_Sensor_Trends.png", dpi=300, bbox_inches="tight")
plt.show()

# Use validation results from Polynomial Regression
actual = y_val
predicted = poly_pred
sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 7))
# Actual vs predicted points
plt.scatter(
    actual,
    predicted,
    color="#18A999",
    alpha=0.35,
    s=22,
    edgecolors="none",
    label="Predictions")
# Perfect prediction reference line
minimum = min(actual.min(), predicted.min())
maximum = max(actual.max(), predicted.max())
plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    color="#8B5CF6",
    linewidth=2.5,
    linestyle="--",
    label="Perfect prediction")
plt.title(
    "Actual vs. Predicted RUL",
    fontsize=17,
    fontweight="bold",
    pad=18)
plt.xlabel("Actual RUL (cycles)", fontsize=12)
plt.ylabel("Predicted RUL (cycles)", fontsize=12)
plt.legend(frameon=True)
plt.grid(alpha=0.25)
sns.despine()
plt.tight_layout()
plt.savefig(CHARTS_DIR / "Actual_vs_Predicted_RUL.png", dpi=300, bbox_inches="tight")
plt.show()

# Calculate prediction errors
residuals = y_val - poly_pred
sns.set_theme(style="whitegrid")
plt.figure(figsize=(11, 6))
plt.scatter(
    poly_pred,
    residuals,
    color="#E76F51",
    alpha=0.4,
    s=24,
    edgecolors="none"
)
# Zero-error reference line
plt.axhline(
    y=0,
    color="#4776E6",
    linestyle="--",
    linewidth=2.5,
    label="Zero error"
)

plt.title(
    "Residual Error Analysis | Polynomial Regression",
    fontsize=16,
    fontweight="bold",
    pad=18
)

plt.xlabel("Predicted RUL (cycles)", fontsize=12)
plt.ylabel("Residual (cycles)", fontsize=12)

plt.legend()
plt.grid(alpha=0.25)
sns.despine()
plt.tight_layout()
plt.show()

# Get feature importance from your trained Random Forest model
importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": model.feature_importances_
})

# Select the top 10 features
importance = importance.sort_values(
    by="Importance",
    ascending=False
).head(10)

# Plot the results
sns.set_theme(style="whitegrid")

plt.figure(figsize=(11, 7))

bars = plt.barh(
    importance["Feature"][::-1],
    importance["Importance"][::-1],
    color=sns.color_palette("magma", n_colors=10)
)

plt.title(
    "Top 10 Features for RUL Prediction",
    fontsize=17,
    fontweight="bold",
    pad=18
)

plt.xlabel("Feature Importance Score", fontsize=12)
plt.ylabel("Engine Measurements", fontsize=12)

# Add values to the bars
for bar in bars:
    value = bar.get_width()
    plt.text(
        value + 0.002,
        bar.get_y() + bar.get_height() / 2,
        f"{value:.3f}",
        va="center",
        fontsize=9
    )
plt.grid(axis="x", alpha=0.25)
sns.despine()
plt.tight_layout()
plt.show()

models = [
    "Linear Regression",
    "Polynomial Regression",
    "Decision Tree",
    "Random Forest",
    "Gradient Boosting"
]
# Validation results
mae = [25.16, 22.56, 35.66, 24.01, 22.70]
rmse = [31.66, 29.03, 49.15, 31.70, 29.76]
r2 = [0.768, 0.804, 0.439, 0.767, 0.794]
# Distinct colours for each model
colors = [
    "#4776E6",
    "#18A999",
    "#E76F51",
    "#E9A23B",
    "#8B5CF6"
]
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 3, figsize=(19, 7))
metrics = [
    (mae, "MAE Comparison", "Error (cycles)"),
    (rmse, "RMSE Comparison", "Error (cycles)"),
    (r2, "R² Score Comparison", "R² Score")
]
for ax, (values, title, xlabel) in zip(axes, metrics):
    # Horizontal bars
    bars = ax.barh(
        models,
        values,
        color=colors,
        height=0.65
    )
    ax.invert_yaxis()
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)

    # Add values to bars
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_width() + max(values) * 0.025,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}" if title.startswith("R²") else f"{value:.2f}",
            va="center",
            fontsize=9,
            fontweight="bold"
        )
    ax.set_xlim(0, max(values) * 1.25)
fig.suptitle(
    "Predictive Maintenance | Model Performance Dashboard",
    fontsize=18,
    fontweight="bold",
    y=1.02
)
sns.despine()
plt.tight_layout()
plt.savefig(
    "Model_Performance_Dashboard.png",
    dpi=300,
    bbox_inches="tight"
)

# 1. Prepare the complete training dataset
X_full = train.drop(columns=["engine_id", "last_cycle", "RUL"])
y_full = train["RUL"]

# 2. Create and train the final Polynomial Regression model
final_poly_model = make_pipeline(
    StandardScaler(),
    PolynomialFeatures(degree=2, include_bias=False),
    Ridge(alpha=1.0)
)

final_poly_model.fit(X_full, y_full)

# 3. Select the last recorded row for each test engine
test_final = (
    test.sort_values(["engine_id", "cycle"])
    .groupby("engine_id")
    .tail(1)
    .sort_values("engine_id")
)

X_test_final = test_final[X_full.columns]

# 4. Predict RUL for the 100 test engines
test_predictions = final_poly_model.predict(X_test_final)

# 5. Load the actual RUL values supplied by NASA
actual_test_rul = rul.iloc[:, 0].to_numpy()

# 6. Calculate evaluation metrics
test_mae = mean_absolute_error(
    actual_test_rul, test_predictions
)

test_mse = mean_squared_error(
    actual_test_rul, test_predictions
)

test_rmse = np.sqrt(test_mse)

test_r2 = r2_score(
    actual_test_rul, test_predictions
)

print("NASA TEST DATA RESULTS")
print("----------------------")
print("Number of engines:", len(test_predictions))
print("MAE:", round(test_mae, 2))
print("MSE:", round(test_mse, 2))
print("RMSE:", round(test_rmse, 2))
print("R2 Score:", round(test_r2, 3))

# Actual NASA RUL and model predictions
actual = actual_test_rul
predicted = test_predictions

sns.set_theme(style="whitegrid")

plt.figure(figsize=(10, 7))

plt.scatter(
    actual,
    predicted,
    color="#18A999",
    alpha=0.75,
    s=65,
    edgecolors="white",
    linewidths=0.6,
    label="Test engine predictions"
)

# Perfect prediction reference line
minimum = min(actual.min(), predicted.min())
maximum = max(actual.max(), predicted.max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    color="#8B5CF6",
    linestyle="--",
    linewidth=2.5,
    label="Perfect prediction"
)
plt.title(
    "NASA Test Results | Actual vs. Predicted RUL",
    fontsize=16,
    fontweight="bold",
    pad=18
)
plt.xlabel("Actual RUL (cycles)", fontsize=12)
plt.ylabel("Predicted RUL (cycles)", fontsize=12)
plt.legend()
plt.grid(alpha=0.25)
sns.despine()
plt.tight_layout()
plt.savefig(
    "NASA_Test_Actual_vs_Predicted.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()

model_bundle = {
    "model": final_poly_model,
    "feature_columns": X_full.columns.tolist()
}

joblib.dump(model_bundle, "predictive_maintenance_model.pkl")

print("Model saved successfully!")

# Load the saved model
saved_bundle = joblib.load("predictive_maintenance_model.pkl")

loaded_model = saved_bundle["model"]
saved_features = saved_bundle["feature_columns"]

# Make predictions using the saved model
test_predictions_check = loaded_model.predict(
    X_test_final[saved_features]
)

print("Model loaded successfully!")
print("First 5 predictions:", test_predictions_check[:5])

# Check whether saved model gives the same predictions
print(
    "Predictions match:",
    np.allclose(test_predictions, test_predictions_check)
)

data = {
    "Model": [
        "Linear Regression",
        "Polynomial Regression",
        "Decision Tree",
        "Random Forest",
        "Gradient Boosting"
    ],
    "MAE": [25.16, 22.56, 35.66, 24.01, 22.70],
    "RMSE": [31.66, 29.03, 49.15, 31.70, 29.76],
    "R2": [0.768, 0.804, 0.439, 0.767, 0.794]
}

df = pd.DataFrame(data)

# Different colors for each model
colors = [
    "#4776E6",
    "#18A999",
    "#E9A23B",
    "#8B5CF6",
    "#E76F51"
]

# Create three dashboard charts
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Chart 1: MAE
axes[0].bar(df["Model"], df["MAE"], color=colors)
axes[0].set_title("Mean Absolute Error (MAE)")
axes[0].set_ylabel("Error in cycles")
axes[0].tick_params(axis="x", rotation=45)

# Chart 2: RMSE
axes[1].bar(df["Model"], df["RMSE"], color=colors)
axes[1].set_title("Root Mean Squared Error (RMSE)")
axes[1].set_ylabel("Error in cycles")
axes[1].tick_params(axis="x", rotation=45)

# Chart 3: R-squared
axes[2].bar(df["Model"], df["R2"], color=colors)
axes[2].set_title("R-squared Score")
axes[2].set_ylabel("R² score")
axes[2].set_ylim(0, 1)
axes[2].tick_params(axis="x", rotation=45)

# Add values above each bar
for ax in axes:
    for bar in ax.patches:
        height = bar.get_height()
        ax.annotate(
            f"{height:.2f}",
            (bar.get_x() + bar.get_width() / 2, height),
            ha="center",
            va="bottom",
            fontsize=8
        )

plt.suptitle(
    "Predictive Maintenance - Model Performance Dashboard", fontsize=16)
plt.tight_layout()
# Save the dashboard image
plt.savefig(
    CHARTS_DIR / "Model_Performance_Dashboard.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()
print(df)
print("\nDashboard saved successfully!")


















