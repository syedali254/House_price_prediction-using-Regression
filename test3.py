import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
data = fetch_california_housing()
X = data.data
y = data.target

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Predict and evaluate
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MSE:", mse)
print("R²:", r2)
print("Features:", data.feature_names)

# -----------------------
# Visualizations
# -----------------------

# 1. Scatter plot: Actual vs Predicted
plt.figure(figsize=(6, 5))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.5)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.grid(True)
plt.show()

# 2. Residuals plot
residuals = y_test - y_pred
plt.figure(figsize=(6, 5))
sns.histplot(residuals, kde=True, color="red", bins=30)
plt.title("Residuals Distribution")
plt.xlabel("Residual")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

# 3. Feature importance (coefficients)
plt.figure(figsize=(10, 5))
coefficients = model.coef_
sns.barplot(x=data.feature_names, y=coefficients)
plt.xticks(rotation=45)
plt.title("Feature Importance (Model Coefficients)")
plt.ylabel("Coefficient Value")
plt.grid(True)
plt.tight_layout()
plt.show()
