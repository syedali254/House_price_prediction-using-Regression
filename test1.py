import numpy as np 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


# .reshape(-1, 1): Converts shape from (5,) to (5,1) as required by scikit-learn


# Data
X = np.array([1000, 1500, 2000, 2500, 3000]).reshape(-1, 1)
Y = np.array([150, 200, 250, 300, 350])

# Train-Test Split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Model Training
model = LinearRegression()
model.fit(X_train, Y_train)

# Predictions
y_pred = model.predict(X_test)
y_full_pred = model.predict(X)  # Predict all for plotting

# Evaluation
mse = mean_squared_error(Y_test, y_pred)
r2 = r2_score(Y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared Score:", r2)

your_size=1800

your_prediction=model.predict([[your_size]])
print(f"Predicted price for a {your_size} sqft house is: ${your_prediction[0]*1000:.2f}")

# Visualization
plt.scatter(X, Y, color='blue', label='Actual Data')       
plt.plot(X, y_full_pred, color='red', linewidth=2, label='Model Prediction Line')  
plt.xlabel('House Size (sqft)')
plt.ylabel('Price ($1000s)')
plt.title('Linear Regression Fit')
plt.legend()
plt.grid(True)
plt.show()
