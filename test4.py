from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np


X=np.array([1, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10]).reshape(-1,1)

Y=np.array([45, 50, 53, 55, 60, 62, 65, 68, 70, 75,77, 80, 82, 85, 87, 89, 92, 95])

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

model=LinearRegression()
model.fit(X_train,Y_train)

y_pred=model.predict(X_test)

mse=mean_squared_error(Y_test,y_pred)
r2=r2_score(Y_test,y_pred)


print("MSE:", mse)
print("R²:", r2)

n=model.predict([[4.2]])
print(n)