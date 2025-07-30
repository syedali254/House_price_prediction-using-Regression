import numpy as np 
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

X = np.array([
    [1000, 2, 1],
    [1200, 3, 1],
    [1500, 3, 2],
    [1700, 4, 2],
    [2000, 4, 2],
    [2200, 5, 2],
    [2500, 4, 3],
    [2800, 5, 2],
    [3000, 6, 2],
    [3200, 5, 3],
    [3500, 6, 3],
    [3800, 6, 2],
    [4000, 7, 2],
    [4200, 7, 3],
    [4500, 8, 3],
])


Y = np.array([
    150000, 180000, 210000, 240000, 270000,
    300000, 330000, 360000, 390000, 410000,
    450000, 470000, 500000, 530000, 570000
])

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

model=LinearRegression()
model.fit(X_train,Y_train)

pred=model.predict(X_test)

from sklearn.metrics import mean_squared_error, r2_score

mse=mean_squared_error(Y_test,pred)
r2=r2_score(Y_test,pred)

print("MSE:", mean_squared_error(Y_test, pred))
print("R²:", r2_score(Y_test, pred))

#new_house=np.array()

n=model.predict([[1800,3,2]])
print(n)