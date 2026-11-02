import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
data=pd.read_csv("california_housing.csv")
X=data["MedInc"]
y=data["MedHouseVal"]
X_train,X_test,y_train,y_test=train_test_split(
    X,y,test_size=0.2,random_state=42
    )
x_mean=np.mean(X_train)
y_mean=np.mean(y_train)
num=np.sum((X_train-x_mean)*(y_train-y_mean))
dem=np.sum((X_train-x_mean)**2)
b1=num/dem
c=y_mean-b1*x_mean
def predict(x):
    return c+b1*x
print("slope:",b1)
print("intercept:",c)
y_predict=predict(X_test)
mse=np.mean((y_test-y_predict)**2)
sse=np.sum((y_test-y_predict)**2)
sst=np.sum((y_test-np.mean(y_test))**2)
r2=1-sse/sst
print("mse:",mse)
print("r2score:",r2)
idx = np.argsort(X_test)

plt.scatter(X_test,y_test, color="red",label="Actual values")
plt.plot(X_test.iloc[idx], y_predict.iloc[idx], color="blue", linewidth=2,label="Regression line")
plt.xlabel("median income")
plt.ylabel("house price")
plt.legend()
plt.show()
