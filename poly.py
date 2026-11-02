import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
data=pd.read_csv("auto-mpg.csv")
degree=5
data["horsepower"]=pd.to_numeric(data["horsepower"],errors="coerce")
data=data.dropna(subset=["horsepower","mpg"])
X=data["horsepower"].values
y=data["mpg"].values
X_train,X_test,y_train,y_test=train_test_split(
    X,y,test_size=0.2,random_state=42
    )
X_train_poly=np.column_stack([
    X_train**i for i in range(degree+1)
    ])
coeff=np.linalg.lstsq(
    X_train_poly,y_train,rcond=None
    )[0]
X_test_poly = np.column_stack([
    X_test**i for i in range(degree + 1)
])
y_pred=X_test_poly@coeff
print(y_pred)
ss_total = np.sum((y_test - np.mean(y_test)) ** 2)
ss_residual = np.sum((y_test - y_pred) ** 2)
r2_poly = 1 - (ss_residual / ss_total)

# Calculate MSE
mse = np.mean((y_test - y_pred) ** 2)

print("Coefficients:", coeff)
print("R2 Score:", r2_poly)
print("MSE:", mse)
