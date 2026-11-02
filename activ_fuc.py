import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
data=pd.read_csv("WineQt.csv")
X=data.drop(columns=["quality","Id"]).values
y=data["quality"].values
X_train,X_test,y_train,y_test=train_test_split(
    X,y,test_size=0.2,random_state=42
    )
mean=np.mean(X_train,axis=0)
std=np.std(X_train,axis=0)
X_train=(X_train - mean)/(std + 0.000001)
X_test=(X_test - mean)/(std + 0.000001)
sigmoid_model=MLPClassifier(
    hidden_layer_sizes=(128,64),
    activation="logistic",
    solver="adam",
    learning_rate_init=0.001,
    batch_size=64,
    max_iter=300,
    early_stopping=True,
    random_state=42
    )
sigmoid_model.fit(X_train,y_train)
relu_model=MLPClassifier(
    hidden_layer_sizes=(128,64),
    activation="relu",
    solver="adam",
    learning_rate_init=0.001,
    batch_size=64,
    max_iter=300,
    early_stopping=True,
    random_state=42
    )
relu_model.fit(X_train,y_train)
tanh_model=MLPClassifier(
    hidden_layer_sizes=(128,64),
    activation="tanh",
    solver="adam",
    learning_rate_init=0.001,
    batch_size=64,
    max_iter=300,
    early_stopping=True,
    random_state=42
    )
tanh_model.fit(X_train,y_train)

sigmoid_train=sigmoid_model.score(X_train,y_train)
sigmoid_test=sigmoid_model.score(X_test,y_test)

relu_train=relu_model.score(X_train,y_train)
relu_test=relu_model.score(X_test,y_test)

tanh_train=tanh_model.score(X_train,y_train)
tanh_test=tanh_model.score(X_test,y_test)

sigmoid_loss=sigmoid_model.loss_curve_
relu_loss=relu_model.loss_curve_
tanh_loss=tanh_model.loss_curve_

sigmoid_final_loss=sigmoid_loss[-1]
relu_final_loss=relu_loss[-1]
tanh_final_loss=tanh_loss[-1]
print("activation---train---test---loss")
print("sigmoid\t",sigmoid_train,"\t",sigmoid_test,"\t",sigmoid_final_loss)
print("relu\t", relu_train, "\t", relu_test, "\t", relu_final_loss)
print("tanh\t", tanh_train, "\t", tanh_test, "\t", tanh_final_loss)
activation=["sigmoid","relu","tanh"]
x=np.arange(3)
train_accuracy=[sigmoid_train,relu_train,tanh_train]
test_accuracy=[sigmoid_test,relu_test,tanh_test]
plt.plot(
    x,
    train_accuracy,
    marker="o",
    label="train_accuracy"
    )
plt.plot(
    x,
    test_accuracy,
    marker="o",
    label="test_accuracy"
    )
plt.legend()
plt.xticks(x,activation)
plt.show()

plt.plot(
    sigmoid_loss,
    label="sigmoid_loss"
    )
plt.plot(
    relu_loss,
    label="relu_loss"
    )
plt.plot(
    tanh_loss,
    label="tanh_loss"
    )
plt.legend()
plt.show()
