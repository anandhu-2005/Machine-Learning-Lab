import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
data=pd.read_csv("breast-cancer.csv")
X=data.drop(columns=["id","diagnosis"]).values
y=data["diagnosis"].map({'M':1,'B':0}).values
X_train,X_test,y_train,y_test=train_test_split(
    X,y,test_size=0.2,random_state=42
    )
mean=np.mean(X_train,axis=0)
std=np.std(X_train,axis=0)
X_train=(X_train-mean)/(std+0.000001)
X_test=(X_test-mean)/(std+0.000001)
def sigmoid(z):
    z=np.clip(z,-500,500)
    return 1/(1+np.exp(-z))
w=np.zeros(X_train.shape[1])
b=0.0
lr=0.01
epochs=5000
n=len(y_train)
print(n)
for i in range(epochs):
    z=np.dot(X_train,w)+b
    y_pred=sigmoid(z)
    dw=np.dot(X_train.T,y_pred-y_train)/n
    db=np.sum(y_pred-y_train)/n
    w=w-lr*dw
    b=b-lr*db
prob=sigmoid(np.dot(X_test,w)+b)
y_pred=(prob>=0.5).astype(int)
print("predicted value:",y_pred[:10])
print("actaul values:",y_test[:10])
tn=0
fp=0
fn=0
tp=0
for i in range(len(y_test)):
    if y_test[i]==0 and y_pred[i]==0:
        tn+=1
    elif y_test[i]==0 and y_pred[i]==1:
        fp+=1
    elif y_test[i]==1 and y_pred[i]==0:
        fn+=1
    elif y_test[i]==1 and y_pred[i]==1:
        tp+=1
print("[[",tn,fp,"]\n[",fn,tp,"]]")
accuracy=(tn+tp)/(tn+tp+fn+fp)
pre=(tn)/(tn+fp)
print("accurcay:",accuracy)
print("precision:",pre)
mse=np.mean((y_test-y_pred)**2)
print(mse)
print(classification_report(
    y_test,
    y_pred,
    target_names=["B", "M"]
))
