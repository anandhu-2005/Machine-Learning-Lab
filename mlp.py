import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
data=pd.read_csv("WineQT.csv")
X = data.drop(columns=["quality", "Id"]).values
y = data["quality"].values

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model1=MLPClassifier(
    hidden_layer_sizes=(64,32),
    activation="relu",
    solver="adam",
    max_iter=300,
    random_state=42
    )
model1.fit(X_train,y_train)
y_pred1=model1.predict(X_test)
acc1=accuracy_score(y_test,y_pred1)
model2 = MLPClassifier(
    hidden_layer_sizes=(128, 64, 32),
    activation="relu",
    solver="adam",
    max_iter=1000,
    random_state=42
)

model2.fit(X_train, y_train)
y_pred2 = model2.predict(X_test)
acc2 = accuracy_score(y_test, y_pred2)
results = pd.DataFrame({
    "Model": ["Model 1", "Model 2"],
    "Architecture": [
        "64 -> 32 -> Output",
        "128 -> 64 -> 32 -> Output"
    ],
    "Activation Function": [
        "ReLU, ReLU",
        "ReLU, ReLU, ReLU"
    ],
    "Test Accuracy (%)": [
        round(acc1 * 100, 2),
        round(acc2 * 100, 2)
    ]
})
print(results)
print(results.to_string(index=True))

if acc1 > acc2:
    print("\nModel 1 performs better.")
elif acc2 > acc1:
    print("\nModel 2 performs better.")
else:
    print("\nBoth models have the same accuracy.")
