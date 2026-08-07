# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Import the required Python libraries and load the Iris dataset.
2.Split the dataset into training and testing sets.
3.Train the SGD Classifier using the training data.
4.Predict the species of the Iris flowers using the test data.
5.Display the accuracy and classification results.
```
## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: MAALINI B N
RegisterNumber:  212224060136
*/
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report
iris = load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model = SGDClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nSample Predictions:")
for i in range(5):
    print("Actual:", y_test[i], "Predicted:", y_pred[i])
```

## Output:
<img width="578" height="442" alt="image" src="https://github.com/user-attachments/assets/0b156840-61fa-467d-994e-8e75dc40dd1d" />



## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
