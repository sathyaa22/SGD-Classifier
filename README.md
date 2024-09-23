# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

Step 1: Start the program.

Step 2: Import Necessary Libraries and Load Data

Step 3: Split Dataset into Training and Testing Sets

Step 4: Train the Model Using Stochastic Gradient Descent (SGD)

Step 5: Make Predictions and Evaluate Accuracy

Step 6: Generate Confusion Matrix

Step 7: Stop the program.

## Program:

Program to implement the prediction of iris species using SGD Classifier.

Developed by: SATHYAA R

RegisterNumber: 212223100052

```

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

#Load the iris dataset
iris = load_iris()

#Create a Pandas DataFrame
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

#Display the first few rows of the dataset
print(df.head())

#Split the data into features (X) and target (Y)
X = df.drop('target', axis=1)
y = df['target']

#Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Create an SGD Classifier with default parameters
sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3)

#Train the classifier on the training data
sgd_clf.fit(X_train, y_train)

#Make predictions on the testing data
y_pred = sgd_clf.predict(X_test)

#Evaluate the classifier's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")

#Calculate the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
```

## Output:


![Screenshot 2024-09-19 085935](https://github.com/user-attachments/assets/0edb7cc4-1ae4-46ca-a1bc-e8557452fbd1)

![Screenshot 2024-09-19 090929](https://github.com/user-attachments/assets/c470f02a-eb31-48ec-a904-7735f19de4ce)

![Screenshot 2024-09-19 091058](https://github.com/user-attachments/assets/c087dcaf-0aa0-4412-bcaa-fba26a6f9c7c)


## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
