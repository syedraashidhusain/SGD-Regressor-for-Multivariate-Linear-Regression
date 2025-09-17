# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
#### 1. Import Libraries

Import required Python libraries: numpy, pandas, sklearn.

#### 2. Load Dataset

Load or create a dataset with features (e.g., area, number of rooms, location index).

Include target variables (house price, number of occupants).

#### 3. Preprocess Data

Handle missing values if present.

Scale features using StandardScaler to speed up convergence.

#### 4. Split Dataset

Use train_test_split to split data into training and testing sets.

#### 5. Train Model with SGD Regressor

Initialize SGDRegressor with a learning rate and max iterations.

Use MultiOutputRegressor for predicting multiple outputs.

Train model using the training set.

#### 6. Make Predictions

Predict house price and number of occupants for the test set.

#### 7. Evaluate Model

Use metrics such as r2_score and mean_squared_error to check performance.

#### 8. Display Results

Print actual vs predicted values for better understanding.

## Program:
```python
# Import libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_squared_error

# -----------------------------
# Step 1: Create synthetic dataset
# -----------------------------
# Features: [Area (sq ft), Number of Rooms, Location Index]
X = np.array([
    [1200, 3, 1],
    [1500, 4, 2],
    [800, 2, 1],
    [2000, 5, 3],
    [1700, 4, 2],
    [1000, 2, 1],
    [2200, 5, 3],
    [1300, 3, 2]
])

# Targets: [Price (in lakhs), Number of Occupants]
y = np.array([
    [50, 4],
    [65, 5],
    [35, 3],
    [90, 7],
    [70, 6],
    [40, 3],
    [100, 8],
    [55, 4]
])

# -----------------------------
# Step 2: Split dataset
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# -----------------------------
# Step 3: Feature Scaling
# -----------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------------
# Step 4: Train SGD Regressor
# -----------------------------
sgd = SGDRegressor(max_iter=1000, tol=1e-3, eta0=0.01, learning_rate='constant')
multi_regressor = MultiOutputRegressor(sgd)
multi_regressor.fit(X_train, y_train)

# -----------------------------
# Step 5: Prediction
# -----------------------------
y_pred = multi_regressor.predict(X_test)

# -----------------------------
# Step 6: Evaluation
# -----------------------------
print("R2 Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))

print("\nActual vs Predicted:")
for actual, predicted in zip(y_test, y_pred):
    print(f"Actual: {actual}, Predicted: {predicted.round(2)}")
```



## Output:
<img width="417" height="132" alt="image" src="https://github.com/user-attachments/assets/72e0d6c3-e189-43d9-98c2-c277efef96d6" />

## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
