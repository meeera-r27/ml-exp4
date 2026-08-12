# Experiment 4: Linear and Polynomial Regression

## Aim

To implement Linear Regression and Polynomial Regression for predicting the Miles Per Gallon (MPG) of vehicles based on engine displacement and compare the performance of the models using Mean Squared Error (MSE) and R² score.

---

## Theory

### Linear Regression

Linear Regression is a supervised machine learning algorithm used to predict a continuous output variable based on one or more input variables.

The equation of simple linear regression is:

y = mx + c

Where:

- y = predicted output
- x = input feature
- m = slope
- c = intercept

In this experiment:

- Input feature = Displacement
- Target variable = MPG

Linear Regression assumes that there is a linear relationship between the input feature and the target variable.

### Polynomial Regression

Polynomial Regression is an extension of Linear Regression that is used to model a non-linear relationship between the input and output variables.

The general equation is:

y = b₀ + b₁x + b₂x² + ... + bₙxⁿ

In this experiment, Polynomial Regression models with degrees 2, 3, and 4 are implemented and compared.

### Model Evaluation

The models are evaluated using two metrics:

#### Mean Squared Error (MSE)

MSE measures the average squared difference between the actual and predicted values.

A lower MSE indicates better model performance.

#### R² Score

R² score measures how well the model explains the variation in the target variable.

A higher R² score indicates better model performance.

---

## Dataset

The experiment uses the **MPG dataset** from the Seaborn dataset repository.

The following attributes are selected:

| Attribute | Description |
|-----------|-------------|
| `displacement` | Engine displacement of the vehicle |
| `mpg` | Miles Per Gallon |

Missing values are removed before training the models.

---

## Software Requirements

- Python 3
- Google Colab / Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

---

## Algorithm

### Linear Regression

1. Import the required libraries.
2. Load the MPG dataset.
3. Select the `mpg` and `displacement` columns.
4. Remove missing values.
5. Select `displacement` as the input feature.
6. Select `mpg` as the target variable.
7. Split the dataset into training and testing sets using an 80:20 ratio.
8. Create a Linear Regression model.
9. Train the model using the training data.
10. Predict MPG values using the test data.
11. Calculate MSE and R² score.

### Polynomial Regression

1. Select polynomial degrees 2, 3, and 4.
2. Create Polynomial Features for each degree.
3. Transform the training and testing data.
4. Create a Linear Regression model.
5. Train the model using the transformed training data.
6. Predict the MPG values.
7. Calculate MSE and R² score.
8. Compare the results of all polynomial models.
9. Select the model with the lowest MSE and highest R² score.

---

## Implementation

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv"
data = pd.read_csv(url)

# Select required columns
data = data[['mpg', 'displacement']]
data.dropna(inplace=True)

# Features and target
X = data[['displacement']]
y = data['mpg']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Linear Regression
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

y_pred_linear = linear_model.predict(X_test)

print("Linear Regression")
print("MSE =", mean_squared_error(y_test, y_pred_linear))
print("R2 =", r2_score(y_test, y_pred_linear))

# Polynomial Regression
degrees = [2, 3, 4]

for degree in degrees:
    poly = PolynomialFeatures(degree)

    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    y_pred = model.predict(X_test_poly)

    print("\nDegree =", degree)
    print("MSE =", mean_squared_error(y_test, y_pred))
    print("R2 =", r2_score(y_test, y_pred))
