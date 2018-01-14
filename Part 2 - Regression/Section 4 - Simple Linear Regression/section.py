"""
Simple Linear Regression
Section
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1:].values.ravel() # oder auch y = dataset.iloc[:, len(dataset.axes[1]) - 1].values -- ravel() unwraps die matrix in einen 1d array.

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

J = 1 / len(y_test) * sum((y_test - y_pred)**2)

# Visualising the Training set results

xax = np.linspace(0, max(X_train) + 1).reshape(-1, 1)

plt.scatter(X_train, y_train, color = 'red')
plt.plot(xax, regressor.predict(xax), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_test, y_pred, color = 'blue')        # ob x-train oder test oder linspace etc ist egal, geht ja nur um die funktion. ich finde hier aber x-test besser, weil es ja sein könnte, dass die range nicht ganz gleich ist..
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()