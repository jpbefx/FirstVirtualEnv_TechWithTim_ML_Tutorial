import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle

data = pd.read_csv("student-mat.csv", sep=";")

print(data.head())

data = data[
    ["G1", "G2", "G3", "studytime", "failures", "absences"]]  # Best to use only integer values without modifying setup

print(data.head())

predict = "G3"  # This is called the 'LABEL'

# Setting up arrays for Attributes, and Labels.
# X = np.array(data.drop([predict], 1))
# Y = np.array(data[predict])

# Splitting the data into Test groups by 10% (test_size=0.1)
# X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)

# The Tutorial has this setup and it results in a slightly different answer. (0.8682470265752752)
x = np.array(data.drop([predict], 1))
y = np.array(data[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

# !!!!!!!!NO MATTER WHAT THE VARIABLE CASE, THE RESULTS CHANGE AFTER EVERY EXECUTION!!!!!!!


# Linear Regression is the same as finding the slope. Y = mx+b, X = mx+b. || Y2 - Y1 / X2 - X1 = m

linear = linear_model.LinearRegression()

# linear.fit(X_train, Y_train)
linear.fit(x_train, y_train)

# acc = linear.score(X_test, Y_test)
acc = linear.score(x_test, y_test)

print(acc)

# Display the result of the slope (m)
print("Coefficient:  \n", linear.coef_)
print("Intercept: \n", linear.intercept_)

# Make a prediction
predictions = linear.predict(x_test)

# x_test[x] is the prediction grade
# y_test[x] is the actual grade
for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])
