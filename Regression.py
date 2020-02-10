# Python code to illustrate
# regression using data set
import operator
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd

# Load CSV and columns
df = pd.read_csv("C:/Users/jingu1.lee/Desktop/share/MachineLearning/regression/Housing.csv")

Y = df['price']
X = df['lotsize']

X = X.values.reshape(len(X), 1)
Y = Y.values.reshape(len(Y), 1)

# For polinomial Regression
polynomial_features = PolynomialFeatures(degree=1)
X_poly = polynomial_features.fit_transform(X)

# Split the data into training/testing sets
X_train = X_poly[:-250]
X_test = X_poly[-250:]

# Split the targets into training/testing sets
Y_train = Y[:-250]
Y_test = Y[-250:]

# Create linear regression object
model = linear_model.LinearRegression()

# Train the model using the training sets
model.fit(X_train, Y_train)

# Plot outputs
plt.title('Test Data')
plt.xlabel('lotsize')
plt.ylabel('price')
plt.xticks(())
plt.yticks(())

# plt.scatter(X[-250:], Y_test, color='black')

# for i in range(0, X_test.shape[0]):
#     if X_test[i] == 5000:
#         print(i)

# Plot outputs
# plt.plot(X[-250:], model.predict(X_test), color='red', linewidth=3)
# plt.show()

y_poly_pred =  model.predict(X_test)

plt.scatter(X[-250:], Y_test, s=10, color='black')
# sort the values of x before line plot
sort_axis = operator.itemgetter(0)
sorted_zip = sorted(zip(X[-250:], y_poly_pred), key=sort_axis)
x, y_poly_pred = zip(*sorted_zip)
plt.plot(x, y_poly_pred, color='red', linewidth=3)
plt.show()