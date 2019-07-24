#print(_doc_)
import numpy as numpy
from sklearn import datasets , linear_model
from sklearn.metrics import mean_squared_error , r2_score
import matplotlib.pyplot as plt
import pandas as pd

#kanay = datasets.load("kanay.csv")
#kanay = pd.read_csv("kanay.csv")
#kanay_X = kanay.data[:, np.newaxis,3]
#x1,x2,x3,y=np.loadtxt('kanay.txt', skiprows=0,unpack=true)
kanay = pd.read_csv('kanay.csv')
kanay_X = kanay.iloc[:, 0:3].values
kanay_y = kanay.iloc[:, 3]
print(kanay_X)
kanay_X_train = kanay_X[:-20]
kanay_X_test = kanay_X[-20:]
kanay_y_train = kanay_y[:-20]
kanay_y_test = kanay_y[-20:]
regr = linear_model.LinearRegression()
regr.fit(kanay_X_train, kanay_y_train)
kanay_y_pred = regr.predict(kanay_X_test)
# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(kanay_y_test, kanay_y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(kanay_y_test, kanay_y_pred))

# print(kanay_X_test, kanay_y_test)
# plt.scatter(kanay_X_test, kanay_y_test,  color='black')
# plt.plot(kanay_X_test, kanay_y_pred, color='blue', linewidth=3)



# plt.xticks(())
# plt.yticks(())

# plt.show()