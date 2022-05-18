# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 16:26:01 2022

@author: Daniel

Solocion de taller 5
"""
import numpy as np # manejo de matrices
import matplotlib.pyplot as plt # gráficos
from sklearn.linear_model import LinearRegression 
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

dt = np.load('data.npy',allow_pickle=True).item(0)
testing_set = dt['testing_set']
training_set = dt['training_set']

casual = training_set[:,7].reshape(500,1)
registered = training_set[:,8].reshape(500,1)
temp = training_set[:,6].reshape(500,1)


X= temp[np.where(training_set[:,4] == 1)[0]]

Y = casual[np.where(training_set[:,4] == 1)[0]]

fig0 = plt.figure() 
plt.scatter(X,Y, c = 'grey',label = 'datos')
plt.legend()
plt.grid()

pf = PolynomialFeatures(degree = 3)
X_t = pf.fit_transform(X)
rl = LinearRegression()
rl.fit(X_t, Y)


w = rl.coef_ 
b = rl.intercept_
y_pred = rl.predict(X_t)


mse = mean_squared_error(y_true = Y, y_pred = y_pred)
rmse = np.sqrt(mse)

r2 = rl.score(X_t, Y)

#testing
x_test = pf.fit_transform(testing_set.reshape(-1,1))
y_test = np.round(rl.predict(x_test))


