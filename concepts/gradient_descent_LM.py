#!/usr/bin/python3

#Gradient Descent-Simple linear regression
#---
#This script demonstrates Gradient Descent to estimate the beta parameters
#(y-intercept and slope) for a simple linear regression model.
#y = mx + b
#loss = (y - yhat)**2

#import libraries
import numpy as np
from matplotlib import pyplot as plt


#Simulate data
#---
n = 10 #sample size
m = 5 #slope
b = 10 #y-intercept
x = np.random.randn(n, 1) #feature variable
y = m * x + b #target variable


#Gradient descent function
#---
#define the gradient-descent algorithm
def gradient_descent(x, y, learning_rate = 0.01, iterations = 1000):
    
    #step 1: define partial derivatives of loss functions
    #slope (m)
    def dldm(x, y, m, b):
        loss_m = -2 * x * (y - (m * x + b))
        return loss_m
    
    #y-intercept (b)
    def dldb(x, y, m, b):
        loss_b = -2 * (y - (m * x + b))
        return loss_b
    
    #step 2: initialize guess for parameters (y-intercept and slope)
    m = 0
    b = 0
    
    #step 3: input parameter values into the loss functions
    for _ in range(iterations):
        loss_m = np.sum(dldm(x, y, m, b))
        loss_b = np.sum(dldb(x, y, m, b))
        
        #step 4: calculate step sizes
        step_size_m = loss_m * learning_rate
        step_size_b = loss_b * learning_rate
        
        #Step 5: update parameter estimates
        m = m - step_size_m
        b = b - step_size_b
    
    return m, b


#Apply gradient descent
#---
m_est, b_est = gradient_descent(x, y)
print(f"sample size: {n}")
print(f"slope: {m_est:.2f}")
print(f"y-intercept: {b_est:.2f}")


#Visualize results
#---
#simple linear regression plot
yhat = m_est * x + b_est
plt.scatter(x, y)
plt.plot(x, yhat, color = 'red', linestyle = '-')
plt.title('simple linear model estimated using gradient descent')
plt.xlabel('feature variable (x)')
plt.ylabel('target variable (y)')
plt.savefig('./gradient_descent_LM.pdf', format = 'pdf')
plt.show()

