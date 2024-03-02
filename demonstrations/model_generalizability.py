#!/usr/bin/python3

#Relationship between train and test accuracies with training set size
#---
#This script simulates a binvariate classification problem and calculates the
#train and test accuracy scores using a logistic regression model with
#increasing training set sizes. This demonstrates the predictive accuracies
#that can be expected with increasing training set sizes.

#import libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import seaborn as sns
from matplotlib import pyplot as plt

#define global variables
max_tr_size, min_tr_size, tr_interval= 1000, 10, 10


#Data simulation
#---
#define a function to simulate a bivariate classification problem
def make_data(size, s = 0.15, theta = np.pi / 3):
    u = np.array([np.cos(theta), np.sin(theta)])
    X = np.random.uniform(size = (size, 2)) - 0.5
    e = s * np.random.normal(size = size)
    y = 1 * (X @ u + e > 0)
    return X, y


#Model training and predictive accuracy
#---
#create the test data set
X_te, y_te = make_data(max_tr_size)

#define the range of sample sizes
ns = list(range(min_tr_size, max_tr_size + 1, tr_interval))

accuracy = {}
for n in ns:
    #create the training dataset
    X_tr, y_tr = make_data(n)
    
    #train the data using logistic regression
    model = LogisticRegression()
    model.fit(X_tr, y_tr)
    
    #make model predictions
    yhat_tr = model.predict(X_tr)
    yhat_te = model.predict(X_te)
    accuracy_tr = accuracy_score(y_tr, yhat_tr)
    accuracy_te = accuracy_score(y_te, yhat_te)
    
    #store the prediction results
    accuracy[n] = [n, accuracy_tr, accuracy_te]

#create a data frame to store accuracy scores
accuracy_df = pd.DataFrame(accuracy).T
accuracy_df.columns = ['n','tr_accuracy','te_accuracy']


#Visualize the results
#---
#create line plot
sns.lineplot(x = ns, y = accuracy_df['tr_accuracy'], label = 'train accuracy')
sns.lineplot(x = ns, y = accuracy_df['te_accuracy'], label = 'test accuracy')
plt.ylabel('model accuracy')
plt.xlabel('training size (n)')
plt.title('Logistic regression model generalizability')
plt.legend()
plt.tight_layout()
plt.savefig('./model_generalizability.pdf', format = 'pdf')
plt.show()

