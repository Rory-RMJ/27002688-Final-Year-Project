# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 20:47:21 2022

@author: roryj
"""


import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn import metrics
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import make_scorer, f1_score, confusion_matrix, roc_auc_score, accuracy_score, precision_score, recall_score
from sklearn.metrics import precision_recall_fscore_support as score

data = pd.read_csv(r"data\KDD2014_donors_10feat_nomissing_normalised.csv")

'''
Data Exploration and Visualisation
'''
#Print the contents of the data frame
print(data)

#print the information of the data frame
print(data.info()) 


#Print a histogram and correlation heatmap showing the distribution on the data set
histogram = data.drop(['class'], 1) 
histogram.hist(figsize = (20, 20), bins = 50, color = 'c', edgecolor = 'black')
plt.show() 

correlation = histogram.corr() 
sns.heatmap(correlation) 
plt.show() 

base = data.copy() 
nominal_count = len(base.loc[data['class'] == 0, 'class'])
outlier_count = len(base.loc[data['class'] == 1, 'class'])
print(f'Size of the outlier data set: {outlier_count}, Size of the normal instance set: {nominal_count}')
plt.figure(figsize = (15, 2)) 
fig = sns.countplot(y='class', data=base, color = 'b') 
plt.show() 

'''
Pre-Processing and Parameter tuning
'''
classes = base['class']
data_training = base.drop(['class'], axis = 1) 

#Split the data into feature training, feature testing, target training and target testing data sets
X_train, X_test, y_train, y_test = train_test_split(data_training, classes,
                                                    test_size = 0.3, random_state = 42) 

#Tune the hyperparameters
clf = IsolationForest() 

#Define the parameters
param_grid = {'n_estimators': list(range(100, 200, 300)),
              'max_samples': list(range(100, 200, 300)),
              'max_features': [5, 10, 15]}

#Initialise the f1 score
f1_parameter_scorer = make_scorer(f1_score, average = 'micro')

#Perform the Cross Validation
parameter_search = GridSearchCV(clf,
                                param_grid,
                                scoring = f1_parameter_scorer,
                                cv = 3) 

#Fit the parameter cross validation model
parameter_search.fit(X_train, y_train) 
print(parameter_search.best_params_) 

'''
Train and Initalise the Model
'''
#Initialise the Isolation Forest model
iForest = IsolationForest(n_estimators = 100,
                          max_features = 5,
                          max_samples = 100) 

#Fit the model
iForest.fit(X_train)


'''Train and Initalise the Model'''

#Initialise the Isolation Forest model
iForest = IsolationForest(n_estimators = 100,
                          max_features = 5,
                          max_samples = 100) 

#Fit the model
iForest.fit(X_train)

def evaluate_performance(iForest, X_test, y_true, map_labels):
    #Make predictions based on the test set
    prediction_set = X_test.copy() 
    prediction_set['Predictions'] = iForest.predict(X_test) 
    if map_labels:
        prediction_set['Predictions'] = prediction_set['Predictions'].map({1: 0, -1: 1})
        
    #Evaulate the performance
    prediction = prediction_set['Predictions']
    cmatrix = confusion_matrix(y_true, prediction) 
    
    #Visualise the Confusion Matrix
    sns.heatmap(pd.DataFrame(cmatrix, columns = ['Actual', 'Predicted']),
                annot = True, fmt = "d", linewidth = 0.5, cmap="YlGnBu") 
    plt.xlabel('\nPredicted Values')
    plt.ylabel('\nActual Values')
    
    #Calculate and print accruacy score
    accuracy = accuracy_score(y_true, prediction) 
    print("Accuracy Score: ", accuracy) 
    
    #Area Under Curve Value
    auc = roc_auc_score(y_true, prediction) 
    print("Area Under Curve: ", auc) 
    
    iForest_score= score(prediction, y_true, average='macro')
    return iForest_score 

name = "Isolation Forest Performance"

map_labels = True 

iForest_score = evaluate_performance(iForest, X_test, y_test, map_labels) 
performance_df = pd.DataFrame().append({'name': 'Donors Data',
                                        'f1_score': iForest_score[0],
                                        'precision': iForest_score[1],
                                        'recall': iForest_score[2]},
                                       ignore_index = True) 

print(performance_df) 