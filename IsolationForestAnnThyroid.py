# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 19:45:11 2022

@author: roryj
"""

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import make_scorer, f1_score, confusion_matrix, classification_report
from sklearn.metrics import precision_recall_fscore_support as score 

data = pd.read_csv(r"data\annthyroid.csv") 
print(data) 

histogram = data.drop(['class'], 1) 
histogram.hist(figsize = (20, 20), bins = 50, color = 'c', edgecolor = 'black')
plt.show() 

correlation = histogram.corr() 
sns.heatmap(correlation) 

data_copy = data.copy() 
nominal_count = len(data_copy.loc[data['class'] == 0, 'class'])
outlier_count = len(data_copy.loc[data['class'] == 1, 'class'])
print(f'size of outlier set: {outlier_count}, size of normal instance set: {nominal_count}')
plt.figure(figsize = (15, 2)) 
fig = sns.countplot(y = 'class', data = data_copy, color = 'b') 
plt.show() 

'''
Pre-Processing
'''
data_class = data_copy['class']
data_train = data_copy.drop(['class'], axis = 1) 

X_train, X_test, y_train, y_test = train_test_split(data_train, data_class,
                                                    test_size = 0.3, random_state = 42) 

#Tune hyper parameters
clf = IsolationForest()  

#Define the parameter grid
param_grid = {'n_estimators': list(range(100, 200, 300)),
              'max_samples': list(range(100, 200, 300)),
              'max_features': [5, 10, 15]}


f1_parameter_scorer = make_scorer(f1_score, average = 'micro') 

parameter_search = GridSearchCV(clf,
                                param_grid,
                                scoring = f1_parameter_scorer,
                                cv = 3) 

parameter_search.fit(X_train, y_train) 
print(parameter_search.best_params_)


'''
Train and initalise the model
'''
iForest = IsolationForest(n_estimators = 100, max_features = 10, max_samples = 400) 


iForest.fit(X_train) 
def evaluate_performance(iForest, X_test, y_test, map_labels):
    prediction_test_set = X_test.copy() 
    prediction_test_set['Pred'] = iForest.predict(X_test) 
    if map_labels:
        prediction_test_set['Pred'] = prediction_test_set['Pred'].map({1: 0, -1: 1})
        
    x_prediction = prediction_test_set['Pred']
    conf_matrix = confusion_matrix(x_prediction, y_test) 
    
    sns.heatmap(pd.DataFrame(conf_matrix, columns = ['Actual', 'Predicted']),
                xticklabels = ['Regular [0]', 'Anomaly [0]'],
                yticklabels = ['Regular [0]', 'Anomaly [1]'],
                annot = True, linewidth = 0.5, cmap = "YlGnBu") 
    plt.ylabel("Predictions") 
    plt.xlabel ("Actual") 
    
    print(classification_report(x_prediction, y_test)) 
    
    iForest_score = score(x_prediction, y_test, average = 'macro')
    
    f1_score_results = np.round(iForest_score[2] * 100, 2) 
    print('f1_score: ', f1_score_results)  
    
    return iForest_score

name = "Isolation Forest Performance on Annthyroid Data set"

map_labels = True

iForest_score = evaluate_performance(iForest, X_test, y_test, map_labels)

performance_df = pd.DataFrame().append({'name': name,
                                        'f1_score': iForest_score[0],
                                        'precision': iForest_score[1],
                                        'recall': iForest_score[2]},
                                       ignore_index = True) 

print(performance_df) 