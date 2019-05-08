# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 20:01:39 2019

@author: Karim Haggagi
"""
import pandas as pd

class Preprocessing :
    dataset=None
    def __init__(self,dataset):
        self.dataset=dataset

    
    def Split(self) :
        dataset = pd.read_csv(self.dataset)
        X = dataset.iloc[:, :-1].values
        Y = dataset.iloc[:, -1].values
        # Missing Data 
        from sklearn.preprocessing import Imputer
        
        imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
        # Fit the imputer object/instance by our predictor matrix X which include 
        # the missing values
        imputer = imputer.fit(X[:, 2:3])
        
        DS = [X[:,0],X[:,1],X[:,2],X[:,3],Y]
        
        # Feature scaling: standardization
        from sklearn.preprocessing import StandardScaler
        # Here we scale all input features 
        #standardization = StandardScaler()
        #X_scaled = standardization.fit_transform(X)
        X_scaled = StandardScaler().fit_transform(X)
        
        
        # Split Data
        from sklearn.model_selection import KFold
        
        #kfold = KFold(n_splits=3, shuffle=False, random_state=None)
        kfold = KFold(n_splits=3, shuffle=True)
        # Returns the number of splitting iterations in the cross-validator
        k = kfold.get_n_splits(X) # or # k = kfold.get_n_splits([X,Y,3])
        
        # Generate indices to split data into training and test set.
        indices = kfold.split(X)
        
        i =1
        X_train=None;y_train=None
        for train_index, test_index in indices:
            i+=1
            X_train, X_test= X[train_index], X[test_index]
            y_train, y_test = Y[train_index], Y[test_index]
        
        return X_train,y_train,X_test,y_test













