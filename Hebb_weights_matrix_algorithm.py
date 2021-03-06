# -*- coding: utf-8 -*-
"""
  Created on Mon Apr  2 21:15:56 2019
  In this code, I implement Hebb Neural Network using alternative method: Weights_Matrix 
  the inputSize,outputSize parameters contain the number of neurons in 
  the input and output layers. 
  So, for example, if we want to create a NN object with 5 neurons in the input layer and  3 neurons in 
  the output layer, we'd do this with the code: net = Network(5,3)
  The weights in the Network object are all claculated based on thetrainig sampels 
  
@author: Sarah Osama
"""

import numpy as np
#import numpy.matlib
class Hebb_Network_Weights_Matrix:
    def __init__(self,inputSize,outputSize):
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.weight_matrix = np.zeros((self.inputSize,self.outputSize))    
    
    # Define the activation function as a statistic method
    @staticmethod        
    def activation_function(net):
        if net>0:
            return 1  
        elif net<0:
            return -1
        return 0
    
    # The following function was defined to construct the weights matrix, and also this function takes 2 parameters
    def create_weight_matrix(self,x,y):
       
        for i in range(len(x)):
            self.weight_matrix+=x[i].T* y[i]
            #print(x[i].T.shape, y[i].shape)

    def feedforward_activation(self,x_i):
        #y_hat = np.zeros(self.outputSize)
        y_hat = np.empty(self.outputSize)
       
        for index in range(self.outputSize):
            net=np.multiply(x_i,self.weight_matrix[:,index]).sum()
            y_hat[index]= self.activation_function(net)
        return y_hat
    
    def training_phase(self,x,y):
        print("Training phase:" )
        #self.create_weight_matrix(x,y)
        for i in range(len(x)):
            y_hat=self.feedforward_activation(x[i,:]) #,self.weight_matrix
            print("x and y_hat: ",x[i,:], y_hat)
        print("final weight : ",self.weight_matrix)
        print(self.weight_matrix)
        return self.weight_matrix  
     
    def testing_phase(self,x,y):
        print("Testing phase:" )
        #print("Final weights: ",self.weight_matrix)
        print("input ",x)
        print("y ",y)
        testList=list()
        for i in range(len(x)):
            y_hat=self.feedforward_activation(x[i,:])
            print("y_hat ", y_hat)  
            testList.append(y_hat)
        return testList    
   
    
    
    
    
    
    
    
    
    
    
    