# -*- coding: utf-8 -*-
"""
  Created on Wed Apr 17 18:13:40 2019
  In this code, I implement Single Layer Perceptron 
  the inputSize,outputSize parameters contain the number of neurons in the input and output layers. 
  Learning_rate refers to the learning rate and epochs parameter refers to numper of epochs 
  So, for example, if we want to create a NN object with 5 neurons in the input layer,  3 neurons in 
  the output layer, learning rate =0.7 and number of epochs =100 we'd do this with the code: net = SLP(5,3,0.7,100)
  
  
@author: Sarah Osama
"""

import numpy as np
#import numpy.matlib
class SLP:
    # The  __init__ is used to initialise newly created instance, and receives parameters
    def __init__(self,inputSize,learningRate=0.01, epochs=50):
        self.inputSize = inputSize
        self.learningRate = learningRate
        self.epochs = epochs
        self.biase= 0
        self.weights = np.zeros(self.inputSize)
        
    
    # activation function
    @staticmethod        
    def activation_function(net):
        if net>0:
            return 1  
        return 0
    
    def feedforward_activation(self,x_i):
        net = (self.weights * x_i).sum()+self.biase
        y_hat= self.activation_function(net) 
        return y_hat

    def learning(self,deltaTerm,o_pi):
        deltaWeights = self.learningRate * deltaTerm * o_pi
        deltaBiase = self.learningRate * deltaTerm 
        self.weights += deltaWeights
        self.biase += deltaBiase
       
    def training_phase(self,x,y):
        print("Training phase using SGD:" ) 
        wChange = True
        current_epoch = 1
        while (wChange == True) and (current_epoch <= self.epochs):            
            wChange = False
            for i in range(len(x)):
                y_hat=self.feedforward_activation(x[i,:]) 
                deltaTerm= y[i]-y_hat
                if deltaTerm != 0.0:
                    self.learning(deltaTerm,x[i,:])
                    wChange = True
            current_epoch +=1
        return self.weights,self.biase  
     
    def testing_phase(self,x,y):
        print("Testing phase:" )
        print("input ",x)
        print("y ",y)
        YS=list()
        for i in range(len(x)):
            y_hat=self.feedforward_activation(x[i,:])
            YS.append(y_hat)
        return YS
