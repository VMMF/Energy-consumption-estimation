# Data wrangling
import pandas as pd
import numpy as np

# Deep learning: 
from keras.models import Sequential
from keras.layers import LSTM, Dense


from data_tools import *


class ModelLSTM(object):
    """
    A class to create a long short-term memory network
    Should inherit from a abstract model class
    """

    def __init__(self,data: pd.DataFrame, Y_var: str,estimate_based_on: int, LSTM_layer_depth: int, epochs=10, batch_size=256,train_validation_split=0 ):
        self.data = data 
        self.Y_var = Y_var 
        self.estimate_based_on = estimate_based_on 
        self.LSTM_layer_depth = LSTM_layer_depth
        self.batch_size = batch_size
        self.epochs = epochs
        self.train_validation_split = train_validation_split
    

    def create_data_for_model(self, use_last_n=None ):
        """
        A method to create data for the neural network model
        It separates train from validation set (no model test set for now)
        # TODO should be inherited from an abstract class
        """
        # Extracting the main variable we want to model/forecast
        y = self.data[self.Y_var].tolist()

        # Subseting the time series if needed
        if use_last_n is not None:
            y = y[-use_last_n:] #assume your time series are the last use_last_n only

        # The X matrix will hold the lags of Y 
        X, Y = timeseries_to_XY(y, self.estimate_based_on)

        # Creating training and test sets 
        X_train = X
        X_test = []

        Y_train = Y
        Y_test = []

        #TODO consider model.validation_split or https://datascience.stackexchange.com/questions/38955/how-does-the-validation-split-parameter-of-keras-fit-function-work
        if self.train_validation_split > 0:

            index = round(len(X) * self.train_validation_split)
            X_train = X[0:(len(X) - index)]
            X_test = X[len(X_train):]     
            
            Y_train = Y[:(len(X) - index)]
            Y_test = Y[len(Y_train):]

        return X_train, X_test, Y_train, Y_test


    def create_model(self,return_metrics = False):
        """
        Creating an LSTM model 
        TODO : Allow passing different model metrics
        TODO : Create model should inherit from an abstract class that would be comon for many models
        Guide: https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
        """
        # Defining the model
        model = Sequential() #We create a Sequential model and add layers one at a time until we are happy with our network architecture.
        
        model.add(LSTM(self.LSTM_layer_depth, activation='relu', input_shape=(self.estimate_based_on, 1)))
        model.add(Dense(1)) # fully-connected network structure, using linear activation (Regression Problem)
        #TODO add val_loss
        # acc and val_acc are only for classification
        model.compile(optimizer='adam', loss= rmse ) # efficient stochastic gradient descent algorithm and mean squared error for a regression problem
 
        # Saving the model to the class 
        self.model = model

        return model

    def train(self,return_metrics = False):
        """
        Train should inherit from an abstract class. Each model would know the right way to train
        Creating an LSTM model 
        TODO : Allow passing different model metrics

        """

        # Getting the data 
        X_train, X_test, Y_train, Y_test = self.create_data_for_model()

        # Defining the model parameter dict 
        keras_dict = {
            'x': X_train,
            'y': Y_train,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'verbose': 2, # Decreasing verbosity level (0,2,1) accelerates training speed
            'shuffle': False #Don't shuffle the training data before each epoch (bad for )
        }

        if self.train_validation_split > 0:
            keras_dict.update({ 'validation_data': (X_test, Y_test) })

        # Training the model 
        print("\n")
        print("Training the model")
        history = self.model.fit( **keras_dict )

        if return_metrics:
            return history




    def validate(self) -> list:
        """
        A method to predict the validation set used in creating the model
        """
        yhat = []

        if(self.train_validation_split > 0):
        
            # Getting the last n time series 

            _, X_test, _, _ = self.create_data_for_model() #TODO consider storing X_test from previous call (inside train method)        


            # Making the prediction list 
            yhat = [y[0] for y in self.model.predict(X_test)]

        return yhat

    #TODO Implement forward chaining validation
    # https://stats.stackexchange.com/questions/14099/using-k-fold-cross-validation-for-time-series-model-selection

    def predict_ahead(self, steps: int):
        """
        A method to predict n time steps ahead
        """    
        X, _, _, _ = self.create_data_for_model( use_last_n=self.estimate_based_on )   # X is (1,estimate_based_on)     

        # Making the prediction list 
        yhat = []

        for _ in range(steps):
            # Predict 1 sample
            fcst = self.model.predict(X) 
            yhat.append(fcst)

            # Putting it at the end of the buffer
            X = np.append(X, fcst)

            # Eliminating the 1st sample in the buffer to keep it with dimensions (1,estimate_based_on)
            X = np.delete(X, 0)

            # Reshaping for compatibility with model.predict on the next iteration
            X = np.reshape(X, (1, len(X), 1))

        return yhat    




            
