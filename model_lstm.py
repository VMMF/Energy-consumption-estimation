# Data wrangling
import pandas as pd
import numpy as np

# Deep learning: 
from keras.models import Sequential
from keras.layers import LSTM, Dense

#scaler tools
from abstract_scaler import AbstractScaler

from data_tools import *


class ModelLSTM(object):
    """
    A class to create a long short-term memory network
    Should inherit from a abstract model class
    """

    def __init__(self,data: pd.DataFrame, Y_var: str,estimate_based_on: int, LSTM_layer_depth: int, epochs=10, batch_size=256,validation_split = 0,test_split = 0, scaler: AbstractScaler = None ): 
        
        self.estimate_based_on = estimate_based_on 
        self.LSTM_layer_depth = LSTM_layer_depth
        self.batch_size = batch_size
        self.epochs = epochs
        self.validation_split = validation_split
        self.test_split = test_split

        self.scaler = scaler

        self.time_series = data[Y_var].tolist() # Extracting the main variable we want to model/forecast
        X_train, Y_train, X_validate, Y_validate, X_test, Y_test = self.create_data_for_model(self.time_series)
        
        self.X_train = X_train
        self.X_validate = X_validate
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_validate = Y_validate
        self.Y_test = Y_test
    


    def create_data_for_model(self, time_series , use_last_n=None ):
        """
        A method to create data for the neural network model
        It separates train, validation set and test sets
        # TODO should be inherited from an abstract class
        """

        # Subseting the time series if needed
        if use_last_n is not None:
            time_series = time_series[-use_last_n:] #assume your time series are the last use_last_n only

        if self.scaler is not None:

            # Performing data scaling based only on training samples portion to prevent data leaking
            training_time_series = time_series [ : round( len(time_series) * (1 - (self.test_split + self.validation_split) ) ) ]
            self.scaler.learn(training_time_series)
            #scaling the whole dataset
            time_series = self.scaler.work(time_series, FWD = True)

        # The X matrix will hold the lags of Y 
        X, Y = timeseries_to_XY(time_series, self.estimate_based_on)

        # Creating training and test sets 
        X_train = X
        X_validate = []
        X_test = []

        Y_train = Y
        Y_validate = []
        Y_test = []

        len_X = len(X)

        #TODO consider model.validation_split or https://datascience.stackexchange.com/questions/38955/how-does-the-validation-split-parameter-of-keras-fit-function-work


            
        # extracting test segment
        index_test = round(len_X * self.test_split)

        X_train_validation = X[ :(len_X - index_test)]
        X_test = X[len(X_train_validation): ]    

        Y_train_validation = Y[ :(len(X) - index_test)]
        Y_test = Y[len(Y_train_validation): ] 

        # extracting validation segment
        index_validation = round( len(X_train_validation) * self.validation_split)

        X_train = X_train_validation[ :(len(X_train_validation) - index_validation)]
        X_validate = X_train_validation[len(X_train): ] 
        
        Y_train = Y_train_validation[ :(len(X_train_validation) - index_validation)]
        Y_validate = Y_train_validation[len(Y_train): ] 




        return X_train, Y_train, X_validate, Y_validate, X_test, Y_test

    def create_model(self):
        """
        Creating an LSTM model 
        TODO : Allow passing different model metrics
        TODO : Create model should inherit from an abstract class that would be comon for many models
        Guide: https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
        """
        # Defining the model
        model = Sequential() #We create a Sequential model and add layers one at a time until we are happy with our network architecture.
        #TODO do not hardcode activation='relu'
        model.add(LSTM(self.LSTM_layer_depth, activation='relu', input_shape=(self.estimate_based_on, 1)))
        model.add(Dense(1)) # fully-connected network structure, using linear activation (Regression Problem)
        # acc and val_acc are only for classification
        model.compile(optimizer='adam', loss= rmse ) # efficient stochastic gradient descent algorithm and mean squared error for a regression problem
 
        # Saving the model to the class 
        self.model = model

        return model

    def train(self, return_metrics = False):
        """
        Train should inherit from an abstract class. Each model would know the right way to train
        Creating an LSTM model 
        TODO : Allow passing different model metrics

        """

        # Defining the model parameter dict 
        keras_dict = {
            'x': self.X_train,
            'y': self.Y_train,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'verbose': 2, # Decreasing verbosity level (0,2,1) accelerates training speed
            'shuffle': False #Don't shuffle the training data before each epoch (bad for )
        }

        if self.validation_split > 0:
            keras_dict.update({ 'validation_data': (self.X_validate, self.Y_validate) })

        # Training the model 
        print("\n")
        print("Training the model")
        history = self.model.fit( **keras_dict )

        if return_metrics:
            return history


    def validate(self,return_metrics = False) -> list:
        """
        A method to predict the validation set used in creating the model
        """
        yhat = []
        x = []
        y = []
        check_split = None

        x = self.X_validate
        y = self.Y_validate
        check_split = self.validation_split

        return self.__validate_or_test(x, y, check_split , return_metrics)



    def test(self,return_metrics = False) -> list:
        """
        A method to predict the validation set used in creating the model
        """
        yhat = []

        x = self.X_test
        y = self.Y_test
        check_split = self.test_split

        return self.__validate_or_test(x, y, check_split , return_metrics)


    def __validate_or_test (self, x, y, check_split , return_metrics = False ):
        """
        A method to predict x as yhat, compare it to y and return yhat (in original scale if previously normalized).
        Optionally return the prediction error of normalized data
        """
                

        if(check_split > 0):
            # Making the prediction list 
            # TODO review self.model(X_validate,training=False)
            yhat = [y[0] for y in self.model.predict(x)] 


        if return_metrics:
            error = rmse(y,yhat)
            if self.scaler is not None and self.scaler.is_fit:
                yhat = self.scaler.work(yhat, FWD = False)
            return yhat, np.round(error.numpy(), decimals=2) 
        else:
            if self.scaler is not None and self.scaler.is_fit:
                yhat = self.scaler.work(yhat, FWD = False)
            return yhat

    

    #TODO Implement forward chaining validation
    # https://stats.stackexchange.com/questions/14099/using-k-fold-cross-validation-for-time-series-model-selection

    def predict_ahead(self, steps: int):
        """
        A method to predict n time steps ahead
        """    
                    
        X_train, _, _, _, _, _ = self.create_data_for_model(self.time_series, use_last_n=self.estimate_based_on )   # X is (1,estimate_based_on)     

        # Making the prediction list 
        yhat = []

        for _ in range(steps):
            # Predict 1 sample
            fcst = self.model.predict(X_train) 
            yhat.append(fcst)

            # Putting it at the end of the buffer
            X_train = np.append(X_train, fcst)

            # Eliminating the 1st sample in the buffer to keep it with dimensions (1,estimate_based_on)
            X_train = np.delete(X_train, 0)

            # Reshaping for compatibility with model.predict on the next iteration
            X_train = np.reshape(X_train, (1, len(X_train), 1))

        return yhat    




            

