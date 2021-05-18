# Data wrangling
from numpy.core.numeric import False_
import pandas as pd
import numpy as np

# Deep learning: 
from keras.models import Sequential
from keras.layers import LSTM, Dense


class DeepModelTS(object):
    """
    A class to create a deep time series model
    """
    def __init__(self,data: pd.DataFrame, Y_var: str,lag: int, LSTM_layer_depth: int, epochs=10, batch_size=256,train_validation_split=0 ):
        self.data = data 
        self.Y_var = Y_var 
        self.lag = lag 
        self.LSTM_layer_depth = LSTM_layer_depth
        self.batch_size = batch_size
        self.epochs = epochs
        self.train_validation_split = train_validation_split

    @staticmethod
    def create_X_Y(ts: list, lag: int) -> tuple:
        """
        A method to create X and Y matrix from a time series list for the training of 
        deep learning models 
        """
        X, Y = [], []

        if len(ts) - lag <= 0:
            X.append(ts) # if not enough samples for the lag, use all the ones you have
        else:
            for i in range(len(ts) - lag):
                Y.append(ts[i + lag]) #start filling after lag samples. Y is [len(ts) - lag] x 1
                X.append(ts[i:(i + lag)]) # fill a buffer of lag samples. X is [len(ts) - lag] x lag

        # each Y sample will have an X buffer of size "previous lag samples" associated to it
        X, Y = np.array(X), np.array(Y)

        # Reshaping the X array to an LSTM input shape 
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))

        return X, Y         

    def create_data_for_NN(self, use_last_n=None ):
        """
        A method to create data for the neural network model
        It separates train from test set (no model validation set for now)
        """
        # Extracting the main variable we want to model/forecast
        y = self.data[self.Y_var].tolist()

        # Subseting the time series if needed
        if use_last_n is not None:
            y = y[-use_last_n:]

        # The X matrix will hold the lags of Y 
        X, Y = self.create_X_Y(y, self.lag)

        # Creating training and test sets 
        X_train = X
        X_test = []

        Y_train = Y
        Y_test = []

        if self.train_validation_split > 0:
            index = round(len(X) * self.train_validation_split)
            X_train = X[0:(len(X) - index)]
            X_test = X[len(X_train):]     
            
            Y_train = Y[:(len(X) - index)]
            Y_test = Y[len(Y_train):]

        return X_train, X_test, Y_train, Y_test

    def LSTModel(self,return_metrics = False):
        """
        A method to fit the LSTM model 
        Guide: https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
        """
        # Getting the data 
        X_train, X_test, Y_train, Y_test = self.create_data_for_NN()

        # Defining the model
        model = Sequential() #We create a Sequential model and add layers one at a time until we are happy with our network architecture.
        
        model.add(LSTM(self.LSTM_layer_depth, activation='relu', input_shape=(self.lag, 1)))
        model.add(Dense(1)) # fully-connected network structure, using linear activation 
        #TODO identify metrics
        model.compile(optimizer='adam', loss='mse') # efficient stochastic gradient descent algorithm and mean squared error for a regression problem
 

        # Defining the model parameter dict 
        keras_dict = {
            'x': X_train,
            'y': Y_train,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'shuffle': False #Don't shuffle the training data before each epoch
        }

        if self.train_validation_split > 0:
            keras_dict.update({ 'validation_data': (X_test, Y_test) })

        # Training the model 
        print("\n")
        print("Training the model")
        history = model.fit( **keras_dict )

        # Saving the model to the class 
        self.model = model

        if return_metrics:
            return model, history
        else:
            return model

    def predict(self) -> list:
        """
        A method to predict using the validation data used in creating the class
        """
        yhat = []

        if(self.train_validation_split > 0):
        
            # Getting the last n time series 
            _, X_test, _, _ = self.create_data_for_NN()        

            # Making the prediction list 
            yhat = [y[0] for y in self.model.predict(X_test)]

        return yhat

    def predict_n_ahead(self, n_ahead: int):
        """
        A method to predict n time steps ahead
        """    
        X, _, _, _ = self.create_data_for_NN(use_last_n=self.lag)        

        # Making the prediction list 
        yhat = []

        for _ in range(n_ahead):
            # Making the prediction
            fc = self.model.predict(X)
            yhat.append(fc)

            # Creating a new input matrix for forecasting
            X = np.append(X, fc)

            # Ommiting the first variable
            X = np.delete(X, 0)

            # Reshaping for the next iteration
            X = np.reshape(X, (1, len(X), 1))

        return yhat    