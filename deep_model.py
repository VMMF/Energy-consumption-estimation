# Data wrangling
from numpy.core.numeric import False_
import pandas as pd
import numpy as np

# Deep learning: 
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Data preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.exceptions import NotFittedError

class DeepModelTS(object):
    """
    A class to create a deep time series model
    """
    #"static" class variable
    _scaler = None

    def __init__(self,data: pd.DataFrame, Y_var: str,estimate_based_on: int, LSTM_layer_depth: int, epochs=10, batch_size=256,train_validation_split=0 ):
        self.data = data 
        self.Y_var = Y_var 
        self.estimate_based_on = estimate_based_on 
        self.LSTM_layer_depth = LSTM_layer_depth
        self.batch_size = batch_size
        self.epochs = epochs
        self.train_validation_split = train_validation_split

    @staticmethod
    def create_X_Y(ts: list, lag: int) -> tuple:
        """
        A method to create X and Y matrix from a time series. 
        Before machine learning can be used, time series forecasting problems must be re-framed as supervised learning problems. 
        From a sequence to pairs of input and output sequences.
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

        # Reshaping the X array to an LSTM input shape (numpy array necessary for model.predict())
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))

        return X, Y         

    def create_data_for_NN(self, use_last_n=None ):
        """
        A method to create data for the neural network model
        It separates train from validation set (no model test set for now)
        """
        # Extracting the main variable we want to model/forecast
        y = self.data[self.Y_var].tolist()

        # Subseting the time series if needed
        if use_last_n is not None:
            y = y[-use_last_n:]

        # The X matrix will hold the lags of Y 
        X, Y = self.create_X_Y(y, self.estimate_based_on)

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


    def CreateModel(self,return_metrics = False):
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
        model.compile(optimizer='adam', loss='mse') # efficient stochastic gradient descent algorithm and mean squared error for a regression problem
 
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
        X_train, X_test, Y_train, Y_test = self.create_data_for_NN()

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
        A method to predict using the validation data used in creating the model
        """
        yhat = []

        if(self.train_validation_split > 0):
        
            # Getting the last n time series 
            _, X_test, _, _ = self.create_data_for_NN() #TODO consider storing X_test from previous call (inside train method)        

            # Making the prediction list 
            yhat = [y[0] for y in self.model.predict(X_test)]

        return yhat

    def predict_n_ahead(self, n_ahead: int):
        """
        A method to predict n time steps ahead
        """    
        X, _, _, _ = self.create_data_for_NN(use_last_n=self.estimate_based_on)        

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


    @staticmethod
    def data_scale(city_name_MW,FWD = True):
        """
        A method to scale and de_scale data
        city_name_MW is a pandas series
        TODO: make sure de_scale is not called before scale
        """   
        
        if FWD:
            # scalling
            DeepModelTS._scaler = MinMaxScaler(feature_range=(0, 1))
            raw_MW = np.array ( city_name_MW.astype('float32') )
            raw_MW = raw_MW.reshape(raw_MW.shape[0], 1) #reshape operation is not in place
            scaled = DeepModelTS._scaler.fit_transform(raw_MW)
            return scaled
        else:
            # scalling back
            recons_MW = np.array ( city_name_MW )
            recons_MW = recons_MW.reshape(recons_MW.shape[0], 1) #reshape operation is not in place
            try:
                de_scaled = DeepModelTS._scaler.inverse_transform(recons_MW)
                return de_scaled
            except NotFittedError as e:
                print("\n")
                print("Make sure to call this method with FWD = True, before calling it with FWD = False")
                print(repr(e))
                print("\n")
            

