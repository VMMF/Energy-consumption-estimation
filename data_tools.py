
import numpy as np

# Data preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.exceptions import NotFittedError
from keras import backend as K



def timeseries_to_XY(samples: list, buffer: int) -> tuple:
    """
    A method to create X and Y matrix from a time series. 
    Before machine learning can be used, time series forecasting problems must be re-framed as supervised learning problems. 
    From a sequence to pairs of input and output sequences.
    """
    x, y = [], []

    if len(samples) - buffer <= 0:
        x.append(samples) # if not enough samples for the lag, use all the ones you have
    else:
        for i in range(len(samples) - buffer):
            y.append(samples[i + buffer]) #start filling after the 1st lag samples. Y is [len(ts) - lag] x 1
            x.append(samples[i:(i + buffer)]) # fill buffers of lag samples. X is [len(ts) - lag] x lag

    # each Y sample will have an X buffer of size "previous lag samples" associated to it
    x, y = np.array(x), np.array(y)

    # Reshaping the X array to be compatible with model.predict()
    x = np.reshape(x, (x.shape[0], x.shape[1], 1))

    return x, y    



# #TODO fix global variable, create a Scaler class
# global scaler
# scaler = MinMaxScaler(feature_range=(0, 1)) # TODO if model activation is tanh use [-1,1], for Relu use [0,1]  


# def scaler_learn(city_name_MW):

#     raw_MW = np.array ( city_name_MW.astype('float32') )
#     raw_MW = raw_MW.reshape(raw_MW.shape[0], 1) #reshape operation is not in place
#     scaler.fit(raw_MW)


# def scaler_work(city_name_MW,FWD = True):
#     """
#     A method to scale and de_scale data
#     city_name_MW is a pandas series
#     TODO: make sure de_scale is not called before scale
#     """   
    
#     if FWD:
#         # scaling
#         raw_MW = np.array ( city_name_MW.astype('float32') )
#         raw_MW = raw_MW.reshape(raw_MW.shape[0], 1) #reshape operation is not in place
#         scaled = scaler.transform(raw_MW)
#         return scaled
#     else:
#         # scalling back
#         recons_MW = np.array ( city_name_MW )
#         recons_MW = recons_MW.reshape(recons_MW.shape[0], 1) #reshape operation is not in place
#         try:
#             de_scaled = scaler.inverse_transform(recons_MW)
#             return de_scaled
#         except NotFittedError as e:
#             print("\n")
#             print("Make sure to call this method with FWD = True, before calling it with FWD = False")
#             print(repr(e))
#             print("\n")


def rmse(y_true, y_pred):
    """
    A method to calculate root mean square error. (RMSE) punishes large errors 
    and results in a score that is in the same units as the forecast data,
    TODO test rmse = sqrt(mean_squared_error(test, predictions))
    https://www.kaggle.com/learn-forum/52081
    """
    
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


#TODO check if timeseries is stationary, otherwise detrend    