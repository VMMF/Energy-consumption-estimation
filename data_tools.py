
import numpy as np

# Data preprocessing





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
    # The LSTM needs data with the format of [samples, time steps and features].
    x = np.reshape(x, (x.shape[0], x.shape[1], 1))

    return x, y    




#TODO check if timeseries is stationary, otherwise detrend    

def timeseries_diff (timeseries,interval=1):
    """
    A method to remove a possible linear trend in the data by differentiation. 
    Can be called repeatedly in case a difference order quadratic, cubic, etc is required.
    For time series with a seasonal component, the lag may be expected to be the period (width) of the seasonality
    TODO be able to specify the order or number of times to perform the differencing operation
    See series.diff()
    """

    diff = list()
    for i in range(interval, len(timeseries)):
        value = timeseries[i] - timeseries[i - interval]
        diff.append(value)
    return diff

# def timeseries_inv_diff (timeseries_diff):
#     inverted = list()
#     for i in range(len(timeseries_diff)):
#         value = yhat[i] + history[-interval]
#         # value = inverse_difference(series, differenced[i], len(series)-i)
#         inverted.append(value)




# from pandas import read_csv
# from pandas import datetime
# from sklearn.linear_model import LinearRegression
# from matplotlib import pyplot
# import numpy
 
# def parser(x):
# 	return datetime.strptime('190'+x, '%Y-%m')
 
# series = read_csv('shampoo-sales.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
# # fit linear model
# X = [i for i in range(0, len(series))]
# X = numpy.reshape(X, (len(X), 1))
# y = series.values
# model = LinearRegression()
# model.fit(X, y)
# # calculate trend
# trend = model.predict(X)
# # plot trend
# pyplot.plot(y)
# pyplot.plot(trend)
# pyplot.show()
# # detrend
# detrended = [y[i]-trend[i] for i in range(0, len(series))]
# # plot detrended
# pyplot.plot(detrended)
# pyplot.show()