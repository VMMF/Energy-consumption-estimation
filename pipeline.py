# Ploting packages
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


from datetime import datetime, timedelta
import pandas as pd 
import numpy as np

# The deep learning class
from model_lstm import ModelLSTM

# Data preprocessing
from min_max_scaler import MinMax

# Reading the neural network parameters configuration file
import yaml

# Directory managment 
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' #force CPU use

# Reading the Deep Neural Network hyper parameters
with open(f'{os.getcwd()}\\DNN_params.yml') as file:
    conf = yaml.load(file, Loader=yaml.FullLoader)

city_name_MW = 'PJME_MW'

# Reading the data from csv. The dataframe will contain one column per column in the csv
df = pd.read_csv('input/' + str(city_name_MW)+'.csv')

# creating Timestamp objects for array elements on the Datetime column
df['Datetime'] = [datetime.strptime(x, '%Y-%m-%d %H:%M:%S') for x in df['Datetime']] # data is in format : 2004-12-31 01:00:00

#replacig any NA value with 0
df[city_name_MW].fillna(0, inplace=True)

# Averaging MW of possible duplicates in Datetime column, not using Datetime columns as new index, keeping 1,2,3...
df = df.groupby('Datetime', as_index=False)[city_name_MW].mean()

#TODO analyse and transform (if required) time series to stationary
#https://machinelearningmastery.com/time-series-forecasting-long-short-term-memory-network-python/

# Sorting the values by Datetime inside the same dataframe
df.sort_values('Datetime', inplace=True)

# Plotting dataset as reference
plt.figure()
plt.plot('Datetime',city_name_MW,data = df)
plt.title("Total consumption " + str(city_name_MW) + " per hour") 
plt.ylabel("MW")
plt.xlabel("hours")
plt.draw()
plt.pause(0.01) #avoid blocking thread while displaying image
#TODO plot in another thread

# Initiating the class 
deep_learner = ModelLSTM(
    data=df, 
    Y_var= city_name_MW,
    estimate_based_on = conf.get('estimate_based_on'),
    LSTM_layer_depth = conf.get('LSTM_layer_depth'),
    batch_size = conf.get('batch_size'),
    # scaler = MinMax(feature_range = (0, 1)),
    epochs = conf.get('epochs'),
    validation_split = conf.get('validation_split'), # The share of data that will be used for validation
    test_split = conf.get('test_split')
)

# Fitting the model 
deep_learner.create_model()
history = deep_learner.train(return_metrics = True)

if(len(history.epoch)>1):
    plt.figure()
    plt.plot(history.history['loss'], label = 'rmse_train')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label = 'rmse_validation')
    plt.legend()
    plt.title("Cost function")  
    plt.draw()
    plt.pause(0.01)
    


# Making the prediction on the validation set
# Only applicable if validation_split in the DNN_params.yml > 0
yhat, error = deep_learner.validate(return_metrics = True)

if len(yhat) > 0:

    # Constructing the forecast dataframe
    validation = deep_learner.Y_validate
    fig1 = plt.figure(figsize=(15, 10))
    ax1 = fig1.add_subplot(111)
    plt.plot(validation)
    plt.plot(yhat)

    # TODO Add the time index to the plot knowing the validation data range in the original timeseries


    # fc = df.tail(len(yhat)).copy() # copying the last yhat rows from the data
    # fc.reset_index(inplace=True) # When we reset the index, the old index is added as a column, and a new sequential index is used
    # fc['forecast'] = yhat #creating a new forecast column

    # # Ploting the forecasts

    # plt.figure(figsize=(12, 8))
    # for dtype in [city_name_MW, 'forecast']:
    #     # the dataframe in fc has the column named Datetime used as x axis in plot
    #     # it also has the columns city_name and forecast used as y axis
    #     plt.plot('Datetime', dtype, data=fc, label=dtype, alpha=0.8 )



    plt.title('Validation set forecast')  
    plt.text(0.5, 0.95, 'Error %:'  + str(error*100) , horizontalalignment='center', verticalalignment='center',transform = ax1.transAxes)
    plt.legend()
    plt.grid()
    plt.draw()
    plt.pause(0.01)  
    


# Making the prediction on the test set
# Only applicable if test_split in the DNN_params.yml > 0
yhat,error = deep_learner.test(return_metrics = True)

if len(yhat) > 0:

    fc = df.tail(len(yhat)).copy() # copying the last yhat rows from the data
    fc.reset_index(inplace=True) # When we reset the index, the old index is added as a column, and a new sequential index is used
    fc['test forecast'] = yhat #creating a new forecast column

    # Ploting the forecasts

    fig2 = plt.figure(figsize=(15, 10))
    ax2 = fig2.add_subplot(111)
    for dtype in [city_name_MW, 'test forecast']:
        # the dataframe in fc has the column named Datetime used as x axis in plot
        # it also has the columns city_name and forecast used as y axis
        plt.plot('Datetime', dtype, data=fc, label=dtype, alpha=0.8 )

    plt.title("Test set forecast")    
    plt.text(0.5, 0.95, 'Error %:'  + str(error*100) , horizontalalignment='center', verticalalignment='center',transform = ax2.transAxes)
    plt.legend()
    plt.grid()
    plt.show()  
    # plt.pause(0.01) 








# # Forecasting n steps ahead   

# # Creating a new model without validation set (full data) and forecasting n steps ahead
# #TODO use previously trained model
# deep_learner = ModelLSTM(
#     data=df, 
#     Y_var= city_name_MW,
#     estimate_based_on=conf.get('estimate_based_on'),
#     LSTM_layer_depth=conf.get('LSTM_layer_depth'),
#     batch_size = conf.get('batch_size'),
#     epochs=conf.get('epochs'),
#     validation_split=0 
# )

# # Fitting the model 
# deep_learner.create_model()
# history = deep_learner.train(return_metrics = True)

# if(len(history.epoch)>1):
#     plt.figure()
#     plt.plot(history.history['loss'], label = 'rmse_train')
#     if 'val_loss' in history.history:
#         plt.plot(history.history['val_loss'], label = 'rmse_validation')
#     plt.legend()
#     plt.title("Cost function")  
#     plt.draw()
#     plt.pause(0.01)

# # Forecasting 1 week ahead 7 * 24h = 168h
# n_ahead = 168 #TODO put this on a separate file
# yhat = deep_learner.predict_ahead(n_ahead) # predicts future MW usage
# yhat = [y[0][0] for y in yhat] # TODO check yhat = np.squeeze(yhat)

# # Constructing the forecast dataframe
# fc = df.tail(504).copy() # 3 weeks of data
# fc['type'] = 'original'

# last_date = max(fc['Datetime'])
# hat_frame = pd.DataFrame({
#     'Datetime': [last_date + timedelta(hours=x + 1) for x in range(n_ahead)], 
#     city_name_MW: yhat,
#     'type': 'forecast'
# })

# fc = fc.append(hat_frame)
# fc.reset_index(inplace=True, drop=True) # don't save new index as column

# # Ploting future values forecasts 
# plt.figure(figsize=(12, 8))
# for col_type in ['original', 'forecast']:
#     plt.plot('Datetime', city_name_MW, data=fc[fc['type']==col_type], label=col_type )

# plt.title("Forecasting 1 week") 
# plt.legend()
# plt.grid()
# plt.show()    