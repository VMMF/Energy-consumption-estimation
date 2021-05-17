# Ploting packages
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


from datetime import datetime, timedelta
import pandas as pd 

# The deep learning class
from deep_model import DeepModelTS

# Reading the neural network parameters configuration file
import yaml

# Directory managment 
import os

# Data preprocessing
from sklearn.preprocessing import MinMaxScaler

# Reading the Deep Neural Network hyper parameters
with open(f'{os.getcwd()}\\DNN_params.yml') as file:
    conf = yaml.load(file, Loader=yaml.FullLoader)

city_name = 'DAYTON_MW'

# Reading the data from csv. The dataframe will contain one column per column in the csv
df = pd.read_csv('input/' + str(city_name)+'.csv')
# creating Timestamp objects for array elements on the Datetime column
df['Datetime'] = [datetime.strptime(x, '%Y-%m-%d %H:%M:%S') for x in df['Datetime']] # data is in format : 2004-12-31 01:00:00

# Averaging MW of possible duplicates in Datetime column, not using Datetime columns as new index, keeping 1,2,3...
df = df.groupby('Datetime', as_index=False)[city_name].mean()

# Sorting the values by Datetime inside the same dataframe
df.sort_values('Datetime', inplace=True)

# #TODO check data scaling
# scaler = MinMaxScaler(feature_range=(0, 1))
# scaler = scaler.fit(df.values)
# normalized = scaler.transform(df.values)

# plt.plot('Datetime',city_name,data = normalized)
# plt.show()

# Initiating the class 
deep_learner = DeepModelTS(
    data=df, 
    Y_var= city_name,
    lag=conf.get('lag'),
    LSTM_layer_depth=conf.get('LSTM_layer_depth'),
    batch_size = conf.get('batch_size'),
    epochs=conf.get('epochs'),
    train_test_split=conf.get('train_test_split') # The share of data that will be used for validation
)

# Fitting the model 
_, history = deep_learner.LSTModel(return_metrics = True)
plt.plot(history.history['loss'])
plt.legend( ['mse'] )
plt.title("Cost function")  
plt.show()

# Making the prediction on the validation set
# Only applicable if train_test_split in the DNN_params.yml > 0
yhat = deep_learner.predict()

if len(yhat) > 0:

    # Constructing the forecast dataframe
    fc = df.tail(len(yhat)).copy() # copying the last yhat rows from the data
    fc.reset_index(inplace=True) # When we reset the index, the old index is added as a column, and a new sequential index is used
    fc['forecast'] = yhat #creating a new forecast column

    # Ploting the forecasts

    plt.figure(figsize=(12, 8))
    for dtype in [city_name, 'forecast']:
        # the dataframe in fc has the column named Datetime used as x axis in plot
        # it also has the columns city_name and forecast used as y axis
        plt.plot('Datetime', dtype, data=fc, label=dtype, alpha=0.8 )
    plt.title("Validation set forecast")    
    plt.legend()
    plt.grid()
    plt.show()   
    
    
# Forecasting n steps ahead   

# Creating the model using full data and forecasting n steps ahead
deep_learner = DeepModelTS(
    data=df, 
    Y_var= city_name,
    lag=conf.get('lag'),
    LSTM_layer_depth=conf.get('LSTM_layer_depth'),
    batch_size = conf.get('batch_size'),
    epochs=conf.get('epochs'),
    train_test_split=0 
)

# Fitting the model 
deep_learner.LSTModel()

# Forecasting n steps ahead
n_ahead = 168
yhat = deep_learner.predict_n_ahead(n_ahead)
yhat = [y[0][0] for y in yhat]

# Constructing the forecast dataframe
fc = df.tail(400).copy() 
fc['type'] = 'original'

last_date = max(fc['Datetime'])
hat_frame = pd.DataFrame({
    'Datetime': [last_date + timedelta(hours=x + 1) for x in range(n_ahead)], 
    city_name: yhat,
    'type': 'forecast'
})

fc = fc.append(hat_frame)
fc.reset_index(inplace=True, drop=True)

# Ploting future values forecasts 
plt.figure(figsize=(12, 8))
for col_type in ['original', 'forecast']:
    plt.plot('Datetime', city_name, data=fc[fc['type']==col_type], label=col_type )

plt.title("Future values forecast") 
plt.legend()
plt.grid()
plt.show()    