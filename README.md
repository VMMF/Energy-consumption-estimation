# Project Description

The aim of this project is to forecast the energy consumption based on pre-existing time series data

It will use a short-term memory (LSTM) deep learning network
https://towardsdatascience.com/illustrated-guide-to-lstms-and-gru-s-a-step-by-step-explanation-44e9eb85bf21
https://colah.github.io/posts/2015-08-Understanding-LSTMs/

# Key concepts of LSTM:
They are a special case of Recurrent Neural Networks (RNN) usel to model sequence data, for instance on speech recognition, speech synthesis, text generation, video caption generation, etc 
These networks were invented to fight the problem of vanishing gradient (which afect early layers) on RNN 
They decide what info to keep in the memory, as opposed to keeping it all like RNN but giving less weight to early samples of the sequence
They are formed by "cells" or groups of neural networks each performing a different role and cooperating as a unit
The cell state is a "spinal cord link" that broadcasts information and in theory has the possibility not to forget. It works like a token ring netwok protocol common bus where all cells have access.
In addition there is also another communication channel from cell to cell in ladder form (from output gate to input gate) known as hidden state
The gates used to forget or keep remembering data have Sigmoids (0-1 output) activation functions

The inputs to a cell are:
1) incoming current input + previous hidden state (used by the 3 gates)
2) the cell state

The forget gate decides what information remains or is forgotten (from incoming current input + previous hidden state) from previous layer
The input gate contributes to updating the cell state. This means it decides what values will be updated (from incoming current input + previous hidden state) 
and is paralleled with a neural netowrk with tanh activation function (the one that would normally modify its weights to learn on a feed-forward neural network). 
The tanh output, known as candidate, is combined with input gate sigmoid output to determined which of the learnt features are kept. This is then added to the output of forget gate to generate the cell state.
The output gate regulates what information (from incoming current input + previous hidden state) is brought to the next layer. Its output is combined with the output of a second tanh network which has the cell state as input. The combination of the 2 conforms the hidden state (laddered communication from cell to cell)



Project developped in VS Code 1.63.2 and Windows 10.0.19043
# Activating the virtual environment

To avoid conflicts between dependencies it is recommended to use anaconda or miniconda

- Download miniconda :
Note: Do not use 32 bit version of miniconda. Tensorflow is only compatible with python 64 bit
https://docs.conda.io/en/latest/miniconda.html . I'm using version 4.10.3 with Python 3.9.5


- Install it. 
Note: https://stackoverflow.com/questions/44515769/conda-is-not-recognized-as-internal-or-external-command/51996934#51996934
I added to the path the following to be able to work with VS code:
C:\Users\WhateverYourUseris\miniconda3\Library\bin
C:\Users\WhateverYourUseris\miniconda3\Scripts
C:\Users\WhateverYourUseris\miniconda3\condabin

Note: WhateverYourUseris in path needs to be replaced by the actual user name

To be able to detect Python 3.9.5 in CMD I removed previous Python version from path and added to the system variables path:
C:\Users\WhateverYourUseris\miniconda3

- Install GPU processing tools (optional)
Note: https://www.tensorflow.org/install/gpu

-Go to the root directory of the project and create a virtual environment, for instance with name series_forecast

pip install pipenv
python -m venv series_forecast

Note : A folder named series_forecast will be created inside your project folder

VS Code will show a prompt (We noticed a new virtual environment has been created. Do you want to select it for the workspace folder?) . Accept yes .

Note: from this point on all the dependencies to be added will be installed inside the virtual environment we have just created

- Activate the virtual environment

cd series_forecast/Scripts
activate

Note: You will see how in the console (series_forecast) will appear in front of the path.

Go back to the project folder with cd.. (twice) 

In VS Code select the Python interpreter inside the virtual environment:

Ctrl+Shift+P 
Python: Select interpreter

Install all the packages from the **dependency_versions.txt** file to the virtual environment:

```
python.exe -m pip install --upgrade pip
pip install -r dependency_versions.txt
```
Note : It will take a while and install many dependencies, fortunately some of them have already been installed in miniconda 

After this process is finished run pipeline.py


# Time series data

The data is taken from: https://www.kaggle.com/robikscube/hourly-energy-consumption. The data is an hourly time series regarding the power consumption (in MW) in different selectable regions.

# References

https://towardsdatascience.com/energy-consumption-time-series-forecasting-with-python-and-lstm-deep-learning-model-7952e2f9a796
https://github.com/Eligijus112/deep-learning-ts
https://www.kaggle.com/robikscube/hourly-energy-consumption?select=NI_hourly.csv

https://github.com/Housiadas/forecasting-energy-consumption-LSTM
https://www.kaggle.com/uciml/electric-power-consumption-data-set

https://cs109-energy.github.io/building-energy-consumption-prediction.html
https://github.com/awslabs/sagemaker-deep-demand-forecast

https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/
https://machinelearningmastery.com/time-series-forecasting-long-short-term-memory-network-python/

https://stackoverflow.com/questions/58378374/why-does-keras-model-predict-slower-after-compile/58385156#58385156