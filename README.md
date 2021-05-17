# Project Description

The aim of this project is to forecast the energy consumption based on pre-existing time series data

It will use a short-term memory (LSTM) deep learning network
https://towardsdatascience.com/illustrated-guide-to-lstms-and-gru-s-a-step-by-step-explanation-44e9eb85bf21

Key concepts of LSTM:
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



Project developped in VS Code 1.56.1 and Windows 10 20H2
# Activating the virtual environment

To avoid conflicts between dependencies it is recommended to use anaconda or miniconda

- Download miniconda :
https://docs.conda.io/en/latest/miniconda.html . I'm using version 3.8 64 bit
Note: Version 3.9 of Python is not currently supported by Tensorflow 2.4.1. Do not use 32 bit version of miniconda. Tensorflow is only compatible with python 64 bit

- Install it. 
Note: https://stackoverflow.com/questions/44515769/conda-is-not-recognized-as-internal-or-external-command/51996934#51996934
I added to the path the following to be able to work with VS code:
C:\Users\Dell\miniconda3\Library\bin
C:\Users\Dell\miniconda3\Scripts
C:\Users\Dell\miniconda3\condabin

To be able to detect Python 3.8.5 in CMD I removed previous Python version from path and added to the system variables path:
C:\Users\Dell\miniconda3

- Install GPU processing tools (optional)
Note: https://www.tensorflow.org/install/gpu

-Create a virtual environment in miniconda (similar to pipenv)
```
conda create python=3.8 --name deep-ts
```

Activate the virtual environment:
```
conda activate deep-ts
or
CALL conda.bat activate deep-ts
```

In VS Code select the Python interpreter inside the conda virtual environment as opposed to the local Python version (possibly not installed through conda)
Command: Ctrl+Shift+P
        Python: Select interpreter
Note: https://code.visualstudio.com/docs/python/environments
Verify that the VS Code settigs.json contains the following ""python.pythonPath": "C:\\Users\\WhateverYourUseris\\miniconda3\\envs\\deep-ts\\python.exe""

Install all the packages from the **dependency_versions.txt** file to the virtual environment:
```
pip install -r dependency_versions.txt
```
It will take a while and install many dependencies 


After this process is finished run pipeline.py


# Time series data

The data is taken from: https://www.kaggle.com/robikscube/hourly-energy-consumption. The data is an hourly time series regarding the power consumption (in MW) in the Dayton region. The data spans from 2004-10-01 to 2018-08-03 (**n=121271**)