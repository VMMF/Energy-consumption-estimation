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