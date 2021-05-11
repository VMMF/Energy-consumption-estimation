# Activating the virtual environment

For the creation of virtual environments (VE) miniconda 3.8 64 bit is used. You can download anaconda here: 
https://docs.conda.io/en/latest/miniconda.html. Version 3.9 of Python is not currently supported by the latest release of Tensorflow. Do not use 32 bit version of miniconda.

After installing anaconda into your system follow the steps bellow to work in a virtual environment.

Creation of the VE:
```
conda create python=3.8 --name deep-ts
```

Activating the VE:
```
CALL conda.bat activate deep-ts
or
conda activate deep-ts
```

Installing all the packages from the **requirements.txt** file to the virtual environment:
```
pip install -r requirements.txt
```

If you are using Microsoft Visual Studio code there may be some additional pop ups indiciating that some packages should be installed (linter or ikernel).

In VS Code select the interpreter inside the conda VE
https://code.visualstudio.com/docs/python/environments

# Time series data

The data is taken from: https://www.kaggle.com/robikscube/hourly-energy-consumption. The data is an hourly time series regarding the power consumption (in MW) in the Dayton region. The data spans from 2004-10-01 to 2018-08-03 (**n=121271**)