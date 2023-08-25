# import pytorch and matplotlib
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn

#check pytorch version
print(torch.__version__)

#setup device agnostic code
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# create some data using linear regression formula of y = weight * x + bias

weight = 0.7    
bias = 0.3

#create range values

start = 0
end = 1
step = 0.02

create 