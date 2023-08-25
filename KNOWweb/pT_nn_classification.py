# import sklearn and make circles

import sklearn
from sklearn.datasets import make_circles
import matplotlib.pyplot as plt
import torch

# make 1000 samples
n_samples = 1000

# create circles
X, y = make_circles(n_samples, noise=0.03, random_state=42)

# plot the circles
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdBu)
plt.show()

#turn X and y into tensors
X = torch.from_numpy(X).float()
y = torch.from_numpy(y).float()

#check the shape of X and y
print(X.shape, y.shape)

# split data into training and testing data
from sklearn.model_selection import train_test_split

# split data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.2, 
                                                    random_state=42)

# build a model, setup device agnostic code
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# create a model
class Cir_Model(nn.Module):
    def __init__(self, input_size=2, hidden_size=5, output_size=1):
        super(Cir_Model, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x
    
# setup a loss function and optimizer

model = Cir_Model()
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# calculate the accuracy - out of 100 examples, how many are correct

def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))
    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    acc = torch.round(acc * 100)
    
    return acc

# test the first 5 values from the model
y_pred = model(X_test)
print(y_pred[:5])

#use sigmoid function to convert the output to a probability
y_pred_prob = torch.sigmoid(y_pred)
print(y_pred_prob[:5])

print(torch.round(y_pred_prob[:5]))

# train the model
torch.manual_seed(42)
epochs = 1000

for epoch in range(epochs):
    model.train()
    
    #1. forward propagation
    y_logits = model(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))

    #2. calculate loss
    loss = criterion(y_logits, y_train)
    acc = binary_acc(y_logits, y_train)
    
    #5. zero the gradients
    optimizer.zero_grad()
    
    #3. backward propagation
    loss.backward()
    
    #4. weight optimization
    optimizer.step()
    
    model.eval()
    with torch.no_grad():
        y_logits_test = model(X_test)
        y_pred_test = torch.round(torch.sigmoid(y_logits_test))
        acc_test = binary_acc(y_logits_test, y_test.unsqueeze(1))

    
    #6. print the loss every 10 epochs
    if epoch % 100 == 0:
        print('Epoch {}: train loss: {}'.format(epoch, loss.item()))
        print('Epoch {}: train acc: {}'.format(epoch, acc.item()))
        print('Epoch {}: test acc: {}'.format(epoch, acc_test.item()))
        print('-----------------------------------------------')

import requests
from pathlib import Path

if Path("helper_functions.py").is_file():
    print("helper_functions.py already exists")
else:
    print("Downloading helper_functions.py")
    request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")  
    with open("helper_functions.py", 'wb') as f:
        f.write(request.content)

import helper_functions
from helper_functions import plot_decision_boundary, plot_predictions

# plot decision boundary for training and testing data
print(f"X_ train:{X_train}, y_train:{y_train}")

plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.title('Train')
plot_decision_boundary(model, X_train, y_train)
plt.subplot(1,2,2)
plt.title('Test')
plot_decision_boundary(model, X_test, y_test)

plt.show()

# create a model with 3 hidden layers
class Cir_Model2(nn.Module):
    def __init__(self, input_size=2, hidden_size=10, output_size=1):
        super(Cir_Model2, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        return x
    
# make model1 using the new model
model1 = Cir_Model2()

# setup loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model1.parameters(), lr=0.1)

# train the model
torch.manual_seed(42)

epochs = 1000

for epoch in range(epochs):
    
    model1.train()

    #1. forward propagation
    y_logits = model1(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))

    #2. calculate loss
    loss = criterion(y_logits, y_train)
    acc = binary_acc(y_train, y_pred)

    #3. zero the gradients
    optimizer.zero_grad()

    #4. backward propagation
    loss.backward()

    #5. weight optimization
    optimizer.step()

    model1.eval()
    with torch.no_grad():
        y_logits_test = model1(X_test).squeeze()
        y_pred_test = torch.round(torch.sigmoid(y_logits_test))
        test_loss = criterion(y_logits_test, y_test)
        acc_test = binary_acc(y_test, y_pred_test)

# print the loss every 100 epochs
    if epoch % 100 == 0:
        print('Epoch {}: train loss: {}'.format(epoch, loss.item()))
        print('Epoch {}: train acc: {}'.format(epoch, acc.item()))
        print('Epoch {}: test loss: {}'.format(epoch, test_loss.item()))
        print('Epoch {}: test acc: {}'.format(epoch, acc_test.item()))
        print('-----------------------------------------------')

# plot decision boundary for training and testing data
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.title('Train')
plot_decision_boundary(model1, X_train, y_train)
plt.subplot(1,2,2)
plt.title('Test')
plot_decision_boundary(model1, X_test, y_test)

plt.show()

# create a dataset with linearly separable data

weight = 0.7
bias = 0.3
start = 0
end = 1
step = 0.01

X_reg = torch.arange(start, end, step).unsqueeze(dim=1)
y_reg = weight * X_reg + bias

# create train and test data split 8:2

X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(X_reg,y_reg, test_size=0.2, random_state=42)

# plot the data
plt.scatter(X_reg_train, y_reg_train, c='b', label='train')
plt.scatter(X_reg_test, y_reg_test, c='r', label='test')
plt.legend()
plt.show()

# a new model for regression similar to model1
class Reg_Model(nn.Module):
    def __init__(self, input_size=1, hidden_size=10, output_size=1):
        super(Reg_Model, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        return x

# loss and optimizer
model_reg = Reg_Model()
criterion = nn.L1Loss()
optimizer = torch.optim.SGD(model_reg.parameters(), lr=0.1)

# train the model

torch.manual_seed(42)

epochs = 1000

for epoch in range(epochs):
        
        model_reg.train()
    
        #1. forward propagation
        y_pred = model_reg(X_reg_train)
    
        #2. calculate loss
        loss = criterion(y_pred, y_reg_train)
    
        #3. zero the gradients
        optimizer.zero_grad()
    
        #4. backward propagation
        loss.backward()
    
        #5. weight optimization
        optimizer.step()
    
        model_reg.eval()
        with torch.no_grad():
            y_pred_test = model_reg(X_reg_test)
            test_loss = criterion(y_pred_test, y_reg_test)

#6. print the loss every 100 epochs
if epoch % 100 == 0:
    print('Epoch {}: train loss: {}'.format(epoch, loss.item()))
    print('Epoch {}: test loss: {}'.format(epoch, test_loss.item()))
    print('-----------------------------------------------')

#plot the inference data
model_reg.eval()
with torch.no_grad():
    y_pred = model_reg(X_reg_test)

#plot the predictions
plot_predictions(X_reg_train, y_reg_train,X_reg_test, y_reg_test,y_pred)
plt.show()

## create a toy multiclass dataset

import numpy as np
from sklearn.datasets import make_blobs

# set the hyperparameters for the data creation
num_class = 4
num_features = 2
random_seed = 42

# create the data
X_blob, y_blob = make_blobs(n_samples=1000, 
                            n_features=num_features, 
                            centers=num_class,
                            cluster_std = 1.5, 
                            random_state=random_seed)

# turn data into tensors
X_blob = torch.from_numpy(X_blob).float()
y_blob = torch.from_numpy(y_blob).long()

# split into train and test data
X_blob_train, X_blob_test, y_blob_train, y_blob_test = train_test_split(X_blob, 
                                                                        y_blob, 
                                                                        test_size=0.2, 
                                                                        random_state=42)

#plot the data
plt.scatter(X_blob[:,0], X_blob[:,1], c=y_blob, cmap=plt.cm.RdYlBu)
plt.show()

# make a model for multiclass classification
class Blob_Model(nn.Module):
    def __init__(self, input_size=2, hidden_size=8, output_size=4):
        super(Blob_Model, self).__init__()
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )        


        
    def forward(self, x):
        logits = self.linear_layer_stack(x)
        return logits
    
# create an instance of the blob model and send it to target device

model_blob = Blob_Model().to(device)

# create a loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_blob.parameters(), lr=0.1)

# create a training loop
torch.manual_seed(42)

epochs = 1000

for epoch in range(epochs):

    model_blob.train()

    #1. forward propagation
    y_logits = model_blob(X_blob_train)
    y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)

    #2. calculate loss
    loss = criterion(y_logits, y_blob_train)
    acc = binary_acc(y_blob_train, y_pred)

    #3. zero the gradients
    optimizer.zero_grad()

    #4. backward propagation
    loss.backward()

    #5. weight optimization
    optimizer.step()

    model_blob.eval()
    with torch.no_grad():
        y_logits_test = model_blob(X_blob_test)
        test_loss = criterion(y_logits_test, y_blob_test)

    #6. print the loss every 100 epochs
    if epoch % 100 == 0:
        print('Epoch {}: train loss: {}'.format(epoch, loss.item()))
        print('Epoch {}: test loss: {}'.format(epoch, test_loss.item()))
        print('-----------------------------------------------')
        print('-----------------------------------------------')

# plot the decision boundary
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.title('Train')
plot_decision_boundary(model_blob, X_blob_train, y_blob_train)
plt.subplot(1,2,2)
plt.title('Test')
plot_decision_boundary(model_blob, X_blob_test, y_blob_test)

plt.show()
