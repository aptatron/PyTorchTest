#import pytorch
import torch
import torch.nn as nn

# import torchvision
import torchvision
from torchvision import transforms
from torchvision import datasets
from torchvision.transforms import ToTensor

# import matplotlib
import matplotlib.pyplot as plt

#check verisons
print(torch.__version__)
print(torchvision.__version__)

#setup training data
from torchvision.datasets import FashionMNIST

train_data = FashionMNIST(root="data", 
                          train=True, 
                          download=True, 
                          transform=ToTensor(),
                          target_transform=None)

test_data = FashionMNIST(root="data",
                            train=False,
                            download=True,
                            transform=ToTensor(),
                            target_transform=None)

# plot a sample
# img, label = train_data[0]
# print(img.shape, label)
# plt.imshow(img.squeeze(), cmap="gray")
# plt.show()

# #plot more samples
# figure = plt.figure(figsize=(8,8))
# cols, rows = 3, 3
# for i in range(1, cols*rows+1):
#     sample_idx = torch.randint(len(train_data), size=(1,)).item()
#     img, label = train_data[sample_idx]
#     figure.add_subplot(rows, cols, i)
#     plt.title(train_data.classes[label])
#     plt.axis("off")
#     plt.imshow(img.squeeze(), cmap="gray")
# plt.show()

# prepare dataloader
from torch.utils.data import DataLoader
train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True)  # shuffle data
test_dataloader = DataLoader(test_data, batch_size=32, shuffle=False)

#create model
class FashionMNISTmodel(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 10),  # 28*28 is the size of the image
            # nn.ReLU(),
            # nn.Linear(512, 512),
            # nn.ReLU(),
            nn.Linear(10, 10)
        )
    
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# create loss and optimizer
model0 = FashionMNISTmodel()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model0.parameters(), lr=0.1)

from helper_functions import accuracy_fn

# create a function to time training
from timeit import default_timer as timer
def print_train_time(start, end):
    total_time = end - start
    print("Training time: ", total_time)
    return total_time


# import tqdm for progress bar

from tqdm.auto import tqdm

#create a training loop and train a model on batch of data
# torch.manual_seed(42)
# train_time_start_on_cpu = timer()

# epochs = 3

# for epoch in tqdm(range(epochs)):
#     print(f"Epoch {epoch+1}\n-------------------------------")
#     ### training
#     train_loss = 0
#     batch_number = 0
#     # add a loop to loop through the batches
#     for batch in train_dataloader:
#         batch_number += 1
#         images, labels = batch

#         model0.train()

#         # forward pass
#         pred = model0(images)
#         # calculate loss
#         loss = loss_fn(pred, labels)
#         train_loss += loss.item()

#         # zero gradients
#         optimizer.zero_grad()
#         # backward pass
#         loss.backward()
#         # update weights
#         optimizer.step()
#         # update loss

#         if batch_number % 400 == 0:
#             print(f"look at {batch_number * len(labels)}/{len(train_dataloader.dataset)} sample:")

#     train_loss /= len(train_dataloader)

#     ### testing
#     test_loss = 0
#     accuracy = 0

#     model0.eval()
#     with torch.no_grad():
#         for X_test, y_test in test_dataloader:
#             pred = model0(X_test)
#             test_loss += loss_fn(pred, y_test).item()
#             accuracy += accuracy_fn(y_test, pred.argmax(dim=1))


#         test_loss /= len(test_dataloader)
#         accuracy /= len(test_dataloader)
    
#     print(f"Test Error: \n Accuracy: {(100*accuracy):>0.1f}%, Avg loss: {test_loss:>8f} \n")
#     print(f"train loss: {train_loss:>8f} \n")


# #calculate train time
# train_time_end_on_cpu = timer()
# total_train_time_on_cpu = print_train_time(train_time_start_on_cpu, 
#                                            train_time_end_on_cpu)

torch.manual_seed(42)
# make a function to evaluate the models
def eval_model(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               accuracy_fn,
               device: torch.device):
    loss,acc = 0, 0
    model.eval()
    with torch.no_grad():
        for X_test, y_test in tqdm(data_loader):
            X ,y = X_test.to(device), y_test.to(device)
            pred = model(X_test)
            loss += loss_fn(pred, y_test).item()
            acc += accuracy_fn(y_test, pred.argmax(dim=1))

        loss /= len(data_loader)
        acc /= len(data_loader)

    return {"model_name": model.__class__.__name__,
            "test_loss": loss,
            "test_accuracy": acc}

# calculate model0 resutls on test dataset
# model0_results = eval_model(model0, test_dataloader, loss_fn, accuracy_fn, device)  # model0 is the model trained on cpu

# print(model0_results)

# setup device agnostic training
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Using {} device".format(device))

# create a modle with non-linearity

class FashionMNISTmodel2(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_shape, hidden_units),  # 28*28 is the size of the image
            nn.ReLU(),
            nn.Linear(hidden_units, output_shape),
            nn.ReLU()
        )
    
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# create a new model with non-linearity
torch.manual_seed(42)
model1 = FashionMNISTmodel2(input_shape=28*28, hidden_units=10, output_shape=10).to(device)

# setup loss function and optimizer
from helper_functions import accuracy_fn
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model1.parameters(), lr=0.1)

# funcitonalizing training and evaluation/testing loops

def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               device: torch.device = device):
    train_loss, train_acc = 0, 0

    model.train()
    
    for batch, (X ,y) in enumerate(data_loader):
        X, y = X.to(device), y.to(device)
        
        y_pred = model(X)

        loss = loss_fn(y_pred, y)
        train_loss += loss.item()
        train_acc += accuracy_fn(y, y_pred.argmax(dim=1))

        optimizer.zero_grad()
        
        loss.backward()

        optimizer.step()

        # print out whatis happening
        if batch % 400 == 0:
            print(f"look at {batch * len(X)}/{len(data_loader.dataset)} sample:")

    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    print(f"Train loss: {train_loss:>8f} \n Train accuracy: {train_acc:>0.1f}% \n")


# create a test function
def test_step(model: torch.nn.Module,
                data_loader: torch.utils.data.DataLoader,
                loss_fn: torch.nn.Module,
                accuracy_fn,
                device: torch.device = device):
        test_loss, test_acc = 0, 0
    
        model.eval()
    
        with torch.no_grad():
            for X, y in data_loader:
                X, y = X.to(device), y.to(device)
    
                y_pred = model(X)
    
                loss = loss_fn(y_pred, y)
                test_loss += loss.item()
                test_acc += accuracy_fn(y, y_pred.argmax(dim=1))
    
            test_loss /= len(data_loader)
            test_acc /= len(data_loader)
    
        print(f"Test loss: {test_loss:>8f} \n Test accuracy: {test_acc:>0.1f}% \n")

# run the training and testing loops
torch.manual_seed(42)

#measure time
train_time_start_on_gpu = timer()

epochs = 3

# set device to cuda if available
device = "cuda" if torch.cuda.is_available() else "cpu"

# for epoch in tqdm(range(epochs)):
#     print(f"Epoch {epoch+1}\n-------------------------------")
#     train_step(model1, train_dataloader, loss_fn, optimizer, accuracy_fn, device)
#     test_step(model1, test_dataloader, loss_fn, accuracy_fn, device)

# calculate train time
train_time_end_on_gpu = timer()
total_train_time_on_gpu = print_train_time(train_time_start_on_gpu,
                                             train_time_end_on_gpu)

# evluiate model1
# model1_results = eval_model(model1, test_dataloader, loss_fn, accuracy_fn,device)

# print(model1_results)

# model 2:Building a convolutional neural network

class FashionMNISTModel3(nn.Module):
    """TinyVGG model for FashionMNIST"""
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_units*7*7, output_shape)
        )
    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        logits = self.classifier(x)
        return logits

# try on a test image
torch.manual_seed(42)


model3 = FashionMNISTModel3(input_shape=1, hidden_units=10, output_shape=10).to(device)

# setup a loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model3.parameters(), lr=0.1)

# measure time
train_time_start_on_gpu = timer()

# train and test model
epochs = 3

for epoch in tqdm(range(epochs)):
    print(f"Epoch {epoch+1}\n-------------------------------")
    train_step(model3, train_dataloader, loss_fn, optimizer, accuracy_fn, device)
    test_step(model3, test_dataloader, loss_fn, accuracy_fn, device) 

# calculate train time
train_time_end_on_gpu = timer()
total_train_time_on_gpu = print_train_time(train_time_start_on_gpu,
                                                train_time_end_on_gpu)

# evaluate model3
model3_results = eval_model(model3, test_dataloader, loss_fn, accuracy_fn, device)

# make and evaluate random predictions with best model

def make_predictions(model: torch.nn.Module,
                     data: list,
                     device: torch.device = device):
    pred_probs = []
    model.eval()
    with torch.no_grad():
        for sample in data:
            sample = torch.unsqueeze(sample, 0).to(device)

            pred_logit = model(sample)

            pred_prob = nn.functional.softmax(pred_logit.squeeze(), dim=0) # convert logit to probability    
            pred_probs.append(pred_prob.cpu()) # convert tensor to numpy array

    return torch.stack(pred_probs) # stack the list of tensors into a single tensor


import random
random.seed(42)

test_samples =[]
test_labels = []

for sample, label in random.sample(list(test_data), 9):
    test_samples.append(sample)
    test_labels.append(label)   

pred_probs = make_predictions(model3, test_samples)

pred_class = pred_probs.argmax(dim=1)

# plot the predictions
figure = plt.figure(figsize=(9,9))
nrows, ncols = 3, 3
for i, sample in enumerate(test_samples):
    figure.add_subplot(nrows, ncols, i+1)
    plt.title(f"True: {test_data.classes[test_labels[i]]}, Pred: {test_data.classes[pred_class[i]]}")
    plt.axis("off")
    plt.imshow(sample.squeeze(), cmap="gray")

plt.show()

# make a confusion matrix
import mlxtend

# Make predictions with a trained model

y_preds = []
model3.eval()
with torch.no_grad():
    for X_test, y_test in tqdm(test_dataloader):
        X ,y = X_test.to(device), y_test.to(device)
        pred = model3(X_test)
        y_preds.append(torch.softmax(pred, dim = 1).argmax(dim=1).cpu())
                       
# concatenate the list of tensors into a single tensor

print (y_preds)
y_preds_tensor = torch.cat(y_preds, dim=0)

print(len(y_preds_tensor))

import torchmetrics
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix

# setup confusion matrix
cm = ConfusionMatrix(num_classes=len(test_data.classes), task="MultiClass")
cm_tensor = cm(y_preds_tensor, test_data.targets)

# plot confusion matrix
fig, ax = plot_confusion_matrix(conf_mat=cm_tensor.numpy(), class_names=test_data.classes, figsize=(10,10))






