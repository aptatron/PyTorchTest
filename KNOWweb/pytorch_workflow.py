# import torch, nn, matplotlib.pyplot as plt, numpy as np.
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

#check pytorch version
print(torch.__version__)

#setup device agnostic code
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#create some data using linear regression formula of y = weight * x + bias

weight = 0.7
bias = 0.3

#create range values

start = 0
end = 1
step = 0.02

#create x and y values

x = torch.arange(start, end, step).unsqueeze(1)
y = weight * x + bias

#create training data and testing data with 8:2 split   

train_size = int(0.8 * len(x))
test_size = len(x) - train_size

train_x = x[:train_size]
train_y = y[:train_size]

test_x = x[train_size:]
test_y = y[train_size:]

print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)

#plot the data

plt.plot(train_x.numpy(), train_y.numpy(), 'ro')
plt.plot(test_x.numpy(), test_y.numpy(), 'bo')
plt.axis([0, 1, 0, 1])
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# create a linear regression model

class LinearRegression(nn.Module):
    def __init__(self, input_size=1, output_size=1):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(in_features=input_size, out_features=output_size)
        
    def forward(self, x):
        return self.linear(x)

# set manual seed

torch.manual_seed(42)
model_pt = LinearRegression()
print(model_pt)
# print(list(model_pt.parameters()))
print(model_pt.state_dict())

#setup the model to use cuda if available

model_pt.to(device)

#setup loss function and optimizer

criterion = nn.L1Loss()
optimizer = torch.optim.SGD(model_pt.parameters(), lr=0.01)

#write a training loop

epochs = 200

for epoch in range(epochs):
    model_pt.train()

    #1. forward propagation
    y_pred = model_pt(train_x)

    #2. calculate loss
    loss = criterion(y_pred, train_y)

    #3. backward propagation
    

    #4. weight update
    optimizer.zero_grad()

    loss.backward()

    #5. zero out the gradients
    optimizer.step()

    model_pt.eval()
    with torch.no_grad():
        y_pred_test = model_pt(test_x)
        loss_test = criterion(y_pred_test, test_y)

    if epoch % 10 == 0:
        print(f'Epoch: {epoch}, Loss: {loss.item():.6f}, loss_test: {loss_test.item():.6f}')

plt.plot(train_x.numpy(), train_y.numpy(), 'ro')
plt.plot(test_x.numpy(), test_y.numpy(), 'bo')
# change the size of the dot to make it vertical bars
plt.plot(test_x.numpy(), y_pred_test.numpy(), "g+"  , ms=20)
plt.axis([0, 1, 0, 1])
plt.xlabel('x')
plt.ylabel('y')
plt.show()

#save the model

torch.save(model_pt.state_dict(), 'model.pt')

#save the linear regression model

torch.save(model_pt, 'model_raw.pt')

#load the linear regression model

model_pt2 = torch.load('model_raw.pt')
model_pt2.load_state_dict(torch.load('model.pt'))
model_pt2.eval()

#predict using the loaded model

with torch.no_grad():
    y_pred_test2 = model_pt2(test_x)
    loss_test = criterion(y_pred_test, test_y)

# plt.plot(train_x.numpy(), train_y.numpy(), 'ro')
# plt.plot(test_x.numpy(), test_y.numpy(), 'bo')
# # change the size of the dot to make it vertical bars
# plt.plot(test_x.numpy(), y_pred_test2.numpy(), "g+"  , ms=20)
# plt.axis([0, 1, 0, 1])
# plt.xlabel('x')
# plt.ylabel('y')
# plt.show()

print(y_pred_test == y_pred_test2)
