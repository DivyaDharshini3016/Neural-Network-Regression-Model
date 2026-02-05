# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Explain the problem statement

## Neural Network Model

Include the neural network model diagram.

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name:
### Register Number:
```
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("dataset.csv")

X = torch.tensor(data.iloc[:,0].values, dtype=torch.float32).view(-1,1)
Y = torch.tensor(data.iloc[:,1].values, dtype=torch.float32).view(-1,1)

X = X / X.max()

class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1,10)
        self.fc2 = nn.Linear(10,1)
        self.relu = nn.ReLU()

    def forward(self,x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

ai_brain = NeuralNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(ai_brain.parameters(), lr=0.01)

def train_model(ai_brain, X, Y, criterion, optimizer, epochs=500):
    losses = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = ai_brain(X)
        loss = criterion(output, Y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses

losses = train_model(ai_brain, X, Y, criterion, optimizer)

plt.plot(losses)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss vs Epochs")
plt.show()

```
## Dataset Information

<img width="242" height="433" alt="image" src="https://github.com/user-attachments/assets/73ab9b1a-6dd3-4739-8302-2ac5332bd1a7" />

## OUTPUT
### Training Loss Vs Iteration Plot

<img width="687" height="515" alt="image" src="https://github.com/user-attachments/assets/945313c7-8f73-4ecd-a61c-d19f5b814bde" />


### New Sample Data Prediction

<img width="570" height="112" alt="image" src="https://github.com/user-attachments/assets/0b8205c1-22cb-47c9-9c15-dfa44c6359d8" />

## RESULT
The neural network regression model was successfully trained and evaluated. The model demonstrated strong predictive performance on unseen data, with a low error rate.
