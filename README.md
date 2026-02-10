# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Problem Statement
Regression problems aim to predict a continuous numerical value based on input features. Traditional regression models may fail to capture complex non-linear relationships. A Neural Network Regression Model uses multiple layers of neurons to learn these non-linear patterns and improve prediction accuracy.

## Neural Network Model

<img width="912" height="753" alt="image" src="https://github.com/user-attachments/assets/0641249a-926d-4615-9754-b8f35bafdee7" />


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
### Name: Divya Dharshini S
### Register Number: 212224240039
```
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
dataset1 = pd.read_csv('/content/dataset.csv')
X = dataset1[['Size']].values
y = dataset1[['Price']].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=33)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 8)
        self.fc2 = nn.Linear(8, 10)
        self.fc3 = nn.Linear(10, 1)
        self.relu = nn.ReLU()
        self.history = {'loss': []}

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  # No activation here since it's a regression task
        return x
ai_brain = NeuralNet()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(ai_brain.parameters(), lr=0.001)
def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = criterion(ai_brain(X_train), y_train)
        loss.backward()
        optimizer.step()

        ai_brain.history['loss'].append(loss.item())
        if epoch % 200 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')
train_model(ai_brain, X_train_tensor, y_train_tensor, criterion, optimizer)
with torch.no_grad():
    test_loss = criterion(ai_brain(X_test_tensor), y_test_tensor)
    print(f'Test Loss: {test_loss.item():.6f}')
loss_df = pd.DataFrame(ai_brain.history)
import matplotlib.pyplot as plt
loss_df.plot()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss during Training")
plt.show()

```
## Dataset Information

<img width="242" height="433" alt="image" src="https://github.com/user-attachments/assets/73ab9b1a-6dd3-4739-8302-2ac5332bd1a7" />

## OUTPUT

<img width="402" height="197" alt="image" src="https://github.com/user-attachments/assets/6935e5f9-eae3-4eb5-8ce7-d7debb08070a" />
<img width="417" height="37" alt="image" src="https://github.com/user-attachments/assets/6bf94893-e2d3-4634-9811-0418c5d8a3fe" />


### Training Loss Vs Iteration Plot

<img width="873" height="520" alt="image" src="https://github.com/user-attachments/assets/9cacc9f0-234c-46fc-8902-d9d768ad3dc6" />

### New Sample Data Prediction

<img width="955" height="130" alt="image" src="https://github.com/user-attachments/assets/d0f5f4b3-9349-4dbf-bb17-2dfacc497e46" />

## RESULT
The neural network regression model was successfully trained and evaluated. The model demonstrated strong predictive performance on unseen data, with a low error rate.
