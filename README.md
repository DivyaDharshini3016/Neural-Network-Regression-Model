# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Problem Statement
Regression problems aim to predict a continuous numerical value based on input features. Traditional regression models may fail to capture complex non-linear relationships. A Neural Network Regression Model uses multiple layers of neurons to learn these non-linear patterns and improve prediction accuracy.

## Neural Network Model

<img width="1132" height="652" alt="image" src="https://github.com/user-attachments/assets/64f7c9b9-7e44-43c4-8c11-132005f2a5bc" />

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
#Name:Divya Dharshini S
#Reg.No:212224240039
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

```
## Dataset Information

<img width="242" height="433" alt="image" src="https://github.com/user-attachments/assets/73ab9b1a-6dd3-4739-8302-2ac5332bd1a7" />

## OUTPUT
<img width="906" height="256" alt="image" src="https://github.com/user-attachments/assets/5749a2d7-640b-4844-aa1a-b362f02e6952" />
<img width="922" height="120" alt="image" src="https://github.com/user-attachments/assets/00f0d3f7-a29c-4951-b89d-623b255da20b" />

### Training Loss Vs Iteration Plot

<img width="960" height="672" alt="image" src="https://github.com/user-attachments/assets/b10cfde1-3c0f-409d-b40c-9465f5f79f39" />


### New Sample Data Prediction

<img width="955" height="130" alt="image" src="https://github.com/user-attachments/assets/d0f5f4b3-9349-4dbf-bb17-2dfacc497e46" />

## RESULT
The neural network regression model was successfully trained and evaluated. The model demonstrated strong predictive performance on unseen data, with a low error rate.
