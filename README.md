# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Problem Statement
Regression problems aim to predict a continuous numerical value based on input features. Traditional regression models may fail to capture complex non-linear relationships. A Neural Network Regression Model uses multiple layers of neurons to learn these non-linear patterns and improve prediction accuracy.

## Neural Network Model

<img width="1077" height="781" alt="image" src="https://github.com/user-attachments/assets/dd404840-8757-4a91-a143-4ff036b7aa5f" />

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
        self.fc2 = nn.Linear(8, 16)
        self.fc3 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        self.history = {'loss': []}

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  # No activation here since it's a regression task
        return x
divya_brain = NeuralNet()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(divya_brain.parameters(), lr=0.001)
def train_model(divya_brain, X_train, y_train, criterion, optimizer, epochs=2000):
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = criterion(divya_brain(X_train), y_train)
        loss.backward()
        optimizer.step()

        divya_brain.history['loss'].append(loss.item())
        if epoch % 200 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')

```
## Dataset Information

<img width="242" height="433" alt="image" src="https://github.com/user-attachments/assets/73ab9b1a-6dd3-4739-8302-2ac5332bd1a7" />

## OUTPUT

<img width="800" height="259" alt="Screenshot 2026-02-23 083322" src="https://github.com/user-attachments/assets/05687648-11e2-4cd4-b149-1859839a3e98" />
<img width="834" height="115" alt="Screenshot 2026-02-23 083350" src="https://github.com/user-attachments/assets/ad32d8ff-c306-4e93-83d9-2ae46b37503a" />

### Training Loss Vs Iteration Plot

<img width="953" height="668" alt="image" src="https://github.com/user-attachments/assets/57d163d7-5a7e-4958-a8e9-9f0a3a14c30b" />


### New Sample Data Prediction

<img width="957" height="127" alt="image" src="https://github.com/user-attachments/assets/8d91dcc2-bb53-4f02-8a86-5a2eea5c3f2f" />

## RESULT
The neural network regression model was successfully trained and evaluated. The model demonstrated strong predictive performance on unseen data, with a low error rate.
