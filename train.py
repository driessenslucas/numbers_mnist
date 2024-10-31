import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Check if GPU is available and set the device to GPU if available; otherwise, use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define hyperparameters for model training
batch_size = 64  # Number of samples processed before updating model parameters
learning_rate = 0.001  # Step size for each update to the weights
num_epochs = 10  # Number of complete passes through the training dataset

# Define the data transformation pipeline, including data augmentation
transform = transforms.Compose([
    transforms.RandomRotation(10),  # Rotates each image randomly within ±10 degrees
    transforms.RandomAffine(0, translate=(0.1, 0.1)),  # Translates each image randomly by up to 10% of image size
    transforms.ToTensor(),  # Converts the image to a tensor and scales pixel values to [0, 1]
    transforms.Normalize((0.5,), (0.5,))  # Normalizes pixel values to [-1, 1] (mean=0.5, std=0.5)
])

# Load the MNIST dataset with transformations applied, and organize it into DataLoader objects
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


# Define the Convolutional Neural Network (CNN) model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # First convolutional layer: accepts 1-channel grayscale images, outputs 32 channels
        # Kernel size of 3x3 with stride 1 and padding 1 to maintain image dimensions
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)

        # Second convolutional layer: inputs 32 channels, outputs 64 channels
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        # Max pooling layer: reduces each feature map dimension by 2x via max value selection
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layer: input is the flattened 64 * 7 * 7 tensor, output is 128 neurons
        self.fc1 = nn.Linear(64 * 7 * 7, 128)

        # Output layer: maps 128 neurons to 10 classes (one for each digit)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Forward pass through conv1, ReLU activation, and max pooling
        x = self.pool(torch.relu(self.conv1(x)))  # Resulting shape: (batch_size, 32, 14, 14)

        # Forward pass through conv2, ReLU activation, and max pooling
        x = self.pool(torch.relu(self.conv2(x)))  # Resulting shape: (batch_size, 64, 7, 7)

        # Flatten the output from the convolutional layers for the fully connected layer
        x = x.view(-1, 64 * 7 * 7)  # Flatten to (batch_size, 3136)

        # Apply first fully connected layer with ReLU activation
        x = torch.relu(self.fc1(x))

        # Output layer: linear transformation to produce logits for each of the 10 classes
        x = self.fc2(x)
        return x


# Instantiate the model, define the loss function (Cross Entropy Loss) and the optimizer (Adam)
model = CNN().to(device)  # Move the model to the GPU if available
criterion = nn.CrossEntropyLoss()  # Cross-entropy loss for multi-class classification
optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Adam optimizer with learning rate = 0.001

# Initialize variable to track the best validation accuracy and model path
best_accuracy = 0.0
best_model_path = 'best_model.pth'

# Training loop: iterate over the dataset for the specified number of epochs
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    for images, labels in train_loader:
        # Move images and labels to the GPU if available
        images, labels = images.to(device), labels.to(device)

        # Zero the parameter gradients before the backward pass
        optimizer.zero_grad()

        # Forward pass: compute the model output and loss
        outputs = model(images)  # Compute model predictions (logits)
        loss = criterion(outputs, labels)  # Compute the cross-entropy loss

        # Backward pass: compute the gradient of the loss with respect to model parameters
        loss.backward()

        # Optimization step: update the model parameters
        optimizer.step()

    # Evaluate the model on the test set after each epoch to track performance
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():  # No need to compute gradients during evaluation
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)  # Compute predictions on test set
            _, predicted = torch.max(outputs.data, 1)  # Get index of the highest logit for each image

            # Accumulate the number of correctly classified samples
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Calculate the accuracy as a percentage
    accuracy = 100 * correct / total
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%')

    # Save the model if the accuracy has improved
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(model.state_dict(), best_model_path)  # Save the model's state_dict
        print(f'Saved best model with accuracy: {best_accuracy:.2f}%')

print(f'Best accuracy achieved: {best_accuracy:.2f}%')
