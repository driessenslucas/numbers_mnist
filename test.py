import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import torch.nn as nn


# Define the CNN model (ensure this matches your trained model)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Load the trained model
model = CNN()
model.load_state_dict(torch.load('best_model.pth'))
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Start video capture
cap = cv2.VideoCapture(0)  # Use 0 for default webcam

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale and apply thresholding
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Get bounding box of the contour
        x, y, w, h = cv2.boundingRect(contour)

        # Filter out small contours to reduce noise
        if w > 10 and h > 10:  # Minimum size of detected digit
            # Extract and preprocess the digit
            digit = thresh[y:y + h, x:x + w]  # Crop the digit
            digit = cv2.resize(digit, (28, 28), interpolation=cv2.INTER_AREA)  # Resize to 28x28

            # Convert to PIL and apply transformations
            pil_img = Image.fromarray(digit)
            img_tensor = transform(pil_img).unsqueeze(0).to(device)

            # Predict digit
            with torch.no_grad():
                output = model(img_tensor)
                _, predicted = torch.max(output, 1)

            # Draw bounding box and prediction on frame
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, str(predicted.item()), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Display the resulting frame
    cv2.imshow('Digit Recognition', frame)

    # Exit loop with 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
