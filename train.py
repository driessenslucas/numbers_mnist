import os
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import logging

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import GridSearchCV, train_test_split
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.utils import to_categorical

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Check if data is already split
if not os.path.exists('./data_split/X_train.npy') and not os.path.exists('./data_split/Y_train.npy') and not os.path.exists('./data_split/X_val.npy') and not os.path.exists('./data_split/Y_val.npy'):
    logging.info('Data not split. Splitting data...')
    
    train = pd.read_csv("./data/mnist_train.csv")
    test = pd.read_csv("./data/mnist_test.csv")

    Y_train = train["label"]
    X_train = train.drop(labels=["label"], axis=1)

    # Normalize the data
    X_train = X_train / 255.0
    test = test / 255.0

    # Reshape image in 3 dimensions (height = 28px, width = 28px, canal = 1)
    X_train = X_train.values.reshape(-1, 28, 28, 1)

    # Encode labels to one-hot vectors
    Y_train = to_categorical(Y_train, num_classes=10)

    # Set the random seed
    random_seed = 42

    # Split the train and validation set for fitting
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state=random_seed)

    # Save the X_train, Y_train, X_val, Y_val
    np.save('./data_split/X_train.npy', X_train)
    np.save('./data_split/Y_train.npy', Y_train)
    np.save('./data_split/X_val.npy', X_val)
    np.save('./data_split/Y_val.npy', Y_val)

    logging.info('Data splitting complete. Files saved.')
else:
    logging.info('Data already split. Loading the data...')

# Load the data
X_train = np.load('./data_split/X_train.npy')
Y_train = np.load('./data_split/Y_train.npy')
X_val = np.load('./data_split/X_val.npy')
Y_val = np.load('./data_split/Y_val.npy')

# Data augmentation
def data_augmentation(X_train, Y_train):
    logging.info('Starting data augmentation...')
    datagen = ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1
    )
    
    # Fit parameters from data
    datagen.fit(X_train)
    
    logging.info('Data augmentation complete.')
    return datagen

datagen = data_augmentation(X_train, Y_train)

# Create the model
def cnn(lr=0.001, hidden_layers=1, filters=16, kernel_size=3):
    logging.info('Creating CNN model...')
    # Create model
    model = keras.Sequential()
   
    # Convolutional layers
    model.add(layers.Conv2D(filters=filters, kernel_size=kernel_size, activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(filters=filters, kernel_size=kernel_size, activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(filters=filters, kernel_size=kernel_size, activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    # Flattening
    model.add(layers.Flatten())
    model.add(layers.Dense(units=128, activation='relu'))

    # Dropout layer to prevent overfitting
    model.add(layers.Dropout(0.5))

    # Output Layer
    model.add(layers.Dense(units=10, activation='softmax'))

    # Compile model
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    logging.info('CNN model created.')
    return model

# Create the model
model = cnn()

# Train the model using the training sets, add checkpoints and tensorboard
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

logging.info('Starting model training...')
history = model.fit(datagen.flow(X_train, Y_train), epochs=10, validation_data=(X_val, Y_val), callbacks=[tensorboard_callback])
logging.info('Model training complete.')

# Save the model
model.save('./model/model.h5')
logging.info('Model saved to ./model/model.h5')

# Plot the training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
# plt.show()
plt.savefig('./plots/loss.png')
logging.info('Loss plot saved to ./plots/loss.png')

# Plot the training and validation accuracy
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
# plt.show()
plt.savefig('./plots/accuracy.png')
logging.info('Accuracy plot saved to ./plots/accuracy.png')
