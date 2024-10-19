import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.base import BaseEstimator, ClassifierMixin

# Load the data
X_train = np.load('./data_split/X_train.npy')
Y_train = np.load('./data_split/Y_train.npy')
X_val = np.load('./data_split/X_val.npy')
Y_val = np.load('./data_split/Y_val.npy')

# Data augmentation
def data_augmentation(X_train, Y_train):
    datagen = ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1
    )
    
    # fit parameters from data
    datagen.fit(X_train)
    
    return datagen

datagen = data_augmentation(X_train, Y_train)

# Create the model
def cnn(lr=0.001, hidden_layers=1, filters=16, kernel_size=3):
    # create model
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
    
    return model

# Custom KerasClassifier
class KerasClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, lr=0.001, filters=32, batch_size=32, epochs=10):
        self.lr = lr
        self.filters = filters
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = None

    def fit(self, X, y):
        # Create the model
        self.model = cnn(lr=self.lr, filters=self.filters)
        
        # Use the data generator for augmentation
        self.datagen = datagen.flow(X, y, batch_size=self.batch_size)

        # Fit the model
        self.model.fit(self.datagen, epochs=self.epochs, validation_data=(X_val, Y_val))
        return self

    def predict(self, X):
        # Use model to predict classes
        return np.argmax(self.model.predict(X), axis=-1)

    def score(self, X, y):
        # Evaluate the model
        return self.model.evaluate(X, y, verbose=0)[1]  # Return accuracy

# Define the parameter grid for GridSearch
param_grid = {
    'lr': [0.001, 0.0001],
    'filters': [32, 64],
    'epochs': [5, 10],
    'batch_size': [32, 64]
}

# Create the KerasClassifier
model = KerasClassifier(cnn)

# Perform Grid Search
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', n_jobs=-1, cv=3)
grid_result = grid.fit(X_train, Y_train)

# Save the best model
best_model = grid_result.best_estimator_
best_model.model.save('./model/best_model.h5')

# Summarize results
print("Best Score: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

# Optionally evaluate the best model on validation data
best_model = grid_result.best_estimator_
val_score = best_model.score(X_val, Y_val)
print("Validation Score: %.2f" % val_score)

