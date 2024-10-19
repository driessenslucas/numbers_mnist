import os
import json
import numpy as np
import tensorflow as tf
import logging
from sklearn.metrics import (confusion_matrix, classification_report, 
                             accuracy_score, precision_recall_fscore_support, 
                             roc_auc_score)
import seaborn as sns
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the data
logging.info('Loading validation data...')
X_val = np.load('./data_split/X_val.npy')
Y_val = np.load('./data_split/Y_val.npy')
logging.info('Validation data loaded successfully.')

# Load the model
logging.info('Loading model...')
model = tf.keras.models.load_model('./model/model.h5')
logging.info('Model loaded successfully.')

# Get predictions
logging.info('Getting predictions...')
Y_pred = model.predict(X_val)
Y_pred_classes = np.argmax(Y_pred, axis=1)
Y_true_classes = np.argmax(Y_val, axis=1)
logging.info('Predictions obtained successfully.')

# Confusion Matrix
cm = confusion_matrix(Y_true_classes, Y_pred_classes)

# Classification Report
report = classification_report(Y_true_classes, Y_pred_classes)
logging.info('Classification report generated:\n%s', report)

# Accuracy
accuracy = accuracy_score(Y_true_classes, Y_pred_classes)
logging.info('Accuracy: %.4f', accuracy)

# Precision, Recall, F1-Score
precision, recall, f1_score, _ = precision_recall_fscore_support(Y_true_classes, Y_pred_classes, average='weighted')
logging.info('Precision: %.4f', precision)
logging.info('Recall: %.4f', recall)
logging.info('F1-Score: %.4f', f1_score)

# ROC AUC score
roc = roc_auc_score(Y_val, Y_pred)
logging.info('ROC AUC: %.4f', roc)

# Save metrics to JSON
metrics = {
    'confusion_matrix': cm.tolist(),
    'classification_report': classification_report(Y_true_classes, Y_pred_classes, output_dict=True),
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1_score': f1_score,
    'roc_auc': roc
}

with open('./metrics/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=4)
logging.info('Metrics saved to metrics.json')

# Plots

# Plotting the first 25 images
logging.info('Plotting the first 25 images...')
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X_val[i], cmap=plt.cm.binary)
    plt.xlabel(f'Predicted: {Y_pred_classes[i]}, True: {Y_true_classes[i]}')
# plt.show()
plt.savefig('./plots/First_25_images.png')
logging.info('First 25 images plotted and saved to First_25_images.png.')

# Plot some false positives and false negatives
false_positives = []
false_negatives = []
for i in range(len(Y_true_classes)):
    if Y_true_classes[i] != Y_pred_classes[i]:
        if Y_true_classes[i] < Y_pred_classes[i]:
            false_positives.append(i)
        else:
            false_negatives.append(i)

if false_positives:
    logging.info('Plotting false positives...')
    plt.figure(figsize=(10, 10))
    for i in range(min(25, len(false_positives))):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(X_val[false_positives[i]], cmap=plt.cm.binary)
        plt.xlabel(f'Predicted: {Y_pred_classes[false_positives[i]]}, True: {Y_true_classes[false_positives[i]]}')
    # plt.show()
    plt.savefig('./plots/Incorrectly_classified_images.png')
    logging.info('False positives plotted and saved to Incorrectly_classified_images.png.')
else:
    logging.info('No false positives found.')

if false_negatives:
    logging.info('Plotting false negatives...')
    plt.figure(figsize=(10, 10))
    for i in range(min(25, len(false_negatives))):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(X_val[false_negatives[i]], cmap=plt.cm.binary)
        plt.xlabel(f'Predicted: {Y_pred_classes[false_negatives[i]]}, True: {Y_true_classes[false_negatives[i]]}')
    # plt.show()
    plt.savefig('./plots/Incorrectly_classified_images.png')
    logging.info('False negatives plotted and saved to Incorrectly_classified_images.png.')
else:
    logging.info('No false negatives found.')

# Plotting the confusion matrix
logging.info('Plotting the confusion matrix...')
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=np.arange(10), yticklabels=np.arange(10))
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
# plt.show()
plt.savefig('./plots/Confusion_Matrix.png')
logging.info('Confusion matrix plotted and saved to Confusion_Matrix.png.')
