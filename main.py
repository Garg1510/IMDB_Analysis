import numpy as np
from keras.preprocessing import sequence
from keras.datasets import imdb
from sklearn.metrics import confusion_matrix, accuracy_score
from model_utils import create_network

# Load IMDB dataset
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=20000)

# Pad sequences
X_train = sequence.pad_sequences(X_train, maxlen=80)
X_test = sequence.pad_sequences(X_test, maxlen=80)

# Create model using utility function from model_utils.py
model = create_network()

# Fit the model
model.fit(X_train, y_train, batch_size=32, epochs=5, verbose=2, validation_split=0.2)

# Predictions
predictions = model.predict(X_test, batch_size=128, verbose=2)
predictions = (predictions >= 0.5).astype(int)

# Evaluate model
cm = confusion_matrix(y_test, predictions)
accuracy = accuracy_score(y_test, predictions)
precision = cm[1][1] / (cm[1][1] + cm[0][1])
recall = cm[1][1] / (cm[1][1] + cm[1][0])
f1_score = 2 * precision * recall / (precision + recall)

print("Confusion Matrix:")
print(cm)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1_score)
