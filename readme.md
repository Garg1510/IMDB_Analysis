# IMDB Sentiment Analysis with LSTM


This project focuses on sentiment analysis using Long Short-Term Memory (LSTM) networks applied to the IMDB movie reviews dataset. The goal is to classify movie reviews as positive or negative based on the text content.

Steps and Components:

1. Data Loading and Preprocessing:

Load the IMDB dataset using Keras.
Preprocess the text data by tokenizing and padding sequences.

2. Model Architecture:

Construct an LSTM model using Keras Sequential API.
Embedding layer with vocabulary size 20,000 and embedding dimension 128.
LSTM layer with 128 units and dropout of 0.2 for regularization.
Output layer with sigmoid activation for binary classification.

3. Model Training:

Train the model using a batch size of 32 and 5 epochs.
Evaluate model performance on a validation set.

4. Evaluation Metrics:

Compute and display metrics such as accuracy, precision, recall, and F1 score.
Generate a confusion matrix to visualize model performance.

5. Utility Functions:

Modularize the model creation using functions for reusability.
Implement cross-validation using KerasClassifier for model validation.
