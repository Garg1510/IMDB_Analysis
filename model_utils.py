from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM

def create_network():
    model = Sequential()
    model.add(Embedding(20000, 128))
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
