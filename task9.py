from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, GRU, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences

vocab_size = 10000
max_len = 200
embedding_dim = 64

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)

X_train = pad_sequences(X_train, maxlen=max_len)
X_test = pad_sequences(X_test, maxlen=max_len)

model_lstm = Sequential()
model_lstm.add(Embedding(vocab_size, embedding_dim, input_length=max_len))
model_lstm.add(LSTM(64))
model_lstm.add(Dense(1, activation='sigmoid'))

model_lstm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("Training LSTM model...")
model_lstm.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.2)

loss, accuracy = model_lstm.evaluate(X_test, y_test)
print(f'LSTM Test Loss: {loss}, Test Accuracy: {accuracy}')

model_gru = Sequential()
model_gru.add(Embedding(vocab_size, embedding_dim, input_length=max_len))
model_gru.add(GRU(64))
model_gru.add(Dense(1, activation='sigmoid'))

model_gru.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("Training GRU model...")
model_gru.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.2)

loss, accuracy = model_gru.evaluate(X_test, y_test)
print(f'GRU Test Loss: {loss}, Test Accuracy: {accuracy}')

