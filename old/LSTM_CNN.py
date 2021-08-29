
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Conv1D, MaxPooling1D, LSTM, Dense

BATCH_SIZE = 128
SEQ_SIZE = 5

N_FEATURES = 5


rnn = Sequential()
rnn.add(InputLayer(input_shape=(SEQ_SIZE, N_FEATURES)))
rnn.add(Conv1D(32))
rnn.add(MaxPooling1D())
rnn.add(LSTM(64))
rnn.add(Dense(1))
