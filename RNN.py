import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
available_devices = tf.config.experimental.list_physical_devices('GPU')
if len(available_devices) > 0:
    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_virtual_device_configuration(gpu, [
            tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5120)])
        # tf.config.experimental.set_memory_growth(gpu, True)

import numpy as np

df = pd.read_csv('procesados.csv')

variable_target = 'INPC_General'
variable_dropeada = 'INPC_Sub' if variable_target == 'INPC_General' else 'INPC_General'
df = df.drop(variable_dropeada, axis=1)

def feature_generator(df, col):
    df['Prev_Day_{}'.format(col)] = df[col].shift(1)
    df['One_Day_Change_{}'.format(col)] = df[col] - df['Prev_Day_{}'.format(col)]
    df['Five_Day_Percentage_Change_{}'.format(col)] = (df[col] - df[col].shift(5))*100/df[col].shift(5)
    return df

# for feature in features:
#     df = feature_generator(df, feature)

print(df.shape)
train = df[(df['Year'] >= 2000) & (df['Year'] < 2021)].reset_index(drop=True)
test = df[df['Year'] == 2021].reset_index(drop=True)

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

scaler = MinMaxScaler().fit(train)
columns = train.columns

train_scaled = pd.DataFrame(scaler.transform(train), columns=columns)
test_scaled = pd.DataFrame(scaler.transform(test), columns=columns)

print(train_scaled)

from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

BATCH_SIZE = 128
SEQ_SIZE = 5


def df_to_generator(df_scaled):
    df_output = df_scaled[variable_target]
    # df_scaled.drop('INPC', inplace=True, axis=1)

    n_features = df_scaled.shape[1]
    df_generator = TimeseriesGenerator(data=np.array(df_scaled), targets=np.array(df_output),
                                       length=SEQ_SIZE, batch_size=n_features)

    return df_scaled, df_output, df_generator


train_scaled, train_output, train_generator = df_to_generator(train_scaled)
# valid_scaled, valid_output, valid_generator = df_to_generator(valid_scaled)
test_scaled, test_output, test_generator = df_to_generator(test_scaled)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM

N_FEATURES = train_scaled.shape[1]


def create_rnn():
    rnn = Sequential()
    rnn.add(LSTM(16, input_shape=(SEQ_SIZE, N_FEATURES)))
    rnn.add(Dense(1))

    rnn.compile(loss=tf.keras.losses.MeanSquaredLogarithmicError(), metrics=METRICS.keys(),
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005))
    history = rnn.fit(train_generator, shuffle=False, epochs=EPOCH, verbose=2, batch_size=BATCH_SIZE)

    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    for metric in METRICS.keys():
        plt.xlabel('Epoch')
        plt.ylabel(METRICS[metric])
        plt.plot(hist['epoch'], hist[metric], label='Training')
        plt.legend()
        plt.show()

    return rnn


EPOCH = 500
METRICS = {'mae': 'Mean Absolute Error (MAE)', 'mse': 'Mean Squared Error (MSE)'}
model = create_rnn()

test_metrics = model.evaluate(test_generator, verbose=0)
print('*** Test metrics ***')
for i, test_metric in enumerate(['loss'] + list(METRICS.keys())):
    print(test_metric, test_metrics[i])

from sklearn.metrics import mean_absolute_error


def inv_scaler(y, current_scaler):
    df = pd.DataFrame(dict(zip(test_scaled.columns, np.ones(y.shape[0]))),
                      index=range(y.shape[0]), columns=test_scaled.columns)
    df[variable_target] = y
    y_esc = pd.DataFrame(current_scaler.inverse_transform(df), columns=test_scaled.columns)[variable_target]
    return y_esc


y_train = np.array(train_output)[SEQ_SIZE:].reshape(-1, 1)
y_prediction_train = model.predict(train_generator).reshape(-1, 1)
y_prediction_train_esc, y_train_esc = inv_scaler(y_prediction_train, scaler), inv_scaler(y_train, scaler)

y_prediction = model.predict(test_generator).reshape(-1, 1)
y_true = np.array(test_output)[SEQ_SIZE:].reshape(-1, 1)
y_prediction_esc, y_true_esc = inv_scaler(y_prediction, scaler), inv_scaler(y_true, scaler)


print(mean_absolute_error(y_true, y_prediction))
print(mean_absolute_error(y_true_esc, y_prediction_esc))

print(mean_absolute_error(y_train, y_prediction_train))
print(mean_absolute_error(y_train_esc, y_prediction_train_esc))


def plot_results(predictions, true_values):
    plt.plot(predictions, 'y', label='Predicted')
    plt.plot(true_values, 'r', label='True')
    plt.title('True and predicted value')
    plt.xlabel('Index')
    plt.ylabel('Dato')
    plt.legend()
    plt.show()


plot_results(y_prediction_esc, y_true_esc)
plot_results(y_prediction_train_esc, y_train_esc)