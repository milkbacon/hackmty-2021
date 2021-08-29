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

variable_target = ['INPC_General', 'INPC_Sub']
print(df)

def feature_generator(df, col):
    df['Prev_Q_{}'.format(col)] = df[col].shift(1)
    df['One_Q_Change_{}'.format(col)] = df[col] - df[col].shift(1)
    df['Q_Percentage_Change_{}'.format(col)] = (df[col] - df[col].shift(1))*100/df[col].shift(1)

    df['Prev_Trim_{}'.format(col)] = df[col].shift(6)
    df['One_Trim_Change{}'.format(col)] = df[col] - df[col].shift(6)
    df['Trim_Percentage_Change_{}'.format(col)] = (df[col] - df[col].shift(6))*100/df[col].shift(6)

    df['Prev_Sem_{}'.format(col)] = df[col].shift(12)
    df['One_Sem_Change{}'.format(col)] = df[col] - df[col].shift(12)
    df['Sem_Percentage_Change_{}'.format(col)] = (df[col] - df[col].shift(12))*100/df[col].shift(12)

    df['Prev_Year_{}'.format(col)] = df[col].shift(24)
    df['One_Year_Change{}'.format(col)] = df[col] - df[col].shift(24)
    df['Year_Percentage_Change_{}'.format(col)] = (df[col] - df[col].shift(24)) * 100 / df[col].shift(24)

    return df

features = list(df.columns)
for feature in features:
    if feature not in ['Day', 'Year', 'Month']:
        df = feature_generator(df, feature)
df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
#df = df.fillna(0)

df_info = df[variable_target + ['Day', 'Year', 'Month']]


from sklearn.ensemble import RandomForestRegressor
def feature_selection(name='rf'):
    if name == 'rf':
        s = pd.Series(index=df.drop(df_info.columns, axis=1).columns,
                      data=RandomForestRegressor(random_state=50).fit(df.drop(df_info.columns, axis=1),
                                                                      df[variable_target[0]]).feature_importances_).sort_values(ascending=False)
    elif name == 'corr':
        s = pd.Series(index=df.drop(df_info.columns, axis=1).columns,
                      data=[np.corrcoef(df[feature], df[variable_target[0]])[0, 1] for feature in df.drop(df_info.columns, axis=1)]).sort_values(ascending=False)
    return s


s = feature_selection('rf')
print(s)
N_FEATURES = 22

df = df.loc[:, s.index[:N_FEATURES]]
print(df)
df = pd.concat([df_info, df], axis=1)
print(df)
df.loc[df.Year >= 2018, 'Cambio2018'] = 1
df.loc[df.Year < 2018, 'Cambio2018'] = 0
print(df.shape)
train = df[(df['Year'] >= 2000) & (df['Year'] < 2019)].reset_index(drop=True)
valid = df[df['Year'] == 2019].reset_index(drop=True)
test = df[df['Year'] == 2020].reset_index(drop=True)

print(train.Year.unique())
print(valid.Year.unique())
print(test.Year.unique())

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

scaler = MinMaxScaler().fit(train)
columns = train.columns

train_scaled = pd.DataFrame(scaler.transform(train), columns=columns)
valid_scaled = pd.DataFrame(scaler.transform(valid), columns=columns)
test_scaled = pd.DataFrame(scaler.transform(test), columns=columns)

print(train_scaled)

from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

BATCH_SIZE = 64
SEQ_SIZE = 5


def df_to_generator(df_scaled):
    df_output = df_scaled.loc[:, [variable_target]]
    # df_scaled.drop('INPC', inplace=True, axis=1)

    n_features = df_scaled.shape[1]
    df_generator = TimeseriesGenerator(data=np.array(df_scaled), targets=np.array(df_output),
                                       length=SEQ_SIZE, batch_size=n_features)
    return df_scaled, df_output, df_generator


train_scaled, train_output, train_generator = df_to_generator(train_scaled)
valid_scaled, valid_output, valid_generator = df_to_generator(valid_scaled)
test_scaled, test_output, test_generator = df_to_generator(test_scaled)

print(train_output.shape, train_scaled.shape)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, InputLayer, Conv1D, MaxPooling1D, Convolution1D

N_FEATURES = train_scaled.shape[1]


def create_rnn(type_model='LSTM'):
    if type_model == 'RNN':
        rnn = Sequential()
        rnn.add(SimpleRNN(16, input_shape=(SEQ_SIZE, N_FEATURES)))
        rnn.add(Dense(1))

    elif type_model == 'LSTM':
        rnn = Sequential()
        rnn.add(LSTM(64, input_shape=(SEQ_SIZE, N_FEATURES)))
        rnn.add(Dense(1))

    # CNN
    elif type_model =='CNN':
        rnn = Sequential()
        rnn.add(InputLayer(input_shape=(SEQ_SIZE, N_FEATURES)))
        rnn.add(Convolution1D(filters=32))
        rnn.add(MaxPooling1D(pool_size=2))
        rnn.add(LSTM(16))
        rnn.add(Dense(1))

    elif type_model == 'CNN2':

        rnn = Sequential()
        rnn.add(LSTM(16, return_sequences=True, input_shape=(SEQ_SIZE, N_FEATURES)))
        rnn.add(Convolution1D(input_shape=(SEQ_SIZE, N_FEATURES), filters=64, kernel_size=2))
        rnn.add(MaxPooling1D(pool_size=2))
        rnn.add(Dense(1))

    elif type_model == 'CNN3':
        rnn = Sequential()
        rnn.add(LSTM(60, return_sequences=True, input_shape=(SEQ_SIZE, N_FEATURES)))
        rnn.add(Convolution1D(filters=32, kernel_size=3, strides=1))
        rnn.add(LSTM(8, return_sequences=False))
        rnn.add(Dense(1))

    rnn.compile(loss=tf.keras.losses.MeanSquaredLogarithmicError(), metrics=METRICS.keys(),
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005))
    history = rnn.fit(train_generator, shuffle=False, epochs=EPOCH, verbose=2, batch_size=BATCH_SIZE,
                      validation_data=valid_generator)

    print(rnn.summary())
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    for metric in METRICS.keys():
        plt.xlabel('Epoch')
        plt.ylabel(METRICS[metric])
        plt.plot(hist['epoch'], hist[metric], label='Training')
        plt.plot(hist['epoch'], hist['val_' + metric], label='Validation')
        a = plt.gca()
        a.set_ylim(0, 0.15)
        plt.legend()
        plt.show()

    return rnn


EPOCH = 750
METRICS = {'mae': 'Mean Absolute Error (MAE)', 'mse': 'Mean Squared Error (MSE)'}
model = create_rnn('LSTM')

test_metrics = model.evaluate(test_generator, verbose=0)
print('*** Test metrics ***')
for i, test_metric in enumerate(['loss'] + list(METRICS.keys())):
    print(test_metric, test_metrics[i])

from sklearn.metrics import mean_absolute_error


def inv_scaler(y, current_scaler):

    df_scaler = pd.DataFrame(dict(zip(df.columns, np.ones(y.shape[0]))), index=range(y.shape[0]), columns=df.columns)
    df_scaler[variable_target[0]], df_scaler[variable_target[1]] = y[:, 0], y[:, 1]
    y_esc = pd.DataFrame(current_scaler.inverse_transform(df_scaler), columns=df.columns)[[variable_target]]
    return y_esc



y_train = np.array(train_output)[SEQ_SIZE:].reshape(-1, 2)
y_prediction_train = model.predict(train_generator).reshape(-1, 2)
y_prediction_train_esc, y_train_esc = inv_scaler(y_prediction_train, scaler), inv_scaler(y_train, scaler)

y_true = np.array(test_output)[SEQ_SIZE:].reshape(-1, 2)
y_prediction = model.predict(test_generator).reshape(-1, 2)
y_prediction_esc, y_true_esc = inv_scaler(y_prediction, scaler), inv_scaler(y_true, scaler)

print(mean_absolute_error(y_true[:, 0], y_prediction[:, 0]))
print(mean_absolute_error(y_true[:, 1], y_prediction[:, 1]))
print(mean_absolute_error(y_true_esc[:, 0], y_prediction_esc[:, 0]))
print(mean_absolute_error(y_true_esc[:, 1], y_prediction_esc[:, 1]))

print(mean_absolute_error(y_train[:, 0], y_prediction_train[:, 0]))
print(mean_absolute_error(y_train[:, 1], y_prediction_train[:, 1]))
print(mean_absolute_error(y_train_esc[:, 0], y_prediction_train_esc[:, 0]))
print(mean_absolute_error(y_train_esc[:, 1], y_prediction_train_esc[:, 1]))

model_rf = RandomForestRegressor(random_state=50).fit(train_scaled.iloc[SEQ_SIZE:,:].drop(variable_target, axis=1), train_output.iloc[SEQ_SIZE:])
train_prediction_rf = inv_scaler(model_rf.predict(train_scaled.iloc[SEQ_SIZE:,:].drop(variable_target, axis=1)), scaler)
test_prediction_rf = inv_scaler(model_rf.predict(test_scaled.iloc[SEQ_SIZE:,:].drop(variable_target, axis=1)), scaler)

print(mean_absolute_error(y_train_esc[:, 0], train_prediction_rf[:, 0]))
print(mean_absolute_error(y_train_esc[:, 1], train_prediction_rf[:, 1]))
print(mean_absolute_error(y_true_esc[:, 0], test_prediction_rf[:, 0]))
print(mean_absolute_error(y_true_esc[:, 1], test_prediction_rf[:, 1]))

from sklearn.ensemble import GradientBoostingRegressor
model_rf = GradientBoostingRegressor(random_state=50).fit(train_scaled.iloc[SEQ_SIZE:,:].drop(variable_target, axis=1), train_output.iloc[SEQ_SIZE:])
train_prediction_gbr = inv_scaler(model_rf.predict(train_scaled.iloc[SEQ_SIZE:,:].drop(variable_target, axis=1)), scaler)
test_prediction_gbr = inv_scaler(model_rf.predict(test_scaled.iloc[SEQ_SIZE:,:].drop(variable_target, axis=1)), scaler)

print(mean_absolute_error(y_train_esc[:, 0], train_prediction_gbr[:, 0]))
print(mean_absolute_error(y_train_esc[:, 1], train_prediction_gbr[:, 1]))
print(mean_absolute_error(y_true_esc[:, 0], test_prediction_gbr[:, 0]))
print(mean_absolute_error(y_true_esc[:, 1], test_prediction_gbr[:, 1]))

# print(mean_absolute_error(np.mean(np.append(y_prediction_train_esc, train_prediction_rf), axis=1), train_prediction_rf))
# print(mean_absolute_error(np.mean(np.append(y_prediction_esc, test_prediction_rf), axis=1), test_prediction_rf))

from scipy.signal import lfilter


def plot_results(predictions, true_values):
    plt.plot(predictions, 'y', label='Predicted')
    plt.plot(true_values, 'r', label='True')
    plt.title('True and predicted value')
    plt.xlabel('Index')
    plt.ylabel('Dato')
    plt.legend()
    plt.show()


plot_results(y_prediction_esc[:,0], y_true_esc[:,0])
plot_results(y_prediction_esc[:,1], y_true_esc[:,1])
plot_results(y_prediction_train_esc[:,0], y_train_esc[:,0])
plot_results(y_prediction_train_esc[:,1], y_train_esc[:,1])


def error_variacion(predictions, true_values, var_predictora,  tipo_dato='training'):
    df_variacion = pd.DataFrame(columns=['variacion_real', 'variacion_pronosticada'])
    if tipo_dato == 'training':
        # true_values = np.append(np.array(df.loc[df.Year == 1999, variable_target]), true_values)
        data_1999 = {'INPC_General': ['40.2940424', '40.64551836', '40.92767545', '41.09961013', '41.29594312',
                                      '41.49342412', '41.6870306', '41.86197932', '41.96430761', '42.08730267',
                                      '42.24416816', '42.35998764', '42.52474674', '42.63841318', '42.77073686',
                                      '42.87191714', '43.17029143', '43.29988858', '43.46077263', '43.55692985',
                                      '43.82114721', '43.97040611', '44.19400744', '44.47702554'],
                     'INPC_Sub': ['43.0504988', '43.49265163', '43.88333973', '44.2422043', '44.59523712',
                                  '44.92030338',
                                  '45.18329684', '45.40433808', '45.64828322', '45.81764455', '46.00781399',
                                  '46.16908444',
                                  '46.32554281', '46.4597995', '46.64884146', '46.83819141', '47.22727646',
                                  '47.39878603',
                                  '47.56509734', '47.741314', '47.87922017', '48.03264253', '48.24723146',
                                  '48.43903368']}
        true_values = np.append(np.array(data_1999[var_predictora]).astype(np.float64), np.array(true_values))
        df_variacion['variacion_real'] = ((np.divide(true_values,
                                                     np.array(pd.Series(true_values).shift(-24)))) - 1) * 100
        df_variacion['variacion_pronosticada'] = ((np.divide(predictions, np.array(
            pd.Series(true_values).shift(-24).dropna()))) - 1) * 100
        n = 10
        b = [1 / n] * n
        a = 1
        df_variacion['variacion_suavizada'] = lfilter(b, a, df_variacion.variacion_pronosticada)

    elif tipo_dato == 'testing':
        n = true_values.shape[0]
        # n = df.loc[df.Year == 2020].shape[0]
        # true_values, predictions = true_values[:n], predictions[:n]
        compared_to = np.array(df.loc[df.Year == 2020, variable_target][:n])
        df_variacion['variacion_real'] = ((np.divide(np.array(true_values), compared_to)) - 1) * 100
        df_variacion['variacion_pronosticada'] = ((np.divide(np.array(predictions), compared_to)) - 1) * 100
        n = 10
        b = [1.5 / n] * n
        a = 1
        df_variacion['variacion_suavizada'] = lfilter(b, a, df_variacion.variacion_pronosticada)
    df_variacion = df_variacion.dropna().reset_index(drop=True)
    print(tipo_dato)
    print(df_variacion)
    print(mean_absolute_error(df_variacion.variacion_real, df_variacion.variacion_pronosticada))
    print(mean_absolute_error(df_variacion.variacion_real, df_variacion.variacion_suavizada))

    df_variacion.plot()
    plt.legend()
    plt.show()


error_variacion(y_prediction_train_esc[:,0], y_train_esc[:,0], tipo_dato='training', var_predictora=variable_target[0])
error_variacion(y_prediction_train_esc[:,0], y_train_esc[:,0], tipo_dato='training', var_predictora=variable_target[0])
error_variacion(y_prediction_esc[:,1], y_true_esc[:,1], tipo_dato='testing', var_predictora=variable_target[1])
error_variacion(y_prediction_esc[:,1], y_true_esc[:,1], tipo_dato='testing', var_predictora=variable_target[1])
