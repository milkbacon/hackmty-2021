
from tensorflow.keras.models import load_model

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

variable_target = 'INPC_Sub'
variable_dropeada = 'INPC_Sub' if variable_target == 'INPC_General' else 'INPC_General'
df = df.drop(variable_dropeada, axis=1)
print(df)

def feature_generator(df, col):
    # Modificado por 1 y -1 para obtener dato posterior o anterior
    signo = 1
    df['Prev_Q_{}'.format(col)] = df[col].shift(1*signo)
    df['One_Q_Change_{}'.format(col)] = df[col] - df[col].shift(1*signo)
    df['Q_Percentage_Change_{}'.format(col)] = (df[col] - df[col].shift(1*signo))*100/df[col].shift(1*signo)

    df['Prev_Trim_{}'.format(col)] = df[col].shift(6*signo)
    df['One_Trim_Change{}'.format(col)] = df[col] - df[col].shift(6*signo)
    df['Trim_Percentage_Change_{}'.format(col)] = (df[col] - df[col].shift(6*signo))*100/df[col].shift(6*signo)

    df['Prev_Sem_{}'.format(col)] = df[col].shift(12*signo)
    df['One_Sem_Change{}'.format(col)] = df[col] - df[col].shift(12*signo)
    df['Sem_Percentage_Change_{}'.format(col)] = (df[col] - df[col].shift(12))*100/df[col].shift(12*signo)

    df['Prev_Year_{}'.format(col)] = df[col].shift(24*signo)
    df['One_Year_Change{}'.format(col)] = df[col] - df[col].shift(24*signo)
    df['Year_Percentage_Change_{}'.format(col)] = (df[col] - df[col].shift(24*signo)) * 100 / df[col].shift(24*signo)

    return df

features = list(df.columns)
for feature in features:
    if feature not in ['Day', 'Year', 'Month']:
        df = feature_generator(df, feature)
df = df.fillna(0).replace([np.inf, -np.inf], np.nan).dropna()
info_columns = [variable_target] + ['Day', 'Year', 'Month']


from sklearn.ensemble import RandomForestRegressor
def feature_selection(name='rf'):
    if name == 'rf':
        s = pd.Series(index=df.drop(info_columns, axis=1).columns,
                      data=RandomForestRegressor(random_state=50, n_jobs=-1).fit(df.drop(info_columns, axis=1),
                                                                      df[variable_target]).feature_importances_).sort_values(ascending=False)
    elif name == 'corr':
        s = pd.Series(index=df.drop(info_columns, axis=1).columns,
                      data=[np.corrcoef(df[feature], df[variable_target])[0, 1] for feature in df.drop(info_columns, axis=1)]).sort_values(ascending=False)
    return s


s = feature_selection('rf')
print(s)
N_FEATURES = 13
# 13 0.46
# 10 2.79
print(list(s.index[:N_FEATURES]))
exit()

df = df.loc[:, list(s.index[:N_FEATURES]) + info_columns]
print(df)
df.loc[df.Year >= 2018, 'Cambio2018'] = 1
df.loc[df.Year < 2018, 'Cambio2018'] = 0
print(df.shape)
train = df[(df['Year'] >= 2000) & (df['Year'] < 2021)].reset_index(drop=True)
# valid = df[df['Year'] == 2020].reset_index(drop=True)
test = df[df['Year'] == 2021].reset_index(drop=True)
print(test)

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

scaler = MinMaxScaler().fit(train)
columns = train.columns

train_scaled = pd.DataFrame(scaler.transform(train), columns=columns)
# valid_scaled = pd.DataFrame(scaler.transform(valid), columns=columns)
test_scaled = pd.DataFrame(scaler.transform(test), columns=columns)

print(train_scaled)

from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

BATCH_SIZE = 64
SEQ_SIZE = 5
n_features = train_scaled.shape[1]

def df_to_generator(df_scaled):
    df_output = df_scaled.loc[:, variable_target]
    # df_scaled.drop('INPC', inplace=True, axis=1)

    n_features = df_scaled.shape[1]

    df_generator = TimeseriesGenerator(data=np.array(df_scaled), targets=np.array(df_output),
                                       length=SEQ_SIZE, batch_size=n_features)
    return df_scaled, df_output, df_generator

train_output = train_scaled.loc[:, variable_target]
train_generator = TimeseriesGenerator(data=np.array(train_scaled), targets=np.array(train_output),
                                   length=SEQ_SIZE, batch_size=n_features)

test_output = test_scaled.loc[:, variable_target]

test_scaled = pd.concat([train_scaled.tail(5).reset_index(drop=True), test_scaled], axis=0, ignore_index=True).reset_index(drop=True)
test_output = pd.concat([train_output.tail(5).reset_index(drop=True), test_output], axis=0, ignore_index=True).reset_index(drop=True)
#test_scaled, test_output = np.append(np.array(train_scaled.tail(5)), np.array(test_scaled)), np.append(np.array(train_output.tail(5)), np.array(test_output))
print(test_scaled.shape)
print(test_output.shape)
print(test_output)
print(test_scaled)
test_generator = TimeseriesGenerator(data=np.array(test_scaled), targets=np.array(test_output),
                                   length=SEQ_SIZE, batch_size=n_features)
# valid_scaled, valid_output, valid_generator = df_to_generator(valid_scaled)


model = load_model('modelo0.98_1.10.h5')

def inv_scaler(y, current_scaler):
    df_scaler = pd.DataFrame(dict(zip(df.columns, np.ones(y.shape[0]))), index=range(y.shape[0]), columns=df.columns)
    df_scaler[variable_target] = y
    y_esc = pd.DataFrame(current_scaler.inverse_transform(df_scaler), columns=df.columns)[variable_target]
    return y_esc


y_train = np.array(train_output)[SEQ_SIZE:].reshape(-1, 1)
y_prediction_train = model.predict(train_generator).reshape(-1, 1)
y_prediction_train_esc, y_train_esc = inv_scaler(y_prediction_train, scaler), inv_scaler(y_train, scaler)

y_true = np.array(test_output)[SEQ_SIZE:].reshape(-1, 1)
y_prediction = model.predict(test_generator).reshape(-1, 1)
y_prediction_esc, y_true_esc = inv_scaler(y_prediction, scaler), inv_scaler(y_true, scaler)


from sklearn.metrics import mean_absolute_error
from scipy.signal import lfilter


def error_variacion(predictions, true_values, tipo_dato='training'):
    df_variacion = pd.DataFrame(columns=['variacion_real', 'variacion_pronosticada'])
    if tipo_dato == 'training':
        # true_values = np.append(np.array(df.loc[df.Year == 1999, variable_target]), true_values)
        data_1999 = {'INPC_General': ['40.2940424', '40.64551836', '40.92767545', '41.09961013', '41.29594312',
                                      '41.49342412', '41.6870306', '41.86197932', '41.96430761', '42.08730267',
                                      '42.24416816', '42.35998764', '42.52474674', '42.63841318', '42.77073686',
                                      '42.87191714', '43.17029143', '43.29988858', '43.46077263', '43.55692985',
                                      '43.82114721', '43.97040611', '44.19400744', '44.47702554'],
                     'INPC_Sub': ['43.0504988', '43.49265163', '43.88333973', '44.2422043', '44.59523712', '44.92030338',
                                  '45.18329684', '45.40433808', '45.64828322', '45.81764455', '46.00781399', '46.16908444',
                                  '46.32554281', '46.4597995', '46.64884146', '46.83819141', '47.22727646', '47.39878603',
                                  '47.56509734', '47.741314', '47.87922017', '48.03264253', '48.24723146', '48.43903368']}
        val_actuales = np.array(true_values)
        true_values = np.append(np.array(data_1999[variable_target]).astype(np.float64), np.array(true_values)[:(true_values.shape[0] - len(data_1999['INPC_General']))])

        df_variacion['variacion_real'] = ((np.divide(val_actuales, np.array(pd.Series(true_values)))) - 1) * 100
        df_variacion['variacion_pronosticada'] = ((np.divide(predictions, np.array(pd.Series(true_values)))) - 1) * 100
        n = 10
        b = [1 / n] * n
        a = 1
        df_variacion['variacion_suavizada'] = lfilter(b, a, df_variacion.variacion_pronosticada)
        print(mean_absolute_error(df_variacion.variacion_real, df_variacion.variacion_suavizada))
    elif tipo_dato == 'testing':
        n = true_values.shape[0]
        # n = df.loc[df.Year == 2020].shape[0]
        # true_values, predictions = true_values[:n], predictions[:n]
        compared_to = np.array(df.loc[df.Year == 2020, variable_target][:n])
        df_variacion['variacion_real'] = ((np.divide(np.array(true_values), compared_to)) - 1) * 100
        df_variacion['variacion_pronosticada'] = ((np.divide(np.array(predictions), compared_to)) - 1) * 100
        print(np.abs(np.array([df_variacion.iloc[i, :]['variacion_real'] - df_variacion.iloc[i, :]['variacion_pronosticada'] for i in range(df_variacion.shape[0])])))
    df_variacion = df_variacion.dropna().reset_index(drop=True)
    print(mean_absolute_error(df_variacion.variacion_real, df_variacion.variacion_pronosticada))
    # print(np.abs(df.variacion_real - df.variacion_pronosticada))
    # print(mean_absolute_error(df_variacion.variacion_real, df_variacion.variacion_suavizada))

    df_variacion.plot()
    plt.legend()
    plt.show()


error_variacion(y_prediction_train_esc, y_train_esc, tipo_dato='training')
error_variacion(y_prediction_esc, y_true_esc, tipo_dato='testing')
