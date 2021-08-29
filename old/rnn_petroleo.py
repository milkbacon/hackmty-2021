import pandas as pd
import os
import matplotlib.pyplot as plt

archivo = 'petroleo'

df = pd.read_excel('no_procesados/no_inegi/{}.xlsx'.format(archivo))
features = list(df.columns)
features.remove('Fecha')
df['Fecha'] = pd.to_datetime(df.Fecha, format='%m/%d/%Y')
df['Year'] = df.Fecha.dt.year
df['Month'] = df.Fecha.dt.month
df['Day'] = df.Fecha.dt.day
df.drop('Fecha', axis=1, inplace=True)

for feature_i in features:
    for feature_j in features:
        if feature_i != feature_j:
            df[feature_i + '-' + feature_j] = df[feature_i] - df[feature_j]
            df[feature_i + '/' + feature_j] = df[feature_i] - df[feature_j]
