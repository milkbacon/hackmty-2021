import pandas as pd
import numpy as np

# archivos para los que sirve el script:
# peso-dolar.xlsx
for archivo in ['peso_dolar', 'petroleo_', 'DJA_', 'gasolina_mensual', 'weekly_economic_index']:

    if archivo == 'peso_dolar':
        df = pd.read_excel('no_procesados/no_inegi/{}.xlsx'.format(archivo), skiprows=17)
        df['Fecha'] = pd.to_datetime(df.Fecha, format='%d/%m/%Y')
        df['Year'] = df.Fecha.dt.year
        df['Month'] = df.Fecha.dt.month
        df['Day'] = df.Fecha.dt.day
        # df.drop('Fecha', axis=1, inplace=True)
    elif archivo == 'petroleo_':
        df = pd.read_excel('no_procesados/no_inegi/{}.xlsx'.format(archivo))
        features = list(df.columns)
        features.remove('Fecha')
        df['Fecha'] = pd.to_datetime(df.Fecha, format='%m/%d/%Y')
        df['Year'] = df.Fecha.dt.year
        df['Month'] = df.Fecha.dt.month
        df['Day'] = df.Fecha.dt.day
        # df.drop('Fecha', axis=1, inplace=True)

        for feature_i in features:
             for feature_j in features:
                 if feature_i != feature_j:
                     df[feature_i + '-' + feature_j] = df[feature_i] - df[feature_j]
                     df[feature_i + '/' + feature_j] = df[feature_i] - df[feature_j]
    elif archivo == 'DJA_':
        df = pd.read_csv('no_procesados/no_inegi/{}.csv'.format(archivo), skiprows=1)
        df['Fecha'] = pd.to_datetime(df.Date, format='%m/%d/%Y')
        del df['Date']
        df['Year'] = df.Fecha.dt.year
        df['Month'] = df.Fecha.dt.month
        df['Day'] = df.Fecha.dt.day
        # df.drop('Fecha', axis=1, inplace=True)
    elif archivo == 'gasolina_mensual':
        df = pd.read_csv('no_procesados/no_inegi/{}.csv'.format(archivo), skiprows=1, encoding='latin-1').iloc[:, -4:].dropna()
        df.columns = ['Fecha', 'Gas87', 'Gas91', 'Diesel']
        features = list(df.columns)
        features.remove('Fecha')
        df['Fecha'] = pd.to_datetime(df.Fecha, format='%d/%m/%Y')
        df['Year'] = df.Fecha.dt.year
        df['Month'] = df.Fecha.dt.month
        df['Day'] = df.Fecha.dt.day

        for feature_i in features:
             for feature_j in features:
                 if feature_i != feature_j:
                     df[feature_i + '-' + feature_j] = df[feature_i] - df[feature_j]
                     df[feature_i + '/' + feature_j] = df[feature_i] - df[feature_j]
        # df.drop('Fecha', axis=1, inplace=True)
    elif archivo == 'weekly_economic_index':
        df = pd.read_excel('no_procesados/no_inegi/{}.xlsx'.format(archivo), skiprows=4, sheet_name=2)
        df.columns = ['Date', 'WEI', 'WEI29', 'WEI28']
        df['Fecha'] = pd.to_datetime(df.Date, format='%m/%d/%Y')
        del df['Date']
        df['Year'] = df.Fecha.dt.year
        df['Month'] = df.Fecha.dt.month
        df['Day'] = df.Fecha.dt.day
        # df.drop('Fecha', axis=1, inplace=True)
    print(df)
    df.to_csv('procesados/no_inegi/{}.csv'.format(archivo))



