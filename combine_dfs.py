
import pandas as pd
import os
from sklearn.impute import KNNImputer
import calendar
import numpy as np
from datetime import datetime, timedelta

def according_granularity():

    df = pd.read_csv('INP_INP20210828141953.csv', skiprows=13, encoding='latin-1', na_values='N/E')
    df = df.iloc[:, :3]

    df.columns = ['Fecha', 'INPC_General', 'INPC_Sub']
    quantil_enconding = {'1Q': 1, '2Q': 15}


    def get_quincena(quincena, mes, anno):
        return 15 if quincena == '1Q' else calendar.monthrange(anno, mes)[1]


    month_encoding = dict(zip(['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun',
                               'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic'], range(1,13)))

    df['Day'] = [get_quincena(fecha[:2], month_encoding[fecha[3:6]], int(fecha[7:11])) for fecha in df.Fecha]
    df['Month'] = [month_encoding[fecha[3:6]] for fecha in df.Fecha]
    df['Year'] = [int(fecha[7:11]) for fecha in df.Fecha]

    df = df.drop('Fecha', axis=1)

    # df['Fecha'] = df.Day.astype('str') + '' df.Month.astype('str') + df.Year.astype('str')

    for file in os.listdir('procesados/datos_corto_plazo/'):
        print(file)
        df_temp = pd.read_csv('procesados/datos_corto_plazo/{}'.format(file), index_col=0)
        df_temp.columns = [(file.split('_')[0] + '_' + col) if (col != 'Month') or (col != 'Year') else col for col in df_temp.columns]
        idx_vars = [file.split('_')[0] + '_Month', file.split('_')[0] + '_Year']
        df = df.merge(df_temp, left_on=['Month', 'Year'], right_on=idx_vars, how='left').drop(idx_vars, axis=1)
        # .interpolate(method='time')

    df['Fecha'] = [df.Day.astype('str')[i] + '/' + df.Month.astype('str')[i] + '/' + df.Year.astype('str')[i] for i in range(df.shape[0])]
    df.index = pd.to_datetime(df.Fecha, format='%d/%m/%Y')
    df.drop('Fecha', axis=1, inplace=True)

    # .interpolate(method='time', limit_direction='forward', axis=0)

    for file in os.listdir('procesados/no_inegi'):
        print(file)
        if file in ['peso_dolar.csv', 'petroleo_.csv']:
            continue

        df_temp = pd.read_csv('procesados/no_inegi/{}'.format(file), index_col=0)
        df_temp = df_temp[df_temp.Year >= 2000].reset_index(drop=True)
        print(df_temp)
        df_temp.columns = [(file.split('_')[0] + '_' + col) for col in df_temp.columns]
        print(df_temp)

        df['Fecha'] = pd.to_datetime([df.Day.astype('int').astype('str')[i] + '/' +
                                      df.Month.astype('int').astype('str')[i] + '/' +
                                      df.Year.astype('int').astype('str')[i] for i in
                                      range(df.shape[0])], format='%d/%m/%Y')

        idx_vars = [file.split('_')[0] + '_Month', file.split('_')[0] + '_Year', file.split('_')[0] + '_Day']
        df_temp['Fecha_Temp'] = pd.to_datetime([df_temp[idx_vars[2]].astype('int').astype('str')[i] + '/' +
                                                df_temp[idx_vars[0]].astype('int').astype('str')[i] + '/' +
                                                df_temp[idx_vars[1]].astype('int').astype('str')[i]
                                                for i in range(df_temp.shape[0])], format='%d/%m/%Y')

        # recorrer ventanas de 1-primerQuincena y primerQ-segundaQ
        for fecha in df.Fecha:
            fecha_anterior = fecha - timedelta(days=15)
            features = list(df_temp.columns)
            for col in ['Fecha_Temp'] + idx_vars:
                features.remove(col)
            df_temp_sub = df_temp[(df_temp.Fecha_Temp < fecha) & (df_temp.Fecha_Temp > fecha_anterior)][[features]].dropna().agg(['mean', 'std', 'median', 'max', 'min'])
            # for feature in df_temp.columns:
            #     if feature not in idx_vars and feature != 'Fecha_Temp':
            #         df_temp.loc[df_temp_sub.index, idx_vars[2]] = fecha.day
            #        df_temp.loc[df_temp_sub.index, feature] = np.nanmean(df_temp_sub[feature])

        df_temp.drop('Fecha_Temp', axis=1, inplace=True)
        df.drop('Fecha', axis=1, inplace=True)

        df = df.merge(df_temp, left_on=['Month', 'Year', 'Day'], right_on=idx_vars, how='left').drop(idx_vars, axis=1)
    df = df.reset_index(drop=True).fillna(0)
    df.to_csv('processed_no_impute_yes_windows.csv')

def solve_impute():
    df = pd.read_csv('processed_no_impute_yes_windows.csv', index_col=0)

    #df_sub = df.iloc[:, :30]


    # df = df[df['Year'] >= 2000]
    # df = df.fillna(0)
    # df = df.interpolate(method='linear', axis=0)
    columns = df.columns
    df = pd.DataFrame(KNNImputer(n_neighbors=2).fit_transform(df), columns=columns)
    df = df.reset_index(drop=True).fillna(0)
    df.to_csv('procesados.csv', index=False)
according_granularity()
# solve_impute()