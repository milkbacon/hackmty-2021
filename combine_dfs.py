
import pandas as pd
import os
from sklearn.impute import KNNImputer
import calendar
import numpy as np
from datetime import datetime, timedelta

pd.set_option('display.max_rows', 500)

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

    df['Fecha'] = pd.to_datetime([df.Day.astype('str')[i] + '/' + df.Month.astype('str')[i] + '/' + df.Year.astype('str')[i] for i in range(df.shape[0])], format='%d/%m/%Y')


    for file in os.listdir('procesados/no_inegi'):
        print(file)

        df_temp = pd.read_csv('procesados/no_inegi/{}'.format(file), index_col=0)
        df_temp['Fecha'] = pd.to_datetime(df_temp['Fecha'], format='%Y-%m-%d')

        df_temp = df_temp[df_temp.Year >= 2000].reset_index(drop=True)
        df_temp.columns = [(file.split('_')[0] + '_' + col) for col in df_temp.columns]

        idx_vars = [file.split('_')[0] + '_Month', file.split('_')[0] + '_Year', file.split('_')[0] + '_Day', file.split('_')[0] + '_Fecha']

        # recorrer ventanas de 1-primerQuincena y primerQ-segundaQ
        agg_funcs = ['mean', 'std', 'median', 'max', 'min']

        features = list(set(df_temp.columns))
        for col in idx_vars:
            features.remove(col)

        licuado = [column + '_' + agg_fun for column in features for agg_fun in agg_funcs]
        df_combined_features = pd.DataFrame(columns=licuado)

        for fecha in df.Fecha:
            fecha_anterior = fecha - timedelta(days=15)
            df_temp_sub = df_temp[(df_temp[idx_vars[3]] < fecha) & (df_temp[idx_vars[3]] > fecha_anterior)].loc[:, features].dropna()
            if len(df_temp_sub.index) == 0:
                results = np.zeros(len(licuado))
            else:
                df_temp_sub = df_temp_sub.agg(agg_funcs).T.stack()
                results = [df_temp_sub[feature][agg_fun] for feature in features for agg_fun in agg_funcs]
            results = dict(zip(licuado, results))
            df_combined_features = df_combined_features.append(results, ignore_index=True)

        all_columns = list(df.columns) + list(df_combined_features.columns)
        df = pd.concat([df, df_combined_features], join='inner', axis=1, ignore_index=True)
        df.columns = all_columns

        # df_temp.drop(idx_vars[3], axis=1, inplace=True)

        # df = df.merge(df_temp, left_on=['Month', 'Year', 'Day'], right_on=idx_vars[:3], how='left').drop(idx_vars, axis=1)
    df = df.reset_index(drop=True).fillna(0)
    df.to_csv('processed_no_impute_yes_windows.csv')

def solve_impute():
    print('solve_impute')
    df = pd.read_csv('processed_no_impute_yes_windows.csv', index_col=0, na_values=0).reset_index(drop=True).drop('Fecha', axis=1)
    # df = pd.read_csv('processed_no_impute_yes_windows.csv', index_col=0)
    print(df)
    #df_sub = df.iloc[:, :30]

    #df.index = pd.to_datetime([df.Day.astype('str')[i] + '/' + df.Month.astype('str')[i] + '/' + df.Year.astype('str')[i] for i in range(df.shape[0])], format='%d/%m/%Y')

    # df = df[df['Year'] >= 2000]
    # df = df.fillna(0)
    # df = df.interpolate(method='linear', axis=0).fillna(0)
    year, month, day = df.Year, df.Month, df.Day

    df = df.drop(['Year', 'Month', 'Day'], axis=1)
    columns = df.columns
    # print(KNNImputer(n_neighbors=2).fit_transform(df))
    df = pd.DataFrame(KNNImputer(n_neighbors=3).fit_transform(np.array(df)), columns=columns).fillna(0)
    df['Year'], df['Month'], df['Day'] = year, month, day

    # df = pd.DataFrame(KNNImputer(n_neighbors=5).fit_transform(df), columns=columns)
    # df = df.reset_index(drop=True).fillna(0)
    df.to_csv('procesados.csv', index=False)
    print(df)
    # df.to_csv('procesados.csv', index=False)
according_granularity()
solve_impute()