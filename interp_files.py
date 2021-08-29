import pandas as pd
import numpy as np
# pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
# pd.set_option('display.width', 1000)

# archivos para los que sirve el script:
# cp_indice 2
# ifb_indice 3
# igae_indice 2
# ivf_var_mensual 2

archivo = 'pobreza_'

df = pd.read_excel('no_procesados/datos_corto_plazo/{}.xlsx'.format(archivo), skiprows=5)

if archivo == 'ivf_var_mensual':
    #df = df.iloc[:(df.shape[0] - 4), :]
    df_nombres = df.iloc[:, 1]
    nombres = list(df_nombres.fillna(''))
    mal_indice = nombres.index('')
    df = df.iloc[:mal_indice, 2:]
elif archivo == 'ifb_indice':
    df_nombres = df.iloc[:, :3]
    df_nombres.columns = ['col' + str(x) for x in range(0, 3)]
    df_nombres.fillna('', inplace=True)
    nombres = list(df_nombres.col0 + df_nombres.col1 + df_nombres.col2)

    mal_indice = nombres.index('')
    df = df.iloc[:mal_indice, 4:]
elif archivo in ['cp_indice', 'igae_indice']:
    df_nombres = df.iloc[:, :2]
    df_nombres.columns = ['col' + str(x) for x in range(0, 2)]
    df_nombres.fillna('', inplace=True)
    nombres = list(df_nombres.col0 + df_nombres.col1)

    mal_indice = nombres.index('')
    df = df.iloc[:mal_indice, 2:]
elif archivo == 'pobreza_':
    df = pd.read_excel('no_procesados/datos_corto_plazo/{}.xlsx'.format(archivo), skiprows=3)
    print(df.shape)
    df = df.iloc[3:(df.shape[0] - 5), [1, 2, 4, 5, 7, 8]].reset_index(drop=True)
    df.columns = ['Year', 'Month', 'CA Rural', 'CA Urbano', 'CANA Rural', 'CANA Urbano']
    df = df.ffill(axis=0)
    month_encoding = dict(zip(['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun',
                               'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic'], range(1, 13)))
    df['Month'] = [month_encoding[var] for var in df.Month]
    df.to_csv('procesados/datos_corto_plazo/{}.csv'.format(archivo))
    print(df)
    exit()

df.index = nombres[:mal_indice]
df = df.T.reset_index()

month_encoding = dict(zip(['ENE', 'FEB', 'MAR', 'ABR', 'MAY', 'JUN',
                           'JUL', 'AGO', 'SEP', 'OCT', 'NOV', 'DIC'], range(1,13)))
df['Year'] = [(int(var[4:]) if len(var) > 3 else 0) + 1993 for var in df['index']]
df['Month'] = [month_encoding[var[:3]] for var in df['index']]
df = df.drop('index', axis=1)

df.to_csv('procesados/datos_corto_plazo/{}.csv'.format(archivo))
print(df.columns)

print(df)
