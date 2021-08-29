
import pandas as pd
import os

df = pd.read_csv('INP_INP20210828141953.csv', skiprows=13, encoding='latin-1', na_values='N/E')
df = df.iloc[:, :3]

df.columns = ['Fecha', 'INPC_General', 'INPC_Sub']
quantil_enconding = {'1Q': 1, '2Q': 15}
month_encoding = dict(zip(['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun',
                           'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic'], range(1,13)))
df['Day'] = [quantil_enconding[fecha[:2]] for fecha in df.Fecha]
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
df = df.interpolate(method='time')


for file in os.listdir('procesados/no_inegi'):
    #if file == 'petroleo_.csv':
    #    continue
    df_temp = pd.read_csv('procesados/no_inegi/{}'.format(file), index_col=0)
    df_temp.columns = [(file.split('_')[0] + '_' + col) if (col != 'Month') or (col != 'Year') or (col != 'Day') else col for col in df_temp.columns]
    idx_vars = [file.split('_')[0] + '_Month', file.split('_')[0] + '_Year', file.split('_')[0] + '_Day']
    df = df.merge(df_temp, left_on=['Month', 'Year', 'Day'], right_on=idx_vars, how='left').drop(idx_vars, axis=1)

# df = df[df['Year'] >= 2000]
# df = df.fillna(0)
# df = df.interpolate(method='linear', axis=0)
df = df.reset_index(drop=True).fillna(0)
df.to_csv('procesados.csv', index=False)