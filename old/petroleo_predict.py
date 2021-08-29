import pandas as pd
import matplotlib.pyplot as plt

df= pd.read_csv('procesados/no_inegi/{}.csv'.format('petroleo_'), index_col=0)
df['Date'] = pd.to_datetime([str(df.Year[i]) + '/' + str(df.Month[i]) for i in range(df.shape[0])], format='%Y/%m')
df = df.drop(['Month', 'Year'])
df.index = df.Date
df.loc[:, ['Mezcla_Mexicana', 'WTI','BRENT','CGPE']].plot()
plt.show()
