import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pmdarima as pm

df = pd.read_csv('data_inflasi.csv', sep=';')

datetimeindex = pd.to_datetime([f'{m_y.split("-")[0]}-{m_y.split("-")[1]}-01' for m_y in df['Bulan']])

df['Bulan'] = datetimeindex

df = df.set_index('Bulan')

y = df['Inflasi'].resample('MS').mean()

y.plot(figsize=(10, 4), title='Inflasi Indonesia', fontsize=14)
plt.show()

from pylab import rcParams
rcParams['figure.figsize'] = 10, 4

decomposition = sm.tsa.seasonal_decompose(y, model='additive')
fig = decomposition.plot()


model = pm.auto_arima(y, seasonal=True, m=12, trace=True,
                      error_action='ignore', suppress_warnings=True)
print(model.summary())