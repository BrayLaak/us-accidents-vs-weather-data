import pandas as pd

df_data = pd.read_csv("WeatherEvents_Jan2016-Dec2021.csv")

print(df_data.columns)

df_data = df_data.drop(df_data.columns[[0,2,5,6,7,8,9,10,13]], axis=1)

print(df_data.columns)
print(df_data.head())

#df_data = df_data.loc[df_data['Defining Parameter'] == '']

