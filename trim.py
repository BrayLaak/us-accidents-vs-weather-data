import pandas as pd

df_data = pd.read_csv("WeatherEvents_Jan2016-Dec2021.csv")
df_data = pd.read_csv("Weather_Events.csv")

print(df_data.columns)

df_data = df_data.drop(df_data.columns[[0,5,6,7,8,9,10,13]], axis=1)


print(df_data.columns)
print(df_data.head())
df_data = df_data.drop(df_data.columns[[0,2,5,6,7,8,9,10,13]], axis=1)

print(df_data.columns)
print(df_data.head())

#df_data = df_data.loc[df_data['Defining Parameter'] == '']

df_data['StartTime(UTC)'] = pd.to_datetime(df_data['StartTime(UTC)'])
df_data['EndTime(UTC)'] = pd.to_datetime(df_data['EndTime(UTC)'])

df_data.rename(columns={'StartTime(UTC)': 'Start_Time', 'EndTime(UTC)': 'End_Time'}, inplace=True)

print(df_data.columns)
print(df_data.head())



#print(df_data.dtypes)