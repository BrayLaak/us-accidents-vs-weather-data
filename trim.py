import pandas as pd

df_data = pd.read_csv("US_Accidents.csv")

df_data = df_data.drop(df_data.columns[[6,7,8,9,10,11,12,13,14,16,17,19,21,22,23,24,25,26,27,30,31,32,33,34,35,36,37,38,39,40,41,42,44,45,46]], axis=1)

print(df_data.columns)


#df_data = df_data.loc[df_data['Defining Parameter'] == '']

#df_data.to_pickle('US_Accidents.pkl')