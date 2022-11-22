import pandas as pd
from matplotlib import pyplot as plt


# Function to generate a simple line graph, add a grid, and save the graph to a file
def generate_graph(df, xAxis, yAxis, graph_title, graph_color):
    df_graph = df.plot(x=xAxis, y=yAxis, title=graph_title, color=graph_color)
    df_graph.grid()
    df_graph.figure.savefig(graph_title + '.pdf')
    return



# Read in accident data
acc_data = pd.read_csv('US_Accidents_Dec21_updated.csv')


acc_data = acc_data.drop(acc_data.columns[[0,3,4,5,6,7,8,9,10,11,12,13,16,17,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46]], axis=1)

acc_data['State, County'] = acc_data['State'].astype(str) + ", " + acc_data['County'].astype(str)
acc_data = acc_data.drop(acc_data.columns[[2,3]], axis=1)
acc_data.dropna()
acc_data.drop(acc_data.loc[acc_data['Timezone'] == 'nan'].index, inplace=True)

acc_data['Start_Time'] = pd.to_datetime(acc_data['Start_Time'])


location_list = acc_data['State, County'].unique()

acc_location_grouped = acc_data.groupby(['State, County'])

acc_tz_grouped = acc_data.groupby(['Timezone'])
# Timezones = 'US/Central', 'US/Eastern', 'US/Mountain', 'US/Pacific'


Eastern_acc = acc_tz_grouped.get_group('US/Eastern')
Eastern_acc['Start_Time'] = pd.to_datetime(Eastern_acc['Start_Time']).dt.tz_localize(tz='US/Eastern', ambiguous='NaT')
Eastern_acc.drop(Eastern_acc.loc[Eastern_acc['Start_Time'] == 'NaT'].index, inplace=True)
Eastern_acc['Start_Time'] = Eastern_acc['Start_Time'].dt.tz_convert('UTC')


Central_acc = acc_tz_grouped.get_group('US/Central')
Central_acc['Start_Time'] = pd.to_datetime(Central_acc['Start_Time']).dt.tz_localize(tz='US/Central', ambiguous='NaT')
print(Central_acc.head())
Central_acc.drop(Central_acc.loc[Central_acc['Start_Time'] == 'NaT'].index, inplace=True)
Central_acc['Start_Time'] = Central_acc['Start_Time'].dt.tz_convert('UTC')
print(Central_acc.head())

Mountain_acc = acc_tz_grouped.get_group('US/Mountain')
Mountain_acc['Start_Time'] = pd.to_datetime(Mountain_acc['Start_Time']).dt.tz_localize(tz='US/Mountain', ambiguous='NaT')
print(Mountain_acc.head())
Mountain_acc.drop(Mountain_acc.loc[Mountain_acc['Start_Time'] == 'NaT'].index, inplace=True)
Mountain_acc['Start_Time'] = Mountain_acc['Start_Time'].dt.tz_convert('UTC')
print(Mountain_acc.head())


Pacific_acc = acc_tz_grouped.get_group('US/Pacific')
Pacific_acc['Start_Time'] = pd.to_datetime(Pacific_acc['Start_Time']).dt.tz_localize(tz='US/Pacific', ambiguous='NaT')
print(Pacific_acc.head())
Pacific_acc.drop(Pacific_acc.loc[Pacific_acc['Start_Time'] == 'NaT'].index, inplace=True)
Pacific_acc['Start_Time'] = Pacific_acc['Start_Time'].dt.tz_convert('UTC')
print(Pacific_acc.head())



#pd.to_datetime(acc_tz_grouped.get_group('US/Eastern')['Timezone']).dt.tz_localize('US/Eastern')
#print(acc_tz_grouped.get_group('US/Eastern').head())

#print(acc_tz_grouped_keys)

#for timezone in acc_tz_grouped_keys:
#    print(acc_tz_grouped.get_group(timezone))
    #acc_tz_grouped.get_group(timezone).column.dt.tz_localize(timezone)


#print(acc_tz_grouped.groups)

#acc_data.index = acc_data.index.tz_localize('UCT')

"""
acc_data.index = acc_data.index.tz_localize('GMT')
acc_data.index = acc_data.index.tz_convert('America/New_York')

This also works similarly for datetime columns, but you need dt after accessing the column:

acc_data['column'] = acc_data['column'].dt.tz_convert('America/New_York')

"""



acc_data['Start_Time'] = pd.to_datetime(acc_data['Start_Time'])

#grouped_acc_data = acc_data.groupby(['State, County'])


""""
weather_data = pd.read_csv("WeatherEvents_Jan2016-Dec2021.csv")


weather_data = weather_data.drop(weather_data.columns[[0,5,6,7,8,9,10,13]], axis=1)


#weather_data = weather_data.loc[weather_data['Defining Parameter'] == '']

weather_data['StartTime(UTC)'] = pd.to_datetime(weather_data['StartTime(UTC)'])
weather_data['EndTime(UTC)'] = pd.to_datetime(weather_data['EndTime(UTC)'])

weather_data.rename(columns={'StartTime(UTC)': 'Start_Time', 'EndTime(UTC)': 'End_Time'}, inplace=True)

weather_data.dropna()

"""

# Merge dataframes on "State, County", create new merged dataframe
#merged_data = acc_data.merge(weather_data, on='State, County', how='outer')

#for location in grouped_acc_data:
    
#print(merged_data.columns)

#count = 0
#grouped_acc_dfs = {}
#for item in location_list:
#    location = location_list[count]
#    
#    df = acc_data.groupby(location)
#    
#    grouped_acc_dfs[location] = df
#    
#    count = count + 1
#
#grouped_acc_dfs = {}
#
#while count < len(location_list):
#    # dynamically create key
#    key = ...
#    # calculate value
#    value = ...
#    a[key] = value 
#    k += 1



#for row in acc_data.head().itertuples():
#    print(row.Index, row.Severity, row.Start_Time)
    


#weather_data = weather_data.loc[weather_data['Defining Parameter'] == '']

