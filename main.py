import pandas as pd
from matplotlib import pyplot as plt
from geopy.distance import great_circle
import pytz


# Function to generate a simple line graph, add a grid, and save the graph to a file
def generate_graph(df, xAxis, yAxis, graph_title, graph_color):
    df_graph = df.plot(x=xAxis, y=yAxis, title=graph_title, color=graph_color)
    df_graph.grid()
    df_graph.figure.savefig(graph_title + '.pdf')
    return

# Function to convert a datetime to UTC based on the timezone
def convert_to_utc(dt, tz):
    local_tz = pytz.timezone(tz)
    local_time = local_tz.localize(dt, is_dst=None)
    utc_time = local_time.astimezone(pytz.utc)
    return utc_time

# Function to calculate the distance between two points
def calculate_distance(lat1, lon1, lat2, lon2):
    return great_circle((lat1, lon1), (lat2, lon2)).miles

# Defines a threshold distance in miles
threshold_distance = 5


# Read in accident data
acc_data = pd.read_csv("US_Accidents.csv")

# Remove NAN values from acc_data
acc_data = acc_data.dropna()


#print(acc_data.columns)

acc_data = acc_data[['ID', 'Severity', 'Start_Time', 'Start_Lat', 'Start_Lng', 'Timezone']]

# Ensures the 'Start_Time' column is in datetime format
acc_data['Start_Time'] = pd.to_datetime(acc_data['Start_Time'])

# Applies the conversion function to the Start_Time column
acc_data['Start_Time_UTC'] = acc_data.apply(lambda row: convert_to_utc(row['Start_Time'], row['Timezone']), axis=1)

# Drops the original Start_Time column and Timezone column
acc_data = acc_data.drop(['Start_Time', 'Timezone'], axis=1)




print(acc_data.head())

#read in weather data
weather_data = pd.read_csv("Weather_Events.csv")

weather_data = weather_data[['EventId', 'Type', 'Severity', 'StartTime(UTC)', 'EndTime(UTC)', 'Precipitation(in)', 'LocationLat', 'LocationLng']]

# drop NAN values from weather_data
weather_data = weather_data.dropna()

# Ensures the 'StartTime(UTC)' column is in datetime format and the 'EndTime(UTC)' column is in time format
weather_data['StartTime(UTC)'] = pd.to_datetime(weather_data['StartTime(UTC)'])
weather_data['EndTime(UTC)'] = pd.to_datetime(weather_data['EndTime(UTC)'])



print(weather_data.head())


'''

# Merge dataframes based on conditional statement
merged_df = pd.merge(df1, df2[(df1['time'] <= df2['stop_time']) & (df1['time'] >= df2['start_time'])], 
                     how='inner', left_index=True, right_index=True)

# Print merged dataframe
print(merged_df)
'''

'''
# Define a function to calculate the distance between two points
def calculate_distance(lat1, lon1, lat2, lon2):
    return great_circle((lat1, lon1), (lat2, lon2)).miles

# Define a threshold distance in miles (e.g., 5 miles)
threshold_distance = 5

# Perform a cross join on the dataframes
df1['key'] = 1
df2['key'] = 1
merged_df = pd.merge(df1, df2, on='key').drop('key', axis=1)

# Calculate the distance between each pair of points
merged_df['distance'] = merged_df.apply(lambda row: calculate_distance(row['latitude_x'], row['longitude_x'], row['latitude_y'], row['longitude_y']), axis=1)

# Filter the merged dataframe based on the threshold distance and the start time conditions
merged_df = merged_df[(merged_df['distance'] <= threshold_distance) &
                      (merged_df['start_time_x'] >= merged_df['start_time_y']) &
                      (merged_df['start_time_x'] <= merged_df['end_time'])]

# Clean up the dataframe by dropping unnecessary columns and resetting the index
merged_df = merged_df.drop(['latitude_y', 'longitude_y', 'distance', 'start_time_y', 'end_time'], axis=1).reset_index(drop=True)
merged_df.columns = ['A', 'start_time', 'latitude', 'longitude', 'B']

print(merged_df)
'''