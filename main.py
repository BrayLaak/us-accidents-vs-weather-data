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
    if pd.isna(lat1) or pd.isna(lon1) or pd.isna(lat2) or pd.isna(lon2):
        return None
    return great_circle((lat1, lon1), (lat2, lon2)).miles


# Function to perform a merge asof on the two dataframes, calculate the distance for each row, and filter the rows based on the threshold distance
def merge_by_state(acc_data, weather_data, threshold_distance):
    result_dataframes = []
    states = acc_data['State'].unique()

    for state in states:
        acc_df = acc_data[acc_data['State'] == state]
        weather_df = weather_data[weather_data['State'] == state]

        # Sorts dataframes by time for merge_asof
        acc_df = acc_df.sort_values('Start_Time_UTC')
        weather_df = weather_df.sort_values('StartTime(UTC)')

        # Performs asof merge on sorted dataframes
        merged_df = pd.merge_asof(acc_df, weather_df, left_on='Start_Time_UTC', right_on='StartTime(UTC)', direction='forward', suffixes=('_acc', '_weather'))

        # Calculates distance for each row
        merged_df['distance'] = merged_df.apply(lambda row: calculate_distance(row['Start_Lat'], row['Start_Lng'], row['LocationLat'], row['LocationLng']), axis=1)
        merged_df = merged_df[merged_df['distance'].notna()]

        # Filters rows based on distance threshold
        merged_df = merged_df[merged_df['distance'] <= threshold_distance]

        result_dataframes.append(merged_df)

    return pd.concat(result_dataframes).reset_index(drop=True)



# Defines a threshold distance in miles
threshold_distance = 5


print("Reading in accident data...")

# Reads in accident data
acc_data = pd.read_csv("US_Accidents.csv")


print('Cleaning data...')
# Removes NAN values from acc_data
acc_data = acc_data.dropna()

# Selects the columns that are needed for the analysis
acc_data = acc_data[['ID', 'Severity', 'State', 'Start_Time', 'Start_Lat', 'Start_Lng', 'Timezone']]

acc_data = acc_data.rename(columns={'Severity': 'Acc_Severity'})

# Ensures the 'Start_Time' column is in datetime format
acc_data['Start_Time'] = pd.to_datetime(acc_data['Start_Time'])

# Applies the conversion function to the Start_Time column
acc_data['Start_Time_UTC'] = acc_data.apply(lambda row: convert_to_utc(row['Start_Time'], row['Timezone']), axis=1)

# Drops the original Start_Time column and Timezone column
acc_data = acc_data.drop(['Start_Time', 'Timezone'], axis=1)

print(acc_data.head())



print("Reading in weather data...")

#read in weather data
weather_data = pd.read_csv("Weather_Events.csv")

print("Cleaning weather data...")

# Select the columns that are needed for the analysis
weather_data = weather_data[['Type', 'State', 'Severity', 'StartTime(UTC)', 'EndTime(UTC)', 'Precipitation(in)', 'LocationLat', 'LocationLng']]

# Rename the 'Severity' column to 'Weather_Severity' to avoid confusion with the 'Severity' column from the accident data
weather_data = weather_data.rename(columns={'Severity': 'Weather_Severity'})

# Rename the 'Type' column to 'Weather_Type'
weather_data = weather_data.rename(columns={'Type': 'Weather_Type'})

# drop NAN values from weather_data
weather_data = weather_data.dropna()

# Ensures the 'StartTime(UTC)' column is in datetime format and the 'EndTime(UTC)' column is in time format
weather_data['StartTime(UTC)'] = pd.to_datetime(weather_data['StartTime(UTC)'])
weather_data['EndTime(UTC)'] = pd.to_datetime(weather_data['EndTime(UTC)'])

# Ensures the 'StartTime(UTC)' and 'EndTime(UTC)' columns are set to UTC timezone
weather_data['StartTime(UTC)'] = weather_data['StartTime(UTC)'].dt.tz_localize('UTC')
weather_data['EndTime(UTC)'] = weather_data['EndTime(UTC)'].dt.tz_localize('UTC')


print(weather_data.head())

print("Grouping data by state...")

# Group the dataframes by state
acc_data_groups = acc_data.groupby('State')
weather_data_groups = weather_data.groupby('State')

# Get unique states from both dataframes
unique_states = set(acc_data['State'].unique()).union(weather_data['State'].unique())

# Initialize a list to store the dataframes for each state
result_dataframes = []


print("Performing merge of data...")

# Iterate through the unique states and perform the merge for each state
for state in unique_states:
    acc_df = acc_data[acc_data['State'] == state]
    weather_df = weather_data[weather_data['State'] == state]

    merged_df = merge_by_state(acc_df, weather_df, threshold_distance)

    # Append the merged_df to the list of dataframes
    result_dataframes.append(merged_df)

# Concatenate the list of dataframes into a single dataframe
result_df = pd.concat(result_dataframes)

# Reset the index of the final dataframe
result_df.reset_index(drop=True, inplace=True)

print(result_df.head())

'''
# Generate a graph for the number of accidents per weather type
accidents_per_weather_type = result_df.groupby('Weather_Type').size().reset_index(name='Count')
generate_graph(accidents_per_weather_type, 'Weather_Type', 'Count', 'Accidents per Weather Type', 'red')

# Generate a graph for the number of accidents per weather severity
accidents_per_weather_severity = result_df.groupby('Weather_Severity').size().reset_index(name='Count')
generate_graph(accidents_per_weather_severity, 'Weather_Severity', 'Count', 'Accidents per Weather Severity', 'blue')

# Generate a graph for the number of accidents per weather type by year
accidents_per_weather_type_by_year = result_df.groupby(['Weather_Type', result_df['Start_Time_UTC'].dt.year]).size().reset_index(name='Count')
generate_graph(accidents_per_weather_type_by_year, 'Weather_Type', 'Count', 'Accidents per Weather Type by Year', 'red')

# Generate a graph for the number of accidents per weather severity by year
accidents_per_weather_severity_by_year = result_df.groupby(['Weather_Severity', result_df['Start_Time_UTC'].dt.year]).size().reset_index(name='Count')
generate_graph(accidents_per_weather_severity_by_year, 'Weather_Severity', 'Count', 'Accidents per Weather Severity by Year', 'blue')
'''