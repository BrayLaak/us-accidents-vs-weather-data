import pandas as pd
from matplotlib import pyplot as plt
from geopy.distance import great_circle
import pytz
import os
import calendar


# Function to generate a configurable graph and save it to a file in a "Graphs" folder
import matplotlib.pyplot as plt
import os

def generate_graph(df, xAxis, yAxis, graph_title, graph_color, graph_type='line', xlabel=None, ylabel=None, grid_style='-', grid_alpha=0.5, file_format='pdf', xticks=None, xtick_rotation=0, xtick_labels=None):
    fig, ax = plt.subplots()
    
    # Plot the data based on the graph_type
    if graph_type == 'line':
        ax.plot(df[xAxis], df[yAxis], color=graph_color)
    elif graph_type == 'bar':
        ax.bar(df[xAxis], df[yAxis], color=graph_color)
    elif graph_type == 'scatter':
        ax.scatter(df[xAxis], df[yAxis], color=graph_color)
    else:
        raise ValueError(f"Unsupported graph_type '{graph_type}'. Supported types are 'line', 'bar', and 'scatter'.")
    
    # Set the title and axis labels
    ax.set_title(graph_title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    
    # Customize the grid
    ax.grid(linestyle=grid_style, alpha=grid_alpha)
    
    # Set custom xticks if provided
    if xticks is not None and len(xticks) > 0:
        ax.set_xticks(xticks)

    # Set the xtick labels if provided
    if xtick_labels is not None and len(xtick_labels) > 0:
        ax.set_xticklabels(xtick_labels)

    # Set the xtick rotation
    if xtick_rotation:
        plt.xticks(rotation=xtick_rotation)

    # Create the "Graphs" folder if it doesn't exist
    graphs_folder = 'Graphs'
    if not os.path.exists(graphs_folder):
        os.makedirs(graphs_folder)

    # Save the graph to a file in the "Graphs" folder
    fig.savefig(os.path.join(graphs_folder, graph_title + '.' + file_format), bbox_inches='tight')
    
    # Show the graph
    plt.show()

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
acc_data = pd.read_csv("US_Accidents_Dec21_Updated.csv")

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


print("Reading in weather data...")

#read in weather data
weather_data = pd.read_csv("WeatherEvents_Jan2016-Dec2021.csv")

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

# Remove nan values from the dataframe
result_df = result_df.dropna()

# create a number of years variable to use for calculating the average number of accidents per year
num_years = result_df['Start_Time_UTC'].dt.year.nunique()


print("Generating graphs...")

# Generate a graph for the number of accidents per weather type
# Group the data by weather type and count the number of accidents for each weather type
accidents_by_weather = result_df.groupby('Weather_Type')['ID'].count().reset_index()

# Sort the data by the number of accidents in descending order
# divide by the number of years to get the average number of accidents per year
accidents_by_weather_sorted = accidents_by_weather.sort_values('ID', ascending=False)
accidents_by_weather_sorted['ID'] = accidents_by_weather_sorted['ID'] / num_years

# Generate a bar graph for the number of accidents per Weather_Type
generate_graph(
    df=accidents_by_weather_sorted,
    xAxis='Weather_Type',
    yAxis='ID',
    graph_title='Avg_Accidents_by_Weather_Type_2016_2021',
    graph_color='blue',
    graph_type='bar',
    xlabel='Weather Type',
    ylabel='Number of Accidents',
    file_format='png'
)


# Generate a graph for the number of accidents per weather severity
# Group the data by Weather_Severity and count the number of accidents for each Weather_Severity
accidents_by_severity = result_df.groupby('Weather_Severity')['ID'].count().reset_index()

# Sort the data by the number of accidents in descending order
# divide by the number of years to get the average number of accidents per year
accidents_by_severity_sorted = accidents_by_severity.sort_values('ID', ascending=False)
accidents_by_severity_sorted['ID'] = accidents_by_severity_sorted['ID'] / num_years

# Generate a bar graph for the number of accidents per Weather_Severity
generate_graph(
    df=accidents_by_severity_sorted,
    xAxis='Weather_Severity',
    yAxis='ID',
    graph_title='Avg_Accidents_by_Weather_Severity_2016_2021',
    graph_color='green',
    graph_type='bar',
    xlabel='Weather Severity',
    ylabel='Number of Accidents',
    file_format='png'
)


# Generate a graph for the number of accidents per month
# Extract the month from the 'Start_Time_UTC' column and create a new column 'Month'
result_df['Month'] = result_df['Start_Time_UTC'].dt.month

# Group the result_df DataFrame by the 'Month' column, count the number of accidents for each month, 
# reset the index, and divide by the number of years to get the average number of accidents per year
accidents_by_month = result_df.groupby('Month')['ID'].count().reset_index()
accidents_by_month['ID'] = accidents_by_month['ID'] / num_years

# Create a DataFrame for all 12 months
all_months = pd.DataFrame({'Month': range(1, 13), 'Month_Name': [calendar.month_name[m] for m in range(1, 13)]})

# Convert the 'Month' column in the 'accidents_by_month' DataFrame to type int64
accidents_by_month['Month'] = accidents_by_month['Month'].astype(int)

# Merge the all_months DataFrame with the accidents_by_month DataFrame to ensure all months are included
accidents_by_month_complete = pd.concat([all_months.set_index('Month'), accidents_by_month.set_index('Month')], axis=1, sort=True).reset_index()

# Generate a graph for the number of accidents per month
# Calculate number of accidents per month
result_df['Month'] = result_df['Start_Time_UTC'].dt.month_name()
accidents_by_month = result_df.groupby('Month')['ID'].count().reset_index()

# Create a month order mapping
month_order = {month: i for i, month in enumerate(calendar.month_name[1:])}

# Sort accidents_by_month DataFrame using the month_order mapping
accidents_by_month['Month_Order'] = accidents_by_month['Month'].map(month_order)
accidents_by_month = accidents_by_month.sort_values('Month_Order').drop('Month_Order', axis=1)

# Generate graph for number of accidents per month
generate_graph(
    df=accidents_by_month,
    xAxis='Month',
    yAxis='ID',
    graph_title='Avg_Accidents_per_Month_2016_2021',
    graph_color='green',
    xlabel='Month',
    ylabel='Number of Accidents',
    xtick_rotation=45,
    file_format='png'
)


# Generate a graph for the number of accidents per day of the week
# Extract the day of the week from the 'Start_Time_UTC' column and create a new column 'Day_of_Week'
result_df['Day_of_Week'] = result_df['Start_Time_UTC'].dt.dayofweek

# Group the result_df DataFrame by the 'Day_of_Week' column, count the number of accidents for each day of the week, 
# reset the index, and divide by the number of years to get the average number of accidents per year
accidents_by_day_of_week = result_df.groupby('Day_of_Week')['ID'].count().reset_index()
accidents_by_day_of_week['ID'] = accidents_by_day_of_week['ID'] / num_years

# Create a DataFrame for all 7 days of the week
all_days = pd.DataFrame({'Day_of_Week': range(0, 7), 'Day_Name': [calendar.day_name[d] for d in range(0, 7)]})

# Merge the all_days DataFrame with the accidents_by_day_of_week DataFrame to ensure all days are included
accidents_by_day_of_week_complete = all_days.merge(accidents_by_day_of_week, on='Day_of_Week', how='left').fillna(0)

# Generate a graph for the number of accidents per day of the week
generate_graph(
    df=accidents_by_day_of_week_complete,
    xAxis='Day_Name',
    yAxis='ID',
    graph_title='Avg_Accidents_per_Day_of_the_Week_2016_2021',
    graph_color='blue',
    xlabel='Day of the Week',
    ylabel='Number of Accidents',
    xtick_rotation=45,
    file_format='png'
)


# Generate a graph for the number of accidents per hour of the day
# Extract the hour of the day from the 'Start_Time_UTC' column and create a new column 'Hour_of_Day'
result_df['Hour_of_Day'] = result_df['Start_Time_UTC'].dt.hour

# Group the result_df DataFrame by the 'Hour_of_Day' column, count the number of accidents (using the 'ID' column), 
# reset the index to create a new DataFrame 'accidents_by_hour_of_day', and divide by the number of years to get the average number of accidents per year
accidents_by_hour_of_day = result_df.groupby('Hour_of_Day')['ID'].count().reset_index()
accidents_by_hour_of_day['ID'] = accidents_by_hour_of_day['ID'] / num_years

# Format the hours in AM and PM
hour_labels = []
for hour in range(0, 24):
    formatted_hour = pd.Timestamp(year=2000, month=1, day=1, hour=hour).strftime('%I %p')
    hour_labels.append(formatted_hour)
    
# Call the generate_graph function to generate the graph
generate_graph(
    df=accidents_by_hour_of_day,
    xAxis='Hour_of_Day',
    yAxis='ID',
    graph_title='Avg_Accidents_per_Hour_of_the_Day_2016_2021',
    graph_color='red',
    xlabel='Hour of the Day',
    ylabel='Number of Accidents',
    xticks=list(range(0, 24)),
    xtick_labels=hour_labels,
    xtick_rotation=80,
    file_format='png'
)
