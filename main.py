import pandas as pd
from matplotlib import pyplot as plt


# Function to remove rows with NAN values from dataframe
def remove_NAN_rows(df):
    return df.dropna()

# Function to generate a simple line graph, add a grid, and save the graph to a file
def generate_graph(df, xAxis, yAxis, graph_title, graph_color):
    df_graph = df.plot(x=xAxis, y=yAxis, title=graph_title, color=graph_color)
    df_graph.grid()
    df_graph.figure.savefig(graph_title + '.pdf')
    return



# Read in accident data
acc_data = pd.read_csv('US_Accidents_Dec21_updated.csv')

print(acc_data.columns)

acc_data = acc_data.drop(acc_data.columns[[0,3,4,5,6,7,8,9,10,11,12,13,16,17,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46]], axis=1)

acc_data['State, County'] = acc_data['State'].astype(str) + ", " + acc_data['County'].astype(str)
acc_data = acc_data.drop(acc_data.columns[[2,3]], axis=1)

location_list = acc_data['State, County'].unique()


grouped_acc_data = acc_data.groupby(['State, County'])

for location in grouped_acc_data:
    


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



for row in acc_data.head().itertuples():
    print(row.Index, row.Severity, row.Start_Time)
    
#print(location_list)
#print(location_list.dtype)


print(acc_data.columns)

#df_data = df_data.loc[df_data['Defining Parameter'] == '']

