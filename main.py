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
acc_data = pd.read_csv("US_Accidents_Dec21_updated.csv")

print(acc_data.columns)

acc_data = acc_data.drop(acc_data.columns[[0,1,3,4,5,6,7,8,9,10,11,12,13,16,17,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46]], axis=1)

print(acc_data.columns)
print(acc_data)

#df_data = df_data.loc[df_data['Defining Parameter'] == '']

