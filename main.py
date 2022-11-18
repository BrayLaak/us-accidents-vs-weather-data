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



# Read in accident data pickle
accident_data = pd.read_pickle("US_Accidents.pkl")
