# us-accidents-vs-weather

## US Traffic Accidents vs. Weather Event Data


# Purpose
This project compares US traffic accidents with weather data collected from across the US. The code is written in Python and uses the pandas and matplotlib libraries.


# Explanation

The graphs generated investigate the accidents that occur per weather severity, the accidents that occur per weather type, the accidents that occur per month, day in the week, and hour in the day to see what trends come up.




In this project it can be seen that the most traffic accidents occur in rainy weather and the second most in fog:


Oddly, the most accidents occur in weather with a light severity:


The most accidents occur in the months of and :


It can also be seen that the Friday has the highest amount of traffic accidents:


And from are the times with the highest numbers of accidents:


# Features

This project will: 

Pull in data from csv files gathered from the sources cited below

Use built-in pandas functions to clean the data

Use custom functions to operate on the data including calculating distances between locations and filtering rows based on a threshold distance.

Use a custom function and data manipulation to generate five basic plots with matplotlib, which visualize the relationship between weather events and traffic accidents. The plots are outputted to PNG files in a "Graphs" subfolder.


# Instructions
This project was created in Python 3.10.4

1. Clone the repository to your local machine.

2. Install all requirements listed in "requirements.txt". If this fails, pip install matplotlib and pandas

3. Download the csv files from the sources below and extract them into the main folder of the repository. Keep the original file names.

4. Run main.py

5. The graphs should now be generated and saved in the "Graphs" subfolder



# Sources
A Kaggle dataset of US accident data 2016-2021:
https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents

A Kaggle dataset of US accident data 2016-2021:
https://www.kaggle.com/datasets/sobhanmoosavi/us-weather-events