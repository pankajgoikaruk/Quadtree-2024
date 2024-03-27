'''
Last Code Update Date: 15/03/2024
Code Updated By: Pankaj Dilip Goikar
Updated Topics:

1. Imported Data (12/02/2024)
2. Create new features and scaling coordinates. (12/02/2024)
3. Combined date and time column together. (12/02/2024)
4. Visulalised Distribution of the Crime Data Points. (12/02/2024)
5. Divide data into 80% and 20% train and test dataframe. (13/02/2024)
6. Create Quadtree file and started implementation of quadtree. (14/02/2024)
7. Worked in insert function.  (15/02/2024) - (17/02/2024)
8. Sorting the issue of data fetching from leaf node. (17/02/2024) - (18/02/2024)
9. Fetched correct leaf node data points and stored in corresponding DCRs with displaying total numbers of data points in each DCRs. (19/02/2024)
10. Update def remove_outlier(self, df):, used quantile techniqe to remove outliers. (20/02/2024)
11. Added some methods called ADF and remove seasonality from time series. (21-2-2024)
12. Added new column call DCR ID to maintain which data points belongs to which DCRs. Addtionally converted list of DCRs into Dataframe. Training and Testing dataset is splited and printed recusrsively. (22/02/2024)
13. Predicted Value added to main test dataframe (24/02/2024)
14. Fixed the issue of train split dataset and performed perfect training and testing process. (25 to 26 /02/2024)
15. Fixed train and test split of orignal dataset. Added new file and class called Make_Quadreee. (25/02/2024)
16. Split data into seen_df and unseen_df and perfomed following process on seen_df. fix the issue of min_points and max_level in make_quadree.py file. (13/03/2024)
17. Set Crime_count as target variable and we scaled its values as well. Created MinMaxScaler method in modelling.py file and perfomed scaler on Crime_count. Created inverse method in modelling.py. 
    Sorted date while fetching the data point from def get_leaf_data_points(self): in quadtree.py file. Updated if isinstance(leaf_data_frames, list): in main.py file to rid off from concat() issue. 
    reset date and time colume by converting from unix timestamp to date and time in modelling.py file, self.unix_timestamps_to_datetime(test_df) (15/03/2024)


Next Target:
1. Perform Data Preprocessing: Data stationarity check. Create count column in each DCRs.
2. lable each rectangle with their ids. 
*  If one or more datapoints are colaborating more then 50 times then change all points colors to red. So we will get very high crime areas. 
3. Sort the issue of combined_leaf_dcrs.csv, becuase here again we are getting more data points.
4. Data split process for training and testing process.
5. Perform recursively variour prediction algorithm on each DCRs.
6. Perform evaluation process.
7. Nearest Neighbor Search: Implement a nearest neighbor search algorithm to find the closest data points to each point in the leaf node. This method ensures that we capture the most relevant data points based on spatial proximity.

'''
# Import the necessary libraries 
import numpy as np
import pandas as pd
from matplotlib import path
from preprocess import Preprocess
from visualise import Visualise
from make_quadtree import Make_Quadtree
###################################### DATA PREPROCESSING ######################################     

# Created object of classes.
prp = Preprocess()
vis = Visualise()
quad = Make_Quadtree()

data_path = 'C:/Users/goikar/Quadtree/data/USA_Crime_2008_to_2009.csv'

# Step 1: Load crime data from csv file.
data = prp.data_import(data_path)
# print('Crime Data Information :------->',data.info())

# Step 2: Get sample data.
data = prp.get_sample_data(data)
df = data[['CMPLNT_FR_DT', 'CMPLNT_FR_TM','Longitude','Latitude']].head(100000)

# Step 3:  Check null values.
df = prp.null_values_check(df)
# print('Total Number of Null Values :------->',df.isnull().sum())

# Step 4:  Combine date and time into a single datetime column and convert date time into DateTime format.
df = prp.combine_datetime(df)

# # Crime Count and add new column       
df = prp.crime_total_count(df)

# Step 5:  Adding some new features to Dataframe and Scaling Longitute and Latitude. 
df = prp.create_new_features(df)

# ###################################### CREATING QUADTREE AND DISTRIBUTING DATA POINTS INTO LIST OF DATA FRAMES ######################################
# Step 7:  Create Quadtree.
quadtree = quad.make_quadtree(df)

# Step 11:  Visualize the quadtree
vis.visualize_quadtree(quadtree)












