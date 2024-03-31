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
8 Quadtree is ready and prediction on parent nodes are working fine. (27-3-2024)
9. Prediction on root node and distrubution in child node is done. Now we are ready to experiment performance on model. Root ID is still not working well need to fix. Level allocation is working fine. (28-3-2024)
10. Divided data into seen and unseen data. Perform inverse Crime_count and Parent_pred column at parent prediction method. (29-3-2024)
11. Now, root, parents and leaf nodes prediction are working fine. Mostly changes has done in modelling.py file. (30-3-2024)




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
from modelling import Modelling
###################################### DATA PREPROCESSING ######################################     

# Created object of classes.
prp = Preprocess()
vis = Visualise() 
quad = Make_Quadtree()
mod = Modelling()

data_path = 'C:/Users/goikar/Quadtree/data/USA_Crime_2008_to_2009.csv'
combined_df_path = 'C:/Users/goikar/Quadtree/12-2-2024/output_29-3-2024/combined_leaf_data_frames.csv'
new_combined_df_path = 'C:/Users/goikar/Quadtree/12-2-2024/output_29-3-2024/new_combined_df_path.csv'
seen_df_evaluation_df_path = 'C:/Users/goikar/Quadtree/12-2-2024/output_29-3-2024/seen_df_evaluation_df.csv'
unseen_df_evaluation_df_path = 'C:/Users/goikar/Quadtree/12-2-2024/output_29-3-2024/unseen_df_evaluation_df.csv'

# Step 1: Load crime data from csv file.
data = prp.data_import(data_path)
# print('Crime Data Information :------->',data.info())

# Step 2: Get sample data.
data = prp.get_sample_data(data)
df = data[['CMPLNT_FR_DT', 'CMPLNT_FR_TM','Longitude','Latitude']].head(1000000)

# Step 3:  Check null values.
df = prp.null_values_check(df)
# print('Total Number of Null Values :------->',df.isnull().sum())

# Step 4:  Combine date and time into a single datetime column and convert date time into DateTime format.
df = prp.combine_datetime(df)

# # Crime Count and add new column       
df = prp.crime_total_count(df)

# Step 5:  Adding some new features to Dataframe and Scaling Longitute and Latitude. 
df = prp.create_new_features(df)

# Step 6: Splitted main data frame into seen and unseen data frame for training and testing purpose.
seen_df, unseen_df = mod.train_val_test_df_split(df, train_size=0.9)

# print(seen_df)

###################################### CREATING QUADTREE AND DISTRIBUTING DATA POINTS INTO LIST OF DATA FRAMES ######################################
# Step 7:  Create Quadtree. The Quadtree object represents the root node of the quadtree
quadtree = quad.make_quadtree(seen_df)

# Step 8: Call the get_leaf_data_points() method to retrieve the list of DataFrames
leaf_data_frames = quadtree.get_leaf_data_points()

# Step 9: Count daily crimes
prp.count_daily_crime(leaf_data_frames)

# prp.get_max_min_daily_crime_count(leaf_data_frames)

# Step 10:  Print Train_df Data
vis.label_and_print_dcrs_list(leaf_data_frames)

# Step 11: Conctinated the list of dataframes and stored in dataframe.
combined_df = pd.concat(leaf_data_frames)

# Step 12:  Save the combined DataFrame to a CSV file
combined_df.to_csv(combined_df_path, index=False)

# Step 13: Collected all leaf node and perfomed prediction on each node.
evaluation_df, new_combined_df = mod.leaf_nodes_predictions(combined_df)

# Step 14: Stored prediction in csv file.
new_combined_df.to_csv(new_combined_df_path, index=False)


evaluation_df.to_csv(seen_df_evaluation_df_path, index=False)

# Step **:  Visualize the quadtree
vis.visualize_quadtree(quadtree)

###################################### MODELLING, TRAINING and TESTING ######################################

















