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

combined_leaf_df_result_path = 'C:/Users/goikar/Quadtree/12-2-2024/output_29-3-2024/combined_leaf_df_result.csv'

# data_path = 'C:/Users/goikar/Quadtree/data/USA_Crime_2008_to_2009.csv'
# combined_leaf_df_path = 'C:/Users/goikar/Quadtree/12-2-2024/output_29-3-2024/combined_leaf_df.csv'
# combined_leaf_df_with_pred_col_path = 'C:/Users/goikar/Quadtree/12-2-2024/output_29-3-2024/combined_leaf_df_with_pred_col.csv'
# seen_evaluation_df_path = 'C:/Users/goikar/Quadtree/12-2-2024/output_29-3-2024/seen_evaluation_df.csv'
# unseen_evaluation_df_path = 'C:/Users/goikar/Quadtree/12-2-2024/output_29-3-2024/unseen_evaluation_df.csv'
# combined_test_df_path = 'C:/Users/goikar/Quadtree/12-2-2024/output_29-3-2024/combined_test_df.csv'

# # Step 1: Load crime data from csv file.
# data = prp.data_import(data_path)
# # print('Crime Data Information :------->',data.info())

# # Step 2: Get sample data.
# data = prp.get_sample_data(data)
# df = data[['CMPLNT_FR_DT', 'CMPLNT_FR_TM','Longitude','Latitude']].head(1000000)

# # Step 3:  Check null values.
# df = prp.null_values_check(df)
# # print('Total Number of Null Values :------->',df.isnull().sum())

# # Step 4:  Combine date and time into a single datetime column and convert date time into DateTime format.
# df = prp.combine_datetime(df)

# # # Crime Count and add new column       
# df = prp.crime_total_count(df)

# # Step 5:  Adding some new features to Dataframe and Scaling Longitute and Latitude. 
# df = prp.create_new_features(df)

# ###################################### CREATING QUADTREE, DISTRIBUTING DATA POINTS INTO LEAF NODES AND PERFORM PREDICTION ######################################
# # Step 7:  CREATE QUADTREE. THE QUADTREE OBJECT REPRESENTS THE ROOT NODE OF THE QUADTREE.
# quadtree = quad.make_quadtree(df)

# # Step 8: COLLECT A LIST OF DATASETS FROM EACH LEAF NODE.
# leaf_data_frames = quadtree.get_leaf_data_points()

# # Step 9: COUNT DAILY CRIMES IN EACH LEAF NODE.
# prp.count_daily_crime(leaf_data_frames)

# # prp.get_max_min_daily_crime_count(leaf_data_frames)

# # Step 10:  PRINT AND LABEL EACH DENSE CRIME REGION
# vis.label_and_print_dcrs_list(leaf_data_frames)

# # Step 11: Conctinated the list of dataframes and stored in dataframe.
# combined_leaf_df = pd.concat(leaf_data_frames)

# # Step 12:  Save the combined DataFrame to a CSV file
# combined_leaf_df.to_csv(combined_leaf_df_path, index=False)

# # Step 13: Collected all leaf node and perfomed prediction on each node.
# combined_leaf_df_with_pred_col, seen_evaluation_df, unseen_evaluation_df, combined_test_df = mod.leaf_nodes_predictions(combined_leaf_df)

# # Step 14: Stored prediction in csv file.
# combined_leaf_df_with_pred_col.to_csv(combined_leaf_df_with_pred_col_path, index=False)

# seen_evaluation_df.to_csv(seen_evaluation_df_path, index=False)

# unseen_evaluation_df.to_csv(unseen_evaluation_df_path, index=False)

# combined_test_df.to_csv(combined_test_df_path, index=False)

# ###################################### RESULT VISUALISATION ######################################

# ################################## Approach-I ###############################################

# # combined_leaf_df = pd.read_csv('C:/Users/goikar/Quadtree/Results/Approach-I/CSV Files/combined_leaf_df.csv')

# # combined_leaf_df_with_pred_col = pd.read_csv('C:/Users/goikar/Quadtree/Results/Approach-I/CSV Files/combined_leaf_df_with_pred_col.csv')

# # combined_test_df = pd.read_csv('C:/Users/goikar/Quadtree/Results/Approach-I/CSV Files/combined_test_df.csv')

# # seen_evaluation_df = pd.read_csv('C:/Users/goikar/Quadtree/Results/Approach-I/CSV Files/seen_evaluation_df.csv')

# # unseen_evaluation_df = pd.read_csv('C:/Users/goikar/Quadtree/Results/Approach-I/CSV Files/unseen_evaluation_df.csv')

# #################################################################################


# # # Step 16:  Visualize the prediction.
# # vis.actual_prediction_plot(combined_test_df)

# # PLOT ALL UNSEEN DATA PREDICTION.
# vis.time_series_plot_all_dcrs(combined_leaf_df_with_pred_col, combined_test_df)

# # PLOT TIME-SERIES DATA FOR EACH DCR
# vis.time_series_plot_each_dcrs(combined_leaf_df_with_pred_col, combined_test_df)

# # vis.actual_vs_prediction_plot(combined_test_df)

# # PLOT GRID BASED HEATMAP. Can Delete.
# # vis.plot_heatmap_for_dcr(combined_test_df)

# # Step **: PLOT SPATIAL HEATMAP FOR ALL DCR IN ONE.
# # PLOT SPATIAL HEATMAP FOR ALL DCRS IN ONE.
# actual_heatmap = vis.plot_spatial_heatmap(combined_test_df, 'Crime_count', 'Crime Density.')
# # predicted_heatmap = vis.plot_spatial_heatmap(combined_test_df, 'unseen_pred', 'Predicted Crime Count Heatmap')
# # Save the maps as HTML files or display them in Jupyter Notebook
# actual_heatmap.save('C:/Users/goikar/Quadtree/12-2-2024/output_img/heatmap/03-04-2024/all_dcrs_actual_heatmap.html')
# # predicted_heatmap.save('C:/Users/goikar/Quadtree/12-2-2024/output_img/heatmap/03-04-2024/all_dcrs_predicted_heatmap.html')

# # # Step **: PLOT SPATIAL HEATMAP FOR EACH DCR.
# vis.plot_spatial_heatmap_for_dcr(combined_test_df, 'Crime_count', 'unseen_pred')
# ######## OR This way############# Leave it commented ##################
# # PLOT SPATIAL HEATMAP FOR EACH DCR INCUDING ACTUAL AND PREDICTED VALUES.
# # html_file_paths = vis.plot_spatial_heatmap_for_dcr(combined_test_df, 'Crime_count')
# # Print the paths of the generated HTML files
# # for path in html_file_paths:
# #     print(f"Spatial heatmap saved as: {path}")
# # Call the function to generate spatial heatmaps for predicted values for each DCR
# # predicted_html_file_paths = vis.plot_spatial_heatmap_for_dcr(combined_test_df, 'unseen_pred')
# # Print the paths of the generated HTML files for predicted values
# # for path in predicted_html_file_paths:
# #     print(f"Spatial heatmap for predicted values saved as: {path}")
# ############################################################################

combined_leaf_df = pd.read_csv('C:/Users/goikar/Quadtree/12-2-2024/output_29-3-2024/combined_leaf_df.csv')

# CRIME DISTRIBUTION IN EACH DENSE CRIME REGION (LEAF NODE).
vis.leaf_node_points_distribution(combined_leaf_df)

# # Scatter Plot Actual Vs Predicted.
# vis.model_performance(combined_test_df)

# # Step 15:  Visualize the quadtree
# vis.visualize_quadtree(quadtree)



###################################### MODEL COMPARISON ######################################
# Read the evaluation dataframes for the three approaches
a1_test_evaluation_df = pd.read_csv('C:/Users/goikar/Quadtree/12-2-2024/output_29-3-2024/unseen_evaluation_df.csv')
a2_test_evaluation_df = pd.read_csv('C:/Users/goikar/Quadtree/12-2-2024/backup_code/2-3-2024/output_files/test_evaluation_df.csv')
a3_test_evaluation_df = pd.read_csv('C:/Users/goikar/Quadtree/12-2-2024/backup_code/4-3-2024/output_files/test_evaluation_df.csv')
pre_auth = pd.read_csv('C:/Users/goikar/Quadtree/12-2-2024/output_29-3-2024/umair_output.csv') 

combined_test_df = pd.read_csv('C:/Users/goikar/Quadtree/12-2-2024/output_29-3-2024/combined_test_df.csv')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add a new column indicating the approach for each dataframe
a1_test_evaluation_df['Approach'] = 'Approach-I'
a2_test_evaluation_df['Approach'] = 'Approach-II'
a3_test_evaluation_df['Approach'] = 'Approach-III'
pre_auth['Approach'] = 'HDBSCAN-SARIMA'

# Merge the evaluation dataframes
merged_df = pd.concat([a1_test_evaluation_df, a2_test_evaluation_df, a3_test_evaluation_df, pre_auth], ignore_index=True)

# # A single dataframe with all evaluation metrics and the approaches
# print(merged_df)

# Bar Plot. To analyse comparison of metric values accross different approaches.
vis.avg_mae_approaches(merged_df, 'MAE')
vis.avg_mae_approaches(merged_df, 'RMSE')
vis.avg_mae_approaches(merged_df, 'MAPE')
vis.avg_mae_approaches(merged_df, 'ME')



# # Step 2: Calculate the average values of MAE, RMSE, MAPE, and ME for each DCR
# average_metrics_a1 = a1_test_evaluation_df.groupby('DCR_ID').mean()

# # Step 3: Plot histograms for each evaluation metric
# plt.figure(figsize=(12, 8))

# plt.subplot(2, 2, 1)
# plt.hist(average_metrics_a1['MAE'], bins=10, color='blue', alpha=0.7)
# plt.xlabel('MAE')
# plt.ylabel('Frequency')
# plt.title('Distribution of MAE for Approach-I')

# plt.subplot(2, 2, 2)
# plt.hist(average_metrics_a1['RMSE'], bins=10, color='orange', alpha=0.7)
# plt.xlabel('RMSE')
# plt.ylabel('Frequency')
# plt.title('Distribution of RMSE for Approach-I')

# plt.subplot(2, 2, 3)
# plt.hist(average_metrics_a1['MAPE'], bins=10, color='green', alpha=0.7)
# plt.xlabel('MAPE')
# plt.ylabel('Frequency')
# plt.title('Distribution of MAPE for Approach-I')

# plt.subplot(2, 2, 4)
# plt.hist(average_metrics_a1['ME'], bins=10, color='red', alpha=0.7)
# plt.xlabel('ME')
# plt.ylabel('Frequency')
# plt.title('Distribution of ME for Approach-I')

# plt.tight_layout()
# plt.show()














