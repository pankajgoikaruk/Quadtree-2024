from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
import pandas as pd
from evaluation import Evaluation
import numpy as np
from visualise import Visualise
import matplotlib.pyplot as plt
import seaborn as sns

vis = Visualise()

# Initialize the scaler
min_max_scaler = MinMaxScaler(feature_range=(1, 10))

XGBreg_model = XGBRegressor(base_score=0.5, booster='gbtree', n_estimators=1000,
                        early_stopping_rounds=50, objective='reg:linear', max_depth=3, learning_rate=0.01)

class Modelling:
    def __init__(self) -> None:
        # self.XGBreg_parent_model = None
        pass

    def train_val_test_df_split(self, df, train_size):
        train_per = train_size

        split_index = int(len(df) * train_per)

        seen_df = df[:split_index]
        unseen_df = df[split_index:]

        return seen_df, unseen_df
    
    # Convert datetime columns to Unix timestamps
    def datetime_to_unix_timestamps(self, data):
        data['CMPLNT_FR_DT'] = data['CMPLNT_FR_DT'].astype('int64') // 10**9 # we used Unix timestamp from nanoseconds to seconds
        data['CMPLNT_DATETIME'] = data['CMPLNT_DATETIME'].astype('int64') // 10**9
        return data['CMPLNT_FR_DT'], data['CMPLNT_DATETIME']
    
    def unix_timestamps_to_datetime(self, data):
        data['CMPLNT_FR_DT'] = pd.to_datetime(data['CMPLNT_FR_DT'], unit='s')
        data['CMPLNT_DATETIME'] = pd.to_datetime(data['CMPLNT_FR_DT'], unit='s')
        return data['CMPLNT_FR_DT'], data['CMPLNT_DATETIME']
    
    # Scale the target values
    def min_max_scale_values(self, df, col_name):
        # Reshape the Crime_count column to a 2D array
        col_counts = df[col_name].values.reshape(-1, 1)

        # Fit and transform the scaled values
        df[col_name] = min_max_scaler.fit_transform(col_counts)
        
        return df[col_name]
    
    # Scale the target values
    def inverse_min_max_scale_values(self, df, col_name):
        # Reshape the Crime_count column to a 2D array
        col_counts = df[col_name].values.reshape(-1, 1)

        # Fit and transform the scaled values
        df[col_name] = min_max_scaler.inverse_transform(col_counts)
        
        return df[col_name]
    
    def root_unseendata_features_target(self):
        # Define features and target variable
        FEATURES = ['CMPLNT_FR_DT', 'CMPLNT_DATETIME', 'Hour_of_crime', 'Scl_Longitude', 'Scl_Latitude',
                    'Dayofweek_of_crime', 'Quarter_of_crime', 'Month_of_crime', 'Dayofyear_of_crime',
                    'Dayofmonth_of_crime', 'Weekofyear_of_crime', 'Year_of_crime', 'Distance_From_Central_Point', 'Longitude_Latitude_Ratio', 'Location_density'] # 'Crime_count', 'Scl_Longitude', 'Scl_Latitude', 
        TARGET = 'Crime_count'
        return FEATURES, TARGET
    
    def parent_leaf_featurs_target(self):
        # Define features and target variable
        FEATURES = ['CMPLNT_FR_DT', 'CMPLNT_DATETIME', 'Hour_of_crime', 'Scl_Longitude', 'Scl_Latitude',
                    'Dayofweek_of_crime', 'Quarter_of_crime', 'Month_of_crime', 'Dayofyear_of_crime',
                    'Dayofmonth_of_crime', 'Weekofyear_of_crime', 'Year_of_crime', 'Distance_From_Central_Point', 'Longitude_Latitude_Ratio','Parent_pred', 'Location_density'] # 'Crime_count', 'Scl_Longitude', 'Scl_Latitude', 
        TARGET = 'Crime_count'
        return FEATURES, TARGET

    # PREDICTION ON ROOT NODE.
    def root_node_prediction(self, df):
        """
        Perform prediction at the root node using a predictive model trained on the provided data.

        Parameters:
        - data: DataFrame containing the data points within the boundary of the root node.
        - features: List of feature column names used for prediction.
        - target: Name of the target variable column used for prediction.

        Returns:
        - Predicted values for the target variable at the root node.
        """

        FEATURES, TARGET = self.root_unseendata_features_target()

        # Passing the date and time to convert into unix_timestamps required for modelling.
        df['CMPLNT_FR_DT'], df['CMPLNT_DATETIME'] = self.datetime_to_unix_timestamps(df)
        # Passing df and column name for scale the values.
        df['Crime_count'] = self.min_max_scale_values(df, col_name='Crime_count')

        # Extract features and target variable from the data
        X_train = df[FEATURES]
        y_train = df[TARGET]
        
        # Fit Root Model
        XGBreg_model.fit(X_train, y_train, eval_set=[(X_train, y_train)], verbose=100)

        # Make Root predictions
        y_pred = XGBreg_model.predict(X_train)

        df['Parent_pred'] = y_pred

        # # Inversed Unix timestamps to data time.
        # df['CMPLNT_FR_DT'], df['CMPLNT_DATETIME'] = self.unix_timestamps_to_datetime(df)
        # df['Crime_count'] = self.inverse_min_max_scale_values(df)

        # # Converted scaled crime_count and parent_pred into orignal values.
        # df['Crime_count'] = self.inverse_min_max_scale_values(df, col_name='Crime_count')
        # df['Parent_pred'] = self.inverse_min_max_scale_values(df, col_name='Parent_pred')

        return df

    # PREDICTION ON EACH PARENT NODES.
    def parent_node_prediction(self, df):
        """
        Perform prediction at the root node using a predictive model trained on the provided data.

        Parameters:
        - data: DataFrame containing the data points within the boundary of the root node.
        - features: List of feature column names used for prediction.
        - target: Name of the target variable column used for prediction.

        Returns:
        - Predicted values for the target variable at the root node.
        """
        FEATURES, TARGET = self.parent_leaf_featurs_target()

        # Note: if root node has already scaled the valued then no need to double scaled here. if root node has inverse the scaled values then you can perfom here.
        # # Passing the date and time to convert into unix_timestamps required for modelling.
        # df['CMPLNT_FR_DT'], df['CMPLNT_DATETIME'] = self.datetime_to_unix_timestamps(df)
        # # Passing df and column name for scale the values.
        # df['Crime_count'] = self.min_max_scale_values(df, col_name='Crime_count')
        # df['Parent_pred'] = self.min_max_scale_values(df, col_name='Parent_pred')

        # Extract features and target variable from the data
        X_train = df[FEATURES]
        y_train = df[TARGET]
        
        # Fit Parent Model
        XGBreg_model.fit(X_train, y_train, eval_set=[(X_train, y_train)], verbose=100)

        # Make Parent predictions
        y_pred = XGBreg_model.predict(X_train)
        
        df['Parent_pred'] = y_pred
        
        # # Inversed Unix timestamps to data time.
        # df['CMPLNT_FR_DT'], df['CMPLNT_DATETIME'] = self.unix_timestamps_to_datetime(df)
        # # df['Crime_count'] = self.inverse_min_max_scale_values(df)

        # # Converted scaled crime_count and parent_pred into orignal values.
        # df['Crime_count'] = self.inverse_min_max_scale_values(df, col_name='Crime_count')
        # df['Parent_pred'] = self.inverse_min_max_scale_values(df, col_name='Parent_pred')

        return df
    
    # PREDICTION ON EACH LEAF NODES.
    def leaf_nodes_predictions(self, combined_df):

        # Initialize an empty list to store dcr_data DataFrames
        all_dcr_data = []
        
        # Initialize an empty DataFrame to store evaluation results
        unseen_evaluation_results = []
        seen_evaluation_results = []

        # To store all unseen data prediction in one df.
        all_test_dfs = []
        
        FEATURES, TARGET = self.parent_leaf_featurs_target()

        for dcr_id in combined_df['DCR_ID'].unique():
            dcr_data = combined_df[combined_df['DCR_ID'] == dcr_id]

            # Split data into train and test sets
            train_df, test_df = self.train_val_test_df_split(dcr_data, train_size=0.8)

            if train_df.empty or test_df.empty:
                continue  # Skip this iteration if train_df or test_df is empty

            X_train = train_df[FEATURES]
            y_train = train_df[TARGET]

            X_test = test_df[FEATURES]
            y_test = test_df[TARGET]

            ############ Training and Prediction ###############
            # Seen Data Fit Leaf nodes Model
            XGBreg_model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=100)

            # Seen Leaf nodes Data Predictions
            seen_y_pred = XGBreg_model.predict(X_train)

            # Unseen Leaf nodes Data predictions
            unseen_y_pred = XGBreg_model.predict(X_test)

            ###### This line of codes will store seen and unseen prediction in new_combined_df
            # Concatenate seen_y_pred and unseen_y_pred into a single array
            seen_unseen_y_pred = np.concatenate([seen_y_pred, unseen_y_pred])
            # Attach seen_y_pred and unseen_y_pred predicted values to the actual dataset
            dcr_data['Predicted'] = seen_unseen_y_pred

            # Inversed dcr_data Unix timestamps to data time and scaled values into orginal values.
            dcr_data['CMPLNT_FR_DT'], dcr_data['CMPLNT_DATETIME'] = self.unix_timestamps_to_datetime(dcr_data)
            dcr_data['Crime_count'] = self.inverse_min_max_scale_values(dcr_data, col_name='Crime_count')
            dcr_data['Predicted'] = self.inverse_min_max_scale_values(dcr_data, col_name='Predicted')

            ############ Seen Data Evaluation and Inversed ###############
            train_df['seen_pred'] = seen_y_pred

            # Inversed Unix timestamps to data time and scaled values into orginal values.
            train_df['CMPLNT_FR_DT'], train_df['CMPLNT_DATETIME'] = self.unix_timestamps_to_datetime(train_df)
            train_df['Crime_count'] = self.inverse_min_max_scale_values(train_df, col_name='Crime_count')
            train_df['seen_pred'] = self.inverse_min_max_scale_values(train_df, col_name='seen_pred')

            seen_evaluation = Evaluation(y_train, seen_y_pred)

            mae = round(seen_evaluation.mean_absolute_error(), 3)
            rmse = round(seen_evaluation.root_mean_squared_error(), 3)
            mape = round(seen_evaluation.mean_absolute_percentage_error(), 3)
            me = round(seen_evaluation.mean_error(), 3)

            dcr_len = len(dcr_data)

            # Append the results to the DataFrame
            seen_evaluation_results.append({'DCR_ID': dcr_id, 'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'ME': me, 'DCR_Length': dcr_len})

            ############ Unseen Data Evaluation and Inversed ###############
            test_df['unseen_pred'] = unseen_y_pred

            # Append the modified test_df to the list
            all_test_dfs.append(test_df)
            
            # Inversed Unix timestamps to data time and scaled values into orginal values.
            test_df['CMPLNT_FR_DT'], test_df['CMPLNT_DATETIME'] = self.unix_timestamps_to_datetime(test_df)
            test_df['Crime_count'] = self.inverse_min_max_scale_values(test_df, col_name='Crime_count')
            test_df['unseen_pred'] = self.inverse_min_max_scale_values(test_df, col_name='unseen_pred')
            
            unseen_evaluation = Evaluation(y_test, unseen_y_pred)

            mae = round(unseen_evaluation.mean_absolute_error(), 3)
            rmse = round(unseen_evaluation.root_mean_squared_error(), 3)
            mape = round(unseen_evaluation.mean_absolute_percentage_error(), 3)
            me = round(unseen_evaluation.mean_error(), 3)

            dcr_len = len(dcr_data)

            # Append the results to the DataFrame
            unseen_evaluation_results.append({'DCR_ID': dcr_id, 'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'ME': me, 'DCR_Length': dcr_len})

            # Append dcr_data to the list
            all_dcr_data.append(dcr_data)

            ############ Visualisation ############

            # # PLOT ALL UNSEEN DATA PREDICTION.
            # vis.plot_actual_vs_predicted(dcr_id, dcr_data, test_df, train_df)           
        
        # Concat all test (unseen) dataset predicted values in single df.
        # Concatenate all modified test_df dataframes into a single dataframe
        combined_test_df = pd.concat(all_test_dfs)
        # Inversed Unix timestamps to data time and scaled values into orginal values.
        combined_test_df['CMPLNT_FR_DT'], combined_test_df['CMPLNT_DATETIME'] = self.unix_timestamps_to_datetime(combined_test_df)
        combined_test_df['Crime_count'] = self.inverse_min_max_scale_values(combined_test_df, col_name='Crime_count')
        combined_test_df['unseen_pred'] = self.inverse_min_max_scale_values(combined_test_df, col_name='unseen_pred')

        # Concatenate all_dcr_data into a single DataFrame
        new_combined_df = pd.concat(all_dcr_data)
        new_combined_df = new_combined_df.sort_values(by='CMPLNT_FR_DT')

        # Converst evaluation results dictionary into dataframe. 
        seen_evaluation_df = pd.DataFrame(seen_evaluation_results)
        unseen_evaluation_df = pd.DataFrame(unseen_evaluation_results)

        return new_combined_df, seen_evaluation_df, unseen_evaluation_df, combined_test_df
    
    
    
    
    







        
















