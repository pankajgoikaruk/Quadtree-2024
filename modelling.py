from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
import pandas as pd
from evaluation import Evaluation
import numpy as np

# Initialize the scaler
min_max_scaler = MinMaxScaler(feature_range=(1, 10))

class Modelling:
    def __init__(self) -> None:
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

    # # Inverse target and predicted values into orignal number.
    # def min_max_inverse_scale_values(self, values):
    #     # Convert pandas Series to numpy array
    #     values_array = values.to_numpy() if isinstance(values, pd.Series) else values

    #     # Reshape the array if it's 1-dimensional
    #     values_reshaped = values_array.reshape(-1, 1) if len(values_array.shape) == 1 else values_array

    #     # Inverse scale the values using the MinMaxScaler's inverse_transform method
    #     inverse_scaled_values = min_max_scaler.inverse_transform(values_reshaped)

    #     return inverse_scaled_values
    
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

        # Define features and target variable
        FEATURES = ['CMPLNT_FR_DT', 'CMPLNT_DATETIME', 'Hour_of_crime', 'Scl_Longitude', 'Scl_Latitude',
                    'Dayofweek_of_crime', 'Quarter_of_crime', 'Month_of_crime', 'Dayofyear_of_crime',
                    'Dayofmonth_of_crime', 'Weekofyear_of_crime', 'Year_of_crime', 'Distance_From_Central_Point', 'Longitude_Latitude_Ratio', 'Location_density'] # 'Crime_count', 'Scl_Longitude', 'Scl_Latitude', 
        TARGET = 'Crime_count'

        # Passing the date and time to convert into unix_timestamps required for modelling.
        df['CMPLNT_FR_DT'], df['CMPLNT_DATETIME'] = self.datetime_to_unix_timestamps(df)
        # Passing df and column name for scale the values.
        df['Crime_count'] = self.min_max_scale_values(df, col_name='Crime_count')

        # Extract features and target variable from the data
        X_train = df[FEATURES]
        y_train = df[TARGET]

        # Initialize XGBoost Model
        XGBreg_root_model = XGBRegressor(base_score=0.5, booster='gbtree', n_estimators=1000,
                        early_stopping_rounds=50, objective='reg:linear', max_depth=3, learning_rate=0.01)
        
        # Fit Model
        XGBreg_root_model.fit(X_train, y_train, eval_set=[(X_train, y_train)], verbose=100)

        # Make predictions
        y_pred = XGBreg_root_model.predict(X_train)

        df['Parent_pred'] = y_pred

        # Inversed Unix timestamps to data time.
        df['CMPLNT_FR_DT'], df['CMPLNT_DATETIME'] = self.unix_timestamps_to_datetime(df)
        # df['Crime_count'] = self.inverse_min_max_scale_values(df)

        # Converted scaled crime_count and parent_pred into orignal values.
        df['Crime_count'] = self.inverse_min_max_scale_values(df, col_name='Crime_count')
        df['Parent_pred'] = self.inverse_min_max_scale_values(df, col_name='Parent_pred')

        return df

    
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

        # Define features and target variable
        FEATURES = ['CMPLNT_FR_DT', 'CMPLNT_DATETIME', 'Hour_of_crime', 'Scl_Longitude', 'Scl_Latitude',
                    'Dayofweek_of_crime', 'Quarter_of_crime', 'Month_of_crime', 'Dayofyear_of_crime',
                    'Dayofmonth_of_crime', 'Weekofyear_of_crime', 'Year_of_crime', 'Distance_From_Central_Point', 'Longitude_Latitude_Ratio','Parent_pred', 'Location_density'] # 'Crime_count', 'Scl_Longitude', 'Scl_Latitude', 
        TARGET = 'Crime_count'

        # Passing the date and time to convert into unix_timestamps required for modelling.
        df['CMPLNT_FR_DT'], df['CMPLNT_DATETIME'] = self.datetime_to_unix_timestamps(df)
        # Passing df and column name for scale the values.
        df['Crime_count'] = self.min_max_scale_values(df, col_name='Crime_count')
        df['Parent_pred'] = self.min_max_scale_values(df, col_name='Parent_pred')

        # Extract features and target variable from the data
        X_train = df[FEATURES]
        y_train = df[TARGET]

        # Initialize XGBoost Model
        XGBreg_parent_model = XGBRegressor(base_score=0.5, booster='gbtree', n_estimators=1000,
                        early_stopping_rounds=50, objective='reg:linear', max_depth=3, learning_rate=0.01)
        
        # Fit Model
        XGBreg_parent_model.fit(X_train, y_train, eval_set=[(X_train, y_train)], verbose=100)

        # Make predictions
        y_pred = XGBreg_parent_model.predict(X_train)
        
        df['Parent_pred'] = y_pred
        
        # Inversed Unix timestamps to data time.
        df['CMPLNT_FR_DT'], df['CMPLNT_DATETIME'] = self.unix_timestamps_to_datetime(df)
        # df['Crime_count'] = self.inverse_min_max_scale_values(df)

        # Converted scaled crime_count and parent_pred into orignal values.
        df['Crime_count'] = self.inverse_min_max_scale_values(df, col_name='Crime_count')
        df['Parent_pred'] = self.inverse_min_max_scale_values(df, col_name='Parent_pred')

        return df
    




    
    # # Train and Test the model.
    # def modelling_prediction(self, combined_df, test_size=0.2):

    #     trained_models = {}  # Initialize dictionary to store trained models
    #     # Initialize an empty DataFrame to store evaluation results
    #     train_evaluation_results = []
    #     test_evaluation_results = []

    #     # Define features and target variable
    #     FEATURES = ['CMPLNT_FR_DT', 'CMPLNT_DATETIME', 'Hour_of_crime',
    #                 'Dayofweek_of_crime', 'Quarter_of_crime', 'Month_of_crime', 'Dayofyear_of_crime',
    #                 'Dayofmonth_of_crime', 'Weekofyear_of_crime', 'Year_of_crime', 'Distance_From_Central_Point', 'Longitude_Latitude_Ratio', 'Location_density'] # 'Crime_count', 'Scl_Longitude', 'Scl_Latitude', 
    #     TARGET = 'Crime_count'

    #     combined_df['CMPLNT_FR_DT'], combined_df['CMPLNT_DATETIME'] = self.datetime_to_unix_timestamps(combined_df)
    #     combined_df['Crime_count'] = self.min_max_scale_values(combined_df)

    #     for dcr_id in combined_df['DCR_ID'].unique():
    #         dcr_data = combined_df[combined_df['DCR_ID'] == dcr_id]

    #         ################ TRAIN DATA FRAME START ################

    #         # Split data into train and test sets
    #         train_df, test_df = train_test_split(dcr_data, test_size=test_size, shuffle=False)

    #         # Split data into train and test sets
    #         train, test = train_test_split(train_df, test_size=test_size, shuffle=False)

    #         # train_test_data[dcr_id] = {'train_X_df': train, 'test_y_df': test}

    #         # Extract features and target variable for training
    #         train_df_X_train = train[FEATURES]
    #         train_df_y_train = train[TARGET]

    #         # Extract features and target variable for testing
    #         train_df_X_test = test[FEATURES]
    #         train_df_y_test = test[TARGET]

    #         # Initialize XGBoost Model
    #         XGBreg_model = XGBRegressor(base_score=0.5, booster='gbtree', n_estimators=1000,
    #                         early_stopping_rounds=50, objective='reg:linear', max_depth=3, learning_rate=0.01)
            
    #         # Fit Model
    #         XGBreg_model.fit(train_df_X_train, train_df_y_train, eval_set=[(train_df_X_train, train_df_y_train), (train_df_X_test, train_df_y_test)], verbose=100)

    #         # Make predictions
    #         train_df_y_pred = XGBreg_model.predict(train_df_X_test)

    #         # Store trained model in dictionary
    #         trained_models[dcr_id] = XGBreg_model

    #         train_df_actual_values = train_df_y_test
    #         train_df_predicted_values = train_df_y_pred

    #         train_evaluation = Evaluation(train_df_actual_values, train_df_predicted_values)

    #         mae = round(train_evaluation.mean_absolute_error(), 3)
    #         rmse = round(train_evaluation.root_mean_squared_error(), 3)
    #         mape = round(train_evaluation.mean_absolute_percentage_error(), 3)
    #         me = round(train_evaluation.mean_error(), 3)

    #         dcr_len = len(dcr_data)

    #         # Append the results to the DataFrame
    #         train_evaluation_results.append({'DCR_ID': dcr_id, 'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'ME': me, 'DCR_Length': dcr_len})

    #         ################ TRAIN DATA FRAME END ################
    #         ################ TEST DATA FRAME START ################

    #         # Extract features and target variable for testing
    #         test_df_X_test = test_df[FEATURES]
    #         test_df_y_test = test_df[TARGET]

    #         # X_val, X_hold, y_val, y_hold = train_test_split(X_test, y_test, test_size=0.5)

    #         # Make predictions
    #         test_df_y_pred = XGBreg_model.predict(test_df_X_test)

    #         # # Store trained model in dictionary
    #         # trained_models[dcr_id] = XGBreg_model

    #         test_df_actual_values = test_df_y_test
    #         test_df_predicted_values = test_df_y_pred

    #         test_evaluation = Evaluation(test_df_actual_values, test_df_predicted_values)

    #         mae = round(test_evaluation.mean_absolute_error(), 3)
    #         rmse = round(test_evaluation.root_mean_squared_error(), 3)
    #         mape = round(test_evaluation.mean_absolute_percentage_error(), 3)
    #         me = round(test_evaluation.mean_error(), 3)

    #         dcr_len = len(test_df_actual_values)

    #         # Append the results to the DataFrame
    #         test_evaluation_results.append({'DCR_ID': dcr_id, 'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'ME': me, 'DCR_Length': dcr_len})
        
    #     # Converted Date and Time back to Datatime format.
    #     combined_df['CMPLNT_FR_DT'], combined_df['CMPLNT_DATETIME'] = self.unix_timestamps_to_datetime(combined_df)
    #     train_df['CMPLNT_FR_DT'], train_df['CMPLNT_DATETIME'] = self.unix_timestamps_to_datetime(train_df)

    #     train['CMPLNT_FR_DT'], train['CMPLNT_DATETIME'] = self.unix_timestamps_to_datetime(train)
    #     test['CMPLNT_FR_DT'], test['CMPLNT_DATETIME'] = self.unix_timestamps_to_datetime(test)



    #         ################ TEST DATA FRAME END ################

    #     # Convert evaluation results to DataFrame
    #     train_evaluation_df = pd.DataFrame(train_evaluation_results)
    #     test_evaluation_df = pd.DataFrame(test_evaluation_results)

    #     # Inverse scaling of actual target values
    #     test_df_actual_values_inverse = self.min_max_inverse_scale_values(test_df_actual_values)

    #     # Inverse scaling of predicted values
    #     test_df_predicted_values_inverse = self.min_max_inverse_scale_values(test_df_predicted_values)

    #     # Concatinate Predicted value into test_df data frame.
    #     test_df['Crime_count'] = test_df_actual_values_inverse
    #     test_df['Predicted_Crime_count'] = np.round(test_df_predicted_values_inverse, 2)
        
    #     # Converted Date and Time back to Datatime format.
    #     test_df['CMPLNT_FR_DT'], test_df['CMPLNT_DATETIME'] = self.unix_timestamps_to_datetime(test_df)

    #     # Print evaluation results
    #     print(train_evaluation_df)
    #     print(test_evaluation_df)        
        
    #     return train_evaluation_df, test_evaluation_df, trained_models, test_df_actual_values_inverse, test_df_predicted_values_inverse, test_df, train, test
    
#######################################################################################################
    
    
    # def perform_high_level_prediction(self, root_points):
        
    #     # Define features and target variable
    #     FEATURES = ['CMPLNT_FR_DT', 'CMPLNT_DATETIME', 'Hour_of_crime',
    #                 'Dayofweek_of_crime', 'Quarter_of_crime', 'Month_of_crime', 'Dayofyear_of_crime',
    #                 'Dayofmonth_of_crime', 'Weekofyear_of_crime', 'Year_of_crime', 'Distance_From_Central_Point', 'Longitude_Latitude_Ratio', 'Location_density']
    #     TARGET = 'Crime_count'

    #     # Create DataFrame from root points
    #     root_df_data = {
    #         feature: [] for feature in FEATURES
    #     }
    #     root_df_data[TARGET] = []

    #     for point in root_points:
    #         for feature in FEATURES:
    #             root_df_data[feature].append(getattr(point, feature))
    #         root_df_data[TARGET].append(point.Crime_count)  # Assuming Crime_count is the target attribute

    #     root_df = pd.DataFrame(root_df_data)

    #     root_df['CMPLNT_FR_DT'], root_df['CMPLNT_DATETIME'] = self.datetime_to_unix_timestamps(root_df)
    #     root_df['Crime_count'] = self.min_max_scale_values(root_df)

    #     # Extract features and target variable for training
    #     root_X_train = root_df[FEATURES]
    #     root_y_train = root_df[TARGET]

    #     # Initialize XGBoost Model
    #     XGBreg_model = XGBRegressor(base_score=0.5, booster='gbtree', n_estimators=1000,
    #                                 early_stopping_rounds=50, objective='reg:linear', max_depth=3, learning_rate=0.01)

    #     # Fit Model
    #     XGBreg_model.fit(root_X_train, root_y_train, eval_set=[(root_X_train, root_y_train)], verbose=100)

    #     # Make predictions
    #     root_predicted_values = XGBreg_model.predict(root_X_train)

    #     return root_predicted_values










        
















