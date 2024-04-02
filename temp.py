def calculate_node_position(node_id, node_level, parent_id=None):
    # Calculate parent ID if not provided
    if parent_id is None:
        parent_id = (node_id - 1) // 4
    
    # Calculate position within level
    position_within_level = node_id - parent_id * 4
    
    return {
        "node_id": node_id,
        "level": node_level,
        "parent_id": parent_id,
        "position_within_level": position_within_level
    }

# Example usage
node_info = calculate_node_position(10, 2)
print(node_info)



#################################################

def unseen_df_prediction(self, df):
        """
        Perform prediction at the Unseen dataset using a trained predictive model.

        Parameters:
        - data: DataFrame containing the data points within the boundary of the root node.
        - features: List of feature column names used for prediction.
        - target: Name of the target variable column used for prediction.

        Returns:
        - Predicted values for the target variable at the unseen dataset.
        """
        evaluation_results = []
        # Getting features for unseen dataset.
        FEATURES, TARGET = self.root_unseendata_features_target()

        # Passing the date and time to convert into unix_timestamps required for modelling.
        df['CMPLNT_FR_DT'], df['CMPLNT_DATETIME'] = self.datetime_to_unix_timestamps(df)
        # Passing df and column name for scale the values.
        df['Crime_count'] = self.min_max_scale_values(df, col_name='Crime_count')

        # Extract features and target variable from the data
        X_train = df[FEATURES]
        y_test = df[TARGET]

        # Make Root predictions
        y_pred = XGBreg_model.predict(X_train)

        df['unseen_df_pred'] = y_pred

        unseen_df_evaluation = Evaluation(df['Crime_count'], df['unseen_df_pred'])

        mae = round(unseen_df_evaluation.mean_absolute_error(), 3)
        rmse = round(unseen_df_evaluation.root_mean_squared_error(), 3)
        mape = round(unseen_df_evaluation.mean_absolute_percentage_error(), 3)
        me = round(unseen_df_evaluation.mean_error(), 3)

        dcr_len = len(df)

        # Append the results to the DataFrame
        evaluation_results.append({'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'ME': me, 'DCR_Length': dcr_len})
        
        unseen_df_evaluation_df = pd.DataFrame(evaluation_results)

        # # Inversed Unix timestamps to data time.
        # df['CMPLNT_FR_DT'], df['CMPLNT_DATETIME'] = self.unix_timestamps_to_datetime(df)
        # df['Crime_count'] = self.inverse_min_max_scale_values(df)

        # # Converted scaled crime_count and parent_pred into orignal values.
        # df['Crime_count'] = self.inverse_min_max_scale_values(df, col_name='Crime_count')
        # df['Parent_pred'] = self.inverse_min_max_scale_values(df, col_name='Parent_pred')

        return df, unseen_df_evaluation_df



# Inverse target and predicted values into orignal number.
    def min_max_inverse_scale_values(self, values):
        # Convert pandas Series to numpy array
        values_array = values.to_numpy() if isinstance(values, pd.Series) else values

        # Reshape the array if it's 1-dimensional
        values_reshaped = values_array.reshape(-1, 1) if len(values_array.shape) == 1 else values_array

        # Inverse scale the values using the MinMaxScaler's inverse_transform method
        inverse_scaled_values = min_max_scaler.inverse_transform(values_reshaped)

        return inverse_scaled_values
