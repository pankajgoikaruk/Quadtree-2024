import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler

# Initialize the scaler
min_max_scaler = MinMaxScaler(feature_range=(1, 10))


class Point:
    def __init__(self, x, y, index, CMPLNT_FR_DT, CMPLNT_DATETIME, Scl_Longitude, Scl_Latitude, Hour_of_crime, Dayofweek_of_crime, Quarter_of_crime, Month_of_crime, Dayofyear_of_crime, Dayofmonth_of_crime, Weekofyear_of_crime, Year_of_crime, Distance_From_Central_Point, Crime_count, parent_pred, Longitude_Latitude_Ratio, Location_density): # , Twelve_Month_Differenced, 
        
        self.x = x # Longitude
        self.y = y # Latitude
        self.index = index
        self.CMPLNT_FR_DT = CMPLNT_FR_DT
        self.CMPLNT_DATETIME = CMPLNT_DATETIME
        self.Scl_Longitude = Scl_Longitude
        self.Scl_Latitude = Scl_Latitude
        self.Hour_of_crime = Hour_of_crime
        self.Dayofweek_of_crime = Dayofweek_of_crime
        self.Quarter_of_crime = Quarter_of_crime
        self.Month_of_crime = Month_of_crime
        self.Dayofyear_of_crime = Dayofyear_of_crime
        self.Dayofmonth_of_crime = Dayofmonth_of_crime
        self.Weekofyear_of_crime = Weekofyear_of_crime
        self.Year_of_crime = Year_of_crime
        self.Distance_From_Central_Point = Distance_From_Central_Point
        # self.Twelve_Month_Differenced = Twelve_Month_Differenced
        self.Crime_count = Crime_count
        self.parent_pred = parent_pred
        self.Longitude_Latitude_Ratio = Longitude_Latitude_Ratio
        self.Location_density = Location_density


class Rectangle:
    def __init__(self, x1, y1, x2, y2):
        """
        Initialize a rectangle with coordinates of bottom-left and top-right corners.
        
        Parameters:
        - x1, y1: Coordinates of the bottom-left corner.
        - x2, y2: Coordinates of the top-right corner.
        """
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
    
    def contains_point(self, x, y):
        """
        Check if a point (x, y) lies within the rectangle.
        
        Parameters:
        - x: The x-coordinate of the point.
        - y: The y-coordinate of the point.
        
        Returns:
        - True if the point lies within the rectangle, False otherwise.
        """
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2

    def intersects(self, other):
        """
        Check if the rectangle intersects with another rectangle.
        
        Parameters:
        - other: Another Rectangle object.
        
        Returns:
        - True if the rectangles intersect, False otherwise.
        """
        return not (self.x2 < other.x1 or self.x1 > other.x2 or self.y2 < other.y1 or self.y1 > other.y2)
    

class Quadtree:
    def __init__(self, boundary, max_points=None, max_levels=None):
        """
        Initialize a quadtree with a boundary, maximum number of points per node, and maximum depth.
        
        Parameters:
        - boundary: A Rectangle object representing the boundary of the quadtree.
        - max_points: Maximum number of points allowed in a node before it subdivides.
        - max_levels: Maximum depth of the quadtree.
        - node_id: Unique identifier for the node.
        - level: Depth level of the node in the quadtree.
        """
        self.boundary = boundary
        self.max_points = max_points if max_points is not None else 4
        self.max_levels = max_levels if max_levels is not None else 10
        self.points = []  # Points stored in the current node
        self.children = []  # Child nodes of the quadtree
        self.next_node_id = 0  # Initialize counter for node IDs
        self.level = 0  # Initialize level of the current node
        self.node_id = 0

        # Ensure that boundary is a Rectangle object
        if not isinstance(self.boundary, Rectangle):
            raise ValueError("Boundary must be a Rectangle object")

    def insert(self, point):
        # Check if the point is within the boundary of the current node
        if not self.boundary.contains_point(point.x, point.y):
            return False

        # Check if the current node is a leaf node and there is space to insert the point
        if self.is_leaf() and len(self.points) < self.max_points:
            self.points.append(point)
            return True

        # If the current node is not a leaf node or it's full, subdivide it
        if not self.children:
            self.subdivide()
        
        # Attempt to insert the point into the child nodes
        for child in self.children:
            if child.insert(point):
                return True

        # If the point cannot be inserted into any child nodes, insert it into the current node
        self.points.append(point)
        return True

    # Subdivide method will create 4 children for current node and this current node will become parent node.
    def subdivide(self):

        # Calculate the dimensions of each child node
        x_mid = (self.boundary.x1 + self.boundary.x2) / 2
        y_mid = (self.boundary.y1 + self.boundary.y2) / 2

        # Create child nodes representing each quadrant
        # NW Quadrant
        nw_boundary = Rectangle(self.boundary.x1, y_mid, x_mid, self.boundary.y2)
        nw_quadtree = Quadtree(nw_boundary, self.max_points, self.max_levels)
        nw_quadtree.node_id = self.next_node_id * 4 + 1  # Assign unique ID
        nw_quadtree.level = self.level + 1  # Increment level
        self.next_node_id += 1  # Increment next_node_id
        self.children.append(nw_quadtree)

        # NE Quadrant
        ne_boundary = Rectangle(x_mid, y_mid, self.boundary.x2, self.boundary.y2)
        ne_quadtree = Quadtree(ne_boundary, self.max_points, self.max_levels)
        ne_quadtree.node_id = self.next_node_id * 4 + 2  # Assign unique ID
        ne_quadtree.level = self.level + 1  # Increment level
        self.next_node_id += 1  # Increment next_node_id
        self.children.append(ne_quadtree)

        # SW Quadrant
        sw_boundary = Rectangle(self.boundary.x1, self.boundary.y1, x_mid, y_mid)
        sw_quadtree = Quadtree(sw_boundary, self.max_points, self.max_levels)
        sw_quadtree.node_id = self.next_node_id * 4 + 3  # Assign unique ID
        sw_quadtree.level = self.level + 1  # Increment level
        self.next_node_id += 1  # Increment next_node_id
        self.children.append(sw_quadtree)

        # SE Quadrant
        se_boundary = Rectangle(x_mid, self.boundary.y1, self.boundary.x2, y_mid)
        se_quadtree = Quadtree(se_boundary, self.max_points, self.max_levels)
        se_quadtree.node_id = self.next_node_id * 4 + 4  # Assign unique ID
        se_quadtree.level = self.level + 1  # Increment level
        self.next_node_id += 1  # Increment next_node_id
        self.children.append(se_quadtree)

        df = self.get_current_node_data(self.points)

        # # Perfome Prediction on each parent level.
        # predictions = self.root_node_prediction(df, col_name='Crime_Count')

        # # Add a new column 'Predicted_Crime_Count' to the DataFrame and assign predicted values to it
        # df['Predicted_Crime_Count'] = predictions

        print(df)

        # # Update the node ID and level for child nodes
        # for child in self.children:
        #     child.node_id = self.node_id * 4 + self.children.index(child) + 1
        #     child.level = self.level + 1

        # This points contails all data points of current parent node which will be recursively distribute to its children. This for loop will distribute spatial data to belongingings child node according to spatio location (longitude and latitude). 
        for point in self.points:
            for child in self.children:
                if child.boundary.contains_point(point.x, point.y):
                    child.insert(point)
                    break       

        # self.points = []
    
    def print_tree(self):
        """
        Print information about each node in the quadtree using depth-first traversal.
        """
        self._print_node(self)

    def _print_node(self, node):
        """
        Helper function to print information about a node and its children recursively.
        """
        print(f"Node ID: {id(node)}, Level: {node.level}, Data Points: {len(node.points)}")

        # Recursively print information about children
        for child in node.children:
            self._print_node(child)

    def is_leaf(self):
        """
        Check if the current node is a leaf node (i.e., it has no children).
        """
        return len(self.children) == 0 
    
    def root_node_prediction(self, df, col_name):
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
                    'Dayofmonth_of_crime', 'Weekofyear_of_crime', 'Year_of_crime', 'Distance_From_Central_Point', 'Longitude_Latitude_Ratio','parent_pred', 'Location_density'] # 'Crime_count', 'Scl_Longitude', 'Scl_Latitude', 
        TARGET = 'Crime_count'

        df['CMPLNT_FR_DT'], df['CMPLNT_DATETIME'] = self.datetime_to_unix_timestamps(df)
        df[col_name] = self.min_max_scale_values(df)

        # Extract features and target variable from the data
        X_train = df[FEATURES]
        y_train = df[TARGET]

        # Initialize XGBoost Model
        XGBreg_model = XGBRegressor(base_score=0.5, booster='gbtree', n_estimators=1000,
                        early_stopping_rounds=50, objective='reg:linear', max_depth=3, learning_rate=0.01)
        
        # Fit Model
        XGBreg_model.fit(X_train, y_train, eval_set=[(X_train, y_train)], verbose=100)

        # Make predictions
        y_pred = XGBreg_model.predict(X_train)

        return y_pred
    
    # Convert datetime columns to Unix timestamps
    def datetime_to_unix_timestamps(self, data):
        data['CMPLNT_FR_DT'] = data['CMPLNT_FR_DT'].astype('int64') // 10**9 # we used Unix timestamp from nanoseconds to seconds
        data['CMPLNT_DATETIME'] = data['CMPLNT_DATETIME'].astype('int64') // 10**9

        return data['CMPLNT_FR_DT'], data['CMPLNT_DATETIME']

    # Scale the target values
    def min_max_scale_values(self, df):
        # Reshape the Crime_count column to a 2D array
        crime_counts = df['Crime_count'].values.reshape(-1, 1)

        # Fit and transform the scaled values
        df['Crime_count'] = min_max_scaler.fit_transform(crime_counts)
        
        return df['Crime_count']

    def get_current_node_data(self, points):
        data = {'index': [], 'CMPLNT_FR_DT': [], 'CMPLNT_DATETIME': [], 'Longitude': [], 'Latitude': [], 'Scl_Longitude': [], 'Scl_Latitude': [], 'Hour_of_crime': [], 'Dayofweek_of_crime': [], 'Quarter_of_crime': [], 'Month_of_crime': [], 'Dayofyear_of_crime': [], 'Dayofmonth_of_crime': [], 'Weekofyear_of_crime': [], 'Year_of_crime': [], 'Distance_From_Central_Point': [], 'Crime_count': [], 'parent_pred': [], 'Longitude_Latitude_Ratio': [], 'Location_density': []} # , 'Twelve_Month_Differenced': []

        # Extract data points from leaf node
        for point in self.points:
            data['index'].append(point.index)
            data['CMPLNT_FR_DT'].append(point.CMPLNT_FR_DT)
            data['CMPLNT_DATETIME'].append(point.CMPLNT_DATETIME)
            data['Longitude'].append(point.x)
            data['Latitude'].append(point.y)
            data['Scl_Longitude'].append(point.Scl_Longitude)
            data['Scl_Latitude'].append(point.Scl_Latitude)
            data['Hour_of_crime'].append(point.Hour_of_crime)
            data['Dayofweek_of_crime'].append(point.Dayofweek_of_crime)
            data['Quarter_of_crime'].append(point.Quarter_of_crime)
            data['Month_of_crime'].append(point.Month_of_crime)
            data['Dayofyear_of_crime'].append(point.Dayofyear_of_crime)
            data['Dayofmonth_of_crime'].append(point.Dayofmonth_of_crime)
            data['Weekofyear_of_crime'].append(point.Weekofyear_of_crime)
            data['Year_of_crime'].append(point.Year_of_crime)
            data['Distance_From_Central_Point'].append(point.Distance_From_Central_Point)
            # data['Twelve_Month_Differenced'].append(point.Twelve_Month_Differenced)
            data['Crime_count'].append(point.Crime_count)
            data['parent_pred'].append(point.parent_pred)
            data['Longitude_Latitude_Ratio'].append(point.Longitude_Latitude_Ratio)
            data['Location_density'].append(point.Location_density)       

        # Create DataFrame
        df = pd.DataFrame(data)
        return df

