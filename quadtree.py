import pandas as pd
import numpy as np
from xgboost import XGBRegressor

class Point:
    def __init__(self, x, y, index, CMPLNT_FR_DT, CMPLNT_DATETIME, Scl_Longitude, Scl_Latitude, Hour_of_crime, Dayofweek_of_crime, Quarter_of_crime, Month_of_crime, Dayofyear_of_crime, Dayofmonth_of_crime, Weekofyear_of_crime, Year_of_crime, Distance_From_Central_Point, Crime_count, Longitude_Latitude_Ratio, Location_density): # , Twelve_Month_Differenced, 
        
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
        """
        self.boundary = boundary
        self.max_points = max_points if max_points is not None else 4
        self.max_levels = max_levels if max_levels is not None else 10
        self.points = []  # Points stored in the current node
        self.children = []  # Child nodes of the quadtree
        self.level = 0  # Level of the current node

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

    def subdivide(self):
        # Calculate the dimensions of each child node
        x_mid = (self.boundary.x1 + self.boundary.x2) / 2
        y_mid = (self.boundary.y1 + self.boundary.y2) / 2

        # Create child nodes representing each quadrant
        # NW Quadrant
        nw_boundary = Rectangle(self.boundary.x1, y_mid, x_mid, self.boundary.y2)
        nw_quadtree = Quadtree(nw_boundary, self.max_points, self.max_levels)
        self.children.append(nw_quadtree)

        # NE Quadrant
        ne_boundary = Rectangle(x_mid, y_mid, self.boundary.x2, self.boundary.y2)
        ne_quadtree = Quadtree(ne_boundary, self.max_points, self.max_levels)
        self.children.append(ne_quadtree)

        # SW Quadrant
        sw_boundary = Rectangle(self.boundary.x1, self.boundary.y1, x_mid, y_mid)
        sw_quadtree = Quadtree(sw_boundary, self.max_points, self.max_levels)
        self.children.append(sw_quadtree)

        # SE Quadrant
        se_boundary = Rectangle(x_mid, self.boundary.y1, self.boundary.x2, y_mid)
        se_quadtree = Quadtree(se_boundary, self.max_points, self.max_levels)
        self.children.append(se_quadtree)

        # Extract all features and target variable for points in the current node
        data = np.array([
            [point.CMPLNT_FR_DT, point.CMPLNT_DATETIME, point.Hour_of_crime, point.Scl_Longitude, point.Scl_Latitude,
            point.Dayofweek_of_crime, point.Quarter_of_crime, point.Month_of_crime, point.Dayofyear_of_crime,
            point.Dayofmonth_of_crime, point.Weekofyear_of_crime, point.Year_of_crime, point.Distance_From_Central_Point,
            point.Longitude_Latitude_Ratio, point.Location_density, point.Crime_count]
            for point in self.points
        ])

        print(f"Length of Data is: {len(data)}")

        predictions = self.predict_at_root(data)

        print(predictions)

        # # Perform prediction using the entire dataset of the current node
        # predictions = self.predictive_model.predict(data)


        # This points contails all data points of current parent node which will be recursively distribute to its children. This for loop will distribute spatial data to belongingings child node according to spatio location (longitude and latitude). 
        for point in self.points:
            for child in self.children:
                if child.boundary.contains_point(point.x, point.y):
                    child.insert(point)
                    break

        # self.points = []
        self.level += 1
    
    def is_leaf(self):
        """
        Check if the current node is a leaf node (i.e., it has no children).
        """
        return len(self.children) == 0 
    
    def predict_at_root(self, data):
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


        # Convert data to a DataFrame
        data_df = pd.DataFrame(data, columns=FEATURES + [TARGET])


        # Extract features and target variable from the data
        X = data_df[FEATURES]
        y = data_df[TARGET]

        print(X)
        print(y)

        # # Initialize XGBoost Model
        # XGBreg_model = XGBRegressor(base_score=0.5, booster='gbtree', n_estimators=1000,
        #                 early_stopping_rounds=50, objective='reg:linear', max_depth=3, learning_rate=0.01)
        
        # # Fit Model
        # XGBreg_model.fit(train_df_X_train, train_df_y_train, eval_set=[(train_df_X_train, train_df_y_train), (train_df_X_test, train_df_y_test)], verbose=100)

        # # Make predictions
        # train_df_y_pred = XGBreg_model.predict(train_df_X_test)




        # Train a predictive model using the data
        # Example: You can use any machine learning model for training
        # model = YourModel()
        # model.fit(X, y)

        # # For demonstration purposes, let's assume a simple mean prediction
        # prediction = np.mean(y)

        prediction = 0 # For demo purpose.

        return prediction
