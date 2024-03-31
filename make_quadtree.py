
from quadtree import Point, Rectangle, Quadtree
from modelling import Modelling

mod = Modelling()

class Make_Quadtree:
    def __init__(self) -> None:
        pass

    def make_quadtree(self, df):
        # Calculate a suitable minimum value for max_points based on the number of records
        min_points = len(df) // 50  # Taking one-tenth of data frame.
        max_levels = min_points // 5
        if max_levels < 5:
            max_levels = 10

        # Prompt the user for the maximum number of points per node
        while True:
            try:
                max_points = int(input(f"Enter the maximum number of points per node (minimum recommended: {min_points}): "))
                if max_points < min_points:
                    print(f"Please enter a value greater than or equal to {min_points}.")
                    continue
                break
            except ValueError:
                print("Please enter a positive integer value for the maximum number of points per node.")

        # Prompt the user for the maximum number of levels in the quadtree
        while True:
            try:
                max_levels = int(input(f"Enter the maximum number of levels in the quadtree (minimum recommended: {max_levels}): "))
                if max_levels <= 1:
                    raise ValueError
                break
            except ValueError:
                print("Please enter a positive integer or more than 0 value for the maximum number of levels in the quadtree.")

        # Create a boundary rectangle for the quadtree
        boundary_rectangle = Rectangle(min(df['Longitude']), min(df['Latitude']),
                                        max(df['Longitude']), max(df['Latitude']))

        # Initialize the quadtree with the boundary rectangle
        quadtree = Quadtree(boundary_rectangle, max_points, max_levels)

        # Root prediciton. Receiving predicted value and store in dataframe. 
        df = mod.root_node_prediction(df)

        # print(df)

        # Extract data points from Longitude and Latitude columns and insert them into the quadtree
        for index, row in df.iterrows():
            longitude = row['Longitude']
            latitude = row['Latitude']
            index = row['index']
            Scl_Longitude = row['Scl_Longitude']
            CMPLNT_FR_DT = row['CMPLNT_FR_DT']
            CMPLNT_DATETIME = row['CMPLNT_DATETIME']
            Scl_Longitude = row['Scl_Longitude']
            Scl_Latitude = row['Scl_Latitude']
            Hour_of_crime = row['Hour_of_crime']
            Dayofweek_of_crime = row['Dayofweek_of_crime']
            Quarter_of_crime = row['Quarter_of_crime']
            Month_of_crime = row['Month_of_crime']
            Dayofyear_of_crime = row['Dayofyear_of_crime']
            Dayofmonth_of_crime = row['Dayofmonth_of_crime']
            Weekofyear_of_crime = row['Weekofyear_of_crime']
            Year_of_crime = row['Year_of_crime']
            Distance_From_Central_Point = row['Distance_From_Central_Point']
            # Twelve_Month_Differenced = row['Twelve_Month_Differenced']
            Crime_count = row['Crime_count']
            Parent_pred = row['Parent_pred']
            Longitude_Latitude_Ratio = row['Longitude_Latitude_Ratio']
            Location_density = row['Location_density']


            point = Point(longitude, latitude, index, CMPLNT_FR_DT, CMPLNT_DATETIME, Scl_Longitude, Scl_Latitude, Hour_of_crime, Dayofweek_of_crime, Quarter_of_crime, Month_of_crime, Dayofyear_of_crime, Dayofmonth_of_crime, Weekofyear_of_crime, Year_of_crime, Distance_From_Central_Point, Crime_count, Parent_pred, Longitude_Latitude_Ratio, Location_density) # , Twelve_Month_Differenced, Crime_count
            quadtree.insert(point) 
        
        return quadtree