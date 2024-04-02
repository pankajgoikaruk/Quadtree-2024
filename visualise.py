import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
color_pal = sns.color_palette()
import folium
from folium.plugins import HeatMap
import os

class Visualise:
    def __init__(self) -> None:
        pass
    
    # Visualise the quadtree.
    def visualize_quadtree(self, quadtree):
        """
        Visualize the quadtree by plotting its boundaries and data points.

        Parameters:
        - quadtree: The Quadtree object to visualize.
        """
        fig, ax = plt.subplots()

        # Recursively plot each rectangle in the quadtree
        def plot_node(node):
            if node is None:
                return
            ax.add_patch(plt.Rectangle((node.boundary.x1, node.boundary.y1),
                                    node.boundary.x2 - node.boundary.x1,
                                    node.boundary.y2 - node.boundary.y1,
                                    fill=False, edgecolor='black'))

            # Color data points and leaf nodes based on density
            if node.points:
                # Calculate density percentage
                num_points = len(node.points)
                density = num_points / quadtree.max_points * 100

                # Assign colors based on density thresholds
                if density > 80:
                    point_color = 'darkred'
                    # node_color = 'lightcoral'
                elif density > 60:
                    point_color = 'darkorange'
                    # node_color = 'orange'
                elif density > 40:
                    point_color = 'orange'
                    # node_color = 'gold'
                elif density > 30:
                    point_color = 'lightgreen'
                    # node_color = 'limegreen'
                else:
                    point_color = 'darkgreen'
                    # node_color = 'green'

                # Plot data points and leaf nodes with assigned colors
                x = [point.x for point in node.points]
                y = [point.y for point in node.points]
                ax.scatter(x, y, color=point_color, s=5)

            for child in node.children:
                plot_node(child)

        # Start plotting from the root node
        plot_node(quadtree)

        # Set plot limits and labels
        ax.set_xlim(quadtree.boundary.x1, quadtree.boundary.x2)
        ax.set_ylim(quadtree.boundary.y1, quadtree.boundary.y2)
        ax.set_aspect('equal', 'box')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title('Quadtree Visualization')
        plt.show()
    
    # Print or process the list of DataFrames in leaf_data_frame
    def label_and_print_dcrs_list(self, leaf_data_frames):
        for i, df in enumerate(leaf_data_frames):
            num_points = len(df)
            print(f"DCR {i + 1}:")
            print(f"DCR {i + 1}: has {num_points} data points")
            df['DCR_ID'] = f"DCR {i + 1}"
            # print(df)
            print()
    
    # PLOT ALL UNSEEN DATA PREDICTION. 
    def plot_actual_vs_predicted(self, dcr_id, dcr_data, test_df, train_df):
            
            # Plot Crime_count
            plt.figure(figsize=(10, 8))
            plt.plot(dcr_data['CMPLNT_FR_DT'], dcr_data['Crime_count'], label='Crime_count')

            # Plot Seen_pred
            plt.plot(train_df['CMPLNT_FR_DT'], train_df['seen_pred'], label='Seen_Data_Prediction')
            
            # Plot Unseen_pred
            plt.plot(test_df['CMPLNT_FR_DT'], test_df['unseen_pred'], label='Unseen_Data_Prediction')
            
            # Set plot title and labels
            plt.title(f'DCR_ID: {dcr_id}')
            plt.xlabel('CMPLNT_FR_DT')
            plt.ylabel('Count')
            plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
            plt.grid(True)
            plt.legend()
            
            # Save the plot
            plt.savefig(f'C:/Users/goikar/Quadtree/12-2-2024/output_img/DCR_ID_{dcr_id}_plot.png')
            plt.close()
    
    # PLOT GRID BASED HEATMAP.
    def plot_heatmap_for_dcr(self, combined_test_df):
        for dcr_id in combined_test_df['DCR_ID'].unique():
            # Filter combined dataframe for the specific DCR ID
            dcr_data = combined_test_df[combined_test_df['DCR_ID'] == dcr_id]

            # Pivot the dataframe for heatmap plotting
            pivot_actual = dcr_data.reset_index().pivot_table(index='CMPLNT_FR_DT', columns='DCR_ID', values='Crime_count')
            pivot_predicted = dcr_data.reset_index().pivot_table(index='CMPLNT_FR_DT', columns='DCR_ID', values='unseen_pred')

            # Plot heatmap for actual crime count
            plt.figure(figsize=(12, 6))
            plt.title(f'Actual Crime Count Heatmap for DCR_ID: {dcr_id}')
            sns.heatmap(pivot_actual, cmap='YlGnBu')
            plt.xlabel('Day of Crime')
            plt.ylabel('Date')
            plt.tight_layout()
            plt.savefig(f'C:/Users/goikar/Quadtree/12-2-2024/output_img/heatmap/02-04-2024/Actual_Crime_Count_Heatmap_DCR_ID_{dcr_id}.png')
            plt.close()

            # Plot heatmap for predicted crime count
            plt.figure(figsize=(12, 6))
            plt.title(f'Predicted Crime Count Heatmap for DCR_ID: {dcr_id}')
            sns.heatmap(pivot_predicted, cmap='YlGnBu')
            plt.xlabel('Day of Crime')
            plt.ylabel('Date')
            plt.tight_layout()
            plt.savefig(f'C:/Users/goikar/Quadtree/12-2-2024/output_img/heatmap/02-04-2024/Predicted_Crime_Count_Heatmap_DCR_ID_{dcr_id}.png')
            plt.close()
    
    # PLOT SPATIAL HEATMAP FOR ALL DCRS IN ONE.
    def plot_spatial_heatmap(self, combined_test_df, value_column, title):
        # Create a Folium map centered around the geographic region of interest
        m = folium.Map(location=[combined_test_df['Latitude'].mean(), combined_test_df['Longitude'].mean()], zoom_start=10)

        # Add heatmap layer for the specified value column
        heatmap_data = combined_test_df[['Latitude', 'Longitude', value_column]].groupby(['Latitude', 'Longitude']).sum().reset_index().values.tolist()
        HeatMap(heatmap_data, radius=15).add_to(m)

        # Add title to the map
        folium.map.Marker(
            [combined_test_df['Latitude'].mean(), combined_test_df['Longitude'].mean()],
            icon=folium.DivIcon(html=f"<div style='font-size: 18pt'>{title}</div>")
        ).add_to(m)

        return m
    
    # # PLOT SPATIAL HEATMAP FOR EACH DCR INCUDING ACTUAL AND PREDICTED VALUES.
    # def plot_spatial_heatmap_for_dcr(self, combined_test_df, column_name):
    #     html_file_paths = []
    #     for dcr_id in combined_test_df['DCR_ID'].unique():
    #         # Filter combined dataframe for the specific DCR ID
    #         dcr_data = combined_test_df[combined_test_df['DCR_ID'] == dcr_id]
        
    #         # Create a Folium map centered at the mean latitude and longitude of the DCR data
    #         m = folium.Map(location=[dcr_data['Latitude'].mean(), dcr_data['Longitude'].mean()], zoom_start=10)
            
    #         # Create a HeatMap layer using the latitude and longitude coordinates and the specified column
    #         heat_data = [[row['Latitude'], row['Longitude'], row[column_name]] for index, row in dcr_data.iterrows()]
    #         HeatMap(heat_data).add_to(m)
            
    #         # Add title to the map
    #         folium.TileLayer('cartodbpositron').add_to(m)  # Add a base map
    #         folium.LayerControl().add_to(m)  # Add layer control
    #         m.add_child(folium.LatLngPopup())  # Add clickable latitude/longitude popup

    #         # Determine if the heatmap is for actual or predicted values
    #         heatmap_type = 'Actual' if column_name == 'Crime_count' else 'Predicted'
            
    #         # Save the map as an HTML file
    #         html_file_path = f"C:/Users/goikar/Quadtree/12-2-2024/output_img/heatmap/01-04-2024/spatial_heatmap_dcr_{dcr_id}_{heatmap_type}.html"
    #         m.save(html_file_path)
            
    #         html_file_paths.append(html_file_path)
        
    #     return html_file_paths


    def generate_heatmap_dashboard(self, html_file_paths, predicted_html_file_paths, combined_test_df):
        # Create a new HTML file to combine the actual and predicted heatmaps
        with open('C:/Users/goikar/Quadtree/12-2-2024/output_img/heatmap/03-04-2024/crime_heatmap_dashboard.html', 'w') as f:
            # Write the header of the HTML file
            f.write('<!DOCTYPE html>\n<html>\n<head>\n<title>Crime Heatmap Dashboard</title>\n</head>\n<body>\n')
            
            # Loop through each HTML file path and corresponding predicted HTML file path
            for html_path, predicted_html_path in zip(html_file_paths, predicted_html_file_paths):
                # Extract the DCR ID from the HTML file path
                dcr_id = html_path.split('_')[-2]
                
                # Write the iframe tag for the actual heatmap
                f.write(f'<h2>DCR ID: {dcr_id}</h2>\n')
                f.write(f'<iframe src="{html_path}" width="1400" height="700"></iframe>\n')
                
                # Write the iframe tag for the predicted heatmap
                f.write(f'<h2>Predicted DCR ID: {dcr_id}</h2>\n')
                f.write(f'<iframe src="{predicted_html_path}" width="1400" height="700"></iframe>\n')

            # Write the closing tags
            f.write('</body>\n</html>')

    def plot_spatial_heatmap_for_dcr(self, combined_test_df, actual_column_name, predicted_column_name):
        html_file_paths = []
        predicted_html_file_paths = []
        for dcr_id in combined_test_df['DCR_ID'].unique():
            # Filter combined dataframe for the specific DCR ID
            dcr_data = combined_test_df[combined_test_df['DCR_ID'] == dcr_id]
        
            # Create a Folium map centered at the mean latitude and longitude of the DCR data
            m = folium.Map(location=[dcr_data['Latitude'].mean(), dcr_data['Longitude'].mean()], zoom_start=10)
            m.add_child(folium.LatLngPopup())  # Add clickable latitude/longitude popup
            
            # Create a HeatMap layer using the latitude and longitude coordinates and the specified column
            actual_heat_data = [[row['Latitude'], row['Longitude'], row[actual_column_name]] for index, row in dcr_data.iterrows()]
            HeatMap(actual_heat_data).add_to(m)

            # Add clickable latitude/longitude popup to show coordinates and crime count on click
            for _, row in dcr_data.iterrows():
                popup_content = f"Latitude: {row['Latitude']}, Longitude: {row['Longitude']}" # , Crime_count: {row['Crime_count']:.0f}
                folium.Marker([row['Latitude'], row['Longitude']], popup=popup_content).add_to(m)
            
            # Save the actual heatmap map as an HTML file
            actual_html_file_path = f"C:/Users/goikar/Quadtree/12-2-2024/output_img/heatmap/03-04-2024/spatial_heatmap_dcr_{dcr_id}_actual.html"
            m.save(actual_html_file_path)

            
            # Create a HeatMap layer for predicted values
            predicted_heat_data = [[row['Latitude'], row['Longitude'], row[predicted_column_name]] for index, row in dcr_data.iterrows()]
            HeatMap(predicted_heat_data).add_to(m)

            # Add clickable latitude/longitude popup to show coordinates and crime count on click
            for _, row in dcr_data.iterrows():
                popup_content = f"Latitude: {row['Latitude']}, Longitude: {row['Longitude']}" # , Predicted_count: {row['unseen_pred']:.0f}
                folium.Marker([row['Latitude'], row['Longitude']], popup=popup_content).add_to(m)

            # Add clickable latitude/longitude popup to show coordinates on click
            m.add_child(folium.LatLngPopup())
            
            # Save the predicted heatmap map as an HTML file
            predicted_html_file_path = f"C:/Users/goikar/Quadtree/12-2-2024/output_img/heatmap/03-04-2024/spatial_heatmap_dcr_{dcr_id}_predicted.html"
            m.save(predicted_html_file_path)
            
            # Append paths to lists
            html_file_paths.append(actual_html_file_path)
            predicted_html_file_paths.append(predicted_html_file_path)
        
        # Generate the heatmap dashboard
        self.generate_heatmap_dashboard(html_file_paths, predicted_html_file_paths, combined_test_df)


    
    
    


