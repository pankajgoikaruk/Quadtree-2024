import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
color_pal = sns.color_palette()
import folium
from folium.plugins import HeatMap
import os

# Set the font scale and style
sns.set_context("paper", font_scale=2)
sns.set_style("whitegrid")

# Path variables initialisations
crime_heatmap_dashboard = 'C:/Users/goikar/Quadtree/12-2-2024/output_img/heatmap/04-04-2024/crime_heatmap_dashboard.html' # Heatmap Dashboard
data_point_distribution = 'C:/Users/goikar/Quadtree/12-2-2024/output_img/points_distribution/Distribution of Crime Data Points in Each DCR.png'
model_performance = 'C:/Users/goikar/Quadtree/12-2-2024/output_img/model_perfromance/model_performance.png'
all_time_series_pred = 'C:/Users/goikar/Quadtree/12-2-2024/output_img/time_series_plot/4-4-2024/time_series_for_all_dcrs.png' # All DCRs Time Series Prediction.
each_dcr_time_series_pred = "C:/Users/goikar/Quadtree/12-2-2024/output_img/time_series_plot/4-4-2024"
approach_compare = "C:/Users/goikar/Quadtree/12-2-2024/output_img/model_perfromance"

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
                density = num_points / quadtree.max_points * 100 # Calculating density compare to maximum point capacity set for each leaf node. 

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
    
    # PRINT AND LABEL EACH DENSE CRIME REGION
    def label_and_print_dcrs_list(self, leaf_data_frames):
        for i, df in enumerate(leaf_data_frames):
            num_points = len(df)
            print(f"DCR {i + 1}:")
            print(f"DCR {i + 1}: has {num_points} data points")
            df['DCR_ID'] = f"DCR {i + 1}"
            # print(df)
            print()
    
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

    # COMBINED ACTUAL AND PREDICTED DCRs IN A SINGLE HEATMAP DASHBOARD.
    def generate_heatmap_dashboard(self, html_file_paths, predicted_html_file_paths, combined_test_df):
        # Create a new HTML file to combine the actual and predicted heatmaps
        with open(crime_heatmap_dashboard, 'w') as f:
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

    # PLOT SPATIAL HEATMAP FOR EACH DCR.
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
                popup_content = f"Latitude: {row['Latitude']}, Longitude: {row['Longitude']}, Crime_count: {row['Crime_count']:.0f}" # , Crime_count: {row['Crime_count']:.0f}
                folium.Marker([row['Latitude'], row['Longitude']], popup=popup_content).add_to(m)
            
            # Save the actual heatmap map as an HTML file
            actual_html_file_path = f"C:/Users/goikar/Quadtree/12-2-2024/output_img/heatmap/04-04-2024/spatial_heatmap_dcr_{dcr_id}_actual.html"
            m.save(actual_html_file_path)

            
            # Create a HeatMap layer for predicted values
            predicted_heat_data = [[row['Latitude'], row['Longitude'], row[predicted_column_name]] for index, row in dcr_data.iterrows()]
            HeatMap(predicted_heat_data).add_to(m)

            # Add clickable latitude/longitude popup to show coordinates and crime count on click
            for _, row in dcr_data.iterrows():
                popup_content = f"Latitude: {row['Latitude']}, Longitude: {row['Longitude']}, Predicted_count: {row['unseen_pred']:.0f}" # , Predicted_count: {row['unseen_pred']:.0f}
                folium.Marker([row['Latitude'], row['Longitude']], popup=popup_content).add_to(m)

            # Add clickable latitude/longitude popup to show coordinates on click
            m.add_child(folium.LatLngPopup())
            
            # Save the predicted heatmap map as an HTML file
            predicted_html_file_path = f"C:/Users/goikar/Quadtree/12-2-2024/output_img/heatmap/04-04-2024/spatial_heatmap_dcr_{dcr_id}_predicted.html"
            m.save(predicted_html_file_path)
            
            # Append paths to lists
            html_file_paths.append(actual_html_file_path)
            predicted_html_file_paths.append(predicted_html_file_path)
        
        # Generate the heatmap dashboard
        self.generate_heatmap_dashboard(html_file_paths, predicted_html_file_paths, combined_test_df)

    # # CRIME DISTRIBUTION IN EACH DENSE CRIME REGION (LEAF NODE).
    # def leaf_node_points_distribution(self, combined_leaf_df):
    #     # Step 1: Group by 'DCR_ID' and count the rows in each group
    #     dcr_counts = combined_leaf_df['DCR_ID'].value_counts()

    #     # Step 2: Create a pie chart
    #     plt.figure(figsize=(8, 8))
    #     plt.pie(dcr_counts, labels=dcr_counts.index, autopct='%1.1f%%', startangle=140, pctdistance=0.85)
    #     plt.title('Crime distribution in each Dense Crime Region.')
    #     plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    #     # Save the pie chart as an image
    #     plt.savefig(data_point_distribution, bbox_inches='tight')
    #     plt.close() 

    # CRIME DISTRIBUTION IN EACH DENSE CRIME REGION (LEAF NODE).
    def leaf_node_points_distribution(self, combined_leaf_df):
        # Step 1: Group by 'DCR_ID' and count the rows in each group
        dcr_counts = combined_leaf_df['DCR_ID'].value_counts()

        plt.figure(figsize=(15, 15))

        # Plot the pie chart
        patches, texts, autotexts = plt.pie(dcr_counts, labels=dcr_counts.index, autopct=lambda pct: f'{pct:.1f}%',
                                            startangle=140, pctdistance=0.85)

        # Bold DCR_ID labels and percentage values
        for text in texts:
            text.set_fontweight('bold')
        for autotext in autotexts:
            autotext.set_fontweight('bold')

        # Set title with a gap
        plt.title('\nCrime distribution in each Dense Crime Region.\n', pad=20, fontweight='bold')
        
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        
        # Save the pie chart as an image
        plt.savefig(data_point_distribution, bbox_inches='tight')
        plt.close()
    

    def model_performance(self, combined_test_df):
        plt.figure(figsize=(8, 8))
        
        # Plotting actual vs predicted values
        plt.scatter(combined_test_df['Crime_count'], combined_test_df['unseen_pred'], color='orange', label='Predicted Crime_count')
        
        # Plot the line y = x for reference
        plt.plot(combined_test_df['Crime_count'], combined_test_df['Crime_count'], color='blue', label='Actual Crime_count')
        
        # Adding labels and legend
        plt.xlabel('Actual Crime_count')
        plt.ylabel('Predicted Crime_count')
        plt.title('Model Performance on Actual vs Predicted Crime Count')
        plt.legend()
        
        # Save the plot
        plt.savefig(model_performance, bbox_inches='tight')
        plt.close()

    # PLOT ALL DCR TIME SERIES PREDICTION IN ONE PLOT. 
    def time_series_plot_all_dcrs(self, combined_leaf_df, combined_test_df):
        combined_leaf_df = combined_leaf_df.set_index('CMPLNT_FR_DT')
        combined_test_df = combined_test_df.set_index('CMPLNT_FR_DT')
        combined_test_df_sorted = combined_test_df.sort_values(by='CMPLNT_FR_DT')
            
        # Plot Crime_count
        plt.figure(figsize=(15, 10))
        plt.plot(combined_leaf_df.index, combined_leaf_df['Crime_count'], label='Actual Crime_count for all DCRs')
        # Plot Unseen_pred
        plt.plot(combined_test_df_sorted.index, combined_test_df_sorted['unseen_pred'], label='Predicted Crime_count for all DCRs')
        
        # Set plot title and labels
        plt.title('Crime Count Over Time for All DCRs.')
        plt.xlabel('Crimes from 2008-2009')
        plt.ylabel('Number of Crimes Per Day')
        plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
        plt.grid(True)
        plt.legend()
        
        # Save the plot
        plt.savefig(all_time_series_pred)
        plt.close()

    # import os
    # # Define the directory to save the plots
    # output_dir = "C:/Users/goikar/Quadtree/12-2-2024/output_img/time_series_plot/4-4-2024"
    # # Create the output directory if it doesn't exist
    # os.makedirs(output_dir, exist_ok=True)

    # PLOT TIME-SERIES DATA FOR EACH DCR
    def time_series_plot_each_dcrs(self, combined_leaf_df, combined_test_df):
        combined_leaf_df = combined_leaf_df.set_index('CMPLNT_FR_DT')
        combined_test_df = combined_test_df.set_index('CMPLNT_FR_DT')
        combined_test_df_sorted = combined_test_df.sort_values(by='CMPLNT_FR_DT')
        
        for dcr_id in combined_test_df_sorted['DCR_ID'].unique():
            # Filter combined dataframe for the specific DCR ID
            dcr_data = combined_test_df_sorted[combined_test_df_sorted['DCR_ID'] == dcr_id]
                        
            # Plot Crime_count and unseen_pred for the DCR
            plt.figure(figsize=(15, 10))
            plt.plot(combined_leaf_df.index, combined_leaf_df['Crime_count'], label='Actual Crime_count for each DCR')
            plt.plot(dcr_data.index, dcr_data['unseen_pred'], label='Predicted Crime_count for each DCR')
            
            # Set plot title and labels
            plt.title(f'Crime Count Over Time for {dcr_id}')
            plt.xlabel('Date')
            plt.ylabel('Number of Crimes Per Day')
            plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
            plt.grid(True)
            plt.legend()
            
            # Save the plot
            output_file_path = os.path.join(each_dcr_time_series_pred, f'each_dcr_time_series_pred{dcr_id}.png')
            plt.savefig(output_file_path)
            plt.close()
    

    ####################### Model Comparison #######################
    
    # CALCULATE THE AVERAGE MAE FOR EACH APPROACH
    def avg_mae_approaches(self, merged_df, col):
        average_mae = merged_df.groupby('Approach')[col].mean()
        std_deviation = merged_df.groupby('Approach')[col].std()

        # Plotting
        plt.figure(figsize=(15, 10))

        # Define bar width
        bar_width = 0.2

        # Plot the average MAE values
        bars = sns.barplot(x=average_mae.index, y=average_mae, ci='sd', palette="husl", capsize=0.1, errwidth=2, errcolor='black')

        # Add standard deviation error bars
        for bar, deviation in zip(bars.patches, std_deviation):
            plt.errorbar(bar.get_x() + bar.get_width() / 2, bar.get_height(), yerr=deviation, fmt='none', ecolor='black')
        
        # Add data labels
        for bar in bars.patches:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.05, round(yval, 2), ha='center', va='bottom')

        # Set plot title and labels
        plt.title(f'Average {col} for Various Approaches', fontweight='bold')
        plt.xlabel('Various Approaches', fontweight='bold')
        plt.ylabel(f'Average {col}', fontweight='bold')

        # Create custom legend with matching colors
        handles = [plt.Rectangle((0,0),1,1, color=bar.get_facecolor()) for bar in bars.patches]
        plt.legend(handles, ['Hierarchical Predictive Single-Model Framework', 
                            'Hierarchical Predictive Multi-Model Framework', 
                            'Predictive Multi-Model Framework',
                            'HDBSCAN-SARIMA Model'], loc='upper left')

        # Save the plot
        output_file_path = os.path.join(approach_compare, f'approach_compare{col}.pdf')
        plt.savefig(output_file_path, bbox_inches='tight')
        plt.close()


