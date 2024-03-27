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
    
    
    


