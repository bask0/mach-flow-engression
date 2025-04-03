import matplotlib.pyplot as plt
import seaborn as sns


def set_matplotlib_config():
    sns.set_theme(style='ticks')

    plt.rcParams['legend.frameon'] = True      # Enable the frame (bounding box)
    plt.rcParams['legend.framealpha'] = 1.0    # Set the transparency of the frame
    plt.rcParams['legend.fancybox'] = False    # Set to False for square corners (not rounded)
    plt.rcParams['legend.edgecolor'] = 'black' # Set the edge color of the bounding box
    plt.rcParams['legend.facecolor'] = 'white' # Set the background color of the bounding box
    plt.rcParams['legend.loc'] = 'best'        # Set the default location of the legend
    plt.rcParams['legend.fontsize'] = 'small' # Set the font size of the legend text



def blend_with_alpha(fg_color, alpha, bg_color=(1.0, 1.0, 1.0)):
    """
    Blend a foreground color with an alpha over a background color (all values 0.0 to 1.0).

    Parameters:
    - fg_color: Tuple of (R, G, B) foreground color, values from 0.0-1.0
    - alpha: Float between 0.0 (fully transparent) and 1.0 (fully opaque)
    - bg_color: Tuple of (R, G, B) background color, default is white

    Returns:
    - Tuple of (R, G, B) blended color
    """
    r = (1 - alpha) * bg_color[0] + alpha * fg_color[0]
    g = (1 - alpha) * bg_color[1] + alpha * fg_color[1]
    b = (1 - alpha) * bg_color[2] + alpha * fg_color[2]
    return (r, g, b)

def blend_legend_color_with_alpha(legend, indices: list[int], alpha: float, bg_color=(1.0, 1.0, 1.0)):
    """
    Blend the colors of a legend with an alpha over a background color.

    Parameters:
    - legend: Matplotlib legend object
    - alpha: Float between 0.0 (fully transparent) and 1.0 (fully opaque)
    - bg_color: Tuple of (R, G, B) background color, default is white
    """

    for i in indices:
        fc = legend.legend_handles[i].get_facecolor()
        fc_new = blend_with_alpha(fc, alpha=alpha)
        legend.legend_handles[i].set_facecolor(fc_new)
