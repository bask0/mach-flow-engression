import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import matplotlib.cm as cm
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def plot_rank_hist(ranks: np.ndarray, title: str, save_file: str = None, display: bool = True) -> None:
    sns.set_theme(style="whitegrid")
    plt.figure()

    rank_hist = sns.histplot(ranks, discrete=True)
    rank_hist.set_xlabel('Rank')

    rank_hist.set_title(title)
    if save_file is not None:
        plt.savefig(save_file)

    if display:
        plt.show()
    plt.close()


def calibration_plot(
    x: np.ndarray,
    cdf_pred: np.ndarray,
    title: str,
    x_name: str = 'Quantile', 
    save_file: str = None,
    display: bool = True
) -> None:
    sns.set_theme(style="whitegrid")
    cdf_perfect_calibration = x

    prob_plot_data = pd.DataFrame({
        f'{x_name}': np.tile(x, 2),
        f'Relative amount of points in {x_name}': np.concatenate([cdf_perfect_calibration, cdf_pred]),
        'line_id': np.repeat(['Perfect calibration', 'Ensemble model'], len(x)),
        'line_style': np.repeat(['Dashed', 'Solid'], len(x))
    })
    
    ax = sns.lineplot(
        data=prob_plot_data, 
        x=f'{x_name}',
        y=f'Relative amount of points in {x_name}',
        hue='line_id',
        style='line_style',
        dashes={"Dashed": (3, 3), "Solid": ""}
    )

    handles, labels = ax.get_legend_handles_labels()
    # Keep only the line labels, starting at index 1 to skip the default title
    handles[1].set_dashes((2, 2))
    plt.legend(handles[1:3], labels[1:3], title=None)
    plt.title(title)

    if save_file is not None:
        plt.savefig(save_file)

    if display:
        plt.show()
    plt.close()


def calibration_plot_range(
    x: np.ndarray,
    cdf_per_station: np.ndarray,
    high_quantile: float,
    low_quantile: float,
    title: str,
    x_name: str = 'Quantile', 
    save_file: str = None,
    display: bool = True
) -> None:
    assert high_quantile >= low_quantile

    sns.set_theme(style="whitegrid")
    cdf_perfect_calibration = x
    median = np.quantile(cdf_per_station, 0.5, axis=0)
    high = np.quantile(cdf_per_station, high_quantile, axis=0)
    low = np.quantile(cdf_per_station, low_quantile, axis=0)

    prob_plot_data = pd.DataFrame({
        f'{x_name}': np.tile(x, 4),
        f'Relative amount of points in {x_name}': np.concatenate([cdf_perfect_calibration, median, high, low]),
        'line_id': np.repeat(['Perfect calibration', 'Median', f'{high_quantile} Quantile', f'{low_quantile} Quantile'], len(x)),
        'line_style': np.repeat(['Dashed', 'Solid', 'Solid', 'Solid'], len(x)),
    })

    ax = sns.lineplot(
        data=prob_plot_data, 
        x=f'{x_name}',
        y=f'Relative amount of points in {x_name}',
        hue='line_id',
        style='line_style',
        palette=['Black', 'Orange', 'C0', 'C0'],
        dashes={"Dashed": (3, 3), "Solid": ""},
        zorder=20
    )

    ax.fill_between(x=x, y1=low, y2=high, color='C0', alpha=0.5, zorder=10)

    handles, labels = ax.get_legend_handles_labels()
    # Keep only the line labels, starting at index 1 to skip the default title
    handles[1].set_dashes((2, 2))
    plt.legend(handles[1:5], labels[1:5], title=None)
    plt.title(title)

    if save_file is not None:
        plt.savefig(save_file)

    if display:
        plt.show()
    plt.close()

def switzerland_map_plot(
    results: gpd.GeoDataFrame,
    col: str,
    save_file: str = None,
    display: bool = True
) -> None:
    fig, ax = plt.subplots(figsize=(15,15))

    # File from https://data.geo.admin.ch/ch.swisstopo.swissboundaries3d/swissboundaries3d_2025-01/swissboundaries3d_2025-01_2056_5728.gpkg.zip
    cantons = gpd.read_file('swissBOUNDARIES3D_1_5_LV95_LN02.gpkg', layer='tlm_kantonsgebiet')

    vmin = results[col].min()
    vmax = results[col].max()
    abs_vmax = max(abs(vmin), abs(vmax))
    if vmin >= 0 or col.startswith('log_'):
        cmap = 'viridis'
        vmin = vmin
        ticks = [vmin, vmax]
    else:
        cmap = 'seismic'
        vmin = -abs_vmax
        vmax = abs_vmax
        ticks = [vmin, 0, vmax]

    ax.axis('off')
    cantons.plot(facecolor='white', edgecolor='black', ax=ax)
    results.plot(
        column=col, 
        vmin=vmin, 
        vmax=vmax, 
        cmap=cmap, 
        markersize=100, 
        edgecolor='black',
        ax=ax
    )

    # Create a ScalarMappable object for the "seismic" colormap
    sm = cm.ScalarMappable(cmap=cmap, norm=colors.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])  # Required for compatibility with colorbar

    cax = inset_axes(
        ax,
        width="60%", 
        height="2%", 
        loc='upper center',
        borderpad=-4)

    # Create the figure and horizontal colorbar
    cbar = plt.colorbar(sm, cax=cax, orientation='horizontal')
    cbar.set_ticks(ticks)

    # Customize the colorbar
    cbar.set_label(col)

    if save_file is not None:
        fig.savefig(save_file)

    if display:
        plt.show(fig)
    plt.close(fig)