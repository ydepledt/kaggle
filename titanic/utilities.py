import random
import warnings

import webcolors

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import FixedLocator
import matplotlib.colors as mcolors
from matplotlib import colormaps

from typing import Any, Dict, List, Tuple, Union

COLORS = {'BLUE': '#3D6FFF',
          'RED': '#FF3D3D',
          'ORANGE': '#FF8E35',
          'PURPLE': '#BB58FF',
          'GREEN': '#32CD32',
          'YELLOW': '#F9DB00',
          'PINK': '#FFC0CB',
          'BROWN': '#8B4513',
          'CYAN': '#00FFFF',
          'SALMON': '#FA8072',
          'LAVENDER': '#E6E6FA',
          'KHAKI': '#F0E68C',
          'TURQUOISE': '#40E0D0',
          'GOLD': '#FFD700',
          'SILVER': '#C0C0C0',
          'CORAL': '#FF7F50',
          'INDIGO': '#4B0082',
          'OLIVE': '#808000',
          'TEAL': '#008080',
          'NAVY': '#000080',

}

def get_random_color() -> str:
    """
    Get a random color in hexadecimal format.

    Returns:
    --------
    str:
        A random color in hexadecimal format.
    """
    return "#{:06x}".format(np.random.randint(0, 0xFFFFFF))

def get_random_colors(n: int) -> List[str]:
    """
    Get a list of n random colors in hexadecimal format.

    Parameters:
    -----------
    n (int):
        The number of random colors to generate.

    Returns:
    --------
    List[str]:
        A list of n random colors in hexadecimal format.
    """
    return [get_random_color() for _ in range(n)]   

def get_random_colors_from_dict(n: int, colors: Dict[str, str] = COLORS) -> List[str]:
    """
    Get a list of n random colors from a dictionary of color names and values.

    Parameters:
    -----------
    n (int):
        The number of random colors to generate.

    colors (Dict[str, str], optional):
        A dictionary of color names and values. Defaults to COLORS.

    Returns:
    --------
    List[str]:
        A list of n random colors in hexadecimal format.
    """
    return [colors[color] for color in random.choices(list(colors.keys()), k=n)]

def adjust_color(color: str, factor: float = 0.5) -> str:
    """
    Adjust the brightness of a color by a specified factor.

    Parameters:
    -----------
    color (str):
        The color to adjust.
    
    factor (float, optional):
        The factor by which to adjust the color. Defaults to 0.5.
        If negative, the color will be lightened.

    
    Returns:
    --------
    str:
        The adjusted color.
    
    Examples:
    ---------
    >>> adjust_color('#0000FF', 0.5)
    >>> adjust_color('blue', -0.5)
    """

    assert factor >= -1 and factor <= 1, "Factor must be between -1 and 1."

    # Convert color name to RGBA format
    rgba_tuple = mcolors.to_rgba(color)
    
    if factor >= 0:
        # Darken the color
        adjusted_rgba = tuple(max(0, min(1, c * factor)) for c in rgba_tuple[:3])
    else:
        # Lighten the color
        adjusted_rgba = tuple(max(0, min(1, c + (1 - c) * factor)) for c in rgba_tuple[:3])

    # Convert adjusted RGBA back to hexadecimal format
    hex_color = mcolors.to_hex(adjusted_rgba)

    return hex_color


def get_colors(size: int, 
               kwargs: Dict[str, Any],
               base_color: str = COLORS['BLUE']) -> None:
    
    if 'color' not in kwargs:
        kwargs['color'] = [base_color] * size
    elif kwargs['color'] == 'random':
        kwargs['color'] = get_random_colors_from_dict(size)
    elif kwargs['color'] == 'random2':
        kwargs['color'] = get_random_colors(size)
    elif 'random' in kwargs['color'] and ('cmap' in kwargs['color'] or 'palette' in kwargs['color'] or 'colormap' in kwargs['color']):
        list_cmap = list(colormaps)
        cmap = np.random.choice(list_cmap)
        kwargs['color'] = sns.color_palette(cmap, size)
    elif isinstance(kwargs['color'], (list, tuple)) and len(kwargs['color']) != size:
        warnings.warn(f"Number of colors provided ({len(kwargs['color'])}) does not match the number of unique values ({size}). Adding default colors.")
        kwargs['color'] += get_random_colors_from_dict(size - len(kwargs['color']))
    elif isinstance(kwargs['color'], (list, tuple)) and len(kwargs['color']) == size:
        pass
    else:
        try:
            kwargs['color'] = sns.color_palette(kwargs['color'], size)
        except ValueError:
            try:
                color_hex = webcolors.name_to_hex(kwargs['color'])
                kwargs['color'] = [color_hex] * size
            except ValueError:
                warnings.warn(f"Color '{kwargs['color']}' is not a valid color name or hex code. Using default color.")
                kwargs['color'] = [base_color] * size
    

def get_edgecolors(edgefactor: float,
                   kwargs: Dict[str, Any]) -> Tuple[str, str]:
        
        colors = kwargs['color']
        nb_colors = len(colors)

        if 'edgecolor' in kwargs:
            edgecolor = kwargs['edgecolor']
            if not (isinstance(edgecolor, tuple) and len(edgecolor) == nb_colors):
                kwargs['edgecolor'] = [edgecolor] * nb_colors
        else:
            kwargs['edgecolor'] = [adjust_color(color, edgefactor) for color in colors]




def save_plot(filepath: str, **kwargs) -> None:
    """
    Save the current plot to a file.

    Parameters:
    -----------
    filepath (str):
        The file path to save the plot.

    **kwargs:
        Additional keyword arguments for saving the plot.

    Returns:
    --------
    None
    """
    if not filepath.endswith('.png'):
        filepath += '.png'
    plt.savefig(filepath, bbox_inches="tight", **kwargs)


def customize_plot_colors(ax: plt.Axes, 
                          axgridx: bool = False,
                          axgridy: bool = False,
                          color_grid: str = None,
                          color_spine: str = None, 
                          color_tick: str = None) -> None:
    """
    Customize the colors of the plot's spines, ticks, and grid.
    
    Parameters:
    -----------
    ax (plt.Axes):
        The plot's axes.

    axgridx (bool, optional):
        Whether to show grid lines along the x-axis. Defaults to False.
        
    axgridy (bool, optional):
        Whether to show grid lines along the y-axis. Defaults to False.
        
    color_grid (str, optional):
        The color of the grid lines. Defaults to 'gray'.
        
    spine_ccolor_spineolor (str, optional):
        The color of the plot's spines. Defaults to None.
        
    color_tick (str, optional):
        The color of the plot's ticks. Defaults to None.
        
    Returns:
    --------
    None
    """
    
    if color_grid is not None and not axgridx and not axgridy:
        raise Warning("color_grid is provided but axgridx and axgridy are False. "
                      "Grid lines will not be shown.")
    
    if (axgridx or axgridy) and color_grid is None:
        color_grid = 'gray'
        raise Warning("axgridx or axgridy is provided but color_grid is not. "
                      "Default color 'gray' will be used for grid lines.")
    
    if axgridx:
        ax.set_axisbelow(True)
        ax.grid(True, axis='x', linestyle='--', alpha=0.6, color=color_grid)
    if axgridy:
        ax.set_axisbelow(True)
        ax.grid(True, axis='y', linestyle='--', alpha=0.6, color=color_grid)
    if color_spine is not None:
        ax.spines['bottom'].set_color(color_spine)
        ax.spines['left'].set_color(color_spine)
        ax.spines['top'].set_color(color_spine)
        ax.spines['right'].set_color(color_spine)
    if color_tick is not None:
        ax.tick_params(axis='x', colors=color_tick)
        ax.tick_params(axis='y', colors=color_tick)


def add_value_labels(ax: plt.Axes, 
                     color: str,
                     percentage: int = 5,
                     frequency: bool = True) -> None:
    """
    Add value labels to the bars in a bar plot.

    Parameters:
    -----------
    ax (plt.Axes):
        The plot's axes.

    color (str):
        The color of the value labels.

    percentage (int, optional):
        The percentage of the bar height at which to place the value labels. Defaults to 5.
    
    frequency (bool, optional):
        If True, the value labels will be displayed as frequencies. Defaults to True.

    Returns:
    --------
    None
    """

    patch_data = [(p.get_x(), p.get_width(), p.get_height()) for p in ax.patches]
    dx_text_height = sum([p[2] for p in patch_data]) / len(patch_data) * percentage / 100

    for x, width, height in patch_data:
        ax.annotate(f'{height:.2f}%' if frequency else f'{height:.0f}',
                    (x + width / 2., height-dx_text_height), 
                    ha='center', va='center', fontsize=8, fontweight='bold', 
                    color=color, xytext=(0, 5), textcoords='offset points')



def plot_missing_data(ax: plt.Axes,
                      dataframe: pd.DataFrame, 
                      nan_values: List[Union[int, float, str]] = None,
                      **kwargs) -> pd.DataFrame:
    """
    Generate a summary of missing data in a DataFrame and visualize it.

    Parameters:
    -----------
    ax (plt.Axes):
        The plot's axes.

    dataframe : pandas DataFrame
        The input DataFrame for which missing data analysis will be performed.

    nan_values : List[int, float, str], optional
        The list of values to be considered as NaN. Defaults to None.

    filepath : str, optional
        The file path where the generated visualization will be saved. 
        If provided, the figure will be saved as a PNG file. Defaults to None.

    **kwargs:
        Additional keyword arguments for customization (e.g., color, alpha, etc.).

    Returns:
    --------
    pandas DataFrame
        DataFrame containing columns 'Total Missing' and 'Percentage Missing',
        sorted by 'Percentage Missing' in descending order.

    This function takes a pandas DataFrame as input and calculates the total number
    and percentage of missing values for each column in the DataFrame. It creates
    a visualization using seaborn to display the percentage of missing values for
    columns with missing data. The function returns a DataFrame summarizing the 
    missing data statistics for all columns in the input DataFrame.
    """

    if nan_values is not None:
        total_missing = dataframe.isnull().sum()
        for nan_value in nan_values:
            total_missing += (dataframe == nan_value).sum()
    else:
        total_missing = dataframe.isnull().sum()

    percentage_missing = total_missing / dataframe.shape[0]
    
    missing_info_df = pd.DataFrame({'Total Missing': total_missing, 
                                    'Percentage Missing': percentage_missing})

    missing_info_df.sort_values(by='Percentage Missing', ascending=False, inplace=True)
    
    filtered_missing_info_df = missing_info_df[missing_info_df['Percentage Missing'] > 0]

    graphcolor = kwargs.pop('graph_color', '#000000')

    get_colors(len(filtered_missing_info_df), kwargs)
    get_edgecolors(0.5, kwargs)

    rotation         = kwargs.pop('rotation', 45)
    axgridx          = kwargs.pop('axgridx', False)
    axgridy          = kwargs.pop('axgridy', True)
    color_label      = kwargs.pop('color_label', adjust_color(graphcolor, 0.3))
    color_spine      = kwargs.pop('color_spine', adjust_color(graphcolor, 0.45))
    color_tick       = kwargs.pop('color_tick', adjust_color(graphcolor, 0.45))
    color_grid       = kwargs.pop('color_grid', adjust_color(graphcolor, -0.4))
    percentage_label = kwargs.pop('percentage_label', 8)
    filepath         = kwargs.pop('filepath', None)

    kwargs['linewidth'] = kwargs.get('linewidth', 1.8)
    kwargs['alpha'] = kwargs.get('alpha', 0.9)


    ax = sns.barplot(y='Percentage Missing', 
                     x=filtered_missing_info_df.index, 
                     data=filtered_missing_info_df, 
                     palette=kwargs.pop('color'),
                     hue=filtered_missing_info_df.index,
                     **kwargs)
    
    for i, patch in enumerate(ax.patches):
        patch.set_edgecolor(kwargs['edgecolor'][i])
    
    ticks = ax.get_yticks()
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.set_yticklabels([f'{(tick):.1f}%' for tick in ticks])

    plt.title('Percentage of Missing Values by Feature', fontweight='bold')
    plt.xlabel('Feature', color=color_label, fontsize=11)
    plt.ylabel('Percentage Missing', color=color_label, fontsize=11)

    rotation, ha = rotation if isinstance(rotation, tuple) else (rotation, 'center')
    plt.xticks(rotation=rotation, ha=ha)

    add_value_labels(ax, '#000000', frequency=True, percentage=percentage_label)
    
    customize_plot_colors(ax, axgridx=axgridx, axgridy=axgridy, color_grid=color_grid,
                          color_spine=color_spine, color_tick=color_tick)

    if filepath:
        save_plot(filepath)

    return missing_info_df


def plot_groupby(ax: plt.Axes,
                 dataframe: pd.DataFrame, 
                 group: str,
                 result_label: str,
                 **kwargs) -> pd.DataFrame:
    """
    Generate a grouped bar plot based on DataFrame aggregation.

    Parameters:
    -----------
    ax (plt.Axes):
        The plot's axes.

    dataframe : pandas DataFrame
        The input DataFrame containing the data for analysis.

    group : str
        The column name by which the DataFrame will be grouped.

    result_label : str
        The column label for which statistics will be calculated and visualized.

    filepath : str, optional
        The file path where the generated visualization will be saved. 
        If provided, the figure will be saved as a PNG file. Defaults to None.

    **kwargs:
        Additional keyword arguments for customization (e.g., color, figsize, alpha, etc.).

    Returns:
    --------
    pandas DataFrame
        DataFrame containing aggregated statistics based on the provided group
        column and the result_label.

    This function groups the input DataFrame based on a specified column ('group')
    and calculates aggregated statistics ('result_label') for each group. It creates
    a grouped bar plot using seaborn to visualize the calculated statistics. The 
    function returns a DataFrame summarizing the aggregated statistics for each group.
    """
    
    groups = dataframe.groupby(group)

    percentage_by_group = groups[result_label].mean() * 100
    number_by_group = groups[result_label].sum()

    total_by_group = groups.size()
    
    df = pd.DataFrame({
        'Group Percentage': percentage_by_group,
        'Total': total_by_group,
        'Grouped total': number_by_group
    })

    graphcolor = kwargs.pop('graph_color', '#000000')

    get_colors(len(df), kwargs)
    get_edgecolors(0.5, kwargs)

    rotation         = kwargs.pop('rotation', 45)
    axgridx          = kwargs.pop('axgridx', False)
    axgridy          = kwargs.pop('axgridy', True)
    color_label      = kwargs.pop('color_label', adjust_color(graphcolor, 0.3))
    color_spine      = kwargs.pop('color_spine', adjust_color(graphcolor, 0.45))
    color_tick       = kwargs.pop('color_tick', adjust_color(graphcolor, 0.45))
    color_grid       = kwargs.pop('color_grid', adjust_color(graphcolor, -0.4))
    percentage_label = kwargs.pop('percentage_label', 8)
    filepath         = kwargs.pop('filepath', None)

    kwargs['linewidth'] = kwargs.get('linewidth', 1.8)
    kwargs['alpha'] = kwargs.get('alpha', 0.9)


    sns.barplot(y='Group Percentage', x=df.index, 
                data=df, palette=kwargs.pop('color'),
                hue=df.index, legend=False, **kwargs)
    
    for i, patch in enumerate(ax.patches):
        patch.set_edgecolor(kwargs['edgecolor'][i])
    
    plt.title(f'{result_label.capitalize()} Percentage by {group.capitalize()} Category', fontweight= 'bold')
    plt.ylabel(f'{result_label.capitalize()} Percentage', color=color_label, fontsize=11)
    plt.xlabel(f'{group.capitalize()} Category', color=color_label, fontsize=11)

    plt.xticks(rotation=rotation)

    add_value_labels(ax, '#000000', frequency=True, percentage=percentage_label)

    customize_plot_colors(ax, axgridx=axgridx, axgridy=axgridy, color_grid=color_grid,
                          color_spine=color_spine, color_tick=color_tick)

    if filepath:
        save_plot(filepath)

    return df




def plot_hist_discrete_feature(ax: plt.Axes,
                               dataframe: pd.DataFrame, 
                               column: str,
                               frequency: bool = False,
                               **kwargs) -> pd.DataFrame:
    """
    Plot a histogram for a specified column in a DataFrame with customizable options.

    Parameters:
    -----------
    ax (plt.Axes):
        The plot's axes.

    df (pd.DataFrame):
        The DataFrame containing the data.

    column (str):
        The name of the column to plot.

    filepath (str, optional):
        The file path to save the plot as an image. If provided, the figure will be saved as a PNG file.
        Defaults to None.

    frequency (bool, optional):
        If True, plot frequencies instead of counts. Defaults to False.

    **kwargs:
        Additional keyword arguments for customization (e.g., alpha, edgecolor, bins, color, etc.).

    Returns:
    --------
    pd.DataFrame:
        DataFrame containing the unique values, counts, and percentages of the specified column.

    This function plots a histogram for the specified column in the DataFrame. It provides options
    to customize the appearance of the plot, such as figure figsize, colors, and transparency. If a file
    path is provided, the plot will be saved as an image in PNG format.
    """

    labels, counts = np.unique(dataframe[column], return_counts=True)
    percentages = counts / len(dataframe) * 100
    
    df_counts = pd.DataFrame({'Labels': labels, 
                              'Counts': counts,
                              'Percentages': percentages})

    df_counts.sort_values(by='Counts', ascending=False, inplace=True)

    graphcolor = kwargs.pop('graph_color', '#000000')

    get_colors(len(labels), kwargs)
    get_edgecolors(0.5, kwargs)

    axgridx          = kwargs.pop('axgridx', False)
    axgridy          = kwargs.pop('axgridy', True)
    color_label      = kwargs.pop('color_label', adjust_color(graphcolor, 0.3))
    color_spine      = kwargs.pop('color_spine', adjust_color(graphcolor, 0.45))
    color_tick       = kwargs.pop('color_tick', adjust_color(graphcolor, 0.45))
    color_grid       = kwargs.pop('color_grid', adjust_color(graphcolor, -0.4))
    percentage_label = kwargs.pop('percentage_label', 8)
    title            = kwargs.pop('title', f'Distribution of {column.capitalize()}')
    title_before     = kwargs.pop('title_before', '')
    title_addition   = kwargs.pop('title_addition', '')
    filepath         = kwargs.pop('filepath', None)

    kwargs['linewidth'] = kwargs.get('linewidth', 1.8)
    kwargs['alpha'] = kwargs.get('alpha', 0.9)

    if frequency:
        sns.barplot(y='Percentages', 
                    x='Labels', 
                    data=df_counts, 
                    palette=kwargs.pop('color'),
                    hue='Labels',
                    legend=False,
                    **kwargs)
        plt.ylabel('Frequency', color=color_label, fontsize=11)
        ticks = ax.get_yticks()
        ax.yaxis.set_major_locator(FixedLocator(ticks))
        ax.set_yticklabels([f'{(tick):.1f}%' for tick in ticks])
    else:
        ax = sns.barplot(y='Counts', 
                         x='Labels', 
                         data=df_counts, 
                         palette=kwargs.pop('color'),
                         hue='Labels',
                         legend=False,
                         **kwargs)
        plt.ylabel('Number', color=color_label, fontsize=11)

    for i, patch in enumerate(ax.patches):
        patch.set_edgecolor(kwargs['edgecolor'][i])

    add_value_labels(ax, '#000000', frequency=frequency, percentage=percentage_label)

    plt.title(title_before + title + title_addition, fontweight='bold')
    plt.xlabel(column.capitalize(), color=color_label, fontsize=11)

    customize_plot_colors(ax, axgridx=axgridx, axgridy=axgridy, color_grid=color_grid,
                          color_spine=color_spine, color_tick=color_tick)

    if filepath:
        save_plot(filepath)

    return df_counts

def plot_feature_importance(ax: plt.Axes,
                            features: pd.Index,
                            importances: np.ndarray, 
                            **kwargs) -> Dict[str, float]:
    """
    Plot feature importances.

    Parameters:
    -----------
    ax (plt.Axes):
        The plot's axes.

    features (pd.Index):
        The features (columns) corresponding to the importances.
    
    importances (np.ndarray):
        Feature importances obtained from a Random Forest classifier.

    **kwargs:
        Additional keyword arguments for customization (e.g., color, alpha, etc.).

    Returns:
    --------
    Dict[str, float]:
        A dictionary containing features and their corresponding importances.

    This function plots the feature importances.
    If a file path is provided, the plot will be saved as an image in PNG format.
    """
    
    df_importance = pd.DataFrame({'Features': features, 
                                  'Importances': importances})
    
    df_importance.sort_values(by='Importances', ascending=False, inplace=True)

    graphcolor = kwargs.pop('graph_color', '#000000')
    
    get_colors(len(df_importance), kwargs)
    get_edgecolors(0.5, kwargs)

    axgridx     = kwargs.pop('axgridx', True)
    axgridy     = kwargs.pop('axgridy', False)
    color_label = kwargs.pop('color_label', adjust_color(graphcolor , 0.3))
    color_spine = kwargs.pop('color_spine', adjust_color(graphcolor , 0.45))
    color_tick  = kwargs.pop('color_tick', adjust_color(graphcolor , 0.45))
    color_grid  = kwargs.pop('color_grid', adjust_color(graphcolor , -0.4))
    bold_max    = kwargs.pop('bold_max', True)
    italic_min  = kwargs.pop('italic_min', True)
    to_dict     = kwargs.pop('to_dict', False)
    filepath    = kwargs.pop('filepath', None)

    kwargs['linewidth'] = kwargs.get('linewidth', 1.8)
    kwargs['alpha'] = kwargs.get('alpha', 0.9)

    sns.barplot(y='Features', 
                x='Importances', 
                data=df_importance, 
                palette=kwargs.pop('color'),
                hue='Features',
                legend=False,
                **kwargs)
    
    for i, patch in enumerate(ax.patches):
        patch.set_edgecolor(kwargs['edgecolor'][i])
    
    plt.title(f'Features importances', fontweight='bold')
    plt.xlabel('Importances', color=color_label, fontsize=11)
    plt.ylabel('Features', color=color_label, fontsize=11)

    customize_plot_colors(ax, axgridx=axgridx, axgridy=axgridy, color_grid=color_grid,
                          color_spine=color_spine, color_tick=color_tick)
    
    if bold_max:
        max_index = np.argmax(df_importance['Importances'])
        y_labels = ax.get_yticklabels()
        y_labels[max_index].set_weight('bold')

    if italic_min:
        min_index = np.argmin(df_importance['Importances'])
        y_labels = ax.get_yticklabels()
        y_labels[min_index].set_style('italic')


    if filepath:
        save_plot(filepath)

    return df_importance.transpose().to_dict() if to_dict else df_importance.transpose()
