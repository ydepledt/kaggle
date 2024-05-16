import warnings
import webcolors
import torch

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import FixedLocator
from matplotlib import colormaps

from typing import Any, Dict, List, Tuple, Union

class ColorGenerator:
    COLORS = {
        'BLUE': '#3D6FFF',
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

    @staticmethod
    def get_random_color() -> str:
        """
        Get a random color in hexadecimal format.

        Returns:
        --------
        str:
            A random color in hexadecimal format.
        """
        return "#{:06x}".format(np.random.randint(0, 0xFFFFFF))

    @staticmethod
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
        return [ColorGenerator.get_random_color() for _ in range(n)]

    @staticmethod
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
        return [colors[color] for color in np.random.choice(list(colors.keys()), n, replace=False)]

    @staticmethod
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
        >>> ColorGenerator.adjust_color('#0000FF', 0.5)
        >>> ColorGenerator.adjust_color('blue', -0.5)
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
               base_color: str = ColorGenerator.COLORS['BLUE']) -> None:
    
    if 'color' not in kwargs:
        kwargs['color'] = [base_color] * size
    elif kwargs['color'] == 'random':
        kwargs['color'] = ColorGenerator().get_random_colors_from_dict(size)
    elif kwargs['color'] == 'random2':
        kwargs['color'] = ColorGenerator().get_random_colors_from_dict(size)
    elif 'random' in kwargs['color'] and ('cmap' in kwargs['color'] or 'palette' in kwargs['color'] or 'colormap' in kwargs['color']):
        list_cmap = list(colormaps)
        cmap = np.random.choice(list_cmap)
        kwargs['color'] = sns.color_palette(cmap, size)
    elif isinstance(kwargs['color'], str) and '#' in kwargs['color']:
        kwargs['color'] = [kwargs['color']] * size
    elif isinstance(kwargs['color'], (list, tuple)) and len(kwargs['color']) != size:
        warnings.warn(f"Number of colors provided ({len(kwargs['color'])}) does not match the number of unique values ({size}). Adding default colors.")
        kwargs['color'] += ColorGenerator().get_random_colors_from_dict(size - len(kwargs['color']))
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
            kwargs['edgecolor'] = [ColorGenerator.adjust_color(color, edgefactor) for color in colors]




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
        warnings.warn("color_grid is provided but axgridx and axgridy are False. "
                      "Grid lines will not be shown.")
    
    if (axgridx or axgridy) and color_grid is None:
        color_grid = 'gray'
        warnings.warn("axgridx or axgridy is provided but color_grid is not. "
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
                     frequency: bool = True,
                     horizontal: bool = False,
                     outside: bool = False) -> None:
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

    horizontal (bool, optional):
        If True, the value labels will be displayed for a horizontal bar plot. Defaults to False.

    Returns:
    --------
    None
    """

    patch_data = [(p.get_x(), p.get_width(), p.get_height()) for p in ax.patches] if not horizontal else \
                 [(p.get_y(), p.get_height(), p.get_width()) for p in ax.patches]

    dx_text = -sum([p[2] for p in patch_data]) / len(patch_data) * percentage / 100 if not outside else \
               sum([p[2] for p in patch_data]) / len(patch_data) * percentage / 100


    if not horizontal:
        for x, width, height in patch_data:
            ax.annotate(f'{height:.2f}%' if frequency else f'{height:.0f}',
                        (x + width / 2., height+dx_text), 
                        ha='center', va='center', fontsize=8, fontweight='bold', 
                        color=color, xytext=(0, 5), textcoords='offset points')
    else:
        for y, height, width in patch_data:
            ax.annotate(f'{width:.2f}%' if frequency else f'{width:.0f}',
                        (width+dx_text, y + height / 2.), 
                        ha='center', va='center', fontsize=8, fontweight='bold', 
                        color=color, xytext=(5, 0), textcoords='offset points')



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
    color_label      = kwargs.pop('color_label', ColorGenerator.adjust_color(graphcolor, 0.3))
    color_spine      = kwargs.pop('color_spine', ColorGenerator.adjust_color(graphcolor, 0.45))
    color_tick       = kwargs.pop('color_tick', ColorGenerator.adjust_color(graphcolor, 0.45))
    color_grid       = kwargs.pop('color_grid', ColorGenerator.adjust_color(graphcolor, -0.4))
    title            = kwargs.pop('title', f'Missing Data')
    title_before     = kwargs.pop('title_before', '')
    title_after      = kwargs.pop('title_after', '')
    percentage_label = kwargs.pop('percentage_label', 8)
    frequency        = kwargs.pop('frequency', True)
    filepath         = kwargs.pop('filepath', None)

    kwargs['linewidth'] = kwargs.get('linewidth', 1.8)
    kwargs['alpha'] = kwargs.get('alpha', 0.9)

    if frequency:
        sns.barplot(y='Percentage Missing', 
                    x=filtered_missing_info_df.index, 
                    data=filtered_missing_info_df, 
                    palette=kwargs.pop('color'),
                    hue=filtered_missing_info_df.index,
                    **kwargs)
        plt.ylabel('Frequency', color=color_label, fontsize=11)
        ticks = ax.get_yticks()
        ax.yaxis.set_major_locator(FixedLocator(ticks))
        ax.set_yticklabels([f'{(tick):.1f}%' for tick in ticks])
    else:
        sns.barplot(y='Total Missing', 
                    x=filtered_missing_info_df.index, 
                    data=filtered_missing_info_df, 
                    palette=kwargs.pop('color'),
                    hue=filtered_missing_info_df.index,
                    **kwargs)
        plt.ylabel('Counts', color=color_label, fontsize=11)

    xtick_labels = [f"{word[:9]}...{word[-9:]}" if len(word) > 18 else word for word in filtered_missing_info_df.index]
    ax.set_xticks(range(len(xtick_labels)))
    ax.set_xticklabels(xtick_labels)
    
    for i, patch in enumerate(ax.patches):
        patch.set_edgecolor(kwargs['edgecolor'][i])

    plt.title(title_before + title + title_after, fontweight='bold')
    plt.xlabel('Feature', color=color_label, fontsize=11)

    rotation, ha = rotation if isinstance(rotation, tuple) else (rotation, 'center')
    plt.xticks(rotation=rotation, ha=ha)

    add_value_labels(ax, '#000000', frequency=frequency, percentage=percentage_label)
    
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
    color_label      = kwargs.pop('color_label', ColorGenerator.adjust_color(graphcolor, 0.3))
    color_spine      = kwargs.pop('color_spine', ColorGenerator.adjust_color(graphcolor, 0.45))
    color_tick       = kwargs.pop('color_tick', ColorGenerator.adjust_color(graphcolor, 0.45))
    color_grid       = kwargs.pop('color_grid', ColorGenerator.adjust_color(graphcolor, -0.4))
    percentage_label = kwargs.pop('percentage_label', 8)
    add_value        = kwargs.pop('add_value', True)
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

    if add_value:
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

    dataframe_formated = kwargs.pop('dataframe_formated', False)

    labels, counts = (np.unique(dataframe[column], return_counts=True) if not dataframe_formated else (dataframe[column].index, dataframe[column].values.ravel()))
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
    color_label      = kwargs.pop('color_label', ColorGenerator.adjust_color(graphcolor, 0.3))
    color_spine      = kwargs.pop('color_spine', ColorGenerator.adjust_color(graphcolor, 0.45))
    color_tick       = kwargs.pop('color_tick', ColorGenerator.adjust_color(graphcolor, 0.45))
    color_grid       = kwargs.pop('color_grid', ColorGenerator.adjust_color(graphcolor, -0.4))
    color_annot      = kwargs.pop('color_annot', ColorGenerator.adjust_color(graphcolor, 0.3))
    percentage_annot = kwargs.pop('percentage_annot', 8)
    outside_annot    = kwargs.pop('outside_annot', False)
    title            = kwargs.pop('title', f'Distribution of {column.capitalize()}')
    title_before     = kwargs.pop('title_before', '')
    title_after      = kwargs.pop('title_after', '')
    filepath         = kwargs.pop('filepath', None)
    add_value        = kwargs.pop('add_value', True)
    horizontaly      = kwargs.pop('horizontaly', False)
    xticks_sep       = kwargs.pop('xticks_sep', 'auto')
    yticks_sep       = kwargs.pop('yticks_sep', 'auto')

    kwargs['linewidth'] = kwargs.get('linewidth', 1.8)
    kwargs['alpha'] = kwargs.get('alpha', 0.9)


    if frequency:
        x_lab, y_lab = ('Percentages', 'Labels') if not horizontaly else ('Labels', 'Percentages')
        sns.barplot(y=x_lab, 
                    x=y_lab, 
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
        x_lab, y_lab = ('Counts', 'Labels') if not horizontaly else ('Labels', 'Counts')
        sns.barplot(y=x_lab, 
                    x=y_lab, 
                    data=df_counts, 
                    palette=kwargs.pop('color'),
                    hue='Labels',
                    legend=False,
                    **kwargs)
        plt.ylabel('Counts', color=color_label, fontsize=11)

    for i, patch in enumerate(ax.patches):
        patch.set_edgecolor(kwargs['edgecolor'][i])

    if add_value:
        add_value_labels(ax, color_annot, frequency=frequency, 
                         percentage=percentage_annot, horizontal=horizontaly,
                         outside=outside_annot)

    if xticks_sep and xticks_sep != 'auto':
        ax.xaxis.set_major_locator(plt.MultipleLocator(xticks_sep))
    if yticks_sep and yticks_sep != 'auto':
        ax.yaxis.set_major_locator(plt.MultipleLocator(yticks_sep))

    plt.title(title_before + title + title_after, fontweight='bold')
    plt.xlabel(column.capitalize(), color=color_label, fontsize=11)

    customize_plot_colors(ax, axgridx=axgridx, axgridy=axgridy, color_grid=color_grid,
                          color_spine=color_spine, color_tick=color_tick)

    if filepath:
        save_plot(filepath)

    return df_counts


def plot_boxplot(ax: plt.Axes,
                 dataframe: pd.DataFrame, 
                 x: str, 
                 y: str, 
                 **kwargs) -> None:
    """
    Plot a boxplot for a specified x and y column in a DataFrame.

    Parameters:
    -----------
    ax (plt.Axes):
        The Axes object to plot the data.

    dataframe (pd.DataFrame):
        The DataFrame containing the data.

    x (str):
        The name of the x-axis column.

    y (str):
        The name of the y-axis column.

    **kwargs:
        Additional keyword arguments for customization (e.g., color, edgecolor, linewidth, etc.).

    Returns:
    --------
    None

    This function plots a boxplot for the specified x and y columns in the DataFrame. It provides options to customize
    the appearance of the plot, such as colors, edge color, and line width.
    """

    graphcolor = kwargs.pop('graph_color', '#000000')

    if 'color' not in kwargs:
        kwargs['color'] = 'inferno'
    get_colors(len(dataframe[x].unique()), kwargs)

    default_median_style = dict(linewidth=1.5, color='auto')
    default_outlier_style = dict(marker='o')


    axgridx           = kwargs.pop('axgridx', False)
    axgridy           = kwargs.pop('axgridy', True)
    color_label       = kwargs.pop('color_label', ColorGenerator.adjust_color(graphcolor, 0.3))
    color_spine       = kwargs.pop('color_spine', ColorGenerator.adjust_color(graphcolor, 0.45))
    color_tick        = kwargs.pop('color_tick', ColorGenerator.adjust_color(graphcolor, 0.45))
    color_grid        = kwargs.pop('color_grid', ColorGenerator.adjust_color(graphcolor, -0.4))
    title             = kwargs.pop('title', f'Boxplot of {x.capitalize()} and {y.capitalize()}')
    title_before      = kwargs.pop('title_before', '')
    title_addition    = kwargs.pop('title_addition', '')
    filepath          = kwargs.pop('filepath', None)
    outlier_style     = {**default_outlier_style, **kwargs.pop('outlier_style', {})}

    if default_median_style['color'] != 'auto':
        median_style = {**default_median_style, **kwargs.pop('median_style', {})}  
    else:
        median_color = default_median_style.pop('color')
        median_style = {**default_median_style, **kwargs.pop('median_style', {})}


    alpha = kwargs.pop('alpha', 0.8)

    customize_plot_colors(ax, axgridx=axgridx, axgridy=axgridy, color_grid=color_grid,
                          color_spine=color_spine, color_tick=color_tick)

    sns.boxplot(x=x, y=y, data=dataframe, palette=kwargs.pop('color'), 
                hue=x, medianprops=median_style, flierprops=outlier_style, **kwargs)
    
    for patch in ax.patches:
        r, g, b, _ = patch.get_facecolor()
        patch.set_facecolor((r, g, b, alpha))
    
    if median_color == 'auto':
        for i, line in enumerate(ax.lines):
            if median_color == 'auto' and i % 6 == 4 and i != 0:
                line.set_color(ColorGenerator.adjust_color(ax.patches[i//6].get_facecolor(), -0.2))

    plt.title(title_before + title + title_addition, fontweight='bold')
    plt.xlabel(x.capitalize(), color=color_label, fontsize=11)
    plt.ylabel(y.capitalize(), color=color_label, fontsize=11)

    if filepath:
        save_plot(filepath)


def plot_kde(ax: plt.Axes,
             dataframe: Union[pd.DataFrame, List[pd.DataFrame]],
             column: str,
             group_column: str = None,
             **kwargs) -> None:
    
    """
    Plot a kernel density estimation (KDE) of a column in a DataFrame.

    Parameters:
    -----------
    ax (plt.Axes):
        The plot's axes.

    dataframe (pd.DataFrame or List[pd.DataFrame]):
        The DataFrame containing the data.

    column (str):
        The name of the column to plot.

    group_column (str, optional):
        The column by which to group the data. Defaults to None.

    **kwargs:
        Additional keyword arguments for customization (e.g., color, alpha, etc.).

    Returns:
    --------
    None

    This function plots a kernel density estimation (KDE) of a column in a DataFrame.
    If a file path is provided, the plot will be saved as an image in PNG format.

    Examples:
    ---------
    >>> plot_kde(ax, df, 'heart_rate', color='red', alpha=0.8, linewidth=0.1)
    """
    
    graphcolor = kwargs.pop('graph_color', '#000000')

    if isinstance(dataframe, list) and group_column:
        raise ValueError("group_column cannot be specified when passing a list of DataFrames. Feature not supported yet.")

    nb_colors = len(dataframe) if isinstance(dataframe, list) else 1
    nb_colors2 = len(dataframe[group_column].unique()) if group_column else 1

    get_colors(max(nb_colors, nb_colors2), kwargs)

    axgridx          = kwargs.pop('axgridx', False)
    axgridy          = kwargs.pop('axgridy', True)
    color_label      = kwargs.pop('color_label', ColorGenerator.adjust_color(graphcolor, 0.3))
    color_spine      = kwargs.pop('color_spine', ColorGenerator.adjust_color(graphcolor, 0.45))
    color_tick       = kwargs.pop('color_tick', ColorGenerator.adjust_color(graphcolor, 0.45))
    color_grid       = kwargs.pop('color_grid', ColorGenerator.adjust_color(graphcolor, -0.4))
    title_before      = kwargs.pop('title_before', '')
    title_after       = kwargs.pop('title_after', '')
    title             = kwargs.pop('title', f'Kernel Density Estimation (KDE) of {column}' + \
                                  (' by ' + group_column if group_column else ''))
    labels            = kwargs.pop('labels', None)
    filepath          = kwargs.pop('filepath', None)

    kwargs['fill']      = kwargs.get('fill', False)
    kwargs['linewidth'] = kwargs.get('linewidth', 0.7)
    kwargs['alpha']     = kwargs.get('alpha', 0.6)

    if isinstance(dataframe, pd.DataFrame):
        dataframe = [dataframe]
    color = kwargs.pop('color', ColorGenerator.COLORS['BLUE'])

    if group_column:
        for i, group in enumerate(dataframe[group_column].unique()):
            sns.kdeplot(dataframe[dataframe[group_column] == group][column], 
                        color=color[i], label=labels[i] if labels else group, **kwargs)
    
        plt.legend()
    else:
        for i, df in enumerate(dataframe):
            sns.kdeplot(df[column], color=color[i], 
                        label=labels[i] if labels else f'DataFrame {i}', **kwargs)
        plt.legend() if len(dataframe) > 1 else None

    plt.title(title_before + title + title_after, fontweight='bold')
    plt.xlabel(column.capitalize(), color=color_label, fontsize=11)
    plt.ylabel('Density', color=color_label, fontsize=11)

    customize_plot_colors(ax, axgridx=axgridx, axgridy=axgridy, color_grid=color_grid,
                          color_spine=color_spine, color_tick=color_tick)
    
    if filepath:
        save_plot(filepath)    


def categorize_column(df: pd.DataFrame,
                      column: str,
                      int_bins: list,
                      categorial_labels: list,
                      handle_nan: bool = False,
                      replace_original: bool = True,
                      inplace: bool = True) -> pd.DataFrame:
    """
    Categorize a numerical column in a DataFrame into specified bins and labels.

    Parameters:
    -----------
    df (pd.DataFrame):
        The DataFrame containing the data.

    column (str):
        The name of the column to categorize.

    int_bins (list):
        The bin edges for categorizing the numerical values in the column.

    categorial_labels (list):
        The labels corresponding to the bins for categorizing the values.

    handle_nan (bool, optional):
        If True, add a category 'Unknown' for NaN values. Defaults to False.

    replace_original (bool, optional):
        If True, replace the original numerical column after categorization. Defaults to True.

    inplace (bool, optional):
        If True, modify the DataFrame in place. If False, create a copy of the DataFrame.
        Defaults to True.

    Returns:
    --------
    pd.DataFrame or None:
        Returns the modified DataFrame if inplace is True, otherwise, returns a new DataFrame.

    This function categorizes the specified numerical column into bins with corresponding labels.
    Optionally, it can handle NaN values by adding a category 'Unknown'. The modified DataFrame is
    returned if inplace is True; otherwise, a new DataFrame is returned.
    """
    
    assert len(int_bins) == len(categorial_labels) + 1, "Length of int_bins and categorial_labels must match."

    if not inplace:
        df = df.copy()

    str_to_add = '' if replace_original else '_bin'


    # Categorize the numerical column into specified bins and labels
    df[column + str_to_add] = pd.cut(df[column], bins=int_bins, labels=categorial_labels, right=False)

    # Deal with NaN values if specified
    if handle_nan:
        df[column + str_to_add] = df[column + str_to_add].cat.add_categories('Unknown').fillna('Unknown')

    # Return the modified DataFrame if inplace is False
    if not inplace:
        return df
    

def plot_PCA(df: pd.DataFrame,
             categorial_column: str = None,
             components: Union[Tuple[int, int], Tuple[int, int, int]] = (1, 2),
             elev: int = 30,
             azim: int = -60,
             figsize: Tuple[int, int] = (12, 10),
             filepath: str = None,
             **kwargs) -> None:
    """
    Plot the Principal Components using PCA on the DataFrame.

    Parameters:
    -----------
    df (pd.DataFrame):
        The DataFrame containing the data.

    categorial_column (str, optional):
        The name of the column to categorize. If None, plot the data without categorization. Defaults to None.

    components (Tuple[int, int] or Tuple[int, int, int], optional):
        The number of principal components to plot. Defaults to (1, 2).

    elev (int, optional):
        The elevation angle of the 3D plot. Defaults to 30.
    
    azim (int, optional):  
        The azimuth angle of the 3D plot. Defaults to -60.

    figsize (Tuple[int, int], optional): 
        The figsize of the plot (width, height). Defaults to (10, 8).

    filepath (str, optional):
        If provided, saves the plot as a PNG file at the specified filepath. Defaults to None.
    
    **kwargs:
        Additional keyword arguments to pass to the plt.scatter function.

    Returns:
    --------
    None

    This function performs Principal Component Analysis (PCA) on the DataFrame and plots
    the Principal Components. It also plots the data with categorization if specified.
    If categorial_column is None, it plots the data without categorization.
    If filepath is provided, it saves the plot as a PNG file.

    Example: 
    --------
    >>> plot_PCA(df, 'target')
    """
    
    assert(len(components) == 2 or len(components) == 3), "components must be a tuple of length 2 or 3."

    X = df.drop(categorial_column, axis=1) if categorial_column else df
    y = df[categorial_column] if categorial_column else None

    pca = PCA(n_components=len(X.columns))
    pca_result = pca.fit_transform(X)
    column_names = [f'PC{i}' for i in range(1, len(X.columns) + 1)]
    df_pca = pd.DataFrame(data=pca_result, columns=column_names)
    
    if categorial_column:
        df_pca[categorial_column] = y


    if len(components) == 2:
        name_column1, name_column2 = f'PC{components[0]}', f'PC{components[1]}'
        if categorial_column:
            categories = df_pca[categorial_column].unique()
            colors = [ColorGenerator.COLORS['BLUE'], ColorGenerator.COLORS['RED'], ColorGenerator.COLORS['ORANGE']] if categorial_column else None

            for category, color in zip(categories, colors):
                subset = df_pca[df_pca[categorial_column] == category]
                plt.scatter(subset[name_column1], subset[name_column2], c=color, label=category, **kwargs)

            plt.legend()
        else:
            plt.scatter(df_pca[name_column1], df_pca[name_column2], **kwargs)

        plt.xlabel(name_column1, fontweight='bold')
        plt.ylabel(name_column2, fontweight='bold')
        plt.title('2D PCA Plot', fontweight='bold')
        plt.grid(True)
        plt.tight_layout()

    elif len(components) == 3:
        name_column1, name_column2, name_column3 = f'PC{components[0]}', f'PC{components[1]}', f'PC{components[2]}'
        ax = plt.axes(projection='3d')

        if categorial_column:
            categories = df_pca[categorial_column].unique()
            colors = [ColorGenerator.COLORS['BLUE'], ColorGenerator.COLORS['RED'], ColorGenerator.COLORS['ORANGE']] if categorial_column else None

            for category, color in zip(categories, colors):
                subset = df_pca[df_pca[categorial_column] == category]
                ax.scatter(subset[name_column1], subset[name_column2], subset[name_column3], c=color, label=category, **kwargs)

            ax.legend()
        else:
            ax.scatter(df_pca[name_column1], df_pca[name_column2], df_pca[name_column3], **kwargs)

        ax.set_xlabel(name_column1, fontweight='bold')
        ax.set_ylabel(name_column2, fontweight='bold')
        ax.set_zlabel(name_column3, fontweight='bold')

        ax.view_init(elev=elev, azim=azim)

        plt.tight_layout()

        ax.set_title('3D PCA Plot', fontweight='bold')

    if filepath:
        save_plot(filepath)

    plt.show()

def plot_PCA_explained_variance(df: pd.DataFrame, 
                                filepath: str = None,
                                **kwargs) -> np.ndarray:
    
    """
    Plot the Explained Variance Ratio by Principal Components using PCA on the DataFrame.

    Parameters:
    -----------
    df (pd.DataFrame):
        The DataFrame containing the data.

    figsize (Tuple[int, int], optional):
        The figsize of the plot (width, height). Defaults to (10, 8).

    filepath (str, optional):
        If provided, saves the plot as a PNG file at the specified filepath. Defaults to None.

    **kwargs:
        Additional keyword arguments to pass to the plt.bar function.

    Returns:
    --------
    np.ndarray:
        Returns the explained variance ratio of Principal Components obtained from PCA.

    This function performs Principal Component Analysis (PCA) on the DataFrame and plots
    the Explained Variance Ratio by Principal Components. It also returns the explained 
    variance ratio for each principal component obtained from PCA.
    """
    
    pca = PCA()
    pca.fit(df)

    explained_variance_ratio = pca.explained_variance_ratio_
    
    principal_components = range(1, len(explained_variance_ratio) + 1)
    

    plt.bar(principal_components, 
            explained_variance_ratio, 
            **kwargs)
    
    plt.xlabel('Principal Components', fontweight='bold')
    plt.ylabel('Explained Variance Ratio', fontweight='bold')
    plt.title('Explained Variance Ratio by Principal Component', fontweight='bold')
    
    plt.xticks(principal_components)

    if filepath:
        save_plot(filepath)

    plt.show()

    return explained_variance_ratio


def plot_cumulative_explained_variance(explained_variance_ratio: np.ndarray,
                                       figsize: Tuple[int, int] = (10, 8),
                                       filepath: str = None,
                                       **kwargs) -> None:
    """
    Plot the Cumulative Explained Variance by Principal Components.

    Parameters:
    -----------
    explained_variance_ratio (np.ndarray):
        The array containing the explained variance ratio for each principal component.

    figsize (Tuple[int, int], optional):
        The figsize of the plot (width, height). Defaults to (10, 8).

    filepath (str, optional):
        If provided, saves the plot as a PNG file at the specified filepath. Defaults to None.

    **kwargs:
        Additional keyword arguments to pass to the plt.plot function.

    Returns:
    --------
    None

    This function plots the Cumulative Explained Variance by Principal Components using the
    explained variance ratio array obtained from PCA.
    """
    
    principal_components = np.arange(1, len(explained_variance_ratio) + 1)
    cumulative_explained_variance = np.cumsum(explained_variance_ratio)
    

    plt.plot(principal_components, cumulative_explained_variance, **kwargs)
    
    plt.xlabel('Number of Principal Components', fontweight='bold')
    plt.ylabel('Cumulative Explained Variance', fontweight='bold')
    plt.title('Cumulative Explained Variance by Principal Components', fontweight='bold')
    
    plt.xticks(principal_components)
    plt.grid(True)
    
    if filepath:
        save_plot(filepath)
    
    plt.show()


def create_gif_from_3D_PCA(df: pd.DataFrame,
                           fig: plt.Figure,
                           categorial_column: str = None, 
                           components: Tuple[int, int, int] = (1, 2, 3),
                           init_elev: int = None,
                           init_azim: int = None,
                           speed: int = 5,
                           filepath: str = './gif_3D_PCA.gif',
                           **kwargs) -> None:
    """
    Create a GIF from a 3D PCA plot of a DataFrame.
    
    Parameters:
    -----------
    df (pd.DataFrame):
        The DataFrame containing the data.
        
    categorial_column (str, optional):
        The name of the column to use for categorization. Defaults to None.
    
    components (Tuple[int, int, int], optional):
        The components to plot. Defaults to (1, 2, 3).
    
    init_elev (int, optional):
        The initial elevation for the plot. Defaults to None.

    init_azim (int, optional):
        The initial azimuth for the plot. Defaults to None.

    speed (int, optional): 
        The speed of the GIF. Defaults to 5.

    figsize (Tuple[int, int], optional):  
        The figsize of the figure (width, height) in inches. Defaults to (10, 8).

    filepath (str, optional):
        The filepath to save the GIF. Defaults to './gif_3D_PCA.gif'.

    **kwargs:
        Additional keyword arguments for the plot.

    Returns:
    --------
    None

    Examples:
    ---------
    >>> create_gif_from_3D_PCA(df, 'target', components=(1, 2, 3), init_elev=30, init_azim=30, speed=5, figsize=(10, 8), filepath='./gif_3D_PCA.gif')
    """

    assert(init_elev or init_azim), "init_elev or init_azim must be provided."

    X = df.drop(categorial_column, axis=1) if categorial_column else df
    y = df[categorial_column] if categorial_column else None

    pca = PCA(n_components=len(X.columns))
    pca_result = pca.fit_transform(X)
    column_names = [f'PC{i}' for i in range(1, len(X.columns) + 1)]
    df_pca = pd.DataFrame(data=pca_result, columns=column_names)

    if categorial_column:
        df_pca[categorial_column] = y


    ax = fig.add_subplot(111, projection='3d')

    name_column1, name_column2, name_column3 = f'PC{components[0]}', f'PC{components[1]}', f'PC{components[2]}'


    if categorial_column:
        categories = df_pca[categorial_column].unique()
        colors = [ColorGenerator.COLORS['BLUE'], ColorGenerator.COLORS['RED'], ColorGenerator.COLORS['ORANGE']] if categorial_column else None

        for category, color in zip(categories, colors):
            subset = df_pca[df_pca[categorial_column] == category]
            ax.scatter(subset[name_column1], subset[name_column2], subset[name_column3], c=color, label=category, **kwargs)

        ax.legend()
    else:
        ax.scatter(df_pca[name_column1], df_pca[name_column2], df_pca[name_column3], **kwargs)

    ax.set_xlabel(name_column1, fontweight='bold')
    ax.set_ylabel(name_column2, fontweight='bold')
    ax.set_zlabel(name_column3, fontweight='bold')

    def update(frame):
        if init_elev:
            ax.view_init(elev=init_elev, azim=frame)
        elif init_azim:
            ax.view_init(elev=frame, azim=init_azim)

    ani = FuncAnimation(fig, update, frames=range(0, 360, speed), interval=40)
    ani.save(filepath, writer='pillow', fps=80)


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
    
    get_colors(len(df_importance), kwargs)
    get_edgecolors(0.5, kwargs)

    color = kwargs['color'] if isinstance(kwargs['color'], str) else kwargs['color'][0]

    axgridx      = kwargs.pop('axgridx', True)
    axgridy      = kwargs.pop('axgridy', False)
    color_label  = kwargs.pop('color_label', ColorGenerator.adjust_color(color , 0.3))
    color_spine  = kwargs.pop('color_spine', ColorGenerator.adjust_color(color , 0.45))
    color_tick   = kwargs.pop('color_tick', ColorGenerator.adjust_color(color, 0.45))
    color_grid   = kwargs.pop('color_grid', ColorGenerator.adjust_color(color, -0.4))
    bold_max     = kwargs.pop('bold_max', True)
    italic_min   = kwargs.pop('italic_min', True)
    title_before = kwargs.pop('title_before', '')
    title_after  = kwargs.pop('title_after', '')
    title        = kwargs.pop('title', 'Features importances')
    to_dict      = kwargs.pop('to_dict', False)
    filepath     = kwargs.pop('filepath', None)

    sns.barplot(y='Features', x='Importances', 
                data=df_importance, palette=kwargs.pop('color'), 
                hue='Importances', legend=False, **kwargs)
    
    plt.title(title_before + title + title_after, fontweight='bold')
    plt.xlabel('Importances', color=color_label, fontweight='bold')
    plt.ylabel('Features', color=color_label, fontweight='bold')

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



def plot_temporal_serie(df: pd.DataFrame, 
                        column_to_plot: str, 
                        column_timestamp: str, 
                        day: str = None,
                        highlights: Union[List[Tuple[str, str, Tuple[float, float]]], 
                                          List[Tuple[str, str, List[float]]]]  = None,
                        scatter_max: bool = False,
                        scatter_min: bool = False,
                        filepath=None, 
                        **kwargs) -> None:
    """
    Plot a temporal serie of a specified column in a DataFrame with customizable options.

    Parameters:
    -----------
    df (pd.DataFrame):
        The DataFrame containing the data.
    
    column_to_plot (str):
        The name of the column to plot.

    column_timestamp (str):
        The name of the column containing the timestamp.

    day (str, optional):
        The day to plot. Defaults to None.

    highlights (List[Tuple[str, str, Tuple[float, float]]] or 
                List[Tuple[str, str, List[float]]], optional):
        A list of tuples containing the label, color, and values to highlight. Defaults to None.
        If the values are a tuple, the first and second values will be used as the beginning and
        end of the highlight, respectively. If the values are a list, the minimum and maximum.

    scatter_max (bool, optional):
        If True, plot a scatter plot with the maximum value highlighted. Defaults to False.

    scatter_min (bool, optional):
        If True, plot a scatter plot with the minimum value highlighted. Defaults to False.

    filepath (str, optional):
        The file path to save the plot as an image. If provided, the figure will be saved as a PNG file.
        Defaults to None.

    **kwargs:
        Additional keyword arguments for customization (e.g., alpha, edgecolor, bins, color, etc.).

    Returns:    
    --------
    None

    This function plots a temporal serie for the specified column in the DataFrame. It provides options
    to customize the appearance of the plot, such as figure figsize, colors, and transparency. If a file
    path is provided, the plot will be saved as an image in PNG format.

    Examples:
    ---------
    >>> plot_temporal_serie(df, 'heart_rate', 'timestamp', day='2021-01-01', 
                            highlights=[('Sports', ColorGenerator.COLORS['BLUE'], (17, 18)), ('Exam', ColorGenerator.COLORS['RED'], (13, 15))], 
                            scatter_max=True, scatter_min=True, color= ColorGenerator.COLORS['BLUE'], linewidth=2)
    >>> plot_temporal_serie(df, 'heart_rate', 'timestamp', day='2021-01-01', 
                            highlights=[('EVA', ColorGenerator.COLORS['PURPLE'], [6, 6.1, 6.2, 6.3, ..., 8])], 
                            scatter_max=True, scatter_min=True)
    """
    df[column_timestamp] = pd.to_datetime(df[column_timestamp])

    df_day = df.copy() if not day else df[df[column_timestamp].dt.date == parser.parse(day).date()]
    time = df_day[column_timestamp].dt.hour + df_day[column_timestamp].dt.minute / 60

    plt.plot(time, df_day[column_to_plot], **kwargs) if 'color' in kwargs else plt.plot(time, df_day[column_to_plot], c=ColorGenerator.COLORS['BLUE'], **kwargs)

    if highlights:
        for label, color, values in highlights:
            if len(values) == 0:
                continue
            beg, end = (values[0], values[1]) if type(values) == tuple else (min(values), max(values))
            plt.axvspan(beg, end, color=color, alpha=0.3, label=label)
        plt.legend()

    if scatter_max:
        scatter_color = ColorGenerator.COLORS['RED'] if  kwargs.get('color') != ColorGenerator.COLORS['RED'] else ColorGenerator.COLORS['BLUE']
        max_value = df_day[column_to_plot].max()
        max_index = df_day[column_to_plot].idxmax()
        plt.scatter(time[max_index], max_value, c=scatter_color, s=30, marker='o')
        plt.text(time[max_index] + 0.35, max_value, f'{max_value:.0f}', color=scatter_color, fontweight='bold')

    if scatter_min:
        scatter_color = ColorGenerator.COLORS['BLUE'] if  kwargs.get('color') != ColorGenerator.COLORS['BLUE'] else ColorGenerator.COLORS['GREEN']
        min_value = df_day[column_to_plot].min()
        min_index = df_day[column_to_plot].idxmin()
        plt.scatter(time[min_index], min_value, c=scatter_color, s=30, marker='o')
        plt.text(time[min_index] + 0.35, min_value, f'{min_value:.0f}', color=scatter_color, fontstyle='italic')

    column_to_plot_name = column_to_plot.replace('_', ' ').title()

    plt.title(f'{column_to_plot_name} vs Time', fontweight='bold')
    plt.xlabel('Time (hours)', fontweight='bold')
    plt.ylabel(f'{column_to_plot_name}', fontweight='bold')

    plt.xticks(np.arange(0, 24, 1), rotation=45)

    plt.grid(True)

    if filepath:
        save_plot(filepath)

    plt.show()



def get_extremum_from_temporal_serie(df: pd.DataFrame,
                                     column_to_get_extremum: str,
                                     column_timestamp: str,
                                     day: str = None,
                                     max: bool = True) -> Tuple[Union[int, float], List[pd.Timestamp]]:
    
    df_day = df.copy() if not day else df[df[column_timestamp].dt.date == pd.to_datetime(day).date()]

    extremum_value = df_day[column_to_get_extremum].max() if max else df_day[column_to_get_extremum].min()
    extremum_timestamps = df_day.loc[df_day[column_to_get_extremum] == extremum_value, column_timestamp].tolist()

    return extremum_value, extremum_timestamps

def get_extremum_from_temporal_serie(df: pd.DataFrame,
                                     column_to_get_extremum: str,
                                     column_timestamp: str,
                                     day: str = None,
                                     max: bool = True) -> Tuple[Union[int, float], List[pd.Timestamp]]:
    
    """
    Get the maximum or minimum value and timestamps from a specified column in a DataFrame.
    
    Parameters:
    -----------
    df (pd.DataFrame):
        The DataFrame containing the data.
    
    column_to_get_extremum (str):
        The name of the column to get the extremum.
    
    column_timestamp (str):
        The name of the column containing the timestamp.
    
    day (str, optional):
        The day to plot. Defaults to None.
        
    max (bool, optional):
        If True, get the maximum value and timestamps, if False, get the minimum value and timestamps. Defaults to True.
    
    Returns:
    --------
    Tuple[Union[int, float], List[pd.Timestamp]]:
        A tuple containing the maximum or minimum value and the corresponding timestamps.
        
    Examples:
    ---------
    >>> get_extremum_from_temporal_serie(df, 'heart_rate', 'timestamp', day='2021-01-01', max=True)
    """
    
    df_day = df.copy() if not day else df[df[column_timestamp].dt.date == pd.to_datetime(day).date()]

    extremum_value = df_day[column_to_get_extremum].max() if max else df_day[column_to_get_extremum].min()
    extremum_timestamps = df_day.loc[df_day[column_to_get_extremum] == extremum_value, column_timestamp].tolist()

    return extremum_value, extremum_timestamps


def get_timestamps_above_threshold(df: pd.DataFrame,
                                   column_to_check: str,
                                   column_timestamp: str,
                                   day: str = None,
                                   threshold: Union[int, float] = 0.0) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    
    """
    Get the timestamps where a specified column is above a specified threshold.
    
    Parameters:
    -----------
    df (pd.DataFrame):
        The DataFrame containing the data.
    
    column_to_check (str):
        The name of the column to check.
        
    column_timestamp (str):
        The name of the column containing the timestamp.
        
    day (str, optional):
        The day to plot. Defaults to None.
    
    threshold (Union[int, float], optional):
        The threshold. Defaults to 0.0.
        
    Returns:
    --------
    List[Tuple[pd.Timestamp, pd.Timestamp]]:
        A list of tuples containing the start and end timestamps where the column is above the threshold.
        
    Examples:
    ---------
    >>> get_timestamps_above_threshold(df, 'heart_rate', 'timestamp', day='2021-01-01', threshold=100)
    """
    
    df_day = df.copy() if not day else df[df[column_timestamp].dt.date == pd.to_datetime(day).date()]

    timestamps_above_threshold = []

    start_timestamp = None
    end_timestamp = None

    for index, row in df_day.iterrows():
        if row[column_to_check] > threshold:
            if start_timestamp is None:
                start_timestamp = row[column_timestamp]
        else:
            if start_timestamp is not None:
                end_timestamp = row[column_timestamp]
                timestamps_above_threshold.append((start_timestamp, end_timestamp))
                start_timestamp = None
                end_timestamp = None

    if start_timestamp is not None:
        end_timestamp = df_day.iloc[-1][column_timestamp]
        timestamps_above_threshold.append((start_timestamp, end_timestamp))

    return timestamps_above_threshold

def plot_classification(ax: plt.Axes,
                        column_x: Union[np.ndarray, torch.Tensor],
                        column_y: Union[np.ndarray, torch.Tensor],
                        column_x_test: Union[np.ndarray, torch.Tensor] = None,
                        column_y_test: Union[np.ndarray, torch.Tensor] = None,
                        y_pred: Union[np.ndarray, torch.Tensor] = None,
                        **kwargs) -> None:
    
    """
    Plot a classification of two columns with customizable options.

    Parameters:
    -----------
    ax (plt.Axes):
        The Axes object to plot the data.

    column_x (Union[np.ndarray, torch.Tensor]):
        The values of the first column.

    column_y (Union[np.ndarray, torch.Tensor]):
        The values of the second column.

    column_x_test (Union[np.ndarray, torch.Tensor], optional):
        The values of the first column for the test set. Defaults to None.

    column_y_test (Union[np.ndarray, torch.Tensor], optional):
        The values of the second column for the test set. Defaults to None.

    y_pred (Union[np.ndarray, torch.Tensor], optional):
        The predicted values. Defaults to None.

    **kwargs:
        Additional keyword arguments for customization (e.g., color, edgecolor, marker, alpha, etc.).

    Returns:
    --------
    None

    This function plots a binary classification of two columns in a DataFrame with customizable options.
    If the test set is provided, it plots the test set as well. If the predicted values are provided, it plots
    the predicted values. The plot can be customized with additional keyword arguments.

    Examples:
    ---------
    >>> plot_classification(column_x, column_y, ax, 
                                   column_x_test, column_y_test, 
                                   y_pred, color=('blue', 'red'), color_test=('cyan', 'salmon'), color_pred='auto',
                                   edgecolor='black', label_class=('Class 1', 'Class 2'), title='Binary Classification',
                                   marker='o', marker_test='x', marker_pred='^', 
                                   alpha=0.7, alpha_pred=0.8, fraction=0.8, fraction_test=0.95)
    """

    NB_CLASSES = len(np.unique(column_y))
    LIST_COLORS = [ColorGenerator.COLORS['BLUE'], ColorGenerator.COLORS['RED'], ColorGenerator.COLORS['ORANGE'],
                   ColorGenerator.COLORS['GREEN'],  ColorGenerator.COLORS['CYAN'], ColorGenerator.COLORS['SALMON'],  
                   ColorGenerator.COLORS['PURPLE'], ColorGenerator.COLORS['YELLOW'], ColorGenerator.COLORS['PINK']] * (NB_CLASSES // 9 + 1)
    
    X_0 = [column_x[column_y == i][:, 0] for i in range(NB_CLASSES)]
    X_1 = [column_x[column_y == i][:, 1] for i in range(NB_CLASSES)]

    if column_x_test is not None:
        assert column_y_test is not None, 'column_y_test must be provided if column_x_test is provided.'
    if column_y_test is not None:
        assert column_x_test is not None, 'column_x_test must be provided if column_y_test is provided.'

    # Convert torch.Tensor to np.ndarray if necessary (with CPU conversion)
    column_x      = column_x.cpu().numpy() if isinstance(column_x, torch.Tensor) else column_x
    column_y      = column_y.cpu().numpy() if isinstance(column_y, torch.Tensor) else column_y
    column_x_test = column_x_test.cpu().numpy() if column_x_test is not None and isinstance(column_x_test, torch.Tensor) else column_x_test
    column_y_test = column_y_test.cpu().numpy() if column_y_test is not None and isinstance(column_y_test, torch.Tensor) else column_y_test
    y_pred        = y_pred.cpu().numpy() if y_pred is not None and isinstance(y_pred, torch.Tensor) else y_pred

    colors     = kwargs.pop('color', [LIST_COLORS[i] for i in range(NB_CLASSES)])
    colors    += LIST_COLORS if len(colors) < NB_CLASSES else []
    color_test = kwargs.pop('color_test', [ColorGenerator.adjust_color(color, -0.2) for color in colors])
    color_pred = kwargs.pop('color_pred', 'auto')
    
    edgefactor = kwargs.pop('edgefactor', 0.5)

    def get_edgecolors(edgecolor: str, kwargs: Dict[str, Any]) -> Tuple[str, str]:
        if edgecolor in kwargs:
            edgecolor = kwargs.pop(edgecolor)
            if isinstance(edgecolor, tuple) and len(edgecolor) == NB_CLASSES:
                return edgecolor
            else:
                return [edgecolor] * NB_CLASSES
        else:
            return [adjust_color(color, edgefactor) for color in colors]
        
    edgecolor = get_edgecolors('edgecolor', kwargs)
    edgecolor_test = get_edgecolors('edgecolor_test', kwargs)

    axgridx                    = kwargs.pop('axgridx', True)
    axgridy                    = kwargs.pop('axgridy', True)
    color_spine                = kwargs.pop('color_spine', 'black')
    color_tick                 = kwargs.pop('color_tick', 'black')
    color_grid                 = kwargs.pop('color_grid', 'lightgrey')
    label_class                = kwargs.pop('label_class', [f'Class {i}' for i in range(NB_CLASSES)])
    title                      = kwargs.pop('title', f'Multi Class Classification' if NB_CLASSES > 2 else f'Binary Classification')
    filepath                   = kwargs.pop('filepath', None)
    legend_facecolor           = kwargs.pop('legend_facecolor', 'white')
    legend_edgecolor           = kwargs.pop('legend_edgecolor', 'grey')
    legend_framealpha          = kwargs.pop('legend_framealpha', 0.5)
    marker                     = kwargs.pop('marker', 'o')
    marker_test                = kwargs.pop('marker_test', 'x')
    marker_pred                = kwargs.pop('marker_pred', '^')
    alpha                      = kwargs.pop('alpha', 0.7)
    alpha_test                 = kwargs.pop('alpha_test', alpha)
    alpha_pred                 = kwargs.pop('alpha_pred', 0.95)
    alpha_contourf             = kwargs.pop('alpha_contour', 0.2)
    fraction                   = kwargs.pop('fraction', 1.)
    fraction_test              = kwargs.pop('fraction_test', 1.)
    decision_boundary          = kwargs.pop('decision_boundary', False)
    model                      = kwargs.pop('model', None)

    if decision_boundary and model is None:
        warnings.warn('model must be provided if decision_boundary is True. Decision boundary will not be plotted.')

    if 'contour_width' in kwargs and not decision_boundary:
        warnings.warn('Decision boundary is not enabled. contour_width will not be used.')
    contour_width = kwargs.pop('contour_width', 0.5)
    
    if 'contour_resolution' in kwargs and not decision_boundary:
        warnings.warn('Decision boundary is not enabled. contour_resolution will not be used.')
    contour_resolution = kwargs.pop('contour_resolution', 0.05)

    if 'contour_color' in kwargs and not decision_boundary:
        warnings.warn('Decision boundary is not enabled. contour_color will not be used.')
    contour_color = kwargs.pop('contour_color', 'black')

    customize_plot_colors(ax, axgridx=axgridx, axgridy=axgridy, color_grid=color_grid,
                          color_spine=color_spine, color_tick=color_tick)
    
    plt.title(title, fontweight='bold')

    name_label = label_class 

    if column_x_test is not None and column_y_test is not None:
        X_0_test = [column_x_test[column_y_test == i][:, 0] for i in range(NB_CLASSES)]
        X_1_test = [column_x_test[column_y_test == i][:, 1] for i in range(NB_CLASSES)]

        warnings.filterwarnings("ignore", message="You passed a edgecolor/edgecolors*")

        if fraction_test < 1.0:
            X_0_test = [X_0_test[i][:int(fraction_test * len(X_0_test[i]))] for i in range(NB_CLASSES)]
            X_1_test = [X_1_test[i][:int(fraction_test * len(X_1_test[i]))] for i in range(NB_CLASSES)]
    
        for i in range(NB_CLASSES):
            ax.scatter(X_0_test[i], X_1_test[i], c=color_test[i], edgecolors=edgecolor_test[i], marker=marker_test, alpha=alpha_test, label=f'{name_label[i]} Test', **kwargs)

        name_label = [f'{name_label[i]} Train' for i in range(NB_CLASSES)]

    if fraction < 1.0:
        X_0 = [X_0[i][:int(fraction * len(X_0[i]))] for i in range(NB_CLASSES)]
        X_1 = [X_1[i][:int(fraction * len(X_1[i]))] for i in range(NB_CLASSES)]

    for i in range(NB_CLASSES):
        ax.scatter(X_0[i], X_1[i], c=colors[i], edgecolors=edgecolor[i], marker=marker, alpha=alpha, label=name_label[i], **kwargs)

    if y_pred is not None:
        for i in range(len(y_pred)):
            if y_pred[i] != column_y_test[i]:
                color_pred_select = color_test[int(column_y_test[i])] if color_pred == 'auto' else color_pred
                edgecolor_pred_select = color_test[int(y_pred[i])]
                
                plt.scatter(column_x_test[i, 0], column_x_test[i, 1], 
                            label= f'Misclassified {int(y_pred[i])} as {int(column_y_test[i])}', 
                            color=color_pred_select, linewidths=2.2, edgecolor=edgecolor_pred_select, 
                            marker=marker_pred, s=200, alpha=alpha_pred, **kwargs)
    
    if decision_boundary and model:
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        xx, yy = np.meshgrid(np.arange(x_min, x_max, contour_resolution), np.arange(y_min, y_max, contour_resolution))
        grid_tensor = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
        with torch.inference_mode():
            Z = model(grid_tensor)
            Z = torch.argmax(Z, dim=1).numpy()
            Z = Z.reshape(xx.shape)

        colors_rgb = [mcolors.hex2color(color) for color in colors]
        cmap = mcolors.LinearSegmentedColormap.from_list('CustomMap', colors_rgb, N=len(colors))
        ax.contourf(xx, yy, Z, alpha=alpha_contourf, cmap=cmap)
        ax.contour(xx, yy, Z, colors=contour_color, linewidths=contour_width)

    # Keep only unique legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), facecolor=legend_facecolor, edgecolor=legend_edgecolor, framealpha=legend_framealpha)

    if filepath:
        save_plot(filepath)


def plot_losses(ax: plt.Axes,
                training_loss: List[int],
                test_loss: List[int] = None,
                **kwargs) -> None:
    
    """
    Plot the training and test losses.

    Parameters:
    -----------
    ax (plt.Axes):
        The Axes object to plot the data.

    training_loss (List[int]):
        The training loss.

    test_loss (List[int], optional):
        The test loss. Defaults to None.

    **kwargs:
        Additional keyword arguments for customization (e.g., color, edgecolor, linewidth, etc.).

    Returns:
    --------
    None

    This function plots the training and test losses. It provides options to customize the appearance of the plot,
    such as colors, line width, and transparency.

    """

    n_epochs = len(training_loss)

    color = kwargs.pop('color', ColorGenerator.COLORS['CYAN'])
    color_test = kwargs.pop('color_test', ColorGenerator.COLORS['SALMON'])

    axgridx                    = kwargs.pop('axgridx', True)
    axgridy                    = kwargs.pop('axgridy', True)
    color_spine                = kwargs.pop('color_spine', 'black')
    color_tick                 = kwargs.pop('color_tick', 'black')
    color_grid                 = kwargs.pop('color_grid', 'lightgrey')
    dataset_name               = kwargs.pop('dataset_name', None)
    title                      = kwargs.pop('title', ('Training and Test Loss' if test_loss else 'Training Loss') + (f' of dataset: {dataset_name}' if dataset_name else ''))
    filepath                   = kwargs.pop('filepath', None)
    legend_facecolor           = kwargs.pop('legend_facecolor', 'white')
    legend_edgecolor           = kwargs.pop('legend_edgecolor', 'grey')
    legend_framealpha          = kwargs.pop('legend_framealpha', 0.5)
    y_log                      = kwargs.pop('y_log', False)
    min_loss                   = kwargs.pop('min_loss', True)
    min_loss_test              = kwargs.pop('min_loss_test', True)

    customize_plot_colors(ax, axgridx=axgridx, axgridy=axgridy, color_grid=color_grid,
                          color_spine=color_spine, color_tick=color_tick)
    
    plt.title(title, fontweight='bold')
    plt.xlabel('Epoch', fontweight='bold')
    plt.ylabel('Loss' + (' (Log Scale)' if y_log else ''), fontweight='bold')

    if y_log:
        plt.yscale('log')
    
    plt.plot(range(1, n_epochs + 1), training_loss, color=color, label='Training Loss', **kwargs)
    if min_loss:
        min_loss_index = np.argmin(training_loss)
        plt.scatter(min_loss_index + 1, training_loss[min_loss_index], c=color, s=50, marker='o')
    if test_loss:
        plt.plot(range(1, n_epochs + 1), test_loss, color=color_test, label='Test Loss', **kwargs)
        if min_loss_test:
            min_loss_test_index = np.argmin(test_loss)
            plt.axvline(min_loss_test_index + 1, color=color_test, linestyle='--', **kwargs)
            plt.scatter(min_loss_test_index + 1, test_loss[min_loss_test_index], c=color_test, s=50, marker='o')
            # Add the epoch with the minimum test loss within the plot (at the top next to the line)
            plt.text(min_loss_test_index + n_epochs * 0.01, test_loss[min_loss_test_index] + max(test_loss) * 0.05, f'{min_loss_test_index + 1}', color=color_test, fontweight='bold', fontsize=10)

    plt.legend(facecolor=legend_facecolor, edgecolor=legend_edgecolor, framealpha=legend_framealpha)

    if filepath:
        save_plot(filepath)


# def plot_groupby(ax: plt.Axes,
#                  dataframe: pd.DataFrame, 
#                  group: str,
#                  target: str,
#                  frequency: bool = False,
#                  **kwargs) -> pd.DataFrame:
    
#     """
#     Plot the distribution of a group and target column in a DataFrame.

#     Parameters:
#     -----------
#     ax (plt.Axes):
#         The Axes object to plot the data.

#     dataframe (pd.DataFrame):
#         The DataFrame containing the data.

#     group (str):
#         The name of the column to group by.

#     target (str):
#         The name of the target column.

#     frequency (bool, optional):
#         If True, get the frequency distribution. If False (default), get the population distribution.

#     **kwargs:
#         Additional keyword arguments for customization (e.g., color, edgecolor, linewidth, etc.).

#     Returns:
#     --------
#     pd.DataFrame:
#         A DataFrame containing the distribution of the group and target columns.

#     This function plots the distribution of a group and target column in a DataFrame. It provides options to customize
#     the appearance of the plot, such as colors, edge color, and line width.
    
#     Examples:
#     ---------
#     >>> plot_groupby(ax, df, 'category', 'target', frequency=True, color='viridis', edgecolor='black', linewidth=1.5)
#     >>> plot_groupby(ax, df, 'category', 'target', frequency=False, color=['red', 'cyan', 'orange'], edgecolor='auto', linewidth=1.5, title_before='Train ')
#     """
    
#     result = dataframe.groupby([group, target]).size().unstack()
#     result = result.reset_index().rename(columns={'index': group})

#     columns_to_sum = [col for col in result.columns if col not in [group, target]]
#     df = result.copy()

#     df['Total'] = df[columns_to_sum].sum(axis=1)

#     if frequency:
#         for i in range(df.shape[0]):
#             for col in columns_to_sum:
#                 percentage = df.loc[i, col] / df.loc[i, 'Total'] * 100
#                 df.loc[i, col] = percentage

#     style = True if kwargs.pop('style', 'bar').lower() == 'multibar' else False

#     if style:
#         result = pd.melt(result, id_vars=group, var_name=target, value_name='Population')

#     graphcolor = kwargs.pop('graph_color', '#000000')

#     if 'color' not in kwargs:
#         kwargs['color'] = 'inferno'
#     get_colors(len(df.columns) - 2, kwargs)
#     edgecolor_chosen = kwargs.pop('edgecolor', 'auto')

#     axgridx           = kwargs.pop('axgridx', False)
#     axgridy           = kwargs.pop('axgridy', True)
#     color_label       = kwargs.pop('color_label', ColorGenerator.adjust_color(graphcolor, 0.3))
#     color_spine       = kwargs.pop('color_spine', ColorGenerator.adjust_color(graphcolor, 0.45))
#     color_tick        = kwargs.pop('color_tick', ColorGenerator.adjust_color(graphcolor, 0.45))
#     color_grid        = kwargs.pop('color_grid', ColorGenerator.adjust_color(graphcolor, -0.4))
#     legend_facecolor  = kwargs.pop('legend_facecolor', 'white')
#     legend_edgecolor  = kwargs.pop('legend_edgecolor', 'grey')
#     legend_framealpha = kwargs.pop('legend_framealpha', 0.5)
#     percentage_label  = kwargs.pop('percentage_label', 18)
#     title             = kwargs.pop('title', f'Distribution of {group.capitalize()} and {target.capitalize()}')
#     title_before      = kwargs.pop('title_before', '')
#     title_after       = kwargs.pop('title_after', '')
#     filepath          = kwargs.pop('filepath', None)

#     kwargs['linewidth'] = kwargs.get('linewidth', 1.8)
#     kwargs['alpha'] = kwargs.get('alpha', 0.9)

#     customize_plot_colors(ax, axgridx=axgridx, axgridy=axgridy, color_grid=color_grid,
#                           color_spine=color_spine, color_tick=color_tick)


#     if style:
#         sns.barplot(data=result, x=group, 
#                     y='Population', palette=kwargs.pop('color'), 
#                     hue=target, dodge=True, saturation=0.75, **kwargs)

#         for i, patch in enumerate(ax.patches):
#             edgecolor = edgecolor_chosen if edgecolor_chosen != 'auto' else ColorGenerator.adjust_color(patch.get_facecolor(), 0.5)
#             patch.set_edgecolor(edgecolor)

#         add_value_labels(ax, '#000000', frequency=frequency, percentage=percentage_label)
        

#     else:
#         col = kwargs.pop('color')
#         for _, row in df.iterrows():
#             bottom = 0
#             for i, category in enumerate(df.columns[1:-1]):
#                 plt.bar(row[group], row[category], bottom=bottom, label=category, color=col[i], **kwargs)
#                 bottom += row[category]

#         plt.ylim(0, max(df['Total']) * 1.06)

        

#     plt.title(title_before + title + title_after, fontweight='bold')
#     plt.xlabel(group.capitalize(), color=color_label, fontsize=11)
#     plt.ylabel('Population')

#     handles, labels = plt.gca().get_legend_handles_labels()
#     by_label = dict(zip(labels, handles))
#     plt.legend(by_label.values(), by_label.keys(), facecolor=legend_facecolor, edgecolor=legend_edgecolor, framealpha=legend_framealpha, title=target)
#     plt.tight_layout()

#     if filepath:
#         save_plot(filepath)

#     return df

def plot_confusion_matrix_heatmap(ax: plt.Axes,
                                  y_true: Union[np.ndarray, torch.Tensor],
                                  y_pred: Union[np.ndarray, torch.Tensor],
                                  target_names: List[str] = None,
                                  **kwargs) -> np.ndarray:
    
    """
    Plot a confusion matrix heatmap.

    Parameters:
    -----------
    ax (plt.Axes):
        The plot's axes.

    y_true (np.ndarray, torch.Tensor):
        The true labels.

    y_pred (np.ndarray, torch.Tensor):
        The predicted labels.

    target_names (List[str], optional):
        The names of the target classes. Defaults to None.

    **kwargs:
        Additional keyword arguments for customization (e.g., color, alpha, etc.).

    Returns:
    --------
    np.ndarray:
        The confusion matrix.

    This function plots a confusion matrix heatmap.
    If a file path is provided, the plot will be saved as an image in PNG format.

    Examples:
    ---------
    >>> plot_confusion_matrix_heatmap(ax, y_true, y_pred, target_names=['A', 'B', 'C'], color='Blues', alpha=0.8, linewidth=0.1)
    """
    
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()

    cm = confusion_matrix(y_true, y_pred)

    if target_names is None:
        target_names = np.unique(y_true)

    graphcolor = kwargs.pop('graph_color', '#000000')

    axgridx           = kwargs.pop('axgridx', False)
    axgridy           = kwargs.pop('axgridy', False)
    color_spine       = kwargs.pop('color_spine', ColorGenerator.adjust_color(graphcolor, 0.45))
    color_tick        = kwargs.pop('color_tick', ColorGenerator.adjust_color(graphcolor, 0.45))
    color_grid        = kwargs.pop('color_grid', None)
    title_before      = kwargs.pop('title_before', '')
    title_after    = kwargs.pop('title_after', '')
    title             = kwargs.pop('title', 'Confusion Matrix')
    filepath          = kwargs.pop('filepath', None)

    kwargs['linewidth'] = kwargs.get('linewidth', 0.1)
    kwargs['linecolor'] = kwargs.get('linecolor', 'black')
    kwargs['alpha']     = kwargs.get('alpha', 0.9)
    kwargs['cmap']      = kwargs.get('cmap', 'Blues')
    kwargs['cbar']      = kwargs.get('cbar', False)

    customize_plot_colors(ax, axgridx=axgridx, axgridy=axgridy, color_grid=color_grid,
                          color_spine=color_spine, color_tick=color_tick)
    
    sns.heatmap(cm, annot=True, fmt='d', 
                xticklabels=target_names, 
                yticklabels=target_names, 
                **kwargs)
    
    plt.title(title_before + title + title_after, fontweight='bold')
    plt.xlabel('Predicted', fontweight='bold')
    plt.ylabel('Actual', fontweight='bold')

    if filepath:
        save_plot(filepath)

    return cm

def plot_correlation_matrix_heatmap(ax: plt.Axes,
                                    dataframe: pd.DataFrame,
                                    **kwargs) -> pd.DataFrame:
    
    """
    Plot a correlation matrix heatmap.

    Parameters:
    -----------
    ax (plt.Axes):
        The plot's axes.

    dataframe (pd.DataFrame):
        The DataFrame containing the data.

    **kwargs:
        Additional keyword arguments for customization (e.g., color, alpha, etc.).

    Returns:
    --------
    pd.DataFrame:
        The correlation matrix.

    This function plots a correlation matrix heatmap.
    If a file path is provided, the plot will be saved as an image in PNG format.

    Examples:
    ---------
    >>> plot_correlation_matrix_heatmap(ax, df, color='coolwarm', alpha=0.8, linewidth=0.1)
    """
    
    graphcolor = kwargs.pop('graph_color', '#000000')

    axgridx           = kwargs.pop('axgridx', False)
    axgridy           = kwargs.pop('axgridy', False)
    color_spine       = kwargs.pop('color_spine', ColorGenerator.adjust_color(graphcolor, 0.45))
    color_tick        = kwargs.pop('color_tick', ColorGenerator.adjust_color(graphcolor, 0.45))
    color_grid        = kwargs.pop('color_grid', None)
    title_before      = kwargs.pop('title_before', '')
    title_after       = kwargs.pop('title_after', '')
    title             = kwargs.pop('title', 'Correlation Matrix')
    filepath          = kwargs.pop('filepath', None)

    kwargs['linewidth'] = kwargs.get('linewidth', 0.1)
    kwargs['linecolor'] = kwargs.get('linecolor', 'black')
    kwargs['alpha']     = kwargs.get('alpha', 0.9)
    kwargs['cmap']      = kwargs.get('cmap', 'coolwarm')
    kwargs['cbar']      = kwargs.get('cbar', False)

    customize_plot_colors(ax, axgridx=axgridx, axgridy=axgridy, color_grid=color_grid,
                          color_spine=color_spine, color_tick=color_tick)
    
    corr = dataframe.corr()
    
    sns.heatmap(corr, annot=True, fmt='.2f', **kwargs)
    
    plt.title(title_before + title + title_after, fontweight='bold')

    if filepath:
        save_plot(filepath)

    return corr