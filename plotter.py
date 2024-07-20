import warnings
import webcolors
import torch

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import FixedLocator
from matplotlib import colormaps

from typing import Any, Dict, List, Tuple, Union


DEFAULT_PARAMETER = {
    # General appearance
    'graphcolor': '#000000',
    'rotation': 0,
    'horizontal': False,
    'linewidth': 0,

    # Pie chart settings
    'size_circle_pie': 0.75,
    'startangle': 0,
    'donnut_style': False,
    'donnut_size': 0.7,
    'donnut_style_edgecolor': 'lightgray',
    'annot_distance': 0.5,
    'label_distance': 1.1,
    'explode': None,
    'shadow': False,
    
    # Title settings
    'title': '',
    'title_before': '',
    'title_after': '',
    'title_fontsize': 14,
    'title_bold': True,
    'title_italic': False,

    # Axis labels
    'x_label_title': '',
    'y_label_title': '',
    'x_label_fontsize': 11,
    'y_label_fontsize': 11,
    'x_label_bold': True,
    'y_label_bold': True,
    'x_label_italic': False,
    'y_label_italic': False,

    # Ticks settings
    'x_ticks_shorten': None,
    'y_ticks_shorten': None,
    'xticks_sep': 'auto',
    'yticks_sep': 'auto',

    # Grid settings
    'axgridx': False,
    'axgridy': True,
    'linestylex': '--',
    'linestyley': '--',

    # Colors settings
    'set_edgecolor': True,
    'customize_plot_colors': True,
    'edgecolor': 'auto',
    'edgefactor': 0.6,
    'color_label': 'auto',
    'color_spine': 'auto',
    'color_tick': 'auto',
    'color_grid': 'auto',
    'color_annot': 'matching',
    'color_multibar': ('deepskyblue', 'blueviolet'),

    # Annotations settings
    'add_value_annot': True,
    'annot_percentage': 8,
    'annot_fontsize': 8,
    'annot_outside': False,
    'annot_bold': True,
    'annot_italic': False,
    'annot_0': False,

    # Other settings
    'frequency': False,
    'filepath': None,
    'labels': None
}

class ColorGenerator:
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
               base_color: str = 'dodgerblue') -> None:
    """
    Get a list of colors based on the specified keyword arguments.

    Parameters:
    -----------
    size (int):
        The number of colors to generate.

    kwargs (Dict[str, Any]):
        The dictionary of keyword arguments.

    base_color (str, optional):
        The base color to use when generating colors. Defaults to 'dodgerblue'.

    Returns:
    --------
    None

    Examples:
    ---------
    >>> get_colors(5, {'color': 'random'})
    >>> get_colors(5, {'color': 'random_cmap'})
    >>> get_colors(5, {'color': 'Blues'})
    >>> get_colors(5, {'color': 'blue'})
    >>> get_colors(5, {'color': '#FF0000'})
    >>> get_colors(5, {'color': ['red', 'blue', 'green']})
    """
    
    # Base color
    if 'color' not in kwargs:
        kwargs['color'] = [base_color] * size
    # Random color
    elif kwargs['color'] == 'random':
        kwargs['color'] = ColorGenerator().get_random_colors(size)
    # Random colormap
    elif 'random' in kwargs['color'] and ('cmap' in kwargs['color'] or 'palette' in kwargs['color'] or 'colormap' in kwargs['color']):
        list_cmap = list(colormaps)
        cmap = np.random.choice(list_cmap)
        kwargs['color'] = sns.color_palette(cmap, size)
    # Single hex color to multiple colors
    elif isinstance(kwargs['color'], str) and '#' in kwargs['color']:
        kwargs['color'] = [kwargs['color']] * size
    # Colors list (not enough colors) -> add default colors
    elif isinstance(kwargs['color'], (list, tuple)) and len(kwargs['color']) != size:
        warnings.warn(f"Number of colors provided ({len(kwargs['color'])}) does not match the number of unique values ({size}). Adding default colors.")
        kwargs['color'] += ColorGenerator().get_random_colors(size - len(kwargs['color']))
    # Colors list (enough colors)
    elif isinstance(kwargs['color'], (list, tuple)) and len(kwargs['color']) == size:
        pass
    # Color name or hex code
    else:
        # Palette name
        try:
            kwargs['color'] = sns.color_palette(kwargs['color'], size)
        except ValueError:
            # Matplotlib color name
            try:
                color_hex = webcolors.name_to_hex(kwargs['color'])
                kwargs['color'] = [color_hex] * size
            # Invalid color name or hex code -> use default color
            except ValueError:
                warnings.warn(f"Color '{kwargs['color']}' is not a valid color name or hex code. Using default color.")
                kwargs['color'] = [base_color] * size



def get_edgecolors(edgefactor: float,
                   params: Dict[str, Any]) -> Tuple[str, str]:
    params['edgecolor'] = 'auto'



def adjust_graph_color(params: Dict[str, Any],
                       factors: Dict[str, float] = {'color_label': 0.3, 
                                                    'color_spine': 0.45, 
                                                    'color_tick': 0.45, 
                                                    'color_grid': -0.4, 
                                                    'color_annot': 0.3}) -> None:
    """
    Adjust the color of the plot's spines, ticks, and grid.

    Parameters:
    -----------
    params (Dict[str, Any]):
        The dictionary of keyword arguments.

    factors (Dict[str, float], optional):
        The factors by which to adjust the colors. Defaults to {'color_label': 0.3, 'color_spine': 0.45, 'color_tick': 0.45, 'color_grid': -0.4, 'color_annot': 0.3}.

    Returns:
    --------
    None
    """

    adjust = lambda color, factor: ColorGenerator.adjust_color(params['graphcolor'], factor) if color == 'auto' else color

    params['color_label'] = adjust(params['color_label'], factors['color_label'])
    params['color_spine'] = adjust(params['color_spine'], factors['color_spine'])
    params['color_tick']  = adjust(params['color_tick'], factors['color_tick'])
    params['color_grid']  = adjust(params['color_grid'], factors['color_grid'])
    params['color_annot'] = adjust(params['color_annot'], factors['color_annot'])


def adjust_kwargs(kwargs: Dict[str, Any],
                  params: Dict[str, Any],
                  fill: bool = False,
                  linewidth: bool = False,
                  alpha: bool = False,
                  linecolor: bool = False,
                  cmap: bool = False,
                  cbar: bool = False) -> None:
    """
    Pop the specified keyword arguments from the kwargs.

    Parameters:
    -----------
    kwargs (Dict[str, Any]):
        The dictionary of keyword arguments.

    params (Dict[str, Any]):
        The dictionary of keyword arguments to pop.

    Returns:
    --------
    None
    """
    if fill:
        kwargs['fill']      = kwargs.get('fill', False)
    if linewidth:
        kwargs['linewidth'] = kwargs.get('linewidth', 1.8)
    if alpha:
        kwargs['alpha']     = kwargs.get('alpha', 0.9)
    if linecolor:
        kwargs['linecolor'] = kwargs.get('linecolor', 'black')
    if cmap:
        kwargs['cmap']      = kwargs.get('cmap', 'Blues')
    if cbar:
        kwargs['cbar']      = kwargs.get('cbar', False)

    for param in params:
        if param in kwargs:
            params[param] = kwargs.pop(param)



def save_plot(filepath: str, 
              **kwargs) -> None:
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
    if filepath is not None:
        if not filepath.endswith('.png'):
            filepath += '.png'
        plt.savefig(filepath, bbox_inches="tight", **kwargs)



def customize_plot_colors(ax: plt.Axes, 
                          axgridx: bool = False,
                          linestylex: str = '--',
                          axgridy: bool = False,
                          linestyley: str = '--',
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

    if linestylex not in ['-', '--', '-.', ':']:
        warnings.warn(f"Invalid linestyle x '{linestylex}'. Default linestyle '--' will be used.")
        linestylex = '--'

    if linestyley not in ['-', '--', '-.', ':']:
        warnings.warn(f"Invalid linestyle y '{linestyley}'. Default linestyle '--' will be used.")
        linestyley = '--'
    
    if color_grid is not None and not axgridx and not axgridy:
        warnings.warn("color_grid is provided but axgridx and axgridy are False. "
                      "Grid lines will not be shown.")
    
    if (axgridx or axgridy) and color_grid is None:
        color_grid = 'gray'
        warnings.warn("axgridx or axgridy is provided but color_grid is not. "
                      "Default color 'gray' will be used for grid lines.")
    
    if axgridx:
        ax.set_axisbelow(True)
        ax.grid(True, axis='x', linestyle=linestylex, alpha=0.6, color=color_grid)
    if axgridy:
        ax.set_axisbelow(True)
        ax.grid(True, axis='y', linestyle=linestyley, alpha=0.6, color=color_grid)
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
                     fontsize: int = 8,
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
    
    color = [ColorGenerator.adjust_color(p.get_facecolor(), 0.5) for p in ax.patches] if \
            color == 'matching' else \
            [color] * len(ax.patches)

    patch_data = [(p.get_x(), p.get_width(), p.get_height()) for p in ax.patches] if not horizontal else \
                 [(p.get_y(), p.get_height(), p.get_width()) for p in ax.patches]

    dx_text = -sum([p[2] for p in patch_data]) / len(patch_data) * percentage / 100 if not outside else \
               sum([p[2] for p in patch_data]) / len(patch_data) * percentage / 100


    if not horizontal:
        for i, (x, width, height) in enumerate(patch_data):
            ax.annotate(f'{height:.2f}%' if frequency else f'{height:.0f}',
                        (x + width / 2., height+dx_text), 
                        ha='center', va='center', fontsize=fontsize, fontweight='bold', 
                        color=color[i], xytext=(0, 5), textcoords='offset points')
    else:
        for i, (y, height, width) in enumerate(patch_data):
            ax.annotate(f'{width:.2f}%' if frequency else f'{width:.0f}',
                        (width+dx_text, y + height / 2.), 
                        ha='center', va='center', fontsize=fontsize, fontweight='bold', 
                        color=color[i], xytext=(5, 0), textcoords='offset points')
            


def final_graph_customization(ax: plt.Axes, 
                              params: Dict[str, Any]) -> None:
    """
    Customize the final appearance of the plot.

    Parameters:
    -----------
    ax (plt.Axes):
        The plot's axes.

    params (Dict[str, Any]):
        The dictionary of keyword arguments.

    Returns:
    --------
    None
    """
    shorten = lambda text, max_length: f"{text[:max_length//2]}...{text[-max_length//2:]}" if len(text) > max_length else text

    adjust_graph_color(params)

    if params.get('x_ticks_shorten') is not None:
        max_length = params['x_ticks_shorten'] if (isinstance(params['x_ticks_shorten'], int) or params['x_ticks_shorten'] == 'auto' or params['x_ticks_shorten'] == True) else 18
        xtick_labels = [shorten(label.get_text(), max_length) for label in ax.get_xticklabels()]
        ax.set_xticks(range(len(xtick_labels)))
        ax.set_xticklabels(xtick_labels)

    if params.get('y_ticks_shorten') is not None:
        max_length = params['y_ticks_shorten'] if (isinstance(params['y_ticks_shorten'], int) or params['y_ticks_shorten'] == 'auto' or params['y_ticks_shorten'] == True) else 18
        ytick_labels = [shorten(label.get_text(), max_length) for label in ax.get_yticklabels()]
        ax.set_yticks(range(len(ytick_labels)))
        ax.set_yticklabels(ytick_labels)

    if params.get('xticks_sep') and params['xticks_sep'] != 'auto':
        ax.xaxis.set_major_locator(plt.MultipleLocator(params['xticks_sep']))
    if params.get('yticks_sep') and params['yticks_sep'] != 'auto':
        ax.yaxis.set_major_locator(plt.MultipleLocator(params['yticks_sep']))

    if params.get('add_value_annot') and not isinstance(ax.patches[0], mpatches.Wedge):
        add_value_labels(ax, 
                         color=params.get('color_annot'), 
                         fontsize=params.get('annot_fontsize'),
                         frequency=params.get('frequency'), 
                         horizontal=params.get('horizontal'),
                         percentage=params.get('annot_percentage'),
                         outside=params.get('annot_outside'))
    
    if params.get('customize_plot_colors'):
        customize_plot_colors(ax, 
                              axgridx=params.get('axgridx'), 
                              linestylex=params.get('linestylex'),
                              axgridy=params.get('axgridy'), 
                              linestyley=params.get('linestyley'),
                              color_grid=params.get('color_grid'),
                              color_spine=params.get('color_spine'), 
                              color_tick=params.get('color_tick'))
        
    if params.get('set_edgecolor'):
        for patch in ax.patches:
            if params['edgecolor'] == 'auto':
                color_of_the_patch = patch.get_facecolor()
                patch.set_edgecolor(ColorGenerator.adjust_color(color_of_the_patch, params['edgefactor']))
                patch.set_linewidth(params['linewidth'])
    
    plt.title(params['title_before'] + params['title'] + params['title_after'], fontsize=params['title_fontsize'], fontweight='bold')
    plt.xlabel(params['x_label_title'], color=params['color_label'], fontsize=params['x_label_fontsize'], fontweight='bold' if params['x_label_bold'] else 'normal')
    plt.ylabel(params['y_label_title'], color=params['color_label'], fontsize=params['y_label_fontsize'], fontweight='bold' if params['y_label_bold'] else 'normal')

    rotation, ha = params['rotation'] if isinstance(params['rotation'], tuple) else (params['rotation'], 'center')
    plt.xticks(rotation=rotation, ha=ha)


def plot_pie_feature(ax: plt.Axes,
                     dataframe: pd.DataFrame,
                     column: str,
                     **kwargs) -> pd.DataFrame:
    
    """
    Plot a pie chart for a specified column in a DataFrame.

    Parameters:
    -----------
    ax (plt.Axes):
        The plot's axes.

    dataframe (pd.DataFrame):
        The DataFrame containing the data.

    column (str):
        The name of the column to plot.

    **kwargs:
        Additional keyword arguments for customization (e.g., colors, explode, shadow, etc.).

    Returns:
    --------
    pd.DataFrame:
        DataFrame containing the unique values, counts, and percentages of the specified column.
    """

    labels, counts = np.unique(dataframe[column], return_counts=True)
    percentages = counts / len(dataframe) * 100

    dict_df_count = {'Labels': labels, 'Counts': counts, 'Percentages': percentages}
    df_counts = pd.DataFrame(dict_df_count)

    get_colors(len(labels), kwargs)

    params = {'title': f'Pie Chart of {column.capitalize()}', 'size_circle_pie': 1.0}
    params = {**DEFAULT_PARAMETER, **params}

    adjust_kwargs(kwargs, params)

    params['labels'] = kwargs.pop('labels', labels)

    color = kwargs.pop('color')

    ax.pie(df_counts['Counts'], labels=params['labels'], autopct='%1.1f%%', startangle=params['startangle'],
           colors=color, explode=params['explode'], shadow=params['shadow'], 
           radius=params['size_circle_pie'], wedgeprops={'edgecolor': 'black'}, 
           textprops={'color': 'w', 'fontsize': params['annot_fontsize']},
           pctdistance=params['annot_distance'], labeldistance=params['label_distance'])  
    
    final_graph_customization(ax, params)

    if params['donnut_style']:
        centre_circle = plt.Circle((0, 0), params['donnut_size'], fc='white', 
                                   edgecolor=params['donnut_style_edgecolor'], linewidth=params['linewidth'])
        ax.add_artist(centre_circle)
    
    color_text = [item for item in color for _ in range(2)]

    for i, text in enumerate(ax.texts):
        text.set_color(ColorGenerator.adjust_color(color_text[i], params['edgefactor']))

    save_plot(params['filepath'])

    return df_counts



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

    get_colors(len(filtered_missing_info_df), kwargs)

    params = {'title': 'Missing Data', 'x_label_title': 'Feature', 'y_label_title': 'Frequency', 'percentage': True}
    params = {**DEFAULT_PARAMETER, **params}

    get_edgecolors(0.5, params)
    adjust_kwargs(kwargs, params)

    params['y_label_title'] = params.get('y_label_title', 'Frequency' if params['frequency'] else 'Counts')

    y_title = 'Percentage Missing' if params['frequency'] else 'Total Missing'

    sns.barplot(y=y_title, 
                x=filtered_missing_info_df.index, 
                data=filtered_missing_info_df, 
                palette=kwargs.pop('color'),
                hue=filtered_missing_info_df.index,
                **kwargs)

    if params['frequency']:
        ticks = ax.get_yticks()
        ax.yaxis.set_major_locator(FixedLocator(ticks))
        ax.set_yticklabels([f'{(tick):.2f}%' for tick in ticks])

    final_graph_customization(ax, params)

    save_plot(params['filepath'])

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
        'Group total': number_by_group
    })

    get_colors(len(df), kwargs)

    params = {'x_label_title': f'{group.capitalize()} Category'}
    params = {**DEFAULT_PARAMETER, **params}

    get_edgecolors(0.5, params)
    adjust_kwargs(kwargs, params)
    

    params['title'] = f"{result_label.capitalize()} {'Percentage' if params['frequency'] else 'Total'} by {group.capitalize()} Category"
    params['y_label_title'] = f"{result_label.capitalize()} {'Percentage' if params['frequency'] else 'Total'}"

    y_title = 'Group Percentage' if params['frequency'] else 'Group total'

    sns.barplot(y=y_title, x=df.index, 
                data=df, palette=kwargs.pop('color'),
                hue=df.index, legend=False, **kwargs)

    final_graph_customization(ax, params)
    save_plot(params['filepath'])

    return df




def plot_hist_discrete_feature(ax: plt.Axes,
                               dataframe: pd.DataFrame, 
                               column: str,
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

    dict_df_count = {'Labels': labels, 'Counts': counts, 'Percentages': percentages}
    df_counts = pd.DataFrame(dict_df_count)

    get_colors(len(labels), kwargs)

    params = {'title': f'Distribution of {column.capitalize()}', 'x_label_title': column.capitalize()}
    params = {**DEFAULT_PARAMETER, **params}

    adjust_kwargs(kwargs, params, 
                  alpha=True, linewidth=True)

    x_lab = 'Percentages' if params['frequency'] else 'Counts'
    y_lab = 'Labels'

    sns.barplot(y=x_lab,
                x=y_lab,
                data=df_counts,
                palette=kwargs.pop('color'),
                hue='Labels',
                legend=False,
                **kwargs)

    if params['frequency']:
        ticks = ax.get_yticks()
        ax.yaxis.set_major_locator(FixedLocator(ticks))
        ax.set_yticklabels([f'{(tick):.1f}%' for tick in ticks])
        
    final_graph_customization(ax, params)
    
    save_plot(params['filepath'])




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

    if 'color' not in kwargs:
        kwargs['color'] = 'inferno'
    get_colors(len(dataframe[x].unique()), kwargs)

    default_median_style = dict(linewidth=1.5, color='auto')
    default_outlier_style = dict(marker='o')

    params = {'title': f'Boxplot of {x.capitalize()} and {y.capitalize()}', 'x_label_title': x.capitalize(), 'y_label_title': y.capitalize(),
              'outlier_style': default_outlier_style, 'median_style': default_median_style, 'alpha': 0.8, 'add_value_annot': False,
              'set_edgecolor': False}
    params = {**DEFAULT_PARAMETER, **params}

    if default_median_style['color'] != 'auto':
        median_style = {**default_median_style, **kwargs.pop('median_style', {})}  
    else:
        median_color = default_median_style.pop('color')
        median_style = {**default_median_style, **kwargs.pop('median_style', {})}

    adjust_kwargs(kwargs, params, alpha=False)
    

    sns.boxplot(x=x, y=y, data=dataframe, palette=kwargs.pop('color'), 
                hue=x, medianprops=median_style, flierprops=params['outlier_style'], **kwargs)
    
    for patch in ax.patches:
        r, g, b, _ = patch.get_facecolor()
        patch.set_facecolor((r, g, b, params['alpha']))
    
    if median_color == 'auto':
        for i, line in enumerate(ax.lines):
            if median_color == 'auto' and i % 6 == 4 and i != 0:
                line.set_color(ColorGenerator.adjust_color(ax.patches[i//6].get_facecolor(), -0.2))

    final_graph_customization(ax, params)

    save_plot(params['filepath'])


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
    if isinstance(dataframe, list) and group_column:
        raise ValueError("group_column cannot be specified when passing a list of DataFrames. Feature not supported yet.")

    nb_colors  = len(dataframe) if isinstance(dataframe, list) else 1
    nb_colors2 = len(dataframe[group_column].unique()) if group_column else 1

    get_colors(max(nb_colors, nb_colors2), kwargs)

    params = {'title': f'Kernel Density Estimation (KDE) of {column}' + (' by ' + group_column if group_column else ''),
              'x_label_title': column.capitalize(), 'y_label_title': 'Density', 'labels': None, 'add_value_annot': False}
    params = {**DEFAULT_PARAMETER, **params}

    adjust_kwargs(kwargs, params, alpha=True, fill=True)
    

    if isinstance(dataframe, pd.DataFrame):
        dataframe = [dataframe]
    color = kwargs.pop('color', 'dodgerblue')

    if group_column:
        for i, group in enumerate(dataframe[0][group_column].unique()):
            sns.kdeplot(dataframe[0][dataframe[0][group_column] == group][column], 
                        color=color[i], label=params['labels'][i] if params['labels'] else group, **kwargs)
    
        plt.legend()
    else:
        for i, df in enumerate(dataframe):
            sns.kdeplot(df[column], color=color[i], 
                        label=params['labels'][i] if params['labels'] else f'DataFrame {i}', **kwargs)
        plt.legend() if len(dataframe) > 1 else None


    final_graph_customization(ax, params)
    
    save_plot(params['filepath'])    



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

    params = {'title': 'Features importances', 'x_label_title': 'Importances', 'y_label_title': 'Features',
              'to_dict': False, 'bold_max': True, 'italic_min': True, 'add_value_annot': False}
    params = {**DEFAULT_PARAMETER, **params}

    get_edgecolors(0.5, params)
    adjust_kwargs(kwargs, params)
    

    sns.barplot(y='Features', x='Importances', 
                data=df_importance, palette=kwargs.pop('color'), 
                hue='Importances', legend=False, **kwargs)
    
    if params['bold_max']:
        max_index = np.argmax(df_importance['Importances'])
        y_labels = ax.get_yticklabels()
        y_labels[max_index].set_weight('bold')

    if params['italic_min']:
        min_index = np.argmin(df_importance['Importances'])
        y_labels = ax.get_yticklabels()
        y_labels[min_index].set_style('italic')

    final_graph_customization(ax, params)

    save_plot(params['filepath'])

    return df_importance.transpose().to_dict() if params['to_dict'] else df_importance.transpose()


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

    params = {'title': 'Confusion Matrix', 'x_label_title': 'Predicted', 'y_label_title': 'Actual', 'labels': None, 
              'border': True, 'add_value_annot': False, 'set_edgecolor': False, 'customize_plot_colors': False}
    params = {**DEFAULT_PARAMETER, **params}

    adjust_kwargs(kwargs, params, False, True, True, True, True, True)

    sns.heatmap(cm, annot=True, fmt='d', 
                xticklabels=target_names, 
                yticklabels=target_names,
                clip_on = False
                **kwargs)
    
    final_graph_customization(ax, params)

    if params['border']:
        ax.axhline(y=0, color='k',linewidth=1.8*params['linewidth'])
        ax.axhline(y=cm.shape[1], color='k',linewidth=1.8*params['linewidth'])
        ax.axvline(x=0, color='k',linewidth=1.8*params['linewidth'])
        ax.axvline(x=cm.shape[0], color='k',linewidth=1.8*params['linewidth'])

    save_plot(params['filepath'])

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
    corr = dataframe.corr()
    
    params = {'title': 'Correlation Matrix', 'x_label_title': '', 'y_label_title': '',
              'border': True, 'add_value_annot': False, 'set_edgecolor': False, 'customize_plot_colors': False}
    params = {**DEFAULT_PARAMETER, **params}

    adjust_kwargs(kwargs, params, False, True, True, True, True, True)
    
    sns.heatmap(corr, annot=True, fmt='.2f', **kwargs)

    final_graph_customization(ax, params)

    if params['border']:
        ax.axhline(y=0, color='k',linewidth=1.8*params['linewidth'])
        ax.axhline(y=corr.shape[1], color='k',linewidth=1.8*params['linewidth'])
        ax.axvline(x=0, color='k',linewidth=1.8*params['linewidth'])
        ax.axvline(x=corr.shape[0], color='k',linewidth=1.8*params['linewidth'])

    save_plot(params['filepath'])

    return corr
