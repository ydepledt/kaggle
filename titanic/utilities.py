import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.ticker import FixedLocator

from typing import List, Tuple, Union

COLORS = {'BLUE': '#3D6FFF',
          'RED': '#FF3D3D',
          'ORANGE': '#FF8E35',
          'PURPLE': '#BB58FF',
          'GREEN': '#32CD32',
          'YELLOW': '#F9DB00',
          'PINK': '#FFC0CB',
          'BROWN': '#8B4513',
          'CYAN': '#00FFFF',
}

def adjust_color(color: str, 
                 factor: float = 0.5) -> str:
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
    >>> adjust_color(COLORS['BLUE'], 0.5)
    >>> adjust_color(COLORS['BLUE'], -0.5)
    """

    assert(factor >= -1 and factor <= 1), "Factor must be between -1 and 1."

    r, g, b = tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
    
    if factor >= 0:
        r = int(r * factor)
        g = int(g * factor)
        b = int(b * factor)
    else:
        factor = abs(factor)
        r = int(r + (255 - r) * factor)
        g = int(g + (255 - g) * factor)
        b = int(b + (255 - b) * factor)

    return '#%02x%02x%02x' % (r, g, b)


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
        ax.grid(True, axis='x', linestyle='--', alpha=0.6, color=color_grid)
    if axgridy:
        ax.grid(True, axis='y', linestyle='--', alpha=0.6, color=color_grid)
    if color_spine is not None:
        ax.spines['bottom'].set_color(color_spine)
        ax.spines['left'].set_color(color_spine)
        ax.spines['top'].set_color(color_spine)
        ax.spines['right'].set_color(color_spine)
    if color_tick is not None:
        ax.tick_params(axis='x', colors=color_tick)
        ax.tick_params(axis='y', colors=color_tick)


def plot_groupby(dataframe: pd.DataFrame, 
                 group: str,
                 result_label: str,
                 filepath: str = None,
                 **kwargs) -> pd.DataFrame:
    """
    Generate a grouped bar plot based on DataFrame aggregation.

    Parameters:
    -----------
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

    kwargs['color'] = kwargs.get('color', COLORS['BLUE'])
    kwargs['edgecolor'] = kwargs.get('edgecolor', adjust_color(kwargs['color'], 0.5))

    figsize     = kwargs.pop('figsize', (10, 6))
    rotation    = kwargs.pop('rotation', 45)
    axgridx     = kwargs.pop('axgridx', False)
    axgridy     = kwargs.pop('axgridy', True)
    color_label = kwargs.pop('color_label', adjust_color(kwargs['color'], 0.3))
    color_spine = kwargs.pop('color_spine', adjust_color(kwargs['color'], 0.45))
    color_tick  = kwargs.pop('color_tick', adjust_color(kwargs['color'], 0.45))
    color_grid  = kwargs.pop('color_grid', adjust_color(kwargs['color'], -0.4))


    plt.figure(figsize=figsize)
    ax = sns.barplot(y='Group Percentage', x=df.index, data=df, **kwargs)
    plt.title(f'{result_label.capitalize()} Percentage by {group.capitalize()} Category', fontweight= 'bold')
    plt.ylabel(f'{result_label.capitalize()} Percentage', color=color_label, fontsize=11)
    plt.xlabel(f'{group.capitalize()} Category', color=color_label, fontsize=11)

    
    ticks = ax.get_yticks()
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.set_yticklabels([f'{(tick):.1f}%' for tick in ticks])

    rotation, ha = rotation if isinstance(rotation, tuple) else (rotation, 'center')
    plt.xticks(rotation=rotation, ha=ha)

    add_value_labels(ax, adjust_color(kwargs['color'], 0.4), 10)

    customize_plot_colors(ax, axgridx=axgridx, axgridy=axgridy, color_grid=color_grid,
                          color_spine=color_spine, color_tick=color_tick)

    if filepath:
        save_plot(filepath)

    plt.show()

    return df


def plot_missing_data(dataframe: pd.DataFrame, 
                      nan_values: List[Union[int, float, str]] = None,
                      filepath: str = None,
                      **kwargs) -> pd.DataFrame:
    """
    Generate a summary of missing data in a DataFrame and visualize it.

    Parameters:
    -----------
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

    percentage_missing = total_missing / dataframe.shape[0] * 100
    
    missing_info_df = pd.DataFrame({'Total Missing': total_missing, 
                                    'Percentage Missing': percentage_missing})

    missing_info_df.sort_values(by='Percentage Missing', ascending=False, inplace=True)
    
    filtered_missing_info_df = missing_info_df[missing_info_df['Percentage Missing'] > 0]

    kwargs['color'] = kwargs.get('color', COLORS['BLUE'])
    kwargs['edgecolor'] = kwargs.get('edgecolor', adjust_color(kwargs['color'], 0.5))

    figsize     = kwargs.pop('figsize', (10, 6))
    rotation    = kwargs.pop('rotation', 45)
    axgridx     = kwargs.pop('axgridx', False)
    axgridy     = kwargs.pop('axgridy', True)
    color_label = kwargs.pop('color_label', adjust_color(kwargs['color'], 0.3))
    color_spine = kwargs.pop('color_spine', adjust_color(kwargs['color'], 0.45))
    color_tick  = kwargs.pop('color_tick', adjust_color(kwargs['color'], 0.45))
    color_grid  = kwargs.pop('color_grid', adjust_color(kwargs['color'], -0.4))
    percentage  = kwargs.pop('percentage', 5)

    plt.figure(figsize=figsize)
    ax = sns.barplot(y='Percentage Missing', 
                     x=filtered_missing_info_df.index, 
                     data=filtered_missing_info_df, 
                     **kwargs)
    
    ticks = ax.get_yticks()
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.set_yticklabels([f'{(tick):.1f}%' for tick in ticks])

    plt.title('Percentage of Missing Values by Feature', fontweight='bold')
    plt.xlabel('Feature', color=color_label, fontsize=11)
    plt.ylabel('Percentage Missing', color=color_label, fontsize=11)

    rotation, ha = rotation if isinstance(rotation, tuple) else (rotation, 'center')
    plt.xticks(rotation=rotation, ha=ha)

    add_value_labels(ax, adjust_color(kwargs['color'], 0.4), percentage=percentage)
    
    customize_plot_colors(ax, axgridx=axgridx, axgridy=axgridy, color_grid=color_grid,
                          color_spine=color_spine, color_tick=color_tick)

    if filepath:
        save_plot(filepath)

    plt.show()

    return missing_info_df

def plot_distribution(dataframe: pd.DataFrame, 
                      column: str, 
                      filepath: str = None,
                      frequency: bool = False,
                      **kwargs) -> None:
    """
    Generate a histogram to visualize the distribution of a numeric column.

    Parameters:
    -----------
    dataframe : pandas DataFrame
        The input DataFrame containing the data for analysis.

    column : str
        The column name for which the distribution will be visualized.

    filepath : str, optional
        The file path where the generated visualization will be saved. 
        If provided, the figure will be saved as a PNG file. Defaults to None.

    frequency : bool, optional
        If True, plot the frequency instead of the number. Defaults to False.

    **kwargs:
        Additional keyword arguments for customization (e.g., color, alpha, bins, etc.).

    Returns:
    --------
    None

    This function takes a pandas DataFrame as input and creates a histogram using
    seaborn to visualize the distribution of a specified numeric column. The function
    displays the histogram and saves it as a PNG file if a file path is provided.
    """

    kwargs['color'] = kwargs.get('color', COLORS['BLUE'])
    kwargs['edgecolor'] = kwargs.get('edgecolor', adjust_color(kwargs['color'], 0.5))

    figsize     = kwargs.pop('figsize', (10, 6))
    axgridx     = kwargs.pop('axgridx', False)
    axgridy     = kwargs.pop('axgridy', True)
    color_label = kwargs.pop('color_label', adjust_color(kwargs['color'], 0.3))
    color_spine = kwargs.pop('color_spine', adjust_color(kwargs['color'], 0.45))
    color_tick  = kwargs.pop('color_tick', adjust_color(kwargs['color'], 0.45))
    color_grid  = kwargs.pop('color_grid', adjust_color(kwargs['color'], -0.4))

    plt.figure(figsize=figsize)

    if frequency:
        ax = sns.histplot(dataframe[column], stat='percent', **kwargs)
        plt.ylabel('Frequency', color=color_label, fontsize=11)
        ticks = ax.get_yticks()
        ax.yaxis.set_major_locator(FixedLocator(ticks))
        ax.set_yticklabels([f'{(tick):.1f}%' for tick in ticks])
    else:
        ax = sns.histplot(dataframe[column], **kwargs)
        plt.ylabel('Number', color=color_label, fontsize=11)

    bin_edges = [patch.get_x() for patch in ax.patches] + [ax.patches[-1].get_x() + ax.patches[-1].get_width()]
    num_tick = len(bin_edges)
    max_xticks = 19 
    if num_tick < max_xticks:
        ax.set_xticks(bin_edges)
        ax.set_xticklabels([f'{tick:.0f}' for tick in bin_edges])

    plt.title(f'Distribution of {column.capitalize()}', fontweight='bold')
    plt.xlabel(column.capitalize(), color=color_label, fontsize=11)

    customize_plot_colors(ax, axgridx=axgridx, axgridy=axgridy, color_grid=color_grid,
                          color_spine=color_spine, color_tick=color_tick)

    if filepath:
        save_plot(filepath)

    plt.show()


def plot_hist_discrete_feature(dataframe: pd.DataFrame, 
                               column: str,
                               filepath: str = None,
                               frequency: bool = False,
                               **kwargs) -> None:
    """
    Plot a histogram for a specified column in a DataFrame with customizable options.

    Parameters:
    -----------
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
    None

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

    print(df_counts)

    graphcolor = kwargs.pop('graph_color', '#000000')

    kwargs['color']      = kwargs.get('color', [COLORS['BLUE']] *  len(labels))
    kwargs['edgecolor']   = kwargs.get('edgecolor', adjust_color(graphcolor, 0.5))

    if len(kwargs['color']) != len(labels):
        print(f"Warning: Number of colors provided ({len(kwargs['color'])}) does not match the number of unique values ({len(labels)}). Adding default colors.")
        kwargs['color'] += [COLORS['BLUE']] * (len(labels) - len(kwargs['color']))

    figsize     = kwargs.pop('figsize', (10, 6))
    axgridx     = kwargs.pop('axgridx', False)
    axgridy     = kwargs.pop('axgridy', True)
    color_label = kwargs.pop('color_label', adjust_color(graphcolor, 0.3))
    color_spine = kwargs.pop('color_spine', adjust_color(graphcolor, 0.45))
    color_tick  = kwargs.pop('color_tick', adjust_color(graphcolor, 0.45))
    color_grid  = kwargs.pop('color_grid', adjust_color(graphcolor, -0.4))

    plt.figure(figsize=figsize)

    if frequency:
        ax = sns.barplot(y='Percentages', 
                        x='Labels', 
                        data=df_counts, 
                        palette=kwargs.pop('color'),
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
                         **kwargs)
        plt.ylabel('Number', color=color_label, fontsize=11)

    add_value_labels(ax, '#000000', frequency=frequency, percentage=8)

    plt.title(f'Distribution of {column.capitalize()}', fontweight='bold')
    plt.xlabel(column.capitalize(), color=color_label, fontsize=11)

    customize_plot_colors(ax, axgridx=axgridx, axgridy=axgridy, color_grid=color_grid,
                          color_spine=color_spine, color_tick=color_tick)

    if filepath:
        save_plot(filepath)

    plt.show()
