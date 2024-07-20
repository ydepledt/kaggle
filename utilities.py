import pandas as pd

from typing import Union, List, Dict

def get_distinct_value(df: pd.DataFrame, 
                       column: Union[str, List[str]] = None) -> Dict[str, Dict[str, int]]:
    
    """
    Get distinct value of a column or columns in a dataframe
    
    Parameters:
    -----------
    df: pd.DataFrame
        Dataframe to get distinct value from

    column: str or list of str
        Column or columns to get distinct value from. If None, all columns will be used. Default is None.

    Returns:
    --------
    Dict[str, Dict[str, int]]
        Dictionary of distinct value of columns and their counts

    Example:
    --------
    >>> df = pd.DataFrame({'A': ['a', 'b', 'a', 'c', 'c'], 'B': ['x', 'y', 'x', 'z', 'z']})
    >>> get_distinct_value(df, column='A')
    {'A': {'a': 2, 'b': 1, 'c': 2}}
    >>> get_distinct_value(df)
    {'A': {'a': 2, 'b': 1, 'c': 2}, 'B': {'x': 2, 'y': 1, 'z': 2}}
    """

    if column is None:
        column = df.columns

    if isinstance(column, str):
        column = [column]

    return {col: dict(df[col].value_counts()) for col in column}

def keep_top(sub_dict: Dict[str, int],
             n: int = 3) -> Dict[str, int]:
    """
    Keep top n items in a dictionary

    Parameters:
    -----------
    sub_dict: Dict[str, int]
        Dictionary to keep top n items from

    n: int
        Number of items to keep. Default is 3.

    Returns:
    --------
    Dict[str, int]
        Dictionary of top n items

    Example:
    --------
    >>> sub_dict = {'a': 2, 'b': 1, 'c': 2}
    >>> keep_top(sub_dict, n=2)
    {'a': 2, 'c': 2}
    """
    return dict(sorted(sub_dict.items(), key=lambda item: item[1], reverse=True)[:n])


