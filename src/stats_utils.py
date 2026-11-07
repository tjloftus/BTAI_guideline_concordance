# -*- coding: utf-8 -*-
"""
Helper functions for descriptive comparisons, from the authors' tjl_utils library
(analytic_functions.py and fill_null.py), included here so the repository is self-contained.
"""

from itertools import combinations

import numpy as np
import pandas as pd
import scipy.stats as sp


def fill_null_with_zero(input_df: pd.DataFrame, input_col_list: pd.Series):
    for col in input_col_list:
        input_df[f'{col}'] = input_df[f'{col}'].replace([np.inf, -np.inf], np.nan)
        input_df[f'{col}'] = input_df[f'{col}'].fillna(0)
    return input_df


def raw_fisher_crosstab(input_df: pd.DataFrame, column_list: list, level_column: str, abbrev_delimeter: str = None) -> pd.DataFrame:
    """
    Perform Fishers Exact test crosstab analyses and generate a dataframe containing results.

    Parameters
    ----------
    input_df: pd.DataFrame
        pandas dataframe that is being used for the statistical analysis
    column_list: list
        customizable list of columns from the input_df
    level_column: str
        the name of the column in the input_df that contains group assignments (e.g., sex)

    Returns
    -------
    output_df: pd.DataFrame
        pandas dataframe containing results of the analysis
    """
    # Define unique levels
    levels: list = input_df[level_column].dropna().unique().tolist()

    # Define pairs
    pairs: list = list(combinations(levels, 2))

    # Instantiate a dataframe and name its columns
    output_df = pd.DataFrame(columns=levels + [f'{x[0]} vs. {x[1]}' for x in pairs])

    # Track columns that cause errors
    problematic_columns = []

    # Iterate through columns that are in the customizable column_list
    for column in column_list:
        try:
            for pair in pairs:
                # Filter the DataFrame based on current pair
                temp_df = input_df.loc[input_df[level_column].isin(pair), [column, level_column]]

                # Create a crosstab contingency table
                temp_df_crosstab = pd.crosstab(temp_df[level_column], temp_df[column])

                # Ensure the crosstab has all necessary columns (0 and 1)
                for value in [0, 1]:
                    if value not in temp_df_crosstab.columns:
                        temp_df_crosstab[value] = 0

                # Perform Fisher's Exact Test
                fisher_result = sp.fisher_exact(temp_df_crosstab)
                output_df.loc[column, f'{pair[0]} vs. {pair[1]}'] = fisher_result[1]

            # Calculate counts and percentages for each level
            for level in levels:
                count = sum((input_df[level_column] == level) & (input_df[column] == 1))
                denominator = sum(input_df[level_column] == level)

                output_df.loc[column, level] = f'{count} ({count / denominator * 100:.1f})' if denominator > 0 else '0 (0.0)'

        except ValueError as ve:
            # Record the column that caused the ValueError
            print(f"Column '{column}' caused a ValueError: {ve}")
            problematic_columns.append(column)

    # If there are problematic columns, print them
    if problematic_columns:
        print(f"The following columns caused issues during the analysis: {problematic_columns}")

    return output_df.reset_index().rename(columns={'index': 'column'})


def format_summary_df(df: pd.DataFrame) -> pd.DataFrame:
    """Format the kruskal_wallace() dataframe."""
    df['value'] = None

    for i, row in df.iterrows():
        df.iloc[i, df.columns.get_loc('value')] = f"{(row['median']):.1f} [{row['q1']:.1f}-{(row['q3']):.1f}]"

    df = df.drop(columns=['median', 'q1', 'q3'])

    return df.T


def cleanup_df(df: pd.DataFrame, index_col: str) -> pd.DataFrame:
    """Reshape the kruskal_wallace() dataframe."""
    df = df[df.level_1 != 'column']

    new_column_names = df[df.level_1 == index_col].iloc[0, :].tolist()

    new_column_names[0] = 'column'
    new_column_names[1] = 'temp'

    df = df[df.level_1 != index_col]

    df.columns = new_column_names

    return df.drop(columns=['temp'])


def raw_kruskal_wallace(input_df: pd.DataFrame, column_list: list, level_column: str, abbrev_delimeter: str = None) -> pd.DataFrame:
    """
    Perform Kruskal-Wallis test crosstab analyses and generate a dataframe containing results.

    Parameters
    ----------
    input_df: pd.DataFrame
        pandas dataframe that is being used for the statistical analysis
    column_list: list
        customizable list of columns from the input_df
    level_column: str
        the name of the column in the input_df that contains group assignments (e.g., sex)

    Returns
    -------
    output_df: pd.DataFrame
        pandas dataframe containing results of the analysis
    """
    statistic_df = pd.DataFrame(columns=['column', 'comparison', 'p_value', 'kruskal_test_statistic'])
    summary_df = pd.DataFrame(columns=['column', 'levels', 'median', 'q1', 'q3'])

    levels: list = input_df[level_column].dropna().unique().tolist()
    pairs: list = list(combinations(levels, 2))

    grouped_df = input_df.groupby(level_column).describe()

    for column in column_list:
        temp_sum = grouped_df.loc[:, (column, ['50%', '25%', '75%']), ].rename(columns={'50%': 'median', '25%': 'q1', '75%': 'q3'}).reset_index().rename(columns={'': 'levels'})
        temp_sum.columns = temp_sum.columns.get_level_values(1)
        temp_sum['column'] = column
        summary_df = pd.concat([summary_df, temp_sum], axis=0, sort=False)

        for pair in pairs:
            group1 = input_df[column][(input_df[level_column] == pair[0]) & input_df[column].notnull()]
            group2 = input_df[column][(input_df[level_column] == pair[1]) & input_df[column].notnull()]

            # Check for zero variance
            if group1.nunique() == 1 and group2.nunique() == 1 and group1.iloc[0] == group2.iloc[0]:
                # Handle the case of identical values
                test_statistic, p_value = None, 1.0
            else:
                test_statistic, p_value = sp.kruskal(group1, group2)

            statistic_df = pd.concat([statistic_df, pd.DataFrame({
                'column': [column],
                'comparison': [f'{pair[0]} vs. {pair[1]}'],
                'kruskal_test_statistic': [test_statistic],
                'p_value': [p_value]
            })], axis=0, sort=False)

    output_df = cleanup_df(df=summary_df.groupby('column').apply(format_summary_df).reset_index(), index_col='levels')\
        .merge(cleanup_df(df=statistic_df.groupby('column').apply(lambda df: df.drop(columns=['kruskal_test_statistic']).T).reset_index(), index_col='comparison'), how='left', on='column')

    return output_df
