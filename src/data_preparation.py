import math
from math import sqrt
from typing import Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns
from dice_ml import Dice
from imblearn.pipeline import Pipeline as ImbPipeline
from raiutils.exceptions import UserConfigValidationException
from scipy.stats import pointbiserialr
from sklearn import inspection
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import KNNImputer
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def generate_histograms(data: pd.DataFrame, hue: str = None, title: str = None):
    """
    Generate histograms for all float-type columns in the given DataFrame.

    :param data: The input DataFrame containing the data for which histograms are to be generated.
    :param hue: A column name from the DataFrame to be used for color encoding the histograms (optional).
    :param title: The overall title for the set of histograms (optional). If provided, it will be displayed at the
    top of the plot.
    :return: None.
    """
    # show histograms in a grid with four columns
    n_cols = 4

    # calculate the number of rows
    n_rows = (data.shape[1] // 4) + 1

    # create the figure and axes
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
    axes = axes.flatten()

    # select only float-type columns from the DataFrame
    data_float = data.select_dtypes(include=["float"])

    # iterate over each float_type column to create histograms
    for i, col in enumerate(data_float.columns):
        if hue is not None:  # plot histograms with hue
            sns.histplot(x=col, data=data, kde=True, ax=axes[i], hue=hue, palette="viridis")
        else:  # plot histograms without hue
            sns.histplot(x=col, data=data, kde=True, ax=axes[i])

    # remove any empty subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # add an overall title if provided
    if title is not None:
        fig.suptitle(title, fontsize=16)

    plt.tight_layout()
    plt.show()


def display_value_counts(diagnosis_values: pd.Series):
    """
    Display a bar plot of the value counts of the diagnosis column. The percentage of each value relative to the total
    count is displayed above each bar.

    :param diagnosis_values: The input Series containing diagnosis values for which the value counts are to be
    displayed.
    :return: None.
    """
    # get the counts of each unique value in the series
    counts = diagnosis_values.value_counts()

    # create the bar plot
    plt.figure(figsize=(8, 6))
    ax = sns.barplot(x=counts.index, y=counts.values, hue=counts.index, palette="viridis",
                     dodge=False, legend=False)

    # calculate the total number of records
    record_number = diagnosis_values.shape[0]

    # annotate each bar with its count and percentage
    for i, value in enumerate(counts.values):
        ax.text(i, value + 0.05, str(value) + " (" + str(round((value / record_number) * 100)) + "%)", ha="center",
                va="bottom")

    # add labels and title to the plot
    ax.set_title("Value counts of diagnosis")
    ax.set_xlabel("Diagnosis")
    ax.set_ylabel("Count")

    plt.show()


def bin_correlations(corr_matrix: pd.DataFrame, absolute_values: bool = True) -> pd.Series:
    """
    Bin the correlation values from a correlation matrix into 0.1 intervals.

    :param corr_matrix: The input correlation matrix whose correlation values are to be binned.
    :param absolute_values: If True, the absolute values of the correlations are used for binning.
        If False, the original correlation values (including negative values) are used.
    :return: A Series containing the counts of correlation values in each bin, divided by 2
        to account for the fact that each correlation is counted twice (once for each pair).
    """
    if absolute_values:  # use absolute values of correlations for binning
        corr_matrix = corr_matrix.abs()
        bins = np.arange(0, 1.1, 0.1)
    else:
        bins = np.arange(-1, 1.1, 0.1)  # use actual correlation values for binning

    # flatten the correlation matrix values and bin them
    categories = pd.cut(corr_matrix.values.flatten(), bins)

    # count the occurrences in each bin
    category_counts = pd.Series(categories).value_counts(sort=False)
    return category_counts / 2  # divide by 2, since each correlation is counted twice


def visualize_correlation_sizes(corr_matrix: pd.DataFrame,
                                absolute_values: bool = True,
                                palette: str = "viridis",
                                differences: bool = False) -> None:
    """
    Display the sizes of correlations in a correlation matrix by binning them and plotting the counts of each bin.

    :param corr_matrix: The correlation matrix to visualize.
    :param absolute_values: If True, consider the absolute values of the correlations. Defaults to True.
    :param palette: The color palette to use for the bar plot. Default is "viridis".
    :param differences: If True, correlation differences are visualized. If False, correlation sizes are visualized.
    Default is False.
    """
    # bin the correlation values from a correlation matrix into 0.1 intervals.
    unique_category_counts = bin_correlations(corr_matrix, absolute_values)

    # create a bar plot to visualize the binned correlation sizes
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x=unique_category_counts.index.astype(str), y=unique_category_counts.values,
                     hue=unique_category_counts.index.astype(str), palette=palette)

    # annotate bars with percentage values
    for i, value in enumerate(unique_category_counts.values):
        percentage = str(round((value / unique_category_counts.sum()) * 100)) + "%"
        ax.text(i, value + 0.05, percentage, ha="center", va="bottom")

    # set plot labels and title
    label = "sizes" if not differences else "differences"
    plt.xlabel("categories")
    plt.ylabel(f"count of correlation {label}")
    plt.title(f"count of correlation {label} per category")
    plt.xticks(rotation=45)
    plt.show()


def calculate_corr_diffs(df: pd.DataFrame, label_encoded: bool = False) -> pd.DataFrame:
    """
    Calculate the difference in correlation sizes between the two diagnostic groups in the DataFrame.

    :param df: The input DataFrame containing the data.
    :param label_encoded: Specifies if the "diagnosis" column is label encoded (1 and 0) or string labels ("M" and "B").
    :return: A DataFrame containing the difference in correlation sizes between the two diagnostic groups.
    """
    # identify unique diagnoses
    diagnoses = df["diagnosis"].unique()

    # initialize a dictionary to store correlation matrices for each diagnosis
    corr_matrices = {}

    # calculate the correlation matrices in the two diagnostic groups separately
    for diagnosis in diagnoses:
        df_subset = df[df["diagnosis"] == diagnosis].select_dtypes(include=["float"])
        corr_matrices[diagnosis] = df_subset.corr(method="kendall")

    # calculate the difference in correlations
    if label_encoded:
        return corr_matrices[1] - corr_matrices[0]
    else:
        return corr_matrices["M"] - corr_matrices["B"]


def filter_corr(corr_diff: pd.DataFrame, diff_size: float = 0.4) -> pd.DataFrame:
    """
    Filter the correlation differences to identify those with a magnitude greater than a specified threshold.

    :param corr_diff: A DataFrame containing the correlation differences for different features.
    :param diff_size: The minimum absolute value of the correlation difference to be considered as significant.
    Default is 0.4.
    :return: A DataFrame containing pairs of features with a correlation difference greater than the specified
    threshold.
    """

    # select correlations with absolute differences greater than the specified threshold
    high_corr_diffs = corr_diff[(np.abs(corr_diff) > diff_size)].stack().reset_index()
    high_corr_diffs.columns = ['Feature1', 'Feature2', 'Correlation']

    # create a sorted pair of features to ensure each pair is unique regardless of order
    high_corr_diffs['sorted_pair'] = high_corr_diffs.apply(
        lambda row: tuple(sorted([row['Feature1'], row['Feature2']])), axis=1)

    # drop duplicates based on the sorted pairs and remove the "sorted_pair" column
    return high_corr_diffs.drop_duplicates(subset=['sorted_pair']).drop(columns=['sorted_pair'])


def get_outlier_bounds(series: pd.Series) -> Tuple[float, float]:
    """
    Calculate the lower and upper bounds for outliers in a given series using teh IQR proximity rule.

    :param series: The input data series.
    :return: A tuple containing the lower bound and upper bound for outliers.
    """

    # calculate the first and third quartiles
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)

    # calculate the inter quartile range (IQR)
    iqr = q3 - q1

    # calculate the lower and upper bounds for outliers
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return lower_bound, upper_bound


def calculate_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate outliers for each column in the data using the IQR proximity rule.

    :param df: The input data.
    :return: A DataFrame indicating the presence of outliers (1 for outlier, 0 for non-outlier) for each feature.
    """
    df_out_base = pd.DataFrame()

    # calculate outliers for each column
    for column in df:
        # get the lower and upper bounds for the current column
        lower_bound, upper_bound = get_outlier_bounds(df[column])

        # identify outliers and convert the boolean Series to integers (1 for outlier, 0 for non-outlier)
        df_out_base[column] = ((df[column] < lower_bound) | (df[column] > upper_bound)).astype(
            int)

    # set the index of the outlier DataFrame to match the input data
    df_out_base.index = df.index
    return df_out_base


def visualize_outliers(data: pd.DataFrame) -> None:
    """
    Visualize the distribution of outliers in the data.

    :param data: The input data containing outlier information.
    """

    # get the number of records in the data
    record_number = data.shape[0]

    # determine labels for the x-axis
    x_label_left = "Number of records per outlier count"
    x_label_right = "Number of outliers per feature"

    # calculate the sum of outliers per row and per column
    row_sum = data.sum(axis=1).value_counts().sort_index()
    col_sum = data.sum(axis=0)

    # create a figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 8))

    # plot the distribution of the number of records per outlier count
    sns.barplot(hue=row_sum.index, x=row_sum.values, palette="viridis", legend=False,
                y=row_sum.index, ax=axes[0], orient="h")
    axes[0].set_xlabel(x_label_left)
    axes[0].set_ylabel("Count")
    axes[0].set_title("Distribution of the number of records per outlier count")

    # annotate the bars with the percentage of records
    for i, value in enumerate(row_sum):
        percentage = f"{(value / record_number) * 100: .1f} %"
        axes[0].text(x=value, y=i, s=percentage, ha="left", va="center")

    # plot the distribution of the number of outliers per feature
    sns.barplot(hue=col_sum.index, x=col_sum.values, palette="viridis", legend=False,
                y=col_sum.index, ax=axes[1])
    axes[1].set_xlabel(x_label_right)
    axes[1].set_ylabel("")
    axes[1].set_title("Distribution of the number of outliers per feature")

    # annotate the bars with the number of outliers
    for i, value in enumerate(col_sum):
        axes[1].text(x=value, y=i, s=f"{value}", ha="left", va="center")

    # adjust spacing between subplots
    plt.subplots_adjust(wspace=0.3)
    plt.show()

    # calculate and print the number of records with more than 5 outliers
    many_outs = row_sum[row_sum.index > 5].sum()
    print(
        f"Number of records with more than 5 outliers: {many_outs} ({(many_outs / record_number) * 100:.2f} %)")


def get_outlier_descriptives(data: pd.DataFrame, col_name: str = None,
                             out_as_na: bool = False) -> pd.DataFrame:
    """
    Get descriptive statistics for outliers in the data.

    :param data: The input data containing outlier information.
    :param col_name: A prefix for the column names in the output. Defaults to None.
    :param out_as_na: If True, treat NaNs as outliers. Defaults to False.
    :return: A DataFrame containing the count and percentage of outliers for each feature.
    """
    # calculate the count of outliers
    out_count = data.isnull().sum() if out_as_na else data.sum()
    # calculate the percentage of outliers
    out_pct = (out_count / data.shape[0]) * 100
    # add a prefix to the column names if provided
    col_name = f"{col_name}_" if col_name is not None else ""
    # create a DataFrame with outlier count and percentage
    out_info = pd.DataFrame({f"{col_name}outlier_count": out_count, f"{col_name}outlier_pct": out_pct})
    return out_info


def calculate_difference_statistics(df_before: pd.DataFrame, df_after: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the percentage difference in descriptive statistics between the two DataFrames. Compute the descriptive
    and transformed DataFrames, and then calculate the percentage difference between them.

    :param df_before: The original DataFrame before transformation.
    :param df_after: The DataFrame after transformation.
    :return: A DataFrame containing the percentage differences in descriptive statistics.
    """
    # calculate descriptive statistics for both DataFrames
    before_desc = df_before.describe()
    after_desc = df_after.describe()

    # calculate the absolute and percentage differences in descriptive statistics
    diff = after_desc - before_desc
    return diff / before_desc * 100


def visualize_difference_statistics(norm_diff: pd.DataFrame) -> None:
    """
    Visualize the percentage differences in mean and median statistics using a bar plot.

    :param norm_diff: a DataFrame containing th percentage differences in statistics.
    """
    # create a copy of the DataFrame and transpose it
    norm_diff_filtered = norm_diff.copy().transpose()

    # filter for "mean" and "50%" (median) statistics
    norm_diff_filtered = norm_diff_filtered[["mean", "50%"]]

    # rename columns for better readability
    norm_diff_filtered.columns = ["Mean", "Median"]

    # reset the index and melt the DataFrame for visualization
    norm_diff_filtered = norm_diff_filtered.reset_index().melt(id_vars="index", var_name="Statistic",
                                                               value_name="Percent Difference")

    # create the horizontal bar plot
    plt.figure(figsize=(10, 10))
    sns.barplot(data=norm_diff_filtered, x='Percent Difference', y='index', hue='Statistic', orient='h',
                palette="viridis")

    # set the labels and title and add a legend 
    plt.xlabel('Percent Difference (%)')
    plt.title('normed percentage difference between median and mean')
    plt.legend(title='Statistic')
    plt.show()


class OutlierTransformer(BaseEstimator, TransformerMixin):
    """
    Identifies and marks outliers in a dataset using the IQR proximity rule.

    Parameters:
    :param only_added: If True, outliers will only be detected in the features that were newly added to the dataset.
    Defaults to False.

    Attributes:
        outlier_bounds (dict): A dictionary that stores the IQR bounds for each column.
    """

    def __init__(self, only_added: bool = False):
        self.outlier_bounds = {}
        self.only_added = only_added

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> object:
        """
        Calculate the IQR bounds for each column in the dataset, which will be used for later outlier identification.

        :param X: The input DataFrame containing the features as columns.
        :param y: Ignored.
        :return: Fitted OutlierTransformer.
        """
        # calculate IQR bounds for each column
        for col in X.columns:
            self.outlier_bounds[col] = get_outlier_bounds(X[col])

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Identify and mark univariate outliers as missing in the dataset.

        :param X: The input DataFrame containing the features.
        :return: The transformed DataFrame with outliers marked as NaN.
        """
        # determine which columns to process
        columns = X.columns if not self.only_added else X.filter(regex="diff_|ratio_|_x_").columns

        # DataFrame to store outlier flags
        df_out = pd.DataFrame(index=X.index)

        for col in columns:
            # retrieve the calculated IQR bounds
            lower_bound, upper_bound = self.outlier_bounds[col]
            # identify outliers
            outliers = (X[col] < lower_bound) | (X[col] > upper_bound)
            # mark outliers in the output DataFrame
            df_out.loc[outliers, col] = 1

        # create a copy of the input DataFrame
        X_out = X.copy()
        # replace outliers with NaN
        X_out[df_out == 1] = np.nan

        return X_out


def display_replacement_statistics(count_replaced: dict, data: pd.DataFrame) -> None:
    """
    Display statistics about the number of values replaced in the DataFrame.

    :param count_replaced: A dictionary containing the count of replaced values for each column.
    :param data: The original DataFrame to compare against.
    """
    # calculate the total number of elements in the DataFrame
    total_elements = data.shape[0] * data.shape[1]

    # calculate the number of rows in the DataFrame
    row_counts = data.shape[0]

    def calculate_pct(value: int | float, denominator: int | float = row_counts) -> float:
        """
        Calculate the percentage of "value" relative to "denominator" and round it to two decimal places.

        :param value: The numerator value.
        :param denominator: The denominator value.
        :return: The percentage value.
        """
        return round(value / denominator * 100, 2)

    # calculate the sum, the mean, the minimum, and the maximum values of replaced values in any column
    count_sum = sum(count_replaced.values())
    count_mean = sum(count_replaced.values()) / len(count_replaced)
    count_min = min(count_replaced.values())
    count_max = max(count_replaced.values())

    # print the statistics
    print(f"Number of values replaced: {count_sum} ({calculate_pct(count_sum, total_elements)} % of the data)")
    print(f"Mean number of values replaced per column: {round(count_mean, 2)} ({calculate_pct(count_mean)} %)")
    print(f"Minimum number of values replaced per column: {count_min} ({calculate_pct(count_min)} %)")
    print(f"Maximum number of values replaced per column: {count_max} ({calculate_pct(count_max)} %)")


class KNNReplacer(BaseEstimator, TransformerMixin):
    """
    Use K-Nearest Neighbors (KNN) to impute missing values in a DataFrame. Fill missing values using the KNNImputer
    from scikit-learn.

    Parameters:
        :param n_neighbors: Number of neighboring samples to use for imputation.
        :param weights: Weight function used in prediction. Possible values: "uniform" or "distance".

    Attributes:
        knn_imputer (KNNImputer): The KNNImputer instance from scikit-learn used to perform the imputation.
    """

    def __init__(self, n_neighbors: int = 5, weights: str = "distance"):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.knn_imputer = KNNImputer(n_neighbors=self.n_neighbors, weights=self.weights)

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> object:
        """
        Fit the KNN imputer on the input DataFrame.

        :param X: The input DataFrame to fit.
        :param y: Ignored.
        :return: Fitted KNNReplacer.
        """
        self.knn_imputer.fit(X)
        return self

    def transform(self, X: pd.DataFrame, y: pd.Series = None, verbose: bool = False) -> pd.DataFrame:
        """
        Transform the input DataFrame by imputing missing values using the KNNImputer.

        :param X: The input DataFrame to transform.
        :param y: Ignored.
        :param verbose: If True, displays statistics about the number of replacements. Defaults to False.
        :return: The transformed DataFrame with imputed values.
        """
        # create a copy of the DataFrame
        X_knn = X.copy()

        # transform the dataset using the fitted KNNImputer
        X_knn = self.knn_imputer.transform(X_knn)

        # retain the original column names
        X_knn = pd.DataFrame(X_knn, columns=X.columns)

        if verbose:
            # display statistics about the number of replacements
            missing_count = X.isnull().sum().to_dict()
            display_replacement_statistics(missing_count, X)

        return X_knn


class IQRCapper(BaseEstimator, TransformerMixin):
    """
    Cap outliers in a DataFrame based on the IQR proximity rule. Calculate the IQR for each feature and cap any
    values outside the lower and upper bounds.

    Attributes:
        outlier_bounds (dict): A dictionary to store the lower and upper bounds for each feature.
    """

    def __init__(self):
        self.outlier_bounds = {}

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> object:
        """
        Compute the IQR bounds for each feature in the DataFrame. These bounds will be used for later outlier capping.

        :param X: The input DataFrame containing features to be capped.
        :param y: Ignored.
        :return: The fitted IQRCapper.
        """
        # calculate IQR bounds for each column
        for col in X.columns:
            self.outlier_bounds[col] = get_outlier_bounds(X[col])

        return self

    def transform(self, X: pd.DataFrame, y: pd.Series = None, verbose: bool = False) -> pd.DataFrame:
        """
        Cap the outliers in the DataFrame based on the computed IQR bounds.

        :param X: The input DataFrame to transform.
        :param y: Ignored.
        :param verbose: If True, displays statistics about the number of replacements. Defaults to False.
        :return: The transformed DataFrame with outliers capped.
        """
        # create a copy of the DataFrame
        X_capped = X.copy()

        # initialize a dictionary to count the number of capped values for each column
        cap_count = {column: 0 for column in X_capped.columns}

        # cap outliers for each feature in the DataFrame
        for col in X_capped.columns:
            lower_bound, upper_bound = self.outlier_bounds[col]
            if verbose:
                # calculate the number of capped values
                lower_cap = X_capped[col] < lower_bound
                upper_cap = X_capped[col] > upper_bound
                cap_count[col] += lower_cap.sum() + upper_cap.sum()
            X_capped[col] = X_capped[col].clip(lower=lower_bound, upper=upper_bound)

        if verbose:
            # display statistics about the number of replacements
            display_replacement_statistics(cap_count, X)

        return X_capped


def calculate_circular_deviation(data: pd.DataFrame, absolute: bool = True) -> pd.DataFrame:
    """
    Calculate the absolute or relative difference between the actual area of tumor cells and the area calculated
    based on radius and perimeter values.

    :param data: DataFrame containing radius, perimeter and area statistics.
    :param absolute: Determines whether to calculate absolute or relative differences. Default is True.
    :return: DataFrame containing the calculated deviations.
    """
    # determine the identifier based on whether the absolute difference or the ratio is required
    identifier = "diff" if absolute else "ratio"

    # list of suffices for different statistics in the data
    statistic_list = ["_worst", "_mean"]

    # initialize an empty DataFrame to store the calculated deviations
    df_area = pd.DataFrame(index=data.index)

    def calculate_area_with_radius(radius: float) -> float:
        """
        Calculate the area of a circle given its radius.

        :param radius: Radius of the circle.
        :return: Area of the circle.
        """
        return math.pi * radius ** 2

    def calculate_area_with_perimeter(perimeter: float) -> float:
        """
        Calculate the area of a circle given its perimeter.

        :param perimeter: Perimeter of the circle.
        :return: Area of the circle.
        """
        radius = perimeter / (2 * math.pi)
        return math.pi * radius ** 2

    # iterate over each statistic in the list
    for statistic in statistic_list:
        # calculate the area using the radius and the perimeter for the current statistic
        area_with_radius = data[f"radius{statistic}"].apply(calculate_area_with_radius)
        area_with_perimeter = data[f"perimeter{statistic}"].apply(calculate_area_with_perimeter)
        if absolute:
            # calculate absolute differences of the areas and store these in the DataFrame
            df_area[f"{identifier}_area_radius{statistic}"] = area_with_radius - data[f"area{statistic}"]
            df_area[f"{identifier}_area_perimeter{statistic}"] = area_with_perimeter - data[f"area{statistic}"]
        else:
            # calculate the ratio of the areas and store these in the DataFrame
            df_area[f"{identifier}_area_radius{statistic}"] = area_with_radius / data[f"area{statistic}"]
            df_area[f"{identifier}_area_perimeter{statistic}"] = area_with_perimeter / data[f"area{statistic}"]

    return df_area


def calculate_worst_mean_deviation(data: pd.DataFrame, absolute: bool = True) -> pd.DataFrame:
    """
    Calculate the absolute or relative deviation between the "worst" and the "mean" characteristic of features in
    a given dataset.

    :param data: DataFrame containing the feature columns with "_worst" and "_mean" suffixes.
    :param absolute: Determines whether to calculate absolute or relative differences. Default is True.
    :return:
    """
    # generate a list of features without suffixes
    base_features = ["radius", "texture", "perimeter", "area", "smoothness", "compactness", "concave points",
                     "symmetry", "fractal_dimension"]

    # determine the identifier based on whether absolute or relative differences are required
    identifier = "diff" if absolute else "ratio"

    # initialize an empty DataFrame to store the calculated deviations
    df_dev = pd.DataFrame(index=data.index)

    # iterate over each base feature to calculate deviations
    for feature in base_features:
        # construct column names for "worst" and "mean" values
        worst_feature = f"{feature}_worst"
        mean_feature = f"{feature}_mean"

        # construct the new feature name for the deviation column
        new_feature_name = f"{feature}_worst_{identifier}_mean"

        # calculate the deviation based on the "absolute" flag and store in the new column
        if absolute:
            df_dev[new_feature_name] = data[worst_feature] - data[mean_feature]
        else:
            df_dev[new_feature_name] = data[worst_feature] / data[mean_feature]

    return df_dev


def add_label(data: pd.DataFrame, label: pd.Series) -> pd.DataFrame:
    """
    Add the label column to the DataFrame and align the index with the label DataFrame.

    :param data: The DataFrame to which the label will be added.
    :param label: The label data to be added as a new column.
    :return: A DataFrame with the label column added.
    """
    # create a copy of the input data to avoid modifying the original DataFrame
    df = data.copy()
    # align the index with the index of the label
    df.index = label.index
    # add the label data as a new column named "diagnosis"
    df["diagnosis"] = label
    return df


def generate_deviation_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Create the deviation features from the given dataset.

    :param data: The input dataset from which to create the deviation features.
    :return: A DataFrame containing the concatenated deviation features.
    """
    # Calculate the absolute and relative difference between the actual area of tumor cells and the area calculated
    #     based on radius and perimeter values.
    X_area_diff = calculate_circular_deviation(data)
    X_area_ratio = calculate_circular_deviation(data, absolute=False)

    # Calculate the absolute or relative deviation between the "worst" and the "mean" characteristic of features in
    #     a given dataset.
    X_dev_diff = calculate_worst_mean_deviation(data)
    X_dev_ratio = calculate_worst_mean_deviation(data, absolute=False)

    # concatenate all the calculated features into a single DataFrame
    return pd.concat([X_area_diff, X_area_ratio, X_dev_diff, X_dev_ratio], axis=1)


def generate_interaction_features(training_data: pd.DataFrame,
                                  label: pd.Series,
                                  diff_size: float = 0.4) -> (pd.DataFrame, List[Tuple]):
    """
    Generate interaction features for those features where there is a significant difference in the correlation
    between the two diagnostic groups.

    :param training_data: The input training data containing the features.
    :param label: The label data.
    :param diff_size: The threshold for filtering correlation differences. Default is 0.4.
    :return: A DataFrame containing the generated interaction features and a list of tuples where each tuple contains
    the pair of features used to generate an interaction feature.
    """
    # add the label to the training data
    df_train = add_label(training_data, label)

    # calculate the correlation differences in the two diagnostic groups
    corr_matrix_diff = calculate_corr_diffs(df_train, label_encoded=True)

    # filter the correlations based on the specified difference threshold
    filtered_corr = filter_corr(corr_matrix_diff, diff_size=diff_size)

    # initialize the interaction pairs list and the DataFrame for interaction features
    interaction_pairs = []
    df_inter = pd.DataFrame()

    # iterate over teh filtered correlations to generate interaction features
    for index, row in filtered_corr.iterrows():
        # determine the feature name
        feature_1 = row.iloc[0]
        feature_2 = row.iloc[1]
        interaction_name = f"{feature_1}_x_{feature_2}"

        # create the interaction feature by multiplying the two features
        df_inter[interaction_name] = training_data[feature_1] * training_data[feature_2]

        # append the pair of features to the interaction pairs list
        interaction_pairs.append((feature_1, feature_2))

    return df_inter, interaction_pairs


def apply_interaction_features(data: pd.DataFrame, interaction_pairs: List[Tuple]) -> pd.DataFrame:
    """
    Generate interaction terms based on the provided interaction pairs.

    :param data: The input data containing the features.
    :param interaction_pairs: A list of tuples where each tuple contains the pair of features to generate an
    interaction feature.
    :return: A DataFrame containing the generated interaction features.
    """
    # initialize the DataFrame that contains the interaction features
    df_inter = pd.DataFrame()

    # iterate over the interaction pairs to generate interaction features
    for feature_1, feature_2 in interaction_pairs:
        interaction_name = f"{feature_1}_x_{feature_2}"

        # create the interaction feature by multiplying the two features
        df_inter[interaction_name] = data[feature_1] * data[feature_2]

    return df_inter


def evaluate_normality(data: pd.DataFrame) -> None:
    """
    Provide histograms and QQ-plots for each feature to enable the evaluation of a feature's normality.

    :param data: The input data containing the features to evaluate.
    """
    # determine the number of variables
    num_vars = data.shape[1]

    # set the number of columns for subplots (each variable will have 2 plots: histogram and QQ-plot)
    cols = 4

    # calculate the number of rows required for the subplots
    rows = (num_vars + cols - 1) // cols

    # create the subplot grid with twice the number of columns
    fig, axes = plt.subplots(rows, cols * 2, figsize=(20, rows * 5))

    # iterate over each feature in the data
    for i, var in enumerate(data.columns):
        row = i // cols
        col = (i % cols) * 2

        # plot the histogram with KDE
        sns.histplot(data[var], kde=True, ax=axes[row, col])
        axes[row, col].set_title(f'Histogram')

        # plot the QQ-plot
        sp.stats.probplot(data[var], dist="norm", plot=axes[row, col + 1])
        axes[row, col + 1].set_title(f'QQ-Plot of {var}')

    # delete empty subplots
    for i in range(num_vars, rows * cols):
        fig.delaxes(axes.flat[i * 2])
        fig.delaxes(axes.flat[i * 2 + 1])

    # adjust layout and plot figure
    plt.tight_layout()
    plt.show()


def get_merit(features: List[str], label: pd.Series, data: pd.DataFrame) -> float:
    """
    Calculate the merit of a given subset of features based on the correlation-based feature selection (CFS) method.
    Function adapted from Fischer (2021).

    :param features: The subset of features to evaluate.
    :param data: The label.
    :param label: The DataFrame containing the features and the label.
    :return: The merit score of the feature subset.
    """
    k = len(features)

    # calculate the average feature-class correlation
    rcf_all = []
    for feature in features:
        coeff = pointbiserialr(label, data[feature])
        rcf_all.append(abs(coeff.correlation))
    rcf = np.mean(rcf_all)

    # calculate the average feature-feature correlation
    corr = data[features].corr()

    # set the lower triangle and diagonal to NaN to ignore them in the mean calculation
    corr.values[np.tril_indices_from(corr.values)] = np.nan
    corr = abs(corr)
    rff = corr.unstack().mean()

    # return the merit score based on the CFS formula
    return (k * rcf) / sqrt(k + k * (k - 1) * rff)


class PriorityQueue:
    """
    A priority queue implementation using a list. Class adapted from Fischer (2021).

    Attributes:
        queue (List[Tuple]): A list to store the items and their respective priorities.
    """

    def __init__(self):
        self.queue = []

    def is_empty(self) -> bool:
        """
        Check if the priority queue is empty.

        :return: True if the queue is empty, False otherwise.
        """
        return len(self.queue) == 0

    def push(self, item: any, priority: float) -> None:
        """
        Add an item to the priority queue with a given priority. If the item already exists with a smaller priority,
        update its priority. If it exists with a higher priority, do nothing.

        :param item: The item to be added to the queue.
        :param priority: The priority of the item.
        """
        for index, (i, p) in enumerate(self.queue):
            if set(i) == set(item):  # if the item is already in the queue
                if p >= priority:  # existing item has a higher or equal priority
                    break
                del self.queue[index]  # remove the existing item
                self.queue.append((item, priority))  # add the item with the new priority
                break
        else:
            self.queue.append((item, priority))  # add the item if it was not found in the queue

    def pop(self) -> Tuple:
        """
        Remove and return the item with the highest priority from the queue.

        :return: The item with the highest priority and its priority.
        """
        if self.is_empty():
            raise IndexError("pop from an empty priority queue")

        max_idx = 0
        # find the index of the item with the highest priority
        for index, (i, p) in enumerate(self.queue):
            if self.queue[max_idx][1] < p:
                max_idx = index

        # remove and return the item with the highest priority
        (item, priority) = self.queue[max_idx]
        del self.queue[max_idx]
        return item, priority


class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Selects features based on specified selection criteria.

    Parameters:
        :param selection_type: The feature selection method. Options are "correlation_based", "worst", and "three_best".
            - correlation_based: Makes use of the correlation based feature selection according to Hall (1990). Code
            for this selection method was adapted from Fischer (2021).
            - worst: Selects only the _worst characteristics of the features.
            - three_best: Selects only the three features that were identified as best in Street et al. (1993, p. 867).
        :param max_backtrack: The maximum number of backtracks allowed for the correlation_based feature selection
        method.

    Attributes:
        selected_features (List[str]): The list of selected features.
    """

    def __init__(self, selection_type: str = "correlation_based", max_backtrack: int = 5):
        self.selection_type = selection_type
        self.max_backtrack = max_backtrack
        self.selected_features = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> object:
        """
        Fit the feature selector to the data.

        :param X: The input DataFrame containing the features as columns.
        :param y: The label Series.
        :return: The fitted feature selector.
        """
        if self.selection_type == "correlation_based":
            # perform correlation based feature selection
            best_value = -1
            best_feature = ""

            # evaluate each feature based on its point-biserial correlation with the label
            for feature in X.columns:
                # calculate the correlation coefficient between the feature and the label
                coeff = pointbiserialr(y, X[feature])

                # take the absolute value of the correlation coefficient
                abs_coeff = abs(coeff.correlation)

                # compare the correlation coefficient with the best value and set it as best, if it is better
                if abs_coeff > best_value:
                    best_value = abs_coeff
                    best_feature = feature

            # initialize a priority queue with the best feature found
            queue = PriorityQueue()
            queue.push([best_feature], best_value)

            visited = []  # list to keep track of evaluated feature subsets
            n_backtrack = 0  # counter for the number of backtracks

            # repeat until the queue is empty or the maximum number of backtracks is reached
            while not queue.is_empty() and n_backtrack < self.max_backtrack:
                subset, priority = queue.pop()  # get the subset with the highest priority

                # check if the current subset's priority is less than the best value found
                if priority < best_value:
                    n_backtrack += 1  # increment backtrack counter
                else:
                    best_value = priority  # update the best value
                    self.selected_features = subset  # update the selected features

                # iterate through all features and see if one can increase the merit
                for feature in X.columns:
                    if feature not in subset:
                        temp_subset = subset + [feature]  # create a new subset with the current feature

                        # check if the subset has already been evaluated
                        for node in visited:
                            if set(node) == set(temp_subset):
                                break
                        else:
                            # mark the subset as visited
                            visited.append(temp_subset)

                            # calculate the merit of the new subset
                            merit = get_merit(temp_subset, y, X)

                            # push the new subset to the queue with its merit as the priority
                            queue.push(temp_subset, merit)

        elif self.selection_type == "worst":
            # select the first (original) 10 features that include "worst" in their names
            self.selected_features = X.filter(like="worst").columns.tolist()[:10]

        elif self.selection_type == "three_best":
            # select the features that were identified as the three best in Street et al. (1993, p. 867)
            # use a regex pattern to account for different prefixes or suffixes
            pattern = "texture_mean|area_worst|smoothness_worst"
            self.selected_features = X.filter(regex=pattern).columns.tolist()[:3]

        else:
            raise ValueError(f"Invalid selection_type '{self.selection_type}'. "
                             f"Choose 'correlation_based', 'worst' or 'three_best'.")

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the data to retain only the selected features.

        :param X: The input data to transform.
        :return: The transformed data with only the selected features.
        """
        return X[self.selected_features]


class LogTransformer(BaseEstimator, TransformerMixin):
    """
    Applies a log transformation to specified features in a dataset.

    Parameters:
        :param transform_all: If True, transforms all features in the dataset. If False, transforms only specified
        features. Defaults to False.
        :param non_normal_vars: List of features to transform. If None, a list of features previously identified as
        non-normal will be used. Defaults to None.
        :param custom_constants: Custom constants to add to features to enable log transformation. If None, constants
        will be calculated based on the training data. Defaults to None.
        :param constant_value: The value to add to all features to enable log transformation. Default is 1.

    Attributes:
        min_values (List[float]): Minimum values of the specified features.
        constants (List[float]): Constants added to each feature to ensure all values are positive.
    """

    def __init__(self, transform_all: bool = False,
                 non_normal_vars: List[str] = None,
                 custom_constants: List[float] = None,
                 constant_value: float = 1):
        self.transform_all = transform_all
        self.non_normal_vars = non_normal_vars
        self.custom_constants = custom_constants
        self.constant_value = constant_value
        self.min_values = []
        self.constants = []

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> object:
        """
        Fit the log transformer to the training data and calculate constants to add before log transformation
        if necessary.

        :param X: The input DataFrame containing the features as columns.
        :param y: Ignored.
        :return: Fitted LogTransformer.
        """
        # determine the features to be transformed
        if self.non_normal_vars is None and not self.transform_all:
            non_normal_vars = ["area_mean", "concavity_mean", "concave points_mean", "radius_se", "perimeter_se",
                               "area_se", "compactness_se", "concavity_se", "symmetry_se", "area_worst",
                               "compactness_worst", "concavity_worst",
                               "diff_area_perimeter_worst", "diff_area_perimeter_mean", "radius_worst_diff_mean",
                               "perimeter_worst_diff_mean", "area_worst_diff_mean"]
            interaction_features = ["radius_mean_x_radius_se",
                                    "perimeter_mean_x_radius_se", "area_mean_x_radius_se", "radius_se_x_radius_worst",
                                    "radius_se_x_perimeter_worst", "radius_se_x_area_worst"]

            # add interaction features to the list of non-normal variables if they are in the dataset
            if interaction_features[0] in X.columns.tolist():
                self.non_normal_vars = non_normal_vars + interaction_features
            else:
                self.non_normal_vars = non_normal_vars

        # if all features are to be transformed, all features are treated as non-normal
        elif self.transform_all:
            self.non_normal_vars = X.columns.tolist()

        # calculate the constants to add to make positivity of features more likely if no constants are provided
        if self.custom_constants is None:
            self.min_values = X[self.non_normal_vars].min()
            self.constants = [abs(val) + self.constant_value if val < 0
                              else self.constant_value for val in self.min_values]
        else:
            self.constants = self.custom_constants
        return self

    def transform(self, X: pd.DataFrame, y: pd.Series = None,
                  verbose: bool = False) -> pd.DataFrame:
        """
        Add the previously specified constants to the data and apply the log transformation to the specified features.

        :param X: The input data to transform.
        :param y: Ignored.
        :param verbose: If True, displays the added constants and the number of features that were log transformed.
        :return: The transformed DataFrame where features were log-transformed as specified.
        """
        X = X.copy()

        # filter for variables specified as non-normal
        X_to_log = X[self.non_normal_vars]

        # add constants to the variables to make them positive
        X_pos = X_to_log + self.constants

        if verbose:
            # display the constants that have been added
            print(f"[INFO] The following constants were added: \n{self.constants}")

        # check whether all values are positive
        all_positive = (X_pos > 0).all().all()

        # display warnings if not all variables are positive after adding constants
        if not all_positive:
            print("[WARNING] Could not perform Log-transformation because not all variables are positive")
            print(f"Expected minimum values: {self.min_values}")
            print(f"Actual minimum values. {X_to_log.min()}")
        else:
            # perform log transformation
            X_log = np.log1p(X_pos)

            # replace the original with the log-transformed variables
            X[self.non_normal_vars] = X_log

            # change variable names to indicate which variables have been log_transformed
            log_transformed_vars = [f"log_{var}" if var in self.non_normal_vars else var for var in X.columns]
            X.columns = log_transformed_vars

            if verbose:
                # display the number of features that have been log_transformed
                print(f"[INFO] Performed log transformation of {len(self.non_normal_vars)} features")

        return X


class AddFeatures(BaseEstimator, TransformerMixin):
    """
    Adds deviation and interaction features to the dataset as specified.

    Parameters:
         :param deviation_features: A list of deviation features to be added.
         :param add_interaction: Whether to add interaction features. Default is True.
         :param interaction_pairs: A list of interaction feature pairs.
         :param diff_size: Threshold for filtering correlation differences. Is relevant for calculating interaction
         features.
         :param in_pipeline: Whether the transformer is used in a pipeline. Default is True.

    Attributes:
        deviation_features (List[str]): A list of deviation features to be added.
        add_interaction (bool): Whether to add interaction features.
        interaction_pairs (List[str]): A list of interaction feature pairs.
        diff_size (float): Threshold for filtering correlation differences. Is relevant for calculating interaction
        features.
        in_pipeline (bool): Whether the transformer is used in a pipeline.
        label_col (str): The label column name.

    """

    def __init__(self, deviation_features: List[str] = None,
                 add_interaction: bool = True,
                 interaction_pairs: List[str] = None,
                 diff_size: float = 0.4,
                 in_pipeline: bool = True):
        self.deviation_features = deviation_features
        self.add_interaction = add_interaction
        self.interaction_pairs = interaction_pairs
        self.diff_size = diff_size
        self.in_pipeline = in_pipeline
        self.label_col = "diagnosis"

        # determine the default interaction pairs
        if self.add_interaction and self.interaction_pairs is None and self.in_pipeline:
            self.interaction_pairs = [
                ("radius_mean", "radius_se"),
                ("perimeter_mean", "radius_se"),
                ("area_mean", "radius_se"),
                ("radius_se", "perimeter_worst"),
                ("radius_se", "radius_worst"),
                ("radius_se", "area_worst")
            ]

        # determine the default deviation features
        if self.deviation_features is None:
            self.deviation_features = ["diff_area_perimeter_worst", "diff_area_perimeter_mean",
                                       "radius_worst_diff_mean", "perimeter_worst_diff_mean", "area_worst_diff_mean",
                                       "radius_worst_ratio_mean", "perimeter_worst_ratio_mean", "area_worst_ratio_mean"]

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> object:
        """
        If the transformer is not used within a pipeline, determine which interaction features to add.

        :param X: The input DataFrame containing the features as columns.
        :param y: Ignored.
        :return: Fitted AddFeatures instance.
        """
        # determine the interaction features to add
        if self.add_interaction and not self.in_pipeline and y is not None:
            # temporarily add the label
            X[self.label_col] = y

            # calculate correlation differences and filter them
            corr_matrix_diff = calculate_corr_diffs(X, label_encoded=True)
            filtered_corr = filter_corr(corr_matrix_diff, diff_size=self.diff_size)

            # append filtered interaction pairs
            for index, row in filtered_corr.iterrows():
                self.interaction_pairs.append((row.iloc[0], row.iloc[1]))

            # remove the temporarily added label
            X.drop(columns=[self.label_col], inplace=True)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the dataset by adding deviation and interaction features as specified.
        :param X: The input DataFrame containing the features.
        :return: The transformed DataFrame with added deviation and interaction features as specified.
        """
        # calculate deviation features
        X_deviation_features = generate_deviation_features(X)

        # add deviation features to the dataset
        X_added = pd.concat([X, X_deviation_features[self.deviation_features]], axis=1)

        if self.add_interaction:
            # calculate interaction terms
            X_inter = apply_interaction_features(X, self.interaction_pairs)

            # add interaction features to the dataset
            X_added = pd.concat([X_added, X_inter], axis=1)

        return X_added


class Scaler(BaseEstimator, TransformerMixin):
    """
    Scales features in a dataset using either standard scaling or min-max scaling.

    Parameters:
        :param method: The scaling method to use. Options are "standard" for standard scaling and "min_max" for
        min-max scaling.

    Attributes:
        standard_scaler (StandardScaler): Scaler object for standard scaling.
        minmax_scaler (MinMaxScaler): Scaler object for min-max scaling.
        method (str): the chosen scaling method.
    """

    def __init__(self, method: str = "standard"):
        self.standard_scaler = StandardScaler()
        self.minmax_scaler = MinMaxScaler()
        self.method = method

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> object:
        """
        Fit the scaler to the data.

        :param X: The input DataFrame containing the features as columns.
        :param y: Ignored.
        :return: Fitted Scaler.
        """
        if self.method not in ["standard", "min_max"]:
            raise ValueError(f"Invalid method '{self.method}'. Choose 'standard' or 'min_max'")
        elif self.method == "standard":
            self.standard_scaler.fit(X)
        else:  # method = min max scaling
            self.minmax_scaler.fit(X)
        return self

    def transform(self, X: pd.DataFrame, y: pd.Series = None,
                  verbose: bool = False) -> pd.DataFrame:
        """
        Apply the scaling transformation to the data.

        :param X: The input DataFrame containing the features.
        :param y: Ignored.
        :param verbose: If True, print additional information. Default is True.
        :return: The scaled data.
        """
        # copy the DataFrame to make changes
        X_scaled = X.copy()

        # transform the dataset
        if self.method == "standard":
            X_scaled = self.standard_scaler.transform(X_scaled)
        else:  # method = min max scaling
            X_scaled = self.minmax_scaler.transform(X_scaled)

        # rename the columns
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

        if verbose:
            print(f"[INFO] Applied {self.method} scaling to the data")

        return X_scaled


class ImputerAdder(BaseEstimator, TransformerMixin):
    """
    Identifies outliers, handles them and optionally adds new features.

    Parameters:
        :param imputer: The method used to handle identified outliers: either an instance of KNNReplacer or IQRCapper.
        :param add_features: An instance of the AddFeatures class. Optional.

    Attribues:
        imputer (Union[KNNReplacer, IQRCapper]): The method used to handle identified outliers.
        imputer_orig (Union[KNNReplacer, IQRCapper]): The imputer used to handle outliers in the original features.
        imputer_added (Union[KNNReplacer, IQRCapper]): The imputer used to handle outliers in the added features.
        add_features (): The instance of the AddFeatures class.
        out_orig (OutlierTransformer()): The outlier transformer used to identify and mark outliers in the original
        features.
        out_added (OutlierTransformer()): The outlier transformer used to identify and mark outliers in the added
        features.

    """

    def __init__(self, imputer: Union[KNNReplacer, IQRCapper], add_features: Union[None, AddFeatures()] = None):
        self.imputer = imputer
        self.imputer_orig = imputer()  # instantiate the imputer for the original features
        self.imputer_added = imputer()  # instantiate the imputer for the added features
        self.add_features = add_features  # instance of the AddFeatures class
        self.out_orig = OutlierTransformer()  # outlier transformer for the original features
        self.out_added = OutlierTransformer(only_added=True)  # outlier transformer for the added features

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> object:
        """
        Fit the OutlierTransformer, the imputer and the AddFeatures instance to the dataset. This means identifying
        and handling outliers in the original features to fit the AddFeatures instance. Features are added to the
        cleaned dataset, outliers are identified and handled to fit the imputer to the added features.

        :param X: The input DataFrame containing the features as columns.
        :param y: Ignored.
        :return: Fitted ImputerAdder.
        """
        X = X.copy()

        # identify outliers as missing if the imputing method is the KNNReplacer
        if isinstance(self.imputer_orig, KNNReplacer):
            self.out_orig.fit(X, y)
            X = self.out_orig.transform(X)

        # fit the imputer for the original features on the dataset
        self.imputer_orig.fit(X, y)

        if self.add_features is not None:
            # impute outliers to clean the dataset before adding new features
            X_oh = self.imputer_orig.transform(X)

            # fit the FeatureAdder and transform the data to determine outliers in the added features
            self.add_features.fit(X_oh, y)
            X_af = self.add_features.transform(X_oh)

            # identify outliers in the added features as missing if the imputing method is the KNNReplacer
            if isinstance(self.imputer_orig, KNNReplacer):
                self.out_added.fit(X_af, y)
                X_af = self.out_added.transform(X_af)

            # fit the imputer for the added features
            self.imputer_added.fit(X_af, y)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Identify and handle outliers in the original features and add features to the cleaned dataset. Then, identify
        and handle outliers in the added features.

        :param X: The input DataFrame containing the features.
        :return: The transformed DataFrame with added features and outliers in all features handled.
        """
        X = X.copy()

        # identify outliers as missing if the imputing method is the KNNReplacer
        if isinstance(self.imputer_orig, KNNReplacer):
            X = self.out_orig.transform(X)

        # handle outliers in the original features
        X_tf = self.imputer_orig.transform(X)

        if self.add_features is not None:
            # add new features
            X_af = self.add_features.transform(X_tf)

            # identify outliers in the new features as missing if the imputing methods is the KNNReplacer
            if isinstance(self.imputer_orig, KNNReplacer):
                X_af = self.out_added.transform(X_af)

            # handle outliers in the added features
            X_af = self.imputer_added.transform(X_af)

        else:
            # if no features are added, use the transformed original dataset
            X_af = X_tf.copy()

        return X_af


def extract_imputer_type(obj: ImputerAdder) -> str:
    """
    Extract the type of imputer from the ImputerAdder object.

    :param obj: An instance of the ImputerAdder class.
    :return: The type of imputer.
    """
    # check the type of imputer and return its name
    if obj.imputer == KNNReplacer:
        return "KNNReplacer"
    elif obj.imputer == IQRCapper:
        return "IQRCapper"
    else:
        return "Unknown"


def format_random_search_results(search_results: RandomizedSearchCV):
    """
    Format the results of a random search for better readability and analysis.

    :param search_results: The object containing the search results.
    :return: A formatted DataFrame with selected columns and sorted by the mean f1 score.
    """
    # convert search results to a DataFrame
    results = pd.DataFrame(search_results.cv_results_)

    # define a regex pattern to filter for relevant columns
    pattern = "param_|mean_test_|rank_test|test_f1"

    # filter columns based on the patterna dn sort by "rank_test_f1"
    results = results.filter(regex=pattern).sort_values(by='rank_test_f1', ascending=True)

    # extract the imputer type from the "param_imputer_adder" column
    results["param_imputer_adder"] = results["param_imputer_adder"].apply(extract_imputer_type)

    # reorder columns to place "mean_test_f1" and "rank_test_f1" at the beginning
    cols = ['mean_test_f1', 'rank_test_f1'] + [col for col in results.columns if
                                               col not in ['mean_test_f1', 'rank_test_f1']]
    results = results[cols]

    return results


def filter_for_high_f1(search_results: RandomizedSearchCV) -> pd.DataFrame:
    """
    Filters the search results to include only those entries where the F1 score for each split is greater than 0.95.

    :param search_results: The object containing the search results.
    :return: A DataFrame containing only the entries with high F1 scores across all splits.
    """
    # format the search results
    formatted_results = format_random_search_results(search_results)
    # filter the DataFrame to include only the rows where all F1 scores are greater than 0.95
    return formatted_results[
        formatted_results[
            ['split0_test_f1', 'split1_test_f1', 'split2_test_f1', 'split3_test_f1', 'split4_test_f1']].apply(
            lambda row: all(row > 0.95), axis=1)]


def get_parameters_by_rank(search_results: RandomizedSearchCV, rank: int) -> List[Dict]:
    """
    Retrieve the hyperparameters of the model at the specified rank based on the mean F1 score.

    :param search_results: The object containing the search results.
    :param rank: The rank of the model whose parameters are to be retrieved.
    :return: A list of dictionaries containing the hyperparameters of the model at the specified rank.
    """
    # extract the cross-validation results from the search results
    results = search_results.cv_results_

    # find the indices of models that have the specified rank
    ranked_indices = np.where(results["rank_test_f1"] == rank)[0]

    # Retrieve the hyperparameters of the models at the specified rank
    params_at_rank = [results["params"][index] for index in ranked_indices]

    return params_at_rank


def display_permutation_importance(permutation_importances: inspection.permutation_importance,
                                   X_train_proc: pd.DataFrame) -> pd.DataFrame:
    """
    Display a boxplot of permutation importances and return a DataFrame with the importance values.

    :param permutation_importances: The permutation importance results.
    :param X_train_proc: The preprocessed training dataset.
    :return: A DataFrame containing the feature names, their importance means, and standard deviations.
    """
    # extract the mean and standard deviation of the permutation importances
    importance_means = permutation_importances.importances_mean
    importance_std = permutation_importances.importances_std

    # save the feature names
    feature_names = X_train_proc.columns

    # sort the features by their importance
    sorted_idx = np.argsort(importance_means)

    # create the boxplot
    plt.figure(figsize=(8, 6))
    plt.boxplot(permutation_importances.importances[sorted_idx].T, vert=False, tick_labels=feature_names[sorted_idx])

    # set labels and title
    plt.xlabel("Decrease in accuracy score")
    plt.title("Permutation Importances")
    plt.axvline(x=0, color="k", linestyle="--")
    plt.tight_layout()

    # display the plot
    plt.show()

    # create a DataFrame with the importance values
    imp_df = pd.DataFrame({
        "features": feature_names,
        "importance": importance_means,
        "importance_std": importance_std
    })

    # return the DataFrame sorted by importance in descending order
    return imp_df.sort_values(by="importance", ascending=False)


def evaluate_model_on_test_data(preprocessing_test: Union[ImbPipeline, Pipeline],
                                X_test: pd.DataFrame,
                                y_test: pd.Series,
                                trained_model: BaseEstimator) -> pd.DataFrame:
    """
    Evaluate a trained model on test data and return the performance metrics.

    :param preprocessing_test: The preprocessing pipeline to apply to the test data.
    :param X_test: The test features.
    :param y_test: The true labels for the test data.
    :param trained_model: The trained model to evaluate.
    :return: A DataFrame containing the evaluation metrics and their corresponding scores.
    """
    # apply the preprocessing pipeline to the test data
    X_test_proc = preprocessing_test.transform(X_test)

    # predict the labels for the preprocessed test data
    y_test_pred = trained_model.predict(X_test_proc)

    # calculate the performance metrics
    f1 = f1_score(y_test, y_test_pred)
    accuracy = accuracy_score(y_test, y_test_pred)
    recall = recall_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred)

    # store the results in a DataFrame
    df_test_results = pd.DataFrame({
        'Metric': ['F1 Score', 'Accuracy', 'Recall', 'Precision'],
        'Score': [f1, accuracy, recall, precision]
    })

    return df_test_results


def get_counterfactuals(dice: Dice,
                        instance_to_explain: pd.DataFrame,
                        total_cfs: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate counterfactual explanations for a given instance by varying each feature individually.

    :param dice: The DiCE explainer object used to generate counterfactuals.
    :param instance_to_explain: The instance for which counterfactual explanations are to be generated.
    :param total_cfs: The number of counterfactual explanations to generate per feature.
    :return: A tuple containing two DataFrames. The first DataFrame contains the generated counterfactuals along with
    their associated probabilities and the feature varied. The second DataFrame contains the original instance for
     which counterfactual explanations are to be generated.
    """
    explainer = []  # list to store the features that were varied
    cf_list = []  # list to store generated counterfactuals
    probabilities = []  # list to store prediction probabilities of counterfactuals
    features = dice.data_interface.continuous_feature_names  # features to vary

    for feature in features:
        try:
            # generate counterfactuals while varying the specified feature
            cf = dice.generate_counterfactuals(instance_to_explain, total_CFs=total_cfs, features_to_vary=[feature],
                                               desired_class="opposite", verbose=True, proximity_weight=1.5,
                                               random_seed=5)

            cf_list.append(cf)  # append generated counterfactuals to the list
            explainer.append([feature] * total_cfs)  # append the specified feature
            probabilities.append(dice.cfs_pred_scores[:, 1])  # extract probabilities

        except UserConfigValidationException:
            print(f"No counterfactuals found when only varying {feature}")

    if len(cf_list) > 0:
        # extract the instance to explain including its actual label
        test_instance = cf_list[0].cf_examples_list[0].test_instance_df

        # concatenate all counterfactuals into a single DataFrame
        cf_features = [cf.cf_examples_list[0].final_cfs_df for cf in cf_list]
        df_cfs = pd.concat(cf_features, ignore_index=True)

        # flatten the list of probabilities
        probabilities_flat = [item for sublist in probabilities for item in sublist]

        # add probabilities to the DataFrame
        df_cfs["probability"] = probabilities_flat

        # flatten the list of the features that were varied
        explainer_flat = [item for sublist in explainer for item in sublist]

        # add the features varied to the DataFrame
        df_cfs["feature_varied"] = explainer_flat

        return df_cfs, test_instance
    else:
        return pd.DataFrame(), pd.DataFrame()
