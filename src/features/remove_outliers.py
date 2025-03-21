# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import math
# import scipy
# from sklearn.neighbors import LocalOutlierFactor 


# df = pd.read_pickle('../../data/processed/01_combined_data.pkl')

# plt.style.use("fivethirtyeight")
# plt.rcParams["figure.figsize"] = (20, 5)
# plt.rcParams["figure.dpi"] = 100

# outlier_columns = [col for col in df.columns if col != 'month']
# print(outlier_columns)
# len(outlier_columns)

# #disply outliers
# for outlier_column in outlier_columns:
#     df[[outlier_column] + ["month"]].boxplot(by="month", figsize=(20, 10))
#     plt.show()

# def plot_binary_outliers(dataset, col, outlier_col, reset_index):
#     """ Plot outliers in case of a binary outlier score. Here, the col specifies the real data
#     column and outlier_col the columns with a binary value (outlier or not).

#     Args:
#         dataset (pd.DataFrame): The dataset
#         col (string): Column that you want to plot
#         outlier_col (string): Outlier column marked with true/false
#         reset_index (bool): whether to reset the index for plotting
#     """

#     # Taken from: https://github.com/mhoogen/ML4QS/blob/master/Python3Code/util/VisualizeDataset.py

#     dataset = dataset.dropna(axis=0, subset=[col, outlier_col])
#     dataset[outlier_col] = dataset[outlier_col].astype("bool")

#     if reset_index:
#         dataset = dataset.reset_index()

#     fig, ax = plt.subplots()

#     plt.xlabel("samples")
#     plt.ylabel("value")

#     # Plot non outliers in default color
#     ax.plot(
#         dataset.index[~dataset[outlier_col]],
#         dataset[col][~dataset[outlier_col]],
#         "+",
#     )
#     # Plot data points that are outliers in red
#     ax.plot(
#         dataset.index[dataset[outlier_col]],
#         dataset[col][dataset[outlier_col]],
#         "r+",
#     )

#     plt.legend(
#         ["outlier " + col, "no outlier " + col],
#         loc="upper center",
#         ncol=2,
#         fancybox=True,
#         shadow=True,
#     )
#     plt.show()

# # --------------------------------------------------------------
# # Interquartile range (distribution based)
# # --------------------------------------------------------------

# # Insert IQR function
# def mark_outliers_iqr(dataset, col):
#     """Function to mark values as outliers using the IQR method.

#     Args:
#         dataset (pd.DataFrame): The dataset
#         col (string): The column you want apply outlier detection to

#     Returns:
#         pd.DataFrame: The original dataframe with an extra boolean column 
#         indicating whether the value is an outlier or not.
#     """

#     dataset = dataset.copy()

#     Q1 = dataset[col].quantile(0.25)
#     Q3 = dataset[col].quantile(0.75)
#     IQR = Q3 - Q1

#     lower_bound = Q1 - 1.5 * IQR
#     upper_bound = Q3 + 1.5 * IQR

#     dataset[col + "_outlier"] = (dataset[col] < lower_bound) | (
#         dataset[col] > upper_bound
#     )

#     return dataset

# # Plot a single column

# col = "Power_kW"
# dataset = mark_outliers_iqr(df, col)
# plot_binary_outliers(dataset=dataset, col=col, outlier_col=col+"_outlier",  reset_index=True)

# # Loop over all columns

# for col in outlier_columns:
#     dataset = mark_outliers_iqr(df, col)
#     plot_binary_outliers(dataset=dataset, col=col, outlier_col=col+"_outlier",  reset_index=True)


# # Check for normal distribution

# # Plot histograms for all columns
# for outlier_column in outlier_columns:
#     df[[outlier_column] + ["month"]].hist(by="month", figsize=(20, 10))
#     plt.show()

# # Insert Chauvenet's function

# def mark_outliers_chauvenet(dataset, col, C=2):
#     """Finds outliers in the specified column of datatable and adds a binary column with
#     the same name extended with '_outlier' that expresses the result per data point.
    
#     Taken from: https://github.com/mhoogen/ML4QS/blob/master/Python3Code/Chapter3/OutlierDetection.py

#     Args:
#         dataset (pd.DataFrame): The dataset
#         col (string): The column you want apply outlier detection to
#         C (int, optional): Degree of certainty for the identification of outliers given the assumption 
#                            of a normal distribution, typicaly between 1 - 10. Defaults to 2.

#     Returns:
#         pd.DataFrame: The original dataframe with an extra boolean column 
#         indicating whether the value is an outlier or not.
#     """

#     dataset = dataset.copy()
#     # Compute the mean and standard deviation.
#     mean = dataset[col].mean()
#     std = dataset[col].std()
#     N = len(dataset.index)
#     criterion = 1.0 / (C * N)

#     # Consider the deviation for the data points.
#     deviation = abs(dataset[col] - mean) / std

#     # Express the upper and lower bounds.
#     low = -deviation / math.sqrt(C)
#     high = deviation / math.sqrt(C)
#     prob = []
#     mask = []

#     # Pass all rows in the dataset.
#     for i in range(0, len(dataset.index)):
#         # Determine the probability of observing the point
#         prob.append(
#             1.0 - 0.5 * (scipy.special.erf(high.iloc[i]) - scipy.special.erf(low.iloc[i]))
#         )
#         # And mark as an outlier when the probability is below our criterion.
#         mask.append(prob[i] < criterion)
#     dataset[col + "_outlier"] = mask
#     return dataset

# # Loop over all columns

# for col in outlier_columns:
#     dataset = mark_outliers_chauvenet(df, col)
#     plot_binary_outliers(dataset=dataset, col=col, outlier_col=col+"_outlier",  reset_index=True)


# # --------------------------------------------------------------
# # Local outlier factor (distance based)
# # --------------------------------------------------------------

# # Insert LOF function
# def mark_outliers_lof(dataset, columns, n=20):
#     """Mark values as outliers using LOF

#     Args:
#         dataset (pd.DataFrame): The dataset
#         col (string): The column you want apply outlier detection to
#         n (int, optional): n_neighbors. Defaults to 20.
    
#     Returns:
#         pd.DataFrame: The original dataframe with an extra boolean column
#         indicating whether the value is an outlier or not.
#     """
    
#     dataset = dataset.copy()

#     lof = LocalOutlierFactor(n_neighbors=n)
#     data = dataset[columns]
#     outliers = lof.fit_predict(data)
#     X_scores = lof.negative_outlier_factor_

#     dataset["outlier_lof"] = outliers == -1
#     return dataset, outliers, X_scores

# # Loop over all columns

# dataset, outliers, X_scores = mark_outliers_lof(df, outlier_columns)

# for col in outlier_columns:
#     plot_binary_outliers(dataset=dataset, col=col, outlier_col="outlier_lof",  reset_index=True)





# class MarkOutliers():
#     def __init__(self):
#         pass

#     def plot_binary_outliers(dataset, col, outlier_col, reset_index):
#         """ Plot outliers in case of a binary outlier score. Here, the col specifies the real data
#         column and outlier_col the columns with a binary value (outlier or not).

#         Args:
#             dataset (pd.DataFrame): The dataset
#             col (string): Column that you want to plot
#             outlier_col (string): Outlier column marked with true/false
#             reset_index (bool): whether to reset the index for plotting
#         """

#         # Taken from: https://github.com/mhoogen/ML4QS/blob/master/Python3Code/util/VisualizeDataset.py

#         dataset = dataset.dropna(axis=0, subset=[col, outlier_col])
#         dataset[outlier_col] = dataset[outlier_col].astype("bool")

#         if reset_index:
#             dataset = dataset.reset_index()

#         fig, ax = plt.subplots()

#         plt.xlabel("samples")
#         plt.ylabel("value")

#         # Plot non outliers in default color
#         ax.plot(
#             dataset.index[~dataset[outlier_col]],
#             dataset[col][~dataset[outlier_col]],
#             "+",
#         )
#         # Plot data points that are outliers in red
#         ax.plot(
#             dataset.index[dataset[outlier_col]],
#             dataset[col][dataset[outlier_col]],
#             "r+",
#         )

#         plt.legend(
#             ["outlier " + col, "no outlier " + col],
#             loc="upper center",
#             ncol=2,
#             fancybox=True,
#             shadow=True,
#         )
#         plt.show() 
    
#     def mark_outliers_iqr(dataset, col):
#         """Function to mark values as outliers using the IQR method.

#         Args:
#             dataset (pd.DataFrame): The dataset
#             col (string): The column you want apply outlier detection to

#         Returns:
#             pd.DataFrame: The original dataframe with an extra boolean column 
#             indicating whether the value is an outlier or not.
#         """

#         dataset = dataset.copy()

#         Q1 = dataset[col].quantile(0.25)
#         Q3 = dataset[col].quantile(0.75)
#         IQR = Q3 - Q1

#         lower_bound = Q1 - 1.5 * IQR
#         upper_bound = Q3 + 1.5 * IQR

#         dataset[col + "_outlier"] = (dataset[col] < lower_bound) | (
#             dataset[col] > upper_bound
#         )

#         return dataset
    
#     def mark_outliers_chauvenet(dataset, col, C=2):
#         """Finds outliers in the specified column of datatable and adds a binary column with
#         the same name extended with '_outlier' that expresses the result per data point.
        
#         Taken from: https://github.com/mhoogen/ML4QS/blob/master/Python3Code/Chapter3/OutlierDetection.py

#         Args:
#             dataset (pd.DataFrame): The dataset
#             col (string): The column you want apply outlier detection to
#             C (int, optional): Degree of certainty for the identification of outliers given the assumption 
#                             of a normal distribution, typicaly between 1 - 10. Defaults to 2.

#         Returns:
#             pd.DataFrame: The original dataframe with an extra boolean column 
#             indicating whether the value is an outlier or not.
#         """

#         dataset = dataset.copy()
#         # Compute the mean and standard deviation.
#         mean = dataset[col].mean()
#         std = dataset[col].std()
#         N = len(dataset.index)
#         criterion = 1.0 / (C * N)

#         # Consider the deviation for the data points.
#         deviation = abs(dataset[col] - mean) / std

#         # Express the upper and lower bounds.
#         low = -deviation / math.sqrt(C)
#         high = deviation / math.sqrt(C)
#         prob = []
#         mask = []

#         # Pass all rows in the dataset.
#         for i in range(0, len(dataset.index)):
#             # Determine the probability of observing the point
#             prob.append(
#                 1.0 - 0.5 * (scipy.special.erf(high.iloc[i]) - scipy.special.erf(low.iloc[i]))
#             )
#             # And mark as an outlier when the probability is below our criterion.
#             mask.append(prob[i] < criterion)
#         dataset[col + "_outlier"] = mask
#         return dataset
    

#     def mark_outliers_lof(dataset, columns, n=20):
#         """Mark values as outliers using LOF

#         Args:
#             dataset (pd.DataFrame): The dataset
#             col (string): The column you want apply outlier detection to
#             n (int, optional): n_neighbors. Defaults to 20.
        
#         Returns:
#             pd.DataFrame: The original dataframe with an extra boolean column
#             indicating whether the value is an outlier or not.
#         """
        
#         dataset = dataset.copy()

#         lof = LocalOutlierFactor(n_neighbors=n)
#         data = dataset[columns]
#         outliers = lof.fit_predict(data)
#         X_scores = lof.negative_outlier_factor_

#         dataset["outlier_lof"] = outliers == -1
#         return dataset, outliers, X_scores


# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import os

# class OutlierDetection:
#     def __init__(self, df, output_dir="data/visualization/"):
#         """
#         Initialize the OutlierDetection class with the dataset.

#         Args:
#             data_path (str): Path to the processed data file.
#             output_dir (str): Directory to save plots.
#         """
#         # self.data_path = data_path
#         # self.df = self._load_data()
#         self.df = df
#         self.outlier_columns = [col for col in self.df.columns if col != 'month']
#         self.output_dir = output_dir

#         # Ensure the output directory exists
#         os.makedirs(self.output_dir, exist_ok=True)

#         # Configure plot settings
#         plt.style.use("fivethirtyeight")
#         plt.rcParams["figure.figsize"] = (20, 5)
#         plt.rcParams["figure.dpi"] = 100
    
#     def get_raw_data_fg():
#         """Returns the raw data for the FG dataset."""
#         hopsworks_feature_store = HopsworksFeatureStore()

#     # def save_boxplots(self):
#     #     """Saves boxplots of all numeric columns grouped by month."""
#     #     for col in self.outlier_columns:
#     #         fig, ax = plt.subplots(figsize=(20, 10))
#     #         self.df[[col, "month"]].boxplot(by="month", ax=ax)
            
#     #         # Save the plot
#     #         save_path = os.path.join(self.output_dir, f"boxplot_{col}.png")
#     #         plt.savefig(save_path)
#     #         plt.close(fig)  # Close the figure to free memory
#     #         print(f"Saved: {save_path}")

#     def save_boxplots(self):
#         """Saves boxplots of all numeric columns grouped by month."""
#         for col in self.outlier_columns:
#             fig, ax = plt.subplots(figsize=(20, 10))
            
#             # Check if there are valid month values to group by
#             if 'month' in self.df.columns and not self.df['month'].empty:
#                 # Make sure month column has valid values for grouping
#                 if self.df['month'].nunique() > 0:
#                     self.df.boxplot(column=col, by='month', ax=ax)
#                 else:
#                     # If month has no unique values, just create a regular boxplot
#                     self.df[col].plot(kind='box', ax=ax)
#                     ax.set_title(f'Boxplot of {col}')
#             else:
#                 # If no month column, just create a regular boxplot
#                 self.df[col].plot(kind='box', ax=ax)
#                 ax.set_title(f'Boxplot of {col}')
            
#             # Save the plot
#             save_path = os.path.join(self.output_dir, f"boxplot_{col}.png")
#             plt.savefig(save_path)
#             plt.close(fig)  # Close the figure to free memory
#             print(f"Saved: {save_path}")

#     def mark_outliers_iqr(self, col):
#         """Identifies outliers in a column using the IQR method.

#         Args:
#             col (str): Column name to analyze.

#         Returns:
#             pd.DataFrame: Dataframe with a new boolean column marking outliers.
#         """
#         dataset = self.df.copy()
#         Q1 = dataset[col].quantile(0.25)
#         Q3 = dataset[col].quantile(0.75)
#         IQR = Q3 - Q1

#         lower_bound = Q1 - 1.5 * IQR
#         upper_bound = Q3 + 1.5 * IQR

#         dataset[col + "_outlier"] = (dataset[col] < lower_bound) | (dataset[col] > upper_bound)
#         return dataset

#     def save_binary_outliers(self, dataset, col, outlier_col, reset_index=True):
#         """Saves outlier plots instead of showing them.

#         Args:
#             dataset (pd.DataFrame): Dataset containing the outlier column.
#             col (str): The column to plot.
#             outlier_col (str): Boolean column indicating outliers.
#             reset_index (bool): Whether to reset the index for plotting.
#         """
#         dataset = dataset.dropna(subset=[col, outlier_col])
#         dataset[outlier_col] = dataset[outlier_col].astype(bool)

#         if reset_index:
#             dataset = dataset.reset_index()

#         fig, ax = plt.subplots()
#         plt.xlabel("Samples")
#         plt.ylabel("Value")

#         # Plot non-outliers
#         ax.plot(dataset.index[~dataset[outlier_col]], dataset[col][~dataset[outlier_col]], "+")
#         # Plot outliers in red
#         ax.plot(dataset.index[dataset[outlier_col]], dataset[col][dataset[outlier_col]], "r+")

#         plt.legend(
#             ["No Outlier " + col, "Outlier " + col],
#             loc="upper center",
#             ncol=2,
#             fancybox=True,
#             shadow=True,
#         )

#         # Save plot
#         save_path = os.path.join(self.output_dir, f"outliers_{col}.png")
#         plt.savefig(save_path)
#         plt.close(fig)  # Close to free memory
#         print(f"Saved: {save_path}")

#     def detect_and_save_outliers(self):
#         """Loops through all numeric columns, detects outliers, and saves plots."""
#         for col in self.outlier_columns:
#             dataset = self.mark_outliers_iqr(col)
#             self.save_binary_outliers(dataset=dataset, col=col, outlier_col=col + "_outlier", reset_index=True)


# # Example usage
# if __name__ == "__main__":

#     # Define dataset path and output directory for plots
#     data_path = 'data/processed/01_combined_data.pkl'
#     output_dir = 'reports/figures' 

#     # Create an instance of the OutlierDetection class
#     outlier_detector = OutlierDetection(data_path, output_dir)

#     # Save boxplots instead of displaying them
#     outlier_detector.save_boxplots()

#     # Detect and save outlier plots
#     outlier_detector.detect_and_save_outliers()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

class OutlierDetection:
    def __init__(self, df, output_dir="data/visualization/"):
        """
        Initialize the OutlierDetection class with the dataset.
        
        Args:
            df (pandas.DataFrame): DataFrame containing the data.
            output_dir (str): Directory to save plots.
        """
        self.df = df
        # Filter to only include numeric columns (excluding 'month')
        self.outlier_columns = [col for col in self.df.select_dtypes(include=np.number).columns 
                               if col != 'month']
        self.output_dir = output_dir
        
        # Ensure the output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Configure plot settings
        plt.style.use("fivethirtyeight")
        plt.rcParams["figure.figsize"] = (20, 5)
        plt.rcParams["figure.dpi"] = 100
    
    def save_boxplots(self):
        """Saves boxplots of all numeric columns grouped by month."""
        for col in self.outlier_columns:
            fig, ax = plt.subplots(figsize=(20, 10))
            
            # Check if there are valid month values to group by
            if 'month' in self.df.columns and not self.df['month'].empty:
                # Make sure month column has valid values for grouping
                if self.df['month'].nunique() > 0:
                    # Ensure we're only working with numeric data
                    valid_data = self.df[[col, 'month']].dropna()
                    if not valid_data.empty:
                        valid_data.boxplot(column=col, by='month', ax=ax)
                    else:
                        ax.text(0.5, 0.5, f"No valid data for {col}", ha='center', va='center')
                else:
                    # If month has no unique values, just create a regular boxplot
                    self.df[col].dropna().astype(float).plot(kind='box', ax=ax)
                    ax.set_title(f'Boxplot of {col}')
            else:
                # If no month column, just create a regular boxplot
                self.df[col].dropna().astype(float).plot(kind='box', ax=ax)
                ax.set_title(f'Boxplot of {col}')
            
            # Save the plot
            save_path = os.path.join(self.output_dir, f"boxplot_{col}.png")
            plt.savefig(save_path)
            plt.close(fig)  # Close the figure to free memory
            print(f"Saved: {save_path}")
    
    def mark_outliers_iqr(self, col):
        """Identifies outliers in a column using the IQR method.
        
        Args:
            col (str): Column name to analyze.
        
        Returns:
            pd.DataFrame: Dataframe with a new boolean column marking outliers.
        """
        dataset = self.df.copy()
        
        # Ensure we're working with numeric data
        try:
            # Convert to numeric and handle NaN values
            numeric_values = pd.to_numeric(dataset[col], errors='coerce')
            
            # Skip columns with all NaN values
            if numeric_values.isna().all():
                dataset[col + "_outlier"] = False
                return dataset
                
            # Calculate quantiles on non-NaN values
            valid_values = numeric_values.dropna()
            Q1 = valid_values.quantile(0.25)
            Q3 = valid_values.quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Identify outliers in the original series (with NaN handling)
            is_outlier = (numeric_values < lower_bound) | (numeric_values > upper_bound)
            dataset[col + "_outlier"] = is_outlier.fillna(False)  # NaN values are not outliers
            
        except (TypeError, ValueError) as e:
            print(f"Error processing column {col}: {e}")
            dataset[col + "_outlier"] = False
            
        return dataset
    
    def save_binary_outliers(self, dataset, col, outlier_col, reset_index=True):
        """Saves outlier plots instead of showing them.
        
        Args:
            dataset (pd.DataFrame): Dataset containing the outlier column.
            col (str): The column to plot.
            outlier_col (str): Boolean column indicating outliers.
            reset_index (bool): Whether to reset the index for plotting.
        """
        # Ensure we're working with numeric data for the column
        try:
            # Convert to numeric and handle NaN values
            dataset[col] = pd.to_numeric(dataset[col], errors='coerce')
            
            # Drop rows where either the column value or the outlier indicator is NaN
            valid_data = dataset.dropna(subset=[col, outlier_col])
            
            # Check if we have any valid data to plot
            if valid_data.empty:
                print(f"No valid data for plotting outliers in column {col}")
                # Create an empty plot with a message
                fig, ax = plt.subplots()
                ax.text(0.5, 0.5, f"No valid data for plotting outliers in {col}", 
                        ha='center', va='center')
                save_path = os.path.join(self.output_dir, f"outliers_{col}.png")
                plt.savefig(save_path)
                plt.close(fig)
                return
                
            valid_data[outlier_col] = valid_data[outlier_col].astype(bool)
            
            if reset_index:
                valid_data = valid_data.reset_index()
            
            fig, ax = plt.subplots()
            plt.xlabel("Samples")
            plt.ylabel("Value")
            
            # Plot non-outliers
            ax.plot(valid_data.index[~valid_data[outlier_col]], valid_data[col][~valid_data[outlier_col]], "+")
            # Plot outliers in red
            ax.plot(valid_data.index[valid_data[outlier_col]], valid_data[col][valid_data[outlier_col]], "r+")
            
            plt.legend(
                ["No Outlier " + col, "Outlier " + col],
                loc="upper center",
                ncol=2,
                fancybox=True,
                shadow=True,
            )
            
            # Save plot
            save_path = os.path.join(self.output_dir, f"outliers_{col}.png")
            plt.savefig(save_path)
            plt.close(fig)  # Close to free memory
            print(f"Saved: {save_path}")
            
        except Exception as e:
            print(f"Error processing outliers for column {col}: {e}")
            # Create an error plot
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, f"Error processing outliers for {col}: {str(e)}", 
                    ha='center', va='center', wrap=True)
            save_path = os.path.join(self.output_dir, f"outliers_{col}_error.png")
            plt.savefig(save_path)
            plt.close(fig)
    
    def detect_and_save_outliers(self):
        """Loops through all numeric columns, detects outliers, and saves plots."""
        for col in self.outlier_columns:
            try:
                dataset = self.mark_outliers_iqr(col)
                self.save_binary_outliers(dataset=dataset, col=col, outlier_col=col + "_outlier", reset_index=True)
            except Exception as e:
                print(f"Failed to process column {col}: {e}")