import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction

df = pd.read_pickle("../../data/processed/01_combined_data.pkl")
df.info()

predictor_columns = [col for col in df.columns if col != 'month']

# --------------------------------------------------------------
# Principal component analysis PCA

df_pca = df.copy()
PCA = PrincipalComponentAnalysis()

pc_values = PCA.determine_pc_explained_variance(df_pca, predictor_columns)

plt.figure(figsize=(10, 10))
plt.plot(range(1, len(predictor_columns) + 1), pc_values)
plt.xlabel("principal component number")
plt.ylabel("explain  variance")
plt.show()

df_pca = PCA.apply_pca(df_pca, predictor_columns, 4)

df_pca[["pca_1", "pca_2", "pca_3", "pca_4"]].plot()

df_pca.to_pickle("../../data/processed/02_feature_data.pkl")


########################################################################
import pandas as pd

class PrincipalComponentAnalysis2:
    def __init__(self):
        self.pca = None

    def normalize_dataset(self, data_table, cols):
        # Normalization (zero mean, unit variance)
        return data_table.copy().apply(lambda x: (x - x.mean()) / x.std() if x.name in cols else x)

    def apply_pca2(self, data_table, cols, number_comp):
        # Normalize the data first
        dt_norm = self.normalize_dataset(data_table, cols)

        # Perform PCA
        from sklearn.decomposition import PCA
        self.pca = PCA(n_components=number_comp)
        self.pca.fit(dt_norm[cols])

        # Transform the original dataset
        new_values = self.pca.transform(dt_norm[cols])

        # Add new PCA columns to the dataframe
        for comp in range(number_comp):
            data_table[f"pca_{comp + 1}"] = new_values[:, comp]

        return data_table

    def get_feature_importance2(self, feature_names):
        if self.pca is None:
            raise ValueError("PCA not yet applied. Call `apply_pca` first.")
        
        # Absolute value of the component loadings to identify influence
        importance = pd.DataFrame(
            self.pca.components_.T,
            columns=[f"pca_{i+1}" for i in range(self.pca.n_components_)],
            index=feature_names
        )
        return importance

# Example usage
# Assuming predictor_columns contains the columns used for PCA
PCA = PrincipalComponentAnalysis2()
df_pca = PCA.apply_pca2(df_pca, predictor_columns, 4)

# Get the importance of each feature in the PCA components
feature_importance = PCA.get_feature_importance2(predictor_columns)

# Print sorted importance for the first principal component
print("Feature importance for PCA components:")
print(feature_importance)

# To find the top features for PCA1
top_features_pca2 = feature_importance["pca_2"].abs().sort_values(ascending=False).head(5)
print("Top features contributing to PCA2:")
print(top_features_pca2)

