# Cryptocurrency Clustering with Principal Component Analysis

This project analyzes and clusters various cryptocurrencies based on their price change percentages over different periods using Principal Component Analysis (PCA).

## Project Description

In this project, we aim to perform clustering on a set of cryptocurrencies using PCA to reduce the dimensionality of the dataset. By visualizing the PCA components, we can better understand the relationships between features and how they influence each component.

## Dataset

The dataset contains information about the percentage price changes of cryptocurrencies over different periods such as 24 hours, 7 days, 30 days, etc. Each row represents a cryptocurrency, and each column represents a different percentage change period.

## Project Workflow

1. **Data Preprocessing**
   - Load the dataset into a pandas DataFrame.
   - Standardize the features using `StandardScaler` to ensure all features have a mean of 0 and a standard deviation of 1.

2. **Principal Component Analysis (PCA)**
   - Apply PCA to the standardized dataset to reduce the dimensionality to a set number of principal components.
   - Extract and visualize the PCA loadings, which show the contributions of each feature to each principal component.

3. **Visualizing PCA Loadings**
   - Create a table with features as indices and principal components as columns.
   - Plot a heatmap to visually represent the strength of influence each feature has on each principal component.

## Dependencies

- Python 3.x
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook

## How to Run the Project

1. **Clone the repository**:

    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2. **Install the required dependencies**:

    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn
    ```

3. **Open the Jupyter Notebook**:

    ```bash
    jupyter notebook Crypto_Clustering.ipynb
    ```

4. **Run the notebook cells** to perform data preprocessing, PCA, and visualizations.

## Example Code

Here is a sample code for performing PCA and visualizing the loadings:

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the dataset
# Assuming 'df' is your loaded DataFrame with price change features
df = pd.read_csv("crypto_dataset.csv")  # Replace with your dataset path

# Step 2: Standardize the features
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# Step 3: Apply PCA with a chosen number of components
pca = PCA(n_components=3)
pca.fit(df_scaled)

# Step 4: Extract and display the loadings
loadings = pd.DataFrame(
    pca.components_.T, 
    columns=[f'PCA{i+1}' for i in range(pca.n_components_)], 
    index=df.columns
)
print("PCA Loadings:")
print(loadings)

# Step 5: Plot the heatmap of PCA loadings
plt.figure(figsize=(10, 8))
sns.heatmap(loadings, annot=True, cmap='coolwarm', cbar_kws={'label': 'Feature Contribution'})
plt.title("PCA Loadings Heatmap")
plt.show()
