import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.cluster import KMeans

from sklearn.datasets import load_wine, load_iris

sns.set()


def dendrogram_heatmap():
    """
    https://seaborn.pydata.org/generated/seaborn.clustermap.html?highlight=clustermap#seaborn.clustermap
    :return:
    """
    data = pd.read_csv('data_regressions/Country_clusters_standardized.csv', index_col='Country')
    x_scaled = data.copy()
    x_scaled.drop('Language', axis=1, inplace=True)

    print(x_scaled)

    sns.clustermap(x_scaled, figsize=(8, 8), cmap='mako')
    plt.show()

    data_market_segmentation = pd.read_csv('data_regressions/3.12._Example.csv')
    x_scaled_markets = data_market_segmentation.copy()
    print(x_scaled_markets)
    x_scaled_markets = preprocessing.scale(x_scaled_markets)
    x_scaled_markets_df = pd.DataFrame(x_scaled_markets)
    x_scaled_markets_df.rename(columns={0: 'Satisfaction', 1: 'Loyalty'}, inplace=True)

    print(x_scaled_markets_df)

    sns.clustermap(x_scaled_markets_df, figsize=(8, 8), cmap='mako')
    plt.show()


if __name__ == '__main__':
    dendrogram_heatmap()

