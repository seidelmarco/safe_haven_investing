import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.cluster import KMeans

from sklearn.datasets import load_wine, load_iris

sns.set()


def market_segmentation_clustering():
    """
    We try to cluster customer-groups of a retail shop based on customer-satisfaction and loyalty.
    Satisfaction is self-assessed/reported by a questionaire: 10 means extremely satisfied
    Brand-loyalty is a tricky metric: there is no widely accepted technique to measure it but there are proxies
    like churn rate, retention rate or customer lifetime value (CLV)
    Type of data here: continuous, range: (-2.5 to 2.5) - data is already standardized
    :return:
    """
    raw_data = pd.read_csv('data_regressions/3.12._Example.csv')
    print(raw_data.describe())
    print(raw_data)

    plt.scatter(raw_data['Satisfaction'], raw_data['Loyalty'], c='green')
    plt.title('Market Segmentation')
    plt.xlabel('Customer-satisfaction')
    plt.ylabel('Brand-loyalty')
    plt.show()

    X = raw_data.copy()
    kmeans = KMeans(2, n_init='auto')
    kmeans.fit(X)
    clusters = X.copy()
    clusters['cluster_pred'] = kmeans.fit_predict(X)
    print(clusters)

    plt.scatter(clusters['Satisfaction'], clusters['Loyalty'], c=clusters['cluster_pred'], cmap='rainbow')
    plt.title('Market Segmentation')
    plt.xlabel('Customer-satisfaction')
    plt.ylabel('Brand-loyalty')
    plt.show()

    """
    Interpretation so far: we see 2 vertical separated clusters which don't make sense. It seems that one
    feature was overestimated.
    Most probably the algorithm ONLY considered satisfaction as a feature. That's due to that we didn't
    standardize the feature!
    """

    x_scaled = preprocessing.scale(X)
    print(x_scaled)

    # cluster and plot again but use the Elbow-Method beforehand:
    wcss = []

    for i in range(1, 10):
        kmeans = KMeans(i, n_init='auto')
        kmeans.fit(x_scaled)
        wcss.append(kmeans.inertia_)

    print(wcss)
    plt.plot(range(1, 10), wcss)
    plt.title('The Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('Within-cluster sum of squared')
    # plt.xlim(0, 10)
    # plt.ylim(0, 1000)
    plt.show()

    kmeans = KMeans(4, n_init='auto')
    kmeans.fit(x_scaled)
    clusters = X.copy()
    clusters['cluster_pred'] = kmeans.fit_predict(x_scaled)
    print(clusters)

    plt.scatter(clusters['Satisfaction'], clusters['Loyalty'], c=clusters['cluster_pred'], cmap='rainbow')
    plt.title('Market Segmentation')
    plt.xlabel('Customer-satisfaction')
    plt.ylabel('Brand-loyalty')
    plt.show()


def species_segmentation_clustering():
    """
    Sepal: Kelchblatt
    :return:
    """
    raw_data = pd.read_csv('data_regressions/iris_dataset.csv')
    # print(raw_data.describe())
    # print(raw_data)

    # X = raw_data.iloc[:, 0:2]
    X = raw_data.copy()
    # print(X)

    plt.scatter(X['sepal_length'], X['sepal_width'], color='C8')
    plt.title('Iris-Data of Sepal length and width')
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.show()

    kmeans = KMeans(3, n_init='auto')
    kmeans.fit(X)
    identified_clusters = kmeans.fit_predict(X)
    # print(identified_clusters)

    data_with_clusters = X.copy()
    data_with_clusters['cluster'] = identified_clusters
    print(data_with_clusters)

    plt.scatter(X['sepal_length'], X['sepal_width'], c=data_with_clusters['cluster'], cmap='rainbow')
    plt.title('Iris-Data of Sepal length and width')
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.show()

    # Elbow-Method not standardized data:

    wcss = []
    cl_num = 11
    for item in range(1, cl_num):
        kmeans = KMeans(item, n_init='auto')
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)

    print(wcss)

    plt.plot(range(1, 11), wcss)
    plt.title('The Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('Within-cluster sum of squared')
    plt.show()

    # cluster again above with new K - like 4, 5, 6

    # Standardizing - normalize length and width data and compare:

    x_scaled = preprocessing.scale(X)
    print(x_scaled)

    kmeans = KMeans(3, n_init='auto')
    kmeans.fit(x_scaled)
    identified_clusters_scaled = kmeans.fit_predict(x_scaled)
    # print(identified_clusters_scaled)

    data_with_clusters_scaled = X.copy()
    data_with_clusters_scaled['cluster'] = identified_clusters_scaled
    print(data_with_clusters_scaled)

    plt.scatter(X['sepal_length'], X['sepal_width'], c=data_with_clusters_scaled['cluster'], cmap='rainbow')
    plt.title('Iris-Data of Sepal length and width - STANDARDIZED')
    plt.xlabel('Sepal length - axis not scaled')
    plt.ylabel('Sepal width - axis not scaled')
    plt.show()

    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(15, 4))
    fig.suptitle('Comparison of clustering not standardized data vs standardized features:')
    fig.supxlabel('Sepal length')
    fig.supylabel('Sepal width')
    ax1.scatter(X['sepal_length'], X['sepal_width'], c=data_with_clusters['cluster'], cmap='rainbow')
    ax2.scatter(X['sepal_length'], X['sepal_width'], c=data_with_clusters_scaled['cluster'], cmap='rainbow')
    plt.show()

    """
    Looks like the two solutions are nearly identical. That is because the original features have very
    similar scales to start with.
    
    Plot the data with 2, 3, 5 clusters. What do you think that means?
    
    Finally, import the csv with the correct answers and check if the clustering worked as expected.
    """

    kmeans2 = KMeans(2, n_init='auto')
    kmeans2.fit(x_scaled)
    identified_2clusters_scaled = kmeans2.fit_predict(x_scaled)
    # print(identified_2clusters_scaled)

    kmeans3 = KMeans(3, n_init='auto')
    kmeans3.fit(x_scaled)
    identified_3clusters_scaled = kmeans3.fit_predict(x_scaled)
    # print(identified_3clusters_scaled)

    kmeans5 = KMeans(5, n_init='auto')
    kmeans5.fit(x_scaled)
    identified_5clusters_scaled = kmeans5.fit_predict(x_scaled)
    # print(identified_5clusters_scaled)

    data_with_clusters_scaled_comp = X.copy()
    data_with_clusters_scaled_comp['2cluster'] = identified_2clusters_scaled
    data_with_clusters_scaled_comp['3cluster'] = identified_3clusters_scaled
    data_with_clusters_scaled_comp['5cluster'] = identified_5clusters_scaled
    print(data_with_clusters_scaled_comp)

    fig2, (ax21, ax22, ax23) = plt.subplots(1, 3, sharey=True, figsize=(18, 6))
    fig2.suptitle('Comparison of 2, 3 and 5 clusters:')
    fig2.supxlabel('Sepal length')
    fig2.supylabel('Sepal width')
    ax21.scatter(X['sepal_length'], X['sepal_width'], c=data_with_clusters_scaled_comp['2cluster'], cmap='rainbow')
    ax22.scatter(X['sepal_length'], X['sepal_width'], c=data_with_clusters_scaled_comp['3cluster'], cmap='rainbow')
    ax23.scatter(X['sepal_length'], X['sepal_width'], c=data_with_clusters_scaled_comp['5cluster'], cmap='rainbow')
    plt.show()

    data_solution = pd.read_csv('data_regressions/iris_with_answers.csv')
    print(data_solution)
    species = data_solution['species'].unique()
    print(species)
    data_solution['species'] = data_solution['species'].map({'setosa': 0, 'versicolor': 1, 'virginica': 2})

    fig3, (ax31, ax32) = plt.subplots(1, 2, sharey=False, figsize=(18, 7))
    fig3.suptitle('Comparison of 2 features influence - Sepal vs. Petal:')
    fig3.supxlabel('\n\nInterpretation: \n The feature petal has more clustering and predictive relevance. \n'
                   'Sepal-clusters are much more intertwined what doesn\'t reflect reality.\n'
                   'The Iris-flower data set is categorized by Petal length and width.')
    ax31.scatter(data_solution['sepal_length'], data_solution['sepal_width'], c=data_solution['species'], cmap='rainbow')
    ax31.set_title('Clustered by Sepal features')
    ax32.scatter(data_solution['petal_length'], data_solution['petal_width'], c=data_solution['species'], cmap='rainbow')
    ax32.set_title('Clustered by Petal features')
    plt.show()

    """
    Interpretation: the feature petal has more clustering and predictive relevance.
    Sepal-clusters are much more intertwined what doesn't reflect reality.
    The Iris-flower data set is categorized by Petal length and width.
    """

    wine_df = load_wine(return_X_y=True, as_frame=True)
    print(wine_df)


if __name__ == '__main__':
    market_segmentation_clustering()
    species_segmentation_clustering()

