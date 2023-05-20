import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

sns.set()


def basic_cluster_analysis():
    """
    Choose the number of clusters (K) and do some adjusting of the observations with the help of
    optimizing the euclidean distance of the cendroids
    :return:
    """
    data = pd.read_csv('data_regressions/3.01._Country_clusters.csv')
    print(data)

    plt.scatter(data['Longitude'], data['Latitude'])
    # change the limits (the natural domain of a function) so that the plot resembles more a real world map:
    plt.xlim(-180, 180)
    plt.ylim(-90, 90)
    plt.show()

    # slice the needed data for the clustering - selection by postition:
    # first argument - rows: we use all rows, so :
    # second argument - columns: we slice second and third (but not 3)
    X = data.iloc[:, 1:3]
    print(X)

    kmeans = KMeans(3, n_init='auto')
    # next line only for checking purposes
    clusters = kmeans.fit(X)

    clusters = kmeans.fit_predict(X)
    print(clusters)

    # plt.scatter(data['Longitude'], data['Latitude'], c=clusters, cmap='rainbow')

    """
    It's saver to create a dataframe with the clusters
    """
    data_with_clusters = data.copy()
    data_with_clusters['Cluster'] = clusters

    # the hack is, in matplotlib we can set the color to be determined by a variable
    plt.scatter(data_with_clusters['Longitude'], data_with_clusters['Latitude'],
                c=data_with_clusters['Cluster'], cmap='rainbow')

    # change the limits (the natural domain of a function) so that the plot resembles a more real world map:
    plt.xlim(-180, 180)
    plt.ylim(-90, 90)
    plt.show()


def basic_cluster_analysis_exercise_elbow_method_without_continents():
    """
    Choose the number of clusters (K) and do some adjusting of the observations with the help of
    optimizing the euclidean distance of the cendroids - play with more country data and several numbers of clusters.
    We have 241 countries :-)
    :return:
    """
    data = pd.read_csv('data_regressions/Countries_exercise.csv')
    print(data)

    X = data.iloc[:, 1:3]
    print(X)

    plt.scatter(X['Longitude'], X['Latitude'])
    plt.xlim(-200, 200)
    plt.ylim(-90, 90)
    plt.show()

    kmeans = KMeans(3, n_init='auto')  # 7 like number of continents
    clusters = kmeans.fit_predict(X)
    # print(clusters)

    data_with_clusters = data.copy()
    data_with_clusters['Cluster'] = clusters
    print(data_with_clusters)

    plt.scatter(data_with_clusters['Longitude'], data_with_clusters['Latitude'],
                c=data_with_clusters['Cluster'], cmap='rainbow')
    plt.xlim(-200, 200)
    plt.ylim(-90, 90)
    plt.show()

    # WCSS
    wcss = []
    # cl_num is a variable that tracks the highest number of clusters we want  to use the WCSS method for.
    # Note that range() doesn't include the upper boundary

    cl_num = 11
    for i in range(1, cl_num):
        kmeans = KMeans(i, n_init='auto')
        kmeans.fit(X)
        wcss_iter = kmeans.inertia_
        wcss.append(wcss_iter)

    number_clusters = range(1, cl_num)
    plt.plot(number_clusters, wcss)
    plt.title('The Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('Within-cluster sum of squared')
    plt.show()


def cluster_analysis_exercise_categories_elbow_method_standardizing():
    """

    :return:
    """

    raw_data = pd.read_csv('data_regressions/Categorical.csv')
    #print(raw_data)

    # map categoricals to numericals:
    print(raw_data['continent'].unique())
    data_mapped = raw_data.copy()
    data_mapped['continent'] = data_mapped['continent'].map({'North America': 0, 'Asia': 1, 'Africa': 2, 'Europe': 3,
                                                             'South America': 4, 'Oceania': 5, 'Antarctica': 7,
                                                             'Seven seas (open ocean)': 6})
    X = data_mapped.drop('name', axis=1)
    # oder (besser)
    X = data_mapped.iloc[:, 1:4]
    print(X)

    # clustering:
    kmeans = KMeans(8, n_init='auto')
    clusters = kmeans.fit(X)
    print(f"""
        Number of our chosen clusters (K):
        {clusters}
        """)
    identified_clusters = kmeans.fit_predict(X)
    print(identified_clusters)

    # we just copy an existing dataframe:
    data_with_clusters = data_mapped.copy()
    data_with_clusters['cluster'] = identified_clusters
    print(data_with_clusters)

    plt.scatter(data_with_clusters['Longitude'], data_with_clusters['Latitude'], c=data_with_clusters['cluster'],
                cmap='rainbow')
    plt.xlim(-180, 180)
    plt.ylim(-90, 90)
    plt.show()

    # choose the right number of clusters - Elbow-Method:
    """
    WCSS - Within-cluster sum of squares
    If we minimize the distance between points in a cluster, we automatically maximize the distance between
    the clusters
    WCSS of 0 seems to be good but if we have 6 countries in 6 clusters (continents) and our WCSS is 0 then
    our clustering is pointless. So we need a decent approximation and middle ground
    """

    wcss = kmeans.inertia_
    print(wcss)
    # we need to do this for every observation:

    rows = data_mapped['name'].count()
    print(rows)

    country_list = [item for item in data_mapped['name']]
    print(country_list)

    wcss_list = []

    for item in range(1, rows):
        kmeans = KMeans(item, n_init='auto')
        kmeans.fit(X)
        wcss_iter = kmeans.inertia_
        wcss_list.append(wcss_iter)

    print(wcss_list)

    plt.plot(wcss_list)
    plt.title('The Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('Within-cluster sum of squared')
    plt.xlim(0, 10)
    #plt.ylim(0, 1000)
    plt.show()

    # Standardizing yes or no:
    """
    There is a dispute on that topic, but: it is a good practice to standardize data before clustering, esp.
    for beginners. When we know that one variable is inherently more important than the other, then we should
    standardize, like homeprices and size - almost always price matters (if you can't afford the price you
    won't care about the size)
    You should not standardize
    """
    scaler = StandardScaler()
    scaler.fit(X)
    scaled_features = scaler.transform(X)
    print(scaled_features)


if __name__ == '__main__':
    basic_cluster_analysis()
    basic_cluster_analysis_exercise_elbow_method_without_continents()
    cluster_analysis_exercise_categories_elbow_method_standardizing()

