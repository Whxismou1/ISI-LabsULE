#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 21:29:11 2024

@author: @whxismou
"""
# import necessary packages
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.cluster.hierarchy import fcluster
from sklearn.cluster import DBSCAN

def main():
    # load the AirlinesCluster dataset
    url = "Data/AirlinesCluster.csv"
    airlines_dataset = pd.read_csv(url)
    # suppress scientific float notation
    np.set_printoptions(precision=5, suppress=True)
    # visualization of the first few records of the dataset
    print(airlines_dataset.head())
    print(airlines_dataset.head(n=2))
    # visualization of the last few records from the dataset
    print(airlines_dataset.tail())
    print(airlines_dataset.tail(n=2))
    # display the different datatypes available in the dataset
    print(airlines_dataset.dtypes)
    # visualization of descriptive stats of the dataset
    print(airlines_dataset.describe())

    # scale data
    dataset_standardized = preprocessing.scale(airlines_dataset)
    # visualization of descriptive stats of the normalized dataset
    dataset_standardized = pd.DataFrame(dataset_standardized)
    print(dataset_standardized.describe().to_string())

    # %%
    # K-MEANS
    plt.figure(figsize=(8, 7))
    # visualization of the original data
    plt.subplot(221)
    plt.scatter(dataset_standardized[0], dataset_standardized[1])
    plt.title("Original unclustered data")
    # train the model using k=2
    kmeans_model = KMeans(n_clusters=2, random_state=42)
    kmeans_model.fit(dataset_standardized)
    kmeans_predictions = kmeans_model.predict(dataset_standardized)
    # visualization of the clustered data using k=2
    plt.subplot(222)
    plt.scatter(dataset_standardized[0],
                dataset_standardized[1], c=kmeans_predictions)
    plt.title("Clustered data, k=2")
    # train the model using k=3
    kmeans_model = KMeans(n_clusters=3, random_state=42)
    kmeans_model.fit(dataset_standardized)
    kmeans_predictions = kmeans_model.predict(dataset_standardized)
    # visualization of the clustered data using k=3
    plt.subplot(223)
    plt.scatter(dataset_standardized[0],
                dataset_standardized[1], c=kmeans_predictions)
    plt.title("Clustered data, k=3")
    # train the model using k=5
    kmeans_model = KMeans(n_clusters=5, random_state=42)
    kmeans_model.fit(dataset_standardized)
    kmeans_predictions = kmeans_model.predict(dataset_standardized)
    # visualization of the clustered data using k=5
    plt.subplot(224)
    plt.scatter(dataset_standardized[0],
                dataset_standardized[1], c=kmeans_predictions)
    plt.title("Clustered data, k=5")
    plt.show()

    # visualization of descriptive stats of the clustered data using k=5
    cluster_kmeans_data = pd.DataFrame(kmeans_predictions+1)
    airlines_dataset['KmeansCluster'] = cluster_kmeans_data
    kmeans_mean_cluster_pd = pd.DataFrame(
        airlines_dataset.groupby('KmeansCluster').mean())
    pd.DataFrame(airlines_dataset.groupby('KmeansCluster').mean())
    print(kmeans_mean_cluster_pd)

    # %%
    # The elbow method
    # Implementation of the Elbow method
    Sum_of_squared_errors = []
    # Definition of the number of clusters to evaluated
    K = range(1, 30)
    for k in K:
        # Training of kmeans model using k clusers
        kmeans_model = KMeans(n_clusters=k, random_state=42)
        kmeans_model.fit(dataset_standardized)
        # Computing the sum of squared errors for the trained model
        Sum_of_squared_error = kmeans_model.inertia_
        Sum_of_squared_errors.append(Sum_of_squared_error)
    # Visualization of the Sum of Squared Error vs the Number of clusters curve
    plt.figure(figsize=(10, 5))
    plt.plot(K, Sum_of_squared_errors, 'bx-')
    plt.xlabel('Number of clusters, k')
    plt.ylabel('Sum of squared errors')
    plt.title('Elbow Method For Optimal k')
    plt.show()

    # %%
    # Hierarchical Clustering
    # Loading and standardizing the data to normal distribution
    url = "Data/AirlinesCluster.csv"
    airlines_dataset = pd.read_csv(url)
    dataset_standardized = preprocessing.scale(airlines_dataset)
    dataset_standardized_pd = pd.DataFrame(dataset_standardized)
    # Creating the linkage matrix and perform hierarchical clustering on samples
    hierarchical_cluster = linkage(dataset_standardized, 'ward')
    # Plot a complete dendrogram
    plt.title('Hierarchical Clustering Dendrogram (complete)')
    plt.xlabel('Sample index or (cluster size)')
    plt.ylabel('Distance')
    dendrogram(hierarchical_cluster,
               leaf_rotation=90,
               leaf_font_size=6,
               )
    plt.show()
    # Plot a truncated dendrogram at 5 clusters
    plt.title('Hierarchical Clustering Dendrogram (truncated)')
    plt.xlabel('Sample index or (cluster size)')
    plt.ylabel('Distance')
    dendrogram(
        hierarchical_cluster,
        truncate_mode='lastp',  # show only the last p merged clusters
        p=5,  # show only the last p merged clusters
        leaf_rotation=90.,
        leaf_font_size=12.,
        show_contracted=True,  # to get a distribution impression in truncated branches
    )
    plt.show()
    # visualization of the clustered data using 5 clusters
    num_clusters = 5
    hierarchical_cluster_predictions = fcluster(hierarchical_cluster, num_clusters,
                                                criterion='maxclust')
    hierarchical_cluster_predictions[0:30:,]
    # Plotting clustered data using independent colors
    plt.figure(figsize=(10, 8))
    dataset_standardized_pd = pd.DataFrame(dataset_standardized)
    plt.scatter(dataset_standardized_pd.iloc[:, 0],
                dataset_standardized_pd.iloc[:, 1], c=hierarchical_cluster_predictions, cmap='prism')
    plt.title('Airline Data - Hierarchical Clutering, 5 clusters')
    plt.show()
    # visualization of descriptive stats of the clustered data using 5 clusters
    cluster_Hierarchical_data = pd.DataFrame(hierarchical_cluster_predictions)
    airlines_dataset['HierarchicalCluster'] = cluster_Hierarchical_data
    hierarchical_cluster_pd = pd.DataFrame(airlines_dataset.groupby('HierarchicalCluster').mean())
    pd.DataFrame(airlines_dataset.groupby('HierarchicalCluster').mean())
    print(hierarchical_cluster_pd)
    # %%
    # DBSCAN
    
    # Create an instance of DBSCAN with eps=0.3 and min_samples=10
    dbscan_a = DBSCAN(eps=0.3, min_samples=10)
    
    # Fit and predict for scenario a
    labels_a = dbscan_a.fit_predict(dataset_standardized)
    
    # Calculate the estimated number of clusters and noise points for scenario a
    n_clusters_a = len(set(labels_a)) - (1 if -1 in labels_a else 0)
    n_noise_a = list(labels_a).count(-1)
    
    # Repeat the same process for scenario b and c
    dbscan_b = DBSCAN(eps=0.3, min_samples=5)
    labels_b = dbscan_b.fit_predict(dataset_standardized)
    n_clusters_b = len(set(labels_b)) - (1 if -1 in labels_b else 0)
    n_noise_b = list(labels_b).count(-1)
    
    dbscan_c = DBSCAN(eps=0.2, min_samples=10)
    labels_c = dbscan_c.fit_predict(dataset_standardized)
    n_clusters_c = len(set(labels_c)) - (1 if -1 in labels_c else 0)
    n_noise_c = list(labels_c).count(-1)
    
    # Print the results
    print("Scenario a - Estimated number of clusters: %d" % n_clusters_a)
    print("Scenario a - Estimated number of noise points: %d" % n_noise_a)
    print()
    print("Scenario b - Estimated number of clusters: %d" % n_clusters_b)
    print("Scenario b - Estimated number of noise points: %d" % n_noise_b)
    print()
    print("Scenario c - Estimated number of clusters: %d" % n_clusters_c)
    print("Scenario c - Estimated number of noise points: %d" % n_noise_c)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
if __name__ == "__main__":
    main()
