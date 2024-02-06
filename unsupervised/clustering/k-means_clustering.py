#!/usr/bin/python3

#K-means clustering
#---
#This script completes an unsupervised cluster analysis using the k-means method.

#import libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt

#define global variables
ind_variables = ['bill_length_mm','bill_depth_mm'] #independent variables of interest
dep_variable = 'species' #dependent variable of interest

#load the data
#This data contains information about the characteristics of Penguins
#collected by Dr. Kristen Gorman in the Palmer station in Antartica.
#https://github.com/tidyverse/nycflights13?tab=readme-ov-file
dir = "/path/to/directory/"
df = pd.read_csv(dir + "data/palmerpenguins.csv")
df = df.dropna() #remove NA values
df = df.reset_index(drop = True) #reset the index


#Scale the continuous variables
#---
scale = StandardScaler()
df_scaled = scale.fit_transform(df[ind_variables])
df_scaled = pd.DataFrame(df_scaled)
df_scaled.columns = df[ind_variables].columns


#Determine the optimal number of clusters
#---
def optimise_k(data, max_k):
    k_means = [] #number of clusters
    inertias = [] #measure of sum of square distances to nearest cluster center
    
    for k in range(1, max_k):
        kmeans = KMeans(n_clusters = k)
        kmeans.fit(data)
        
        k_means.append(k)
        inertias.append(kmeans.inertia_)
    
    return pd.DataFrame({'k_mean': k_means, 'inertia': inertias})

#visualize the elbow plot
elbow_df = optimise_k(df_scaled, 10)

plt.plot(elbow_df['k_mean'], elbow_df['inertia'], 'o-')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()


#Complete k-means clustering
#---
k = 3
kmeans = KMeans(n_clusters = k, random_state = 123)
clusters = kmeans.fit_predict(df_scaled)
df['clusters'] = clusters #add clusters to original data

#unscaled centroids
centroids = kmeans.cluster_centers_
unscaled_centroids = scale.inverse_transform(centroids)


#Visualize the results
#---
fig, (ax1, ax2) = plt.subplots(nrows = 2, ncols = 1, figsize = (8, 10))

#create data subplot
sns.scatterplot(x = ind_variables[0], y = ind_variables[1], data = df,
    hue = dep_variable,
    palette = sns.color_palette('hls', k),
    size = 1,
    ax = ax1)
ax1.set_xlabel(ind_variables[0])
ax1.set_ylabel(ind_variables[1])
ax1.set_title(f"Penguin {ind_variables[0]} and {ind_variables[1]} by {dep_variable}")

#create k-means cluster plot
sns.scatterplot(x = ind_variables[0], y = ind_variables[1], data = df,
    hue = 'clusters',
    palette = sns.color_palette('hls', k),
    size = 1,
    ax = ax2)
ax2.scatter(x = unscaled_centroids[:,0], y = unscaled_centroids[:,1],
    s = 100, c ='red', marker = 'o', label = 'Centroids')
ax2.set_xlabel(ind_variables[0])
ax2.set_ylabel(ind_variables[1])
ax2.set_title(f"K-means clustering of penguins by {dep_variable}")

plt.tight_layout
plt.show()

fig.savefig(dir + "unsupervised/clustering/k-means_clustering_results.pdf", format = 'pdf')
