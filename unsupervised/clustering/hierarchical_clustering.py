#!/usr/bin/python3

#Hierarchical clustering
#---
#This script completes an unsupervised cluster analysis using the hierarchical clustering method.

#import libraries
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform
import seaborn as sns
import matplotlib.pyplot as plt

#define global variables
ind_variables = ['bill_length_mm','bill_depth_mm'] #independent variables of interest
dep_variable = 'species' #dependent variable of interest

#load the data
#This data contains information about the characteristics of Penguins
#collected by Dr. Kristen Gorman in the Palmer station in Antartica.
#https://github.com/allisonhorst/palmerpenguins
dir = "/path/to/directory/"
df = pd.read_csv(dir + "data/palmerpenguins.csv")
df = df.dropna() #remove NA values
df = df.reset_index(drop = True) #reset the index


#Hierarchical clustering
#---
#calculate the distance matrix
dist_mat = squareform(pdist(df[ind_variables], metric = 'euclidean'))

#perform the hierarchical clustering
hclust = linkage(dist_mat, method = 'complete')

#plot the dendrogram
dendrogram(hclust,
           orientation = 'top',
           distance_sort = 'descending',
           labels = None,
           show_leaf_counts = True)
plt.title(f"Dendrogram of penguin {ind_variables[0]} and {ind_variables[1]}")
plt.xlabel('Penguin')
plt.ylabel('Distance')
plt.xticks([])
plt.savefig(dir + "unsupervised/clustering/hierarchical_clustering_dendrogram.pdf", format = 'pdf')
plt.show()

#add the cluster labels
k = 3
clusters = fcluster(hclust, k, criterion = 'maxclust')
df['clusters'] = clusters


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
ax2.set_xlabel(ind_variables[0])
ax2.set_ylabel(ind_variables[1])
ax2.set_title(f"Hierarchical clustering of penguins by {dep_variable}")

plt.tight_layout
plt.show()

fig.savefig(dir + "unsupervised/clustering/hierarchical_clustering_results.pdf", format = 'pdf')
