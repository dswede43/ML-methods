#!/usr/bin/python3

#Principle Component Analysis (PCA)
#---
#This script completes an unsupervised dimensionality reduction using principle component analysis.

#import libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt

#define global variables
dep_variable = 'species'

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
categorical_vars = ['species','island','sex','year'] #define the categorical variables
scale = StandardScaler()
df_scaled = scale.fit_transform(df.drop(categorical_vars, axis = 1))
df_scaled = pd.DataFrame(df_scaled)
df_scaled.columns = df.drop(categorical_vars, axis = 1).columns


#Run the PCA
#---
pca = PCA(n_components = 2)
principle_components = pca.fit_transform(df_scaled)
pca_df = pd.DataFrame(principle_components)
pca_df = pd.concat([df[categorical_vars], pca_df], axis = 1)
pca_df.columns = categorical_vars + ['PC1', 'PC2']

#calculate the explained variance
var = pca.explained_variance_ratio_


#Visualize the results
#---
#create feature vectors for biplot
feature_vectors = pd.DataFrame(pca.components_.T)
feature_vectors.columns = ['x','y']
feature_vectors['features'] = df.drop(categorical_vars, axis = 1).columns

#calculate the vector lengths
diagonals = []
for index, row in feature_vectors.iterrows():
    x, y = row['x'], row['y']
    diagonal = np.sqrt(x**2 + y**2)
    diagonals.append(diagonal)

feature_vectors['diagonal'] = diagonals
feature_vectors = feature_vectors.sort_values(by = ['diagonal'], ascending = False)

#set x and y axis boundaries
xs = np.array(pca_df['PC1'])
ys = np.array(pca_df['PC2'])

#create the PCA biplot
sns.scatterplot(x = 'PC1', y = 'PC2', data = pca_df,
    hue = dep_variable,
    palette = sns.color_palette('hls', len(df[dep_variable].unique())),
    size = 1,
    alpha = 0.5)
plt.xlabel(f"PC1 - {var[0] * 100:.2f}%")
plt.ylabel(f"PC2 - {var[1] * 100:.2f}%")
plt.title(f"PCA of Penguin characteristics coloured by {dep_variable}")

#add the feature vectors
for i in range(len(feature_vectors)):
    plt.arrow(0, 0, feature_vectors.iloc[i,0] * max(xs),
        feature_vectors.iloc[i,1] * max(ys),
        color='r', width=0.005, head_width=0.05)
    plt.text(feature_vectors.iloc[i,0] * max(xs),
        feature_vectors.iloc[i,1] * max(ys),
        feature_vectors.iloc[i,2], fontsize = 10)

#save plot as pdf
plt.savefig(dir + "unsupervised/dimension reduction/PCA_biplot.pdf", format = 'pdf')

plt.show()
