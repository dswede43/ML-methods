#!/usr/bin/python3

#t-distributed Stochastic Neighbor Embedding (t-SNE)
#---
#This script completes an unsupervised dimensionality reduction using the t-SNE algorithm.

#import libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
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


#Perform the t-SNE
#---
tsne = TSNE(n_components = 2, random_state = 123)
tsne_components = tsne.fit_transform(df_scaled)
tsne_df = pd.DataFrame(tsne_components)
tsne_df = pd.concat([df[categorical_vars], tsne_df], axis = 1)
tsne_df.columns = categorical_vars + ['tSNE1', 'tSNE2']


#Visualize the results
#---
sns.scatterplot(x = 'tSNE1', y = 'tSNE2', data = tsne_df,
    hue = dep_variable,
    palette = sns.color_palette('hls', len(df[dep_variable].unique())),
    size = 1,
    alpha = 0.5)
plt.xlabel("t-SNE1")
plt.ylabel("t-SNE2")
plt.title(f"t-SNE of Penguin characteristics coloured by {dep_variable}")
plt.savefig(dir + "unsupervised/dimension reduction/tSNE_plot.pdf", format = 'pdf')
plt.show()
