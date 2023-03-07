#import packages
import requests  
import re  
import pandas as pd  
#importing packages to support visualization
import matplotlib.pyplot as plt

#Seaborn is widely used for visualition
import seaborn as sns

import numpy as np   
from pandas import DataFrame
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
#PCA in sklearn package allows dimensionality reduction functionality
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer

clust_cv = pd.read_csv("/Users/rika/Documents/TM/clust_tfidf.csv", index_col=0) 
df_clust = pd.read_csv("/Users/rika/Documents/TM/dense_csv.csv", index_col=0) 

#plotting silhouette score to determine the best number of clusters
silhouette_coefficients = []
kmeans_kwargs = {
    "init": "random",
    "n_init": 10,
    "max_iter": 300,
    "random_state": 42,
}
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(clust_cv)
    score = silhouette_score(clust_cv, kmeans.labels_)
    silhouette_coefficients.append(score)
    
    
#plt.style.use("fivethirtyeight")
plt.plot(range(2, 11), silhouette_coefficients)
plt.xticks(range(2, 11))
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Coefficient")
plt.title("Silhouette Interpretation")
plt.show()

#running kmeans model on the tfidf dataframe
kmeans_cv = KMeans(n_clusters=6)
kmeans_cv.fit(clust_cv)
labels_cv = kmeans_cv.predict(clust_cv)
print(labels_cv)


#tfidf to create words frequency array
vect = TfidfVectorizer(stop_words='english', max_features=5000)

X = vect.fit_transform(df_clust['Headline'].values.astype('U'))

#X = vectorizer.fit_transform(text_clust['Clean_Text'].values.astype('U'))

# initialize PCA with 3 components
pca = PCA(n_components=2, random_state=42)
# pass our X to the pca and store the reduced vectors into pca_vecs
pca_vecs = pca.fit_transform(X.toarray())
# save our two dimensions into x0 and x1
x0 = pca_vecs[:, 0]
x1 = pca_vecs[:, 1]
df_clust['x0'] = x0
df_clust['x1'] = x1



# initialize kmeans with 3 centroids
#iterator=1
for i in range(2,10):
    kmeans = KMeans(n_clusters=i, random_state=42)
    # fit the model
    kmeans.fit(X)
    col_name = 'cluster' + str(i)
    # store cluster labels in a variable
    clusters = kmeans.labels_
    df_clust.loc[:, col_name] = clusters 
    #iterator += 1 
    def get_top_keywords(n_terms):
        """This function returns the keywords for each centroid of the KMeans"""
        print("\n")
        print("When K = ",i)
        print("\n")
        print('CENTROIDS ARE', ['bold'])
      
        df = pd.DataFrame(X.todense()).groupby(clusters).mean() # groups the TF-IDF vector by cluster
        terms = vect.get_feature_names_out() # access tf-idf terms
        for k,r in df.iterrows():
            print('Cluster {}'.format(k))
            # for each row of the dataframe, find the n terms that have the highest tf idf score
            print(','.join([terms[t] for t in np.argsort(r)[-n_terms:]])) 
            
    get_top_keywords(20)

#Now we have three K means results.
# Lets map clusters to appropriate labels 
cluster_map = {0: "Cluster1", 1: "Cluster2", 2: "Cluster3", 3: "Cluster4", 4: "Cluster5", 5: "Cluster6", 6: "Cluster7", 7: "Cluster8", }
# apply mapping to cluster3 column
df_clust['cluster6_label'] = df_clust['cluster6'].map(cluster_map)

plt.title("Count of each label - No of Clusters = 6", fontdict={"fontsize": 18})
df_clust.groupby("cluster6_label")["cluster6"].count().plot.pie(figsize=(5,5),autopct='%1.1f%%',label='cluster3_label')
plt.legend()

clust_6 = df_clust[["Headline", "cluster6"]]


plt.figure(figsize=(10, 10))
sns.lmplot(x='x0', y='x1', data=df_clust, hue='cluster6', fit_reg=False).set(title='K-Means Clustering K=6 ')









#def get_top_keywords(n_terms):
  #  """This function returns the keywords for each centroid of the KMeans"""
 #   df = pd.DataFrame(X.todense()).groupby(kmeans.labels_).mean() # groups the TF-IDF vector by cluster
  #  terms = vect.get_feature_names_out() # access tf-idf terms
   # for i,r in df.iterrows():
    #    print('\nCluster {}'.format(i))
     #   print(','.join([terms[t] for t in np.argsort(r)[-n_terms:]])) # for each row of the dataframe, find the n terms that have the highest tf idf score
            
#get_top_keywords(10)















