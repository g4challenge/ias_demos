
#%%
import pandas as pd
import matplotlib.pyplot as plt

data = {
    "x1": [1,2,1,3,4,3,2,10,11,12,11,10,13,12],
    "x2": [2,2,6,1,3,4,5,10,10,11,12,13,12,14]
}

df = pd.DataFrame(data)
df.plot(
    x="x1",
    y="x2",  
    kind="scatter",  title="Datenset",  
    figsize=(10,10)
)

for item in range(len(data["x1"])):  
    plt.text(
        x=data["x1"][item]+0.2,
        y=data["x2"][item]+0.2,  s=item
)

# %%
from scipy.cluster.hierarchy import  dendrogram, linkage

Z = linkage(df, 'ward')

plt.figure(figsize=(10,10))  
dendrogram(Z)
plt.title("Agglomerative Hierarchical Clustering")  
plt.show()
plt.savefig("dendrogram.png")

# %%
from sklearn.cluster import KMeans

clusterer = KMeans(n_clusters=2)  
clusterer.fit(df)
cluster_prediction = clusterer.predict(df)  
centroids = clusterer.cluster_centers_.reshape((-1))

df1 = pd.Series(cluster_prediction, name="cluster")  
df2 = pd.concat([df, df1], axis=1)

plt.figure(figsize=(10,10))  
plt.scatter(
    x=df2.x1.loc[df2.cluster==0],  
    y=df2.x2.loc[df2.cluster==0],  
    label="Cluster 0"
)

plt.scatter(
    x=df2.x1.loc[df2.cluster==1],  
    y=df2.x2.loc[df2.cluster==1],  
    label="Cluster 1"
)
plt.scatter(
    x=[centroids[0], centroids[2]],  
    y=[centroids[1], centroids[3]],  
    label="Centroids",
    marker="^",  s=300
)

plt.title("K-Means Clustering with k=2")  
plt.legend(loc=0)

# %%
