un# Clustering

K-Means and agglomerative clustering are rather partitioning, rather than clustering. 

## K-Means Clustering

- with *K centroid*
- simplicity of algorithm
- scales well with large datasets
- *with-in* sum of squre: cost function
- Cluster mean: a vector

### How to find the HP *K*

- Elbow method: test different amount of clusters K, and calculate performance metric. 
- Silhouette plot (knives): silhouette coefficients (-1, 1)

### Drawbacks of K-Mean clustering
- cluster everything, including noise
- Not good at clustering: 
	- data with different density; 
	- data clustered not round-shaped. not able to capture complex structure

**Note**: scale variables before clustering

## Hierarchical Clustering

### HP:
- distance metric *d*
- similarity metric *D*

### Agglomerative algorithm
  a buttom-up clustering; starting with everything with one cluster; stop at certain number of cluster.

### Viz
#### Dendrogram
- Hight: distance between merging clusters
- 

### Drawbacks
- gets messy with large datasets.
- not able to capture complex structure
- not able to capture noise

## DBSCAN
	Density-Based Spatial Clustering of Applications with Noise

### HP:
- *eps*: epsilon, max distance between points of a cluster
- *min_samples* (of a cluster)
- every points gets a label in the end: *Core*, *Bounday* or *Noise*

### Advantages
- identify complex cluster 
- able to capture complex structure
- Able to distinguish noise








---
Useful links: 
[naftaliharris.com](https://www.naftaliharris.com/blog/)
[yellowbrick library](https://www.scikit-yb.org/en/latest/quickstart.html): not sure applicable for M1, recommended lib for error analysis

