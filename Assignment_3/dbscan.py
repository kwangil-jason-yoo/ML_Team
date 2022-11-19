import os, sys
import numpy as np
import math

from data import readDataLabels
from utils import plot, euclidean_distance

from sklearn.cluster import DBSCAN as sklearnDBSCAN     # Only jerks will use this in assignment!


def test_sklearn(X):
    # Lets see how sklearn performs.
    db = sklearnDBSCAN(eps=0.2, min_samples=10).fit(X)
    plot(X, db.labels_)


class DBSCAN():

    """

        eps  - The radius within which samples are considered neighbors.
        min_samples - The minimum number of samples required by a point to
                        be considered core point. 
    """
    def __init__(self, eps=1.0, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples


    def get_neighbors(self, sample_i): #TODO
        """ Returns a list of indices of neighoring samples
            A sample_a is considered a neighebor of sample_b if the distance 
            between them less than eps """
        neighbors = []
        #idxs = np.arange(len(self.X))
        # for i, _sample in enumerate(self.X[idxs != sample_i]):
        
        #     # TODO
        #     if euclidean_distance(self.X[sample_i], _sample) < self.eps:
        #         neighbors.append(i)
        for i, _sample in enumerate(self.X):
            if euclidean_distance(self.X[sample_i], _sample) <= self.eps and list(self.X[i]) != list(self.X[sample_i]):
                neighbors.append(i)
            
        return np.asarray(neighbors)


    def expand_cluster(self, sample_i, neighbors): #TODO
        """ Recursively expand cluster until the border of the dense area is reached
            Dense area is determined by eps and min_samples """
        cluster = [sample_i]
        # Iterate through neighbors
        for neighbor_i in neighbors:
            if not neighbor_i in self.visited_samples:
                self.visited_samples.append(neighbor_i)
                # Finish this function
                # Hint:
                # Fetch the sample's distant neighbors (neighbors of neighbor)
                # Make sure the neighbor's neighbors are more than min_samples
                # (If this is true the neighbor is a core point) -> call this function to expand the cluster.
                # Else, the neighbor is not a core point so just add it to the cluster
                neighbors_of_neighbor_i = self.get_neighbors(neighbor_i)
                if len(neighbors_of_neighbor_i) >= self.min_samples:
                    cluster+= self.expand_cluster(neighbor_i, neighbors_of_neighbor_i)
                else:
                    if neighbor_i not in cluster:
                        cluster.append(neighbor_i)

        return cluster


    def det_cluster_labels(self): #TODO
        """" Return the sample labels as the index of the cluster in which they are contained """
        # Set default value to number of clusters
        # This will make sure all outliers have same cluster label
        labels = np.full(shape=self.X.shape[0], fill_value=len(self.clusters))
        # Finish this
        for i,j in enumerate(self.clusters):
            #print(j)
            #print(i)
            for k in j:
                labels[k] = i

        return labels


    def predict(self, X): #TODO
        self.X = X 
        self.clusters = []
        self.visited_samples = []
        self.neighbors = {}

        n_samples = np.shape(self.X)[0]

        for sample in range(n_samples):
            # If a saample is visited, then do not bother with it.
            if sample in self.visited_samples:
                continue
            else:

            # TODO
            # Iterate through un-visited samples and expand clusters from them
            # if they have more neighbors than self.min_samples
                #print(sample)
                self.neighbors[sample] = self.get_neighbors(sample)
                #print(self.neighbors[sample])
                if len(self.neighbors[sample]) >= self.min_samples:
                # Core point is a sample that has more than min_samples neighbors. Its the beginning of a new cluster!
                # If core point => mark as visited
                    
                # Sample has more neighbors than self.min_samples => expand
                # cluster from sample
                    #print(sample)
                    #print(self.neighbors[sample])
                    new_cluster = self.expand_cluster(sample, self.neighbors[sample])
                # Add cluster to list of clusters
                    self.clusters.append(new_cluster)
                    if sample not in self.visited_samples:
                        self.visited_samples.append(sample) 
                else:
                    continue
                

        for cluster_id, cluster in enumerate(self.clusters):
            self.clusters[cluster_id] = list(set(cluster))
        cluster_labels = self.det_cluster_labels()
        return cluster_labels


def main():

    dbscan = DBSCAN() #add custom parameters for eps and min_samples

    X,y = readDataLabels()
    plot(X, y)

    # Run prediction over the data
    # Cluster the data using DBSCAN
    clf = DBSCAN(eps=0.17, min_samples=5)   # You can experiment with different parameters
    #(eps=0.2, samples=5) is giving good clustering and accuracy for this is 100 percent.
    y_pred = clf.predict(X) 
    

    # Plot and compare to ground truth
    plot(X, y_pred)
    accuracy = np.sum(y == y_pred, axis=0) / len(y)
    print("Accuracy is :",accuracy)
    # For reference... This is how sklearn performs!
    #test_sklearn(X)


if __name__ == "__main__":
    main()



    





