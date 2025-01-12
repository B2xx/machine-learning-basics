# Machine Learning Basics - collection of common models

## Brief

* This project documents models I built and experimented with in the Machine Learning course (CS-360). 
* Most of them could run directly in Python environment; some have imported data from external source such as Kaggle.
* Most of them are built by scratch instead of using scikit-learn or any other packages, except for the Vit Model I built in the competition, in which I directly used the model structure. 
* This collection also does not involve any model with pre-trained weights. I think it is a great source for me (or other potential users/students) to reflect on the logic of building model and why some models are good for specific tasks.


## Part 1 - KNN and K Means Clustering
This folder contains two foundational machine learning algorithms: **K-Nearest Neighbors (KNN) and K-Means Clustering**.
### K-Nearest Neighbors (KNN)
* Key Idea:  In this project, KNN classifies the Iris flower dataset by considering the K nearest points in the training set. The value of K determines how many neighbors are examined to decide on the class assignment.
* How the algorithm works:
  1. **Distance Calculation:** For a new data point, compute its distance to every point in the training set (commonly using Euclidean distance).
  2. **Neighbor Selection:** Identify the K closest neighbors based on the smallest distances.
  3. **Majority Vote:** Assign the most frequent class among those K neighbors to the new data point.
* How the algorithm make prediction:
  1. Calculate the distance between this new point and every training point.
  2. Select the K nearest neighbors.
  3. Output the most common class label among these neighbors.
### K Means Clustering
* Key Idea: K-Means is an unsupervised learning algorithm used to partition n observations into K clusters. Each data point is assigned to the cluster whose mean (the cluster centroid) is closest.
* How the algorithm works:
  1. **Initialization:** Select K initial centroids, often chosen at random from the data.
  2. **Assignment Step:** Assign each data point to its nearest centroid to form clusters.
  3. **Update Step:** Recalculate centroids by taking the mean of all data points assigned to each cluster.
  4. **Iteration:** Repeat the assignment and update steps until cluster assignments no longer change (or until a specified maximum number of iterations is reached).
* How the algorithm make prediction:
  1. Calculate Distance to Centroids: For the new data point, compute the distance to each of the K cluster centroids.
  2. Assign Cluster: Assign the new data point to the cluster with the closest centroid.

#### Some analysis:
* From the making K Means clustering, it could be messy as the clusters could overlapping each other, which makes it prone to make mistakes while making predictions.
* KNN is non-parametric (does not need learning), and works well in low-dimension for complex decision surfaces but it suffers a lot from **curse of dimensionality**.

## Part 2 - Linear Regression, Ridge Regression, and Lasso Regression

## Part 3 - Logistic Regression

## Part 4 - Random Forest and GBDT

## Competition - Image Classification Task (CNN(ResNet), ViT)
