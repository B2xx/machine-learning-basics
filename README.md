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
* How the algorithm makes predictions:
  1. Calculate the distance between this new point and every training point.
  2. Select the K nearest neighbors.
  3. Output the most common class label among these neighbors.
### K Means Clustering
* Key Idea: K-Means is an unsupervised learning algorithm used to partition n observations into K clusters. Each data point is assigned to the cluster whose mean (the cluster centroid) is closest.
* How the algorithm works:
  1. **Initialization:** Select K initial centroids, often chosen randomly from the data.
  2. **Assignment Step:** Assign each data point to its nearest centroid to form clusters.
  3. **Update Step:** Recalculate centroids by taking the mean of all data points assigned to each cluster.
  4. **Iteration:** Repeat the assignment and update steps until cluster assignments no longer change (or until a specified maximum number of iterations is reached).
* How the algorithm makes predictions:
  1. Calculate Distance to Centroids: For the new data point, compute the distance to each of the K cluster centroids.
  2. Assign Cluster: Assign the new data point to the cluster with the closest centroid.

#### Some analysis:
* The graph of K Means clustering could be messy as the clusters may overlap, which makes it prone to make mistakes while making predictions.
* KNN is non-parametric (does not need learning) and works well in low dimensions for complex decision surfaces, but it suffers from the curse of dimensionality.

## Part 2 - Linear Regression, Ridge Regression, and Lasso Regression

This folder introduces three fundamental regression algorithms:**Linear Regression**, **Ridge Regression**  ,**Lasso Regression**

### Linear Regression
* **Key Idea**  
  Linear Regression models the relationship between a scalar response (dependent variable) and one or more explanatory variables (independent variables) by fitting a linear equation to observed data.

* **How the Algorithm Works**  
  1. **Model Representation**: \( y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_n x_n \).  
  2. **Cost Function**: Typically uses **Mean Squared Error (MSE)** to measure how well the line fits the data.  
  3. **Optimization**: Adjust the coefficients \(\beta\) to minimize the MSE (often through **Gradient Descent** or other optimization methods).

* **How the Algorithm Makes Predictions**  
  1. **Plug-In Values**: Given new input features \((x_1, x_2, \dots, x_n)\), compute \( \hat{y} = \beta_0 + \beta_1 x_1 + \dots + \beta_n x_n \).  
  2. **Output**: The scalar value \(\hat{y}\) is the predicted target value.

* **Some Analysis**  
  - **Advantages**: Simple to implement and interpret; works well when data has a linear relationship.  
  - **Disadvantages**: Prone to overfitting if too many predictors are used without regularization; sensitive to outliers.

---

### Ridge Regression
* **Key Idea**  
  Ridge Regression is a variant of Linear Regression that adds an \(L2\) penalty to the cost function to shrink coefficient values and reduce overfitting.

* **How the Algorithm Works**  
  1. **Cost Function with \(L2\) Regularization**:  
     \[
     \text{Cost} = \text{MSE} + \lambda \sum_{j=1}^{n} \beta_j^2
     \]
     where \(\lambda\) controls the strength of regularization.  
  2. **Optimization**: Similar to Linear Regression, but includes the additional penalty term on the coefficients.  
  3. **Coefficient Shrinkage**: Larger \(\lambda\) values produce smaller (but never zero) coefficients, thereby reducing variance in the model.

* **How the Algorithm Makes Predictions**  
  1. **Use the Learned Coefficients**: After fitting, the model produces coefficients \(\beta_j\).  
  2. **Compute Predicted Value**: Same linear equation as standard Linear Regression:  
     \(\hat{y} = \beta_0 + \beta_1 x_1 + \dots + \beta_n x_n.\)

* **Some Analysis**  
  - **Advantages**: Reduces model complexity and collinearity among features; helps prevent overfitting.  
  - **Disadvantages**: All coefficients are shrunk towards zero but not exactly zero, which might not be ideal if strong feature selection is desired.

---

### Lasso Regression
* **Key Idea**  
  Lasso (Least Absolute Shrinkage and Selection Operator) is a variant of Linear Regression that adds an \(L1\) penalty, encouraging sparsity in the coefficients (some coefficients may become zero).

* **How the Algorithm Works**  
  1. **Cost Function with \(L1\) Regularization**:  
     \[
     \text{Cost} = \text{MSE} + \lambda \sum_{j=1}^{n} |\beta_j|
     \]
  2. **Optimization**: Adjust \(\beta_j\) to minimize the cost function, where \(\lambda\) controls the level of regularization.  
  3. **Coefficient Sparsity**: Under sufficient regularization, some coefficients will become zero, effectively performing feature selection.

* **How the Algorithm Makes Predictions**  
  1. **Learned Sparse Coefficients**: After training, some \(\beta_j\) may be zero.  
  2. **Compute Predicted Value**: Use the non-zero coefficients in the linear function:
     \(\hat{y} = \beta_0 + \beta_1 x_1 + \dots + \beta_n x_n.\)

* **Some Analysis**  
  - **Advantages**: Can perform feature selection by driving some coefficients to zero; helpful when many features are irrelevant.  
  - **Disadvantages**: Can suffer from high bias if \(\lambda\) is too large; not always stable if features are highly correlated.

---

## Part 3 - Logistic Regression

This folder covers **Logistic Regression**, a widely-used classification algorithm.

### Logistic Regression
* **Key Idea**  
  Logistic Regression models the probability that a given input belongs to a certain class (often binary), using the logistic (sigmoid) function:
  \[
  P(y=1 | x) = \frac{1}{1 + e^{-z}}, \quad z = \beta_0 + \beta_1 x_1 + \dots + \beta_n x_n.
  \]

* **How the Algorithm Works**  
  1. **Sigmoid Function**: Converts the linear combination of features \(\beta_0 + \beta_1 x_1 + \dots\) into a probability between 0 and 1.  
  2. **Cost Function**: Uses **Cross-Entropy Loss (Log Loss)** instead of MSE:
     \[
     \text{Cost} = -\frac{1}{N} \sum_{i=1}^{N} [y_i \log(\hat{p_i}) + (1 - y_i) \log(1 - \hat{p_i})].
     \]
  3. **Optimization**: Adjust \(\beta\) via Gradient Descent (or similar methods) to minimize the logistic loss.

* **How the Algorithm Makes Predictions**  
  1. **Probability Thresholding**: For a new input \(x\), compute \(P(y=1 | x)\).  
  2. **Classification**: If \(P(y=1 | x) \geq 0.5\), predict class 1; otherwise, predict class 0.

* **Some Analysis**  
  - **Advantages**: Outputs well-calibrated probabilities; easy to interpret; extends to multiclass problems (e.g., One-vs-Rest).  
  - **Disadvantages**: May underfit if the decision boundary is highly non-linear; sensitive to outliers.

---

## Part 4 - Random Forest and GBDT

This folder introduces two powerful ensemble methods:
1. **Random Forest**  
2. **Gradient Boosted Decision Trees (GBDT)**

### Random Forest
* **Key Idea**  
  A Random Forest is an ensemble of **Decision Trees**. Each tree is trained on a random subset of the data (bagging) and uses random subsets of features (feature bagging). Final predictions aggregate results from all trees (e.g., by majority vote for classification or averaging for regression).

* **How the Algorithm Works**  
  1. **Bootstrap Sampling**: Draw multiple random samples from the training data.  
  2. **Tree Training**: Train a decision tree on each bootstrap sample, randomly selecting features at each split.  
  3. **Ensemble**: Combine predictions from all trees to produce the final output.

* **How the Algorithm Makes Predictions**  
  1. **Collect Individual Tree Predictions**: For classification, each tree votes for a class; for regression, each tree outputs a numerical value.  
  2. **Aggregate Results**:  
     - **Classification**: Predict the class with the most votes.  
     - **Regression**: Predict the average of all tree outputs.

* **Some Analysis**  
  - **Advantages**: Generally robust to overfitting; can handle large feature spaces; works well out-of-the-box.  
  - **Disadvantages**: May require more computation and memory; less interpretable than a single decision tree.

---

### Gradient Boosted Decision Trees (GBDT)
* **Key Idea**  
  GBDT trains decision trees **sequentially**, where each new tree attempts to correct errors made by the previous ensemble. Common implementations include **XGBoost**, **LightGBM**, and **CatBoost**.

* **How the Algorithm Works**  
  1. **Initial Model**: Start with a simple model (often a constant prediction).  
  2. **Sequential Boosting**: Iteratively fit new decision trees to the negative gradient (residual errors) of the loss function.  
  3. **Update Ensemble**: Add each new tree’s output to the overall model, weighted by a learning rate \(\eta\).

* **How the Algorithm Makes Predictions**  
  1. **Summation of Tree Outputs**: For a new data point, each tree in the sequence provides a prediction.  
  2. **Final Score**: Sum these predictions (scaled by the learning rate) to obtain the final prediction.  
  3. **Classification or Regression**:  
     - **Classification**: Convert the final score via a logistic function (for binary) or apply a softmax (for multi-class).  
     - **Regression**: The final score is the numeric output.

* **Some Analysis**  
  - **Advantages**: Often achieves state-of-the-art performance in structured/tabular data; flexible with various loss functions.  
  - **Disadvantages**: More sensitive to hyperparameters (learning rate, number of trees, tree depth, etc.); can overfit without careful tuning.

## Competition - Image Classification Task (CNN(ResNet), ViT)
**Primary task:** Image Classification
**Data: ** We offer a dataset with 3600 images (3200 images for training/validation, and 400 images for testing). The images are of size 512⨉512, but you are free to choose the resolution for model training. In particular, among the 3600 images, there are 900 AI-generated images. The full set of classes is as follows:
{0: Tang(唐), 1: Song(宋), 2: Yuan(元),
3: Ming(明), 4: Qing(清), 5: AI}
You have access to the class labels on the training/validation set, and your task is to train a model to make class label predictions on the test set. Noticing that the labels on the training/validation set are designed to be noisy, where only half the AI-generated images are labelled explicitly, whereas the rest of them have been assigned to some random labels (i.e., one of the five non-AI classes). Hence, you may consider training a dedicated model for data cleaning first.

For detailed documentation of this part, please see my documentation here [https://github.com/B2xx/machine-learning-basics/blob/main/Image%20Classifiation%20Model%20Training%20(ViT%20model)/NYUSH_ML_Competition_Report.pdf]
