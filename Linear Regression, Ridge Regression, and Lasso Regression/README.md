This Readme file is specifically built for the coding file under this folder. The task in this folder involves using linear and ridge regression to predict Boston House pricing and using Lasso regresion for generated data and make prediction for star from the text in Yelp comment.

## Boston House Price Prediction: Linear and Ridge Regression

**Task**  
Use both linear regression and ridge regression to estimate Boston house prices based on various features.

**Key Findings**  
- **Ridge vs. Linear Regression**  
  Ridge regression generally performs better on the test set due to reduced overfitting.  
- **Feature Selection**  
  Using only the three most strongly correlated features gives a similar RMSE to using all 13 features in linear regression. This suggests that a smaller feature subset can reduce computational cost without significantly harming performance.  
- **Limitations of Linear Regression**  
  Linear regression works best when relationships between features and the target are primarily linear. If the data exhibits more complex patterns, linear regression may fail to capture those nuances.

## Lasso Regression

**Task**
The optimal function: $argmin_{\theta,\theta_0} F(\theta, \theta_0)\quad \text{where} \quad F(\theta,\theta_0) = \frac{1}{2} \sum_{i=1}^n \Bigl(\bigl\langle x^{(i)}, \theta \bigr\rangle + \theta_0 - y^{(i)}\Bigr)^2+ \lambda \sum_{j=1}^m \lvert \theta_j \rvert$

1. **Synthetic Data**  
   - Generate data with a known true weight vector (including an intercept).  
   - Fit a Lasso model via coordinate descent.  
   - Evaluate results using RMSE (root mean squared error), precision, recall, and sparsity of learned coefficients.  
   - Vary the regularization parameter $(\(\lambda\))$ to trace out the **Lasso path** (how weights change with different regularization strengths).

2. **Yelp Dataset**  
   - Predict star ratings from text features (represented as numeric vectors).  
   - Split the data into training, validation, and test sets.  
   - Select the best $\(\lambda\)$ using the validation set, then report performance on the test set.  
   - Identify top features that have the largest (absolute) weights in the final model.

**Key Findings**
- **Sparsity**: As $\(\lambda\)$ increases, more coefficients shrink to zero, reducing model complexity at the cost of potential underfitting.  
- **Trade-offs**: There is a trade-off between **precision** and **recall** in recovering the true non-zero weights, especially under higher noise $(\(\sigma\))$.  
- **Overfitting vs. Underfitting**: Lower $\(\lambda\)$ can overfit the data, while very high \(\lambda\) can underfit by shrinking most coefficients to zero.  


