This Readme file is specifically built for the coding file under this folder. 

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
This is the function we aim to solve: $argmin_{\theta,\theta_0} F(\theta, \theta_0)\quad \text{where} \quad F(\theta,\theta_0) = \frac{1}{2} \sum_{i=1}^n \Bigl(\bigl\langle x^{(i)}, \theta \bigr\rangle + \theta_0 - y^{(i)}\Bigr)^2+ \lambda \sum_{j=1}^m \lvert \theta_j \rvert$

**Key Findings**

