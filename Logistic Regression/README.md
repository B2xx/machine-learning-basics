Below is an example **README.md** that follows the *structure* of the previous Boston House example, but updated to reflect **Logistic Regression**, **learning rate**, and **batch size** concepts from your code. Feel free to modify the language or formatting as needed.

---

# Logistic Regression: Learning Rate and Batch Size Trade-off

This folder contains an example of **Logistic Regression**—both **batch** and **stochastic** gradient descent (SGD)—demonstrating how **learning rate** and **batch size** can impact training behavior and final performance. We use the **scikit-learn digits dataset** (handwritten digits) to show how parameter tuning affects accuracy and convergence.

---

## Task

1. **Full-Batch Logistic Regression (Gradient Descent)**
   - Train a multi-class logistic regression model via **batch gradient descent**.
   - Investigate how different **learning rates** influence convergence speed and model accuracy.

2. **Stochastic Gradient Descent (SGD)**
   - Train the same logistic model using **mini-batches** of various sizes.
   - Show how **batch size** and **learning rate** together affect the loss curve and accuracy.

---

## Key Findings

1. **Learning Rate**
   - Higher learning rates can converge faster but risk overshooting or oscillating.  
   - More conservative learning rates converge more slowly yet may produce a smoother descent.  

2. **Batch Size**
   - **Smaller Batches**: More frequent updates can help the model escape local minima but may produce noisy gradients.  
   - **Larger Batches**: More stable gradient updates at the cost of increased compute time per iteration.

3. **Learning Rate and Batch Size**
   - There is a **trade-off**: a larger batch size often allows a slightly bigger learning rate, but if the rate is too high, convergence can still become unstable.  
   - For certain hyperparameter combinations, adaptive strategies (e.g., reducing learning rate after plateaus) help maintain stable progress.
