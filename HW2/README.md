# Q1. Root Node Selection Using Gini Index 

## Decision
**Weather** is selected as the root node because it has the **lowest Gini Index** (0.1667 vs. Wind's 0.4444).

---

## Key Results
| Feature   | Weighted Gini Index | Outcome                     |
|-----------|---------------------|-----------------------------|
| **Weather** | 0.1667              | ✅ **Chosen as root node**  |
| Wind      | 0.4444              | ❌ Higher impurity          |


# Q2. Banknote Authentication using Decision Trees

This project implements a **Decision Tree Classifier** to authenticate banknotes based on image-derived statistical features. The dataset is sourced from the UCI Machine Learning Repository.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Features Calculated](#features-calculated)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Repository Structure](#repository-structure)
- [Acknowledgments](#acknowledgments)

---

## Project Overview
The goal is to classify banknotes as **authentic (1)** or **fake (0)** using four numerical features extracted from wavelet-transformed images:
1. Variance
2. Skewness
3. Kurtosis
4. Entropy

We use a decision tree algorithm (`sklearn.tree.DecisionTreeClassifier`) to train and evaluate the model. Key steps include:
- Exploratory data analysis (EDA) with pairwise feature visualization.
- Hyperparameter tuning (`max_depth`, `criterion`, `min_samples_split`).
- Model evaluation using accuracy, precision, recall, F1-score, and confusion matrices.
- Visualization of the decision tree and feature importances.

---

## Dataset
**Source**: [UCI Banknote Authentication Dataset](https://archive.ics.uci.edu/dataset/267/banknote+authentication)  
**Samples**: 1,372 instances (762 authentic, 610 fake)  
**Features**:
1. `variance`: Spread of pixel values.
2. `skewness`: Asymmetry of pixel distribution.
3. `kurtosis`: "Tailedness" of pixel distribution.
4. `entropy`: Complexity/randomness of the image.

---

## Features Calculated
1. **Variance**: Measures how far pixel values are from the mean.
2. **Skewness**: Quantifies asymmetry in the pixel distribution.
   - Positive skew: Tail on the right.
   - Negative skew: Tail on the left.
3. **Kurtosis**: Indicates heaviness of the tails in the distribution.
4. **Entropy**: Measures randomness/complexity (higher entropy = more texture).

---

## Installation
1. **Google Colab**:  
   The notebook is designed to run directly in Google Colab. No local installation is required.  

2. **Local Setup (Optional)**:  
   Clone the repository and install dependencies:
   ```bash
   git clone https://github.com/SamilEmc1/SamilEmc1/tree/main/HW2
   cd banknote-authentication
   
   ```

## Results  
The decision tree model achieves high accuracy (~98%) in classifying banknotes as authentic or fake. Key metrics include precision, recall, and F1-scores exceeding 97% for both classes. The confusion matrix shows minimal misclassifications, with most predictions aligning with the true labels. Feature importance analysis reveals **variance** and **entropy** as the most critical predictors. A shallow tree (e.g., `max_depth=3`) balances interpretability and performance, showing clear decision rules.

---

## Repository Structure  
The repository contains:  
- `decision_tree.ipynb`: The complete Colab notebook with code, visualizations, and analysis.  
- `README.md`: This documentation file.  
- `images/` (optional): Folder containing output plots like the decision tree visualization and confusion matrix.  

---

## Acknowledgments  
- Dataset: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/banknote+authentication).  
- Tools: Built with `scikit-learn`, `pandas`, `matplotlib`, and `seaborn`.  
