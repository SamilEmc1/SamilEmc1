# Wine Classification with k-Nearest Neighbors (k-NN) from Scratch

A machine learning implementation of the k-Nearest Neighbors algorithm from scratch, applied to the UCI Wine Recognition dataset. Includes comprehensive analysis and visualization components.

## Table of Contents
- [Dataset Description](#dataset-description)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)


## Dataset Description
The **UCI Wine Recognition Dataset** contains:
- 178 samples of 3 distinct wine classes (cultivars)
- 13 chemical analysis features:
  - Alcohol
  - Malic acid
  - Ash
  - Alcalinity of ash  
  - Magnesium
  - Total phenols
  - Flavanoids
  - Nonflavanoid phenols
  - Proanthocyanins
  - Color intensity
  - Hue
  - OD280/OD315 of diluted wines
  - Proline

## Features
- **Custom k-NN Implementation**:
  - Supports Euclidean and Manhattan distance metrics
  - Flexible k-value configuration (1, 3, 5, 7, 9)
  - Efficient NumPy-based calculations
- **Model Evaluation**:
  - Classification reports (precision/recall/F1-score)
  - Confusion matrix visualization
  - Accuracy comparison across parameters
- **Data Analysis**:
  - Feature distribution boxplots
  - Metric vs. performance comparisons
  - Class separation visualization

## Installation

1. **Clone Repository**:
```bash
git clone
cd wine-knn-classification

2. **Dataset Setup**:
-   Download wine.csv from UCI Machine Learning Repository
-   Place in project root directory

## Usage
- **Run Full Analysis**:
  ```bash
  python knn.py

  Tests all k-values (1,3,5,7,9) with both distance metrics
  Generates console reports and confusion matrice

- **2. Explore Interactive Analysis:**:
  ```bash
  jupyter notebook analysis.ipynb

  Contains:
    - Feature distribution visualizations
    - Metric comparison plots
    - Model performance analysis
  
## Results

### Optimal Performance
- **Best Accuracy**: 97%  
- **Configuration**: k=3 with Euclidean distance  
- **Key Discriminative Features**:  
  - Flavanoids  
  - Color intensity  
  - Proline content  

### Full Accuracy Comparison
| k-value | Euclidean Distance | Manhattan Distance |
|---------|--------------------|--------------------|
| 1       | 92%                | 89%                |
| 3       | 97%                | 94%                |
| 5       | 94%                | 91%                |
| 7       | 94%                | 91%                |
| 9       | 91%                | 89%                |

![Confusion Matrix Example](images/confusion_matrix_k3.png)  
*Confusion matrix for optimal configuration (k=3)*
