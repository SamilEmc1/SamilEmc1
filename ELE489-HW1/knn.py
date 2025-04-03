import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class KNN:
    def __init__(self, k=3, metric='euclidean'):
        self.k = k
        self.metric = metric.lower()
        
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        
    def _distance(self, x1, x2):
        if self.metric == 'euclidean':
            return np.sqrt(np.sum((x1 - x2) ** 2))
        elif self.metric == 'manhattan':
            return np.sum(np.abs(x1 - x2))
        else:
            raise ValueError("Invalid metric. Choose 'euclidean' or 'manhattan'")
            
    def predict(self, X_test):
        predictions = []
        for x in X_test:
            distances = [self._distance(x, x_train) for x_train in self.X_train]
            k_indices = np.argsort(distances)[:self.k]
            k_labels = self.y_train[k_indices]
            most_common = Counter(k_labels).most_common(1)
            predictions.append(most_common[0][0])
        return np.array(predictions)

# Load and preprocess data
def load_data():
    df = pd.read_csv('/Users/samilemec/SamilEmc1/wine/wine.csv', header=None)
    X = df.iloc[1:, 1:].values.astype(float)  # Features
    y = df.iloc[1:, 0].values.astype(int)     # Class labels
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Evaluation function
def evaluate(y_true, y_pred):
    print("Classification Report:")
    print(classification_report(y_true, y_pred))
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()
    
    # Test different k values
    for k in [1, 3, 5, 7, 9]:
        print(f"\nEvaluating k={k} with Euclidean distance:")
        knn = KNN(k=k)
        knn.fit(X_train, y_train)
        preds = knn.predict(X_test)
        evaluate(y_test, preds)