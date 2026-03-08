import pandas as pd
from sklearn.datasets import make_classification

X,y= make_classification(n_samples=1000, n_features=5,n_informative=5,n_redundant=0, n_classes=2, random_state=42)
df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
df['label'] = y
df.to_csv('synthetic_dataset.csv', index=False)