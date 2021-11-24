import pandas as pd
import numpy as np
from SoftmaxRegression import SoftmaxRegression

# Read csv
df = pd.read_csv('dataset/iris.data')
df.columns = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'target']
df.drop(['sepal-length', 'sepal-width'], axis=1, inplace=True) 

# Encode targets
df.replace('Iris-setosa', 0, inplace=True)
df.replace('Iris-versicolor', 1, inplace=True)
df.replace('Iris-virginica', 2, inplace=True)

# Shuffle
df = df.sample(frac=1).reset_index(drop=True)

# Split train test set
def train_test_split(df, test_ratio = 0.25):
    thresh = int(df.shape[0] * test_ratio)
    labels = df['target']
    df.drop('target', axis=1, inplace=True)
    return df.iloc[:thresh], df[thresh:], labels[:thresh], labels[thresh:]

test_x, train_x, test_y, train_y = train_test_split(df)

model = SoftmaxRegression()
model.fit(train_x, train_y)
pred = model.predict(test_x)
print(np.sum(pred == test_y) / len(pred))