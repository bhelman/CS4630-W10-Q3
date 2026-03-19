import pandas as pd              # data handling
import numpy as np               # numerical ops

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.metrics import classification_report, confusion_matrix

from imblearn.over_sampling import SMOTE  # for imbalance handling

df = pd.read_csv("CS4630-W10-Q3\data.csv", sep = ';')
df.columns = df.columns.str.strip()

X = df.drop("Target", axis=1)
y = df["Target"]

print(y.value_counts())

smote = SMOTE(random_state = 42)
X_resampled, y_resampled = smote.fit_resample(X, y)

print("\nAfter SMOTE:")             # balance dataset
print(y_resampled.value_counts())

X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Decision Tree
dt = DecisionTreeClassifier(max_depth=5, random_state=42)
dt.fit(X_train, y_train)

y_pred_dt = dt.predict(X_test)

print("Decision Tree Results:")
print(classification_report(y_test, y_pred_dt))

# K-Nearest Neighbors
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred_knn = knn.predict(X_test)

print("KNN Results:")
print(classification_report(y_test, y_pred_knn))

# Naive Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)

y_pred_nb = nb.predict(X_test)

print("Naive Bayes Results:")
print(classification_report(y_test, y_pred_nb))

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['Target_encoded'] = le.fit_transform(df['Target'])

corr = df.corr(numeric_only=True)

print(corr['Target_encoded'].sort_values(ascending=False))