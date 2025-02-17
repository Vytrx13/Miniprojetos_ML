import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.inspection import permutation_importance
# Variable Name	Role	Type	Description	Units	Missing Values
# buying	Feature	Categorical	buying price		no
# maint	Feature	Categorical	price of the maintenance		no
# doors	Feature	Categorical	number of doors		no
# persons	Feature	Categorical	capacity in terms of persons to carry		no
# lug_boot	Feature	Categorical	the size of luggage boot		no
# safety	Feature	Categorical	estimated safety of the car		no
# class	Target	Categorical	evaulation level (unacceptable, acceptable, good, very good)		no
# fonte: https://archive.ics.uci.edu/dataset/19/car+evaluation


columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']


df = pd.read_csv('car.csv', header=None, names=columns)


categorical_columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']
X = df.drop('class', axis=1)
y = df['class']

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), categorical_columns)], remainder='passthrough')
X = ct.fit_transform(X)

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
classifier = SVC(kernel='rbf', random_state=0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# Calculating the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Calculating the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(f'Confusion Matrix:\n{cm}')

X_arr = X_test.toarray()

# Calculando a importância das features
result = permutation_importance(classifier, X_arr, y_test, n_repeats=10)

# Exibindo a importância das features
importance = result.importances_mean
for i, feature in enumerate(df.drop('class', axis=1).columns):
    print(f'{feature}: {importance[i]}')