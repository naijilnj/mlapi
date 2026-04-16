import numpy as np

from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

iris = datasets.load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = SVC()

hyperparam_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto']
}

grid_search = GridSearchCV(
    estimator=model,
    param_grid=hyperparam_grid,
    scoring='accuracy',
    cv=5,
    verbose=2,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

best_hyperparameters = grid_search.best_estimator_
best_score = grid_search.best_score_

print("Best Model (GridSearch):", best_hyperparameters)
print("Best Score (GridSearch):", best_score)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

print("Predictions vs Actual:\n", list(zip(y_test, y_pred)))

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy (GridSearch):", accuracy)


# -------------------------------
# Randomized Search
# -------------------------------
random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=hyperparam_grid,
    scoring='accuracy',
    cv=5,
    verbose=2,
    n_jobs=-1,
    n_iter=10,
    random_state=42
)

random_search.fit(X_train, y_train)

best_hyperparameters = random_search.best_estimator_
best_score = random_search.best_score_

print("\nBest Model (RandomSearch):", best_hyperparameters)
print("Best Score (RandomSearch):", best_score)

# Evaluate
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)

print("Predictions vs Actual:\n", list(zip(y_test, y_pred)))

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy (RandomSearch):", accuracy)