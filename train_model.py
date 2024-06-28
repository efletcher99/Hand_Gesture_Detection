from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
import pickle
import numpy as np

# Load data
data_dict = pickle.load((open('./data.pickle', 'rb')))
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Split data
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Standardize data
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Initialize model with desired parameters
model = RandomForestClassifier(n_estimators=50, max_depth=5, min_samples_split=10, min_samples_leaf=5)

# Cross-validation to evaluate model
cv_scores = cross_val_score(model, x_train, y_train, cv=5)
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean cross-validation score: {cv_scores.mean()}")

# Train model
model.fit(x_train, y_train)

# Predict on test set
y_predict = model.predict(x_test)

# Evaluate model
score = accuracy_score(y_predict, y_test)
print('{}% of samples were classified correctly.'.format(score * 100))

# Save model
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
