import pickle
from sklearn import svm, neighbors, linear_model, tree

# Classifiers
svm1 = svm.SVC()
svm2 = svm.SVC()

knn1 = neighbors.KNeighborsClassifier(n_neighbors=10)
knn2 = neighbors.KNeighborsClassifier(n_neighbors=10)

logistic1 = linear_model.LogisticRegression()
logistic2 = linear_model.LogisticRegression()

tree1 = tree.DecisionTreeClassifier()
tree2 = tree.DecisionTreeClassifier()

# Load feature vector and labels
with open('features.pkl', 'rb') as f:
    fvec = pickle.load(f)
with open('augmented_lbl.pkl', 'rb') as f:
    lbl = pickle.load(f)

# Train classifiers
svm1.fit(fvec, lbl)
svm2.fit(fvec[:50000], lbl[:50000])
print('SVM trained')

knn1.fit(fvec, lbl)
knn2.fit(fvec[:50000], lbl[:50000])
print('KNN trained')

logistic1.fit(fvec, lbl)
logistic2.fit(fvec[:50000], lbl[:50000])
print('Logistic Regression trained')

tree1.fit(fvec, lbl)
tree2.fit(fvec[:50000], lbl[:50000])
print('Decision Tree trained')

# Load test data
with open('test_vectors.pkl', 'rb') as f:
    test_fvec = pickle.load(f)

# Predict labels for test data
svm1_pred = svm1.predict(test_fvec)
svm2_pred = svm2.predict(test_fvec)

knn1_pred = knn1.predict(test_fvec)
knn2_pred = knn2.predict(test_fvec)

logistic1_pred = logistic1.predict(test_fvec)
logistic2_pred = logistic2.predict(test_fvec)

tree1_pred = tree1.predict(test_fvec)
tree2_pred = tree2.predict(test_fvec)

# Save predictions
with open('svm1_pred.pkl', 'wb') as f:
    pickle.dump(svm1_pred, f)

with open('svm2_pred.pkl', 'wb') as f:
    pickle.dump(svm2_pred, f)

with open('knn1_pred.pkl', 'wb') as f:
    pickle.dump(knn1_pred, f)

with open('knn2_pred.pkl', 'wb') as f:
    pickle.dump(knn2_pred, f)

with open('logistic1_pred.pkl', 'wb') as f:
    pickle.dump(logistic1_pred, f)

with open('logistic2_pred.pkl', 'wb') as f:
    pickle.dump(logistic2_pred, f)

with open('tree1_pred.pkl', 'wb') as f:
    pickle.dump(tree1_pred, f)

with open('tree2_pred.pkl', 'wb') as f:
    pickle.dump(tree2_pred, f)
