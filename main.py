from sklearn import tree

# Training data: [height, weight, shoe size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37],
     [166, 65, 40], [190, 90, 47], [175, 64, 39], [177, 70, 40],
     [159, 55, 37], [171, 75, 42], [181, 65, 43]]

# Labels corresponding to the training data
Y = ['male', 'female', 'female', 'female', 'male', 'male',
     'male', 'female', 'male', 'female', 'male']

# Initialize and train the decision tree classifier
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)

# Make a prediction for new data
prediction = clf.predict([[190, 75, 30]])

print(prediction)
