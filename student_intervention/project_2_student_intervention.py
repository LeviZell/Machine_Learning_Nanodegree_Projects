# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 14:35:50 2016

@author: LeviZ
"""

# Import libraries
import numpy as np
import pandas as pd
from time import time
from sklearn.metrics import f1_score

# Read student data
student_data = pd.read_csv("student-data.csv")
print "Student data read successfully!"

# TODO: Calculate number of students
#n_students = student_data.sum('n_students')
n_students = len(student_data.index)

# TODO: Calculate number of features
n_features = len(student_data.count(axis=0, level=None, numeric_only=False))


# TODO: Calculate passing students
n_passed = len(student_data[student_data.passed == 'yes'])

# TODO: Calculate failing students
n_failed = len(student_data[student_data.passed == 'no'])

# TODO: Calculate graduation rate
grad_rate = ((len(student_data[student_data.passed == 'yes']))/float(n_students))*100

# Print the results
print "Total number of students: {}".format(n_students)
print "Number of features: {}".format(n_features)
print "Number of students who passed: {}".format(n_passed)
print "Number of students who failed: {}".format(n_failed)
print "Graduation rate of the class: {:.2f}%".format(grad_rate)

# Extract feature columns
feature_cols = list(student_data.columns[:-1])

# Extract target column 'passed'
target_col = student_data.columns[-1] 

# Show the list of columns
print "Feature columns:\n{}".format(feature_cols)
print "\nTarget column: {}".format(target_col)

# Separate the data into feature data and target data (X_all and y_all, respectively)
X_all = student_data[feature_cols]
y_all = student_data[target_col]

# Show the feature information by printing the first five rows
print "\nFeature values:"
print X_all.head()

def preprocess_features(X):
    ''' Preprocesses the student data and converts non-numeric binary variables into
        binary (0/1) variables. Converts categorical variables into dummy variables. '''
    
    # Initialize new output DataFrame
    output = pd.DataFrame(index = X.index)

    # Investigate each feature column for the data
    for col, col_data in X.iteritems():
        
        # If data type is non-numeric, replace all yes/no values with 1/0
        if col_data.dtype == object:
            col_data = col_data.replace(['yes', 'no'], [1, 0])

        # If data type is categorical, convert to dummy variables
        if col_data.dtype == object:
            # Example: 'school' => 'school_GP' and 'school_MS'
            col_data = pd.get_dummies(col_data, prefix = col)  
        
        # Collect the revised columns
        output = output.join(col_data)
    
    return output

X_all = preprocess_features(X_all)
print "Processed feature columns ({} total features):\n{}".format(len(X_all.columns), list(X_all.columns))

# TODO: Import any additional functionality you may need here
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split, ShuffleSplit

# TODO: Set the number of training points
num_train = 300

# Set the number of testing points
num_test = X_all.shape[0] - num_train


# TODO: Shuffle and split the dataset into the number of training and testing points above
def shuffle_split_data(X, y):
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.25, random_state=0) 
    return X_train, y_train, X_test, y_test

X_train, y_train, X_test, y_test = shuffle_split_data(X_all,y_all)


# Show the results of the split
print "Training set has {} samples.".format(X_train.shape[0])
print "Testing set has {} samples.".format(X_test.shape[0])

def train_classifier(clf, X_train, y_train):
    ''' Fits a classifier to the training data. '''
    
    # Start the clock, train the classifier, then stop the clock
    start = time()
    clf.fit(X_train, y_train)
    end = time()
    
    # Print the results
    print "Trained model in {:.4f} seconds".format(end - start)

    
def predict_labels(clf, features, target):
    ''' Makes predictions using a fit classifier based on F1 score. '''
    
    # Start the clock, make predictions, then stop the clock
    start = time()
    y_pred = clf.predict(features)
    end = time()
    
    # Print and return results
    print "Made predictions in {:.4f} seconds.".format(end - start)
    return f1_score(target.values, y_pred, pos_label='yes')


def train_predict(clf, X_train, y_train, X_test, y_test):
    ''' Train and predict using a classifer based on F1 score. '''
    
    # Indicate the classifier and the training set size
    print "Training a {} using a training set size of {}. . .".format(clf.__class__.__name__, len(X_train))
    
    # Train the classifier
    train_classifier(clf, X_train, y_train)
    
    # Print the results of prediction for both training and testing
    print "F1 score for training set: {:.4f}.".format(predict_labels(clf, X_train, y_train))
    print "F1 score for test set: {:.4f}.".format(predict_labels(clf, X_test, y_test))
    
# TODO: Import the three supervised learning models from sklearn
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

# TODO: Initialize the three models
clf_A = svm.SVC()
clf_B = LogisticRegression()
clf_C = GaussianNB()

# TODO: Set up the training set sizes
X_train_100 = X_train[:100]
y_train_100 = y_train[:100]

X_train_200 = X_train[:200]
y_train_200 = y_train[:200]

X_train_300 = X_train[:300]
y_train_300 = y_train[:300]

# TODO: Execute the 'train_predict' function for each classifier and each training set size
train_predict(clf_A, X_train_100, y_train_100, X_test, y_test)
print "\n"

train_predict(clf_B, X_train_100, y_train_100, X_test, y_test)
print "\n"

train_predict(clf_C, X_train_100, y_train_100, X_test, y_test)
print "\n"

print "------------------------------------------"
print "\n"

train_predict(clf_A, X_train_200, y_train_200, X_test, y_test)
print "\n"

train_predict(clf_B, X_train_200, y_train_200, X_test, y_test)
print "\n"

train_predict(clf_C, X_train_200, y_train_200, X_test, y_test)
print "\n"

print "------------------------------------------"
print "\n"

train_predict(clf_A, X_train_300, y_train_300, X_test, y_test)
print "\n"

train_predict(clf_B, X_train_300, y_train_300, X_test, y_test)
print "\n"

train_predict(clf_C, X_train_300, y_train_300, X_test, y_test)

# TODO: Import 'GridSearchCV' and 'make_scorer'
from sklearn import svm, grid_search, datasets

from sklearn.metrics import fbeta_score, make_scorer
# TODO: Create the parameters list you wish to tune
parameters = [{'kernel':['linear', 'rbf'], 'C':[0.01, 0.5, 1, 10, 100, 1000]}]
              
#              , {'kernel':['rbf'], 'gamma':[0.001, 0.0001], 
#               'C':[0.01, 0.5, 1, 10, 100, 1000]}, {'kernel':['sigmoid'], 
#               'C':[0.01, 0.5, 1, 10, 100, 1000]}, {'kernel':['poly'], 
#               'C':[0.01, 0.5, 1, 10, 100, 1000]}]
# , 'gamma':[0.001, 0.0001], 'random_state':[None,40]}

# TODO: Initialize the classifier
svr = svm.SVC()
clf = grid_search.GridSearchCV(svr, param_grid=parameters)
clf.fit(X_train, y_train)

# TODO: Make an f1 scoring function using 'make_scorer' 
f1_scorer = make_scorer(f1_score, pos_label='yes')

# TODO: Perform grid search on the classifier using the f1_scorer as the scoring method
grid_obj = grid_search.GridSearchCV(svr, param_grid=parameters, scoring=f1_scorer)

# TODO: Fit the grid search object to the training data and find the optimal parameters
grid_obj = grid_obj.fit(X_train, y_train)

# Get the estimator
clf = grid_obj.best_estimator_

# Report the final F1 score for training and testing after parameter tuning
print "Tuned model has a training F1 score of {:.4f}.".format(predict_labels(clf, X_train, y_train))
print "Tuned model has a testing F1 score of {:.4f}.".format(predict_labels(clf, X_test, y_test))

