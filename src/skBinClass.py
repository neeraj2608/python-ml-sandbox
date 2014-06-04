# using scikit-learn for binary classification

from sklearn.preprocessing import StandardScaler
from sklearn import datasets
from sklearn import metrics
from sklearn.linear_model.stochastic_gradient import SGDClassifier
from sklearn.cross_validation import cross_val_score, KFold, train_test_split
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as pt
from scipy.stats import sem # standard error of mean
import numpy as np

iris = datasets.load_iris()
x, y = iris.data[:,:2], iris.target # use only two features - sepal length and sepal width
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42) # 25% reserved for validation
NUMFOLDS = 5

if __name__ == '__main__':
    DEBUG = 1
else:
    DEBUG = 0

def withPipeline():
    clfer = Pipeline([('scaler',StandardScaler()),('linear_model',SGDClassifier())])
    cval = KFold(x.shape[0], NUMFOLDS, shuffle=True, random_state=42)
    score = cross_val_score(clfer, x, y, cv=cval) # reports estimator accuracy
    print "%2.3f (+/- %2.3f)" % (np.mean(score), sem(score))

def withoutPipeline():
    scaler = StandardScaler().fit(x_train) # for each x, (x - mean(all x))/std. dev. of x
                                                         # this step computes the mean and std. dev.
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    clfer = SGDClassifier()
    clfer.fit(x_train, y_train) # this will try to separate the three classes based
                                # on the two features we gave it. Hence, we will get
                                # back three lines. I.e., three sets of coefficients
                                # and three intercepts
    if(DEBUG):
        #print clfer.coef_
        #print clfer.intercept_
        #print clfer.predict(scaler.transform([[4.7, 3.1]]))
        #print clfer.decision_function(scaler.transform([[4.7, 3.1]])) # the algorithm evaluates distance from all three
                                                                      # lines and picks the largest one (in this case [0])
        pass

    # validate results
    y_predict_train = clfer.predict(x_train)
    print "% Correct results on training set:"
    print metrics.accuracy_score(y_train, y_predict_train)
    y_predict_test = clfer.predict(x_test)
    print "\n% Correct results on testing set:"
    print metrics.accuracy_score(y_test, y_predict_test)
    # Understanding the classification report:
    # Precision: TP/(TP + FP) - ideal 1 - all instances reported as x were x. In other words,
    #                                     there were no instances reported as x that were NOT x
    # Recall:    TP/(TP + FN) - ideal 1 - all instances OF x were reported as x
    # Although, accuracy does not appear in the report, it is important to know what it means:
    # Accuracy: (TP + TN) / (TP + TN + FP + FN)
    print "\nClassification Report:"
    print metrics.classification_report(y_test, y_predict_test)
    # Understanding the confusion matrix
    # how many of class i were predicted as j
    # ideal. an Identity matrix
    print "Confusion Matrix:"
    print metrics.confusion_matrix(y_test, y_predict_test)

def createPlot(x_train, y_train):
    # Plotting
    colors = ['red','green','blue']
    for index in xrange(len(colors)):
        xs = x_train[:,0][y_train==index]
        ys = x_train[:,1][y_train==index]
        pt.scatter(xs,ys,c=colors[index])
    pt.legend(iris.target_names)
    pt.xlabel('Sepal Length')
    pt.xlabel('Sepal Width')

if __name__ == '__main__':
    withPipeline()
