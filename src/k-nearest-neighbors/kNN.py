# k-Nearest Neighbors

from numpy import *
import operator
import sys

def createdataset():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','C']
    return group, labels

def classify(inp, dset, classes, k):
    """ Perform k-nn classification for an input point, given a dataset and
    a class for each instance
    inp = input to classify
    dset = dataset
    classes = class of each data instance. Size of classes must be equal to
              size of dset
    k = number of neighbors to use for kNN calculation. Must be smaller than
        size of dset
    returns the label of the most likely class as chosen by kNN
    """
    dsize = dset.shape[0]
    sqdiff = sum((tile(inp,[dsize,1]) - dset)**2,axis=1) # sum of squares of dist
    dist = sqdiff**0.5 # sq root of sum of squares of dist
    sorteddistindices = dist.argsort() # sort in ascending order
    classvotes = {}
    # each of the nearest k neighbors votes for its own class
    for i in range(k):
        label = classes[sorteddistindices[i]]
        classvotes[label] = classvotes.get(label,0) + 1 # second arg to get is the value to
                                                        # return if the element is not in the dict
    sortedclassvotes = sorted(classvotes.iteritems(),key=operator.itemgetter(1),reverse=True) # itemgetter(1) gets the dict's values.
                                                                                              # itemgetter(0) would get the dict's keys
    return sortedclassvotes[0][0]

if __name__ == "__main__":
    sys.dont_write_bytecode = True # don't want me no pyc files. You can also say 'python -B'
    print "Running k nearest neighbors"
    g, l = createdataset()
    inp = [0,0]
    print "Classification of %s is %s " % (inp, classify(inp,g,l,4))
