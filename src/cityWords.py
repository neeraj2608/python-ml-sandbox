# Compare most frequently used words specific to a particular city
# using naive bayes
# Uses the fact that the highest features are weighted the most.

import naiveBayes
import numpy as np
import operator

if __name__ == '__main__':
    DEBUG = 0
else:
    DEBUG = 0

def parse(inp):
    import re
    listTokens = re.split(r'\W*', inp)
    return [elem.lower() for elem in listTokens]

def runClassification(trainingData, trainingClassVec):
    # split training and test data
    TESTINGDATASIZE = 10
    testingData = []
    actualTestingVec = []
    for index in range(0,TESTINGDATASIZE):
        import random
        i = int(random.uniform(0,len(trainingData)))
        testingData.append(trainingData[i])
        actualTestingVec.append(trainingClassVec[i])
        del(trainingData[i])
        del(trainingClassVec[i])

    trainingVocabList, (pC0,pWGivenC0), (pC1,pWGivenC1), pWs = naiveBayes.trainData(trainingData, trainingClassVec)

    predictedTestingVec = []
    topPC0 = []
    topPC1 = []
    for testData in testingData:
        testDataVector = np.array(naiveBayes.bagOfWordsToVector(trainingVocabList, testData))
        pC0GivenData = testDataVector * pWGivenC0 * pC0 + 1
        pC1GivenData = testDataVector * pWGivenC1 * pC1 + 1
        topPC0 = addUnique(topPC0, getTopN(trainingVocabList, pC0GivenData, 30))
        topPC1 = addUnique(topPC1, getTopN(trainingVocabList, pC1GivenData, 30))
        if sum(np.log(pC0GivenData)) > sum(np.log(pC1GivenData)):
            predictedTestingVec.append(0)
        else:
            predictedTestingVec.append(1)

    i = 0
    error = 0
    misClassified = []
    for predicted in predictedTestingVec:
        if (actualTestingVec[i] != predicted):
            error += 1
            misClassified.append(testingData[i])
        i += 1

    if(DEBUG):
        print predictedTestingVec
        print actualTestingVec
        print 'num errors: %d' % error
        print 'misclassified:'
        print misClassified

    return getTopNFromList(topPC0,30), getTopNFromList(topPC1,30), float(error)/TESTINGDATASIZE

def addUnique(list1, list2):
    temp = dict(list1)
    for (x,y) in list2:
        if(x in temp):
            if(temp[x] < y):
                temp[x] = y
        else:
            temp[x] = y
    return temp.items()

def getTopNFromList(lst, n):
    sortedlst = sorted(lst,key=operator.itemgetter(1),reverse=True)
    return sortedlst[:n]

def getTopN(vlst, plst, n):
    d = {}
    index = 0
    for word in vlst:
        d[word] = plst[index]
        index += 1
    sortedlst = sorted(d.iteritems(),key=operator.itemgetter(1),reverse=True)
    return sortedlst[:n]

if __name__ == '__main__':
    import feedparser
    ny = feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
    sf = feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')
    minlen = min(len(ny),len(sf))
    trainingData = []
    trainingClassVec = []
    for index in range(0,minlen):
        trainingData.append(parse(ny['entries'][index]['summary']))
        trainingClassVec.append(1) # 1 is ny
        trainingData.append(parse(sf['entries'][index]['summary']))
        trainingClassVec.append(0) # 0 is sf

    error = 0
    NUMRUNS = 1
    topPC0 = []
    topPC1 = []
    for index in range(0,NUMRUNS):
        tPC0, tPC1, e = runClassification(list(trainingData), list(trainingClassVec))
        error += e
        topPC0 += tPC0
        topPC1 += tPC1

    topPC0 = getTopNFromList(topPC0, 30)
    topPC1 = getTopNFromList(topPC1, 30)
    print "Most common words for New York:"
    print '\n'.join([x for (x,y) in topPC0])
    print "\nMost common words for SF:"
    print '\n'.join([x for (x,y) in topPC1])
