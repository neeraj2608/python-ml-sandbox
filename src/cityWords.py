# Compare most frequently used words specific to a particular city
# using naive bayes
# Uses the fact that the highest-frequency features are weighted the most.

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

def runClassification(trainingVocabList, fullData, fullClassVec):
    # split into training and test data
    trainingData = fullData
    trainingClassVec = fullClassVec
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

    (pC0,pWGivenC0), (pC1,pWGivenC1), pWs = naiveBayes.trainData(trainingVocabList, trainingData, trainingClassVec)

    topPC0 = []
    topPC1 = []
    for testData in testingData:
        testDataVector = np.array(naiveBayes.bagOfWordsToVector(trainingVocabList, testData))
        pC0GivenData = testDataVector * pWGivenC0 * pC0 + 1
        pC1GivenData = testDataVector * pWGivenC1 * pC1 + 1
        topPC0 = addUnique(topPC0, getTopN(trainingVocabList, pC0GivenData, 30)) # make a UNIQUE list of the most frequent words
        topPC1 = addUnique(topPC1, getTopN(trainingVocabList, pC1GivenData, 30)) # make a UNIQUE list of the most frequent words

    return getTopNFromList(topPC0,30), getTopNFromList(topPC1,30)

def removeNMostFrequentWords(vlst, allWords, n):
    x = {}
    for word in vlst:
        x[word] = allWords.count(word)
    sortedvlst = sorted(x.iteritems(), key=operator.itemgetter(1))
    return [x for (x,y) in sortedvlst[:-n]]

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
    fullData = []
    fullClassVec = []
    allWords = []
    for index in range(0,minlen):
        words = parse(ny['entries'][index]['summary'])
        fullData.append(words)
        allWords.extend(words)
        fullClassVec.append(1) # 1 is ny
        words = parse(sf['entries'][index]['summary'])
        fullData.append(words)
        allWords.extend(words)
        fullClassVec.append(0) # 0 is sf

    # remove the most frequent words (combined in both cities).
    trainingVocabList = naiveBayes.createVocabList(fullData)
    trainingVocabList = removeNMostFrequentWords(trainingVocabList, allWords, 30)

    NUMRUNS = 2
    topPC0 = []
    topPC1 = []
    for index in range(0,NUMRUNS):
        tPC0, tPC1 = runClassification(trainingVocabList, list(fullData), list(fullClassVec))
        topPC0 += tPC0
        topPC1 += tPC1

    topPC0 = getTopNFromList(topPC0, 30)
    topPC1 = getTopNFromList(topPC1, 30)
    print "Most common words for New York:"
    print '\n'.join([x for (x,y) in topPC0])
    print "\nMost common words for SF:"
    print '\n'.join([x for (x,y) in topPC1])
