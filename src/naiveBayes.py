import numpy as np

if __name__ == '__main__':
    DEBUG = 1
else:
    DEBUG = 0

def createTrainingData():
    postingList = [['my','dog','has','dog',\
                    'problems','help','please'],
                   ['my','dalmation','is','so',\
                    'cute','I','love','him'],
                   ['mr','licks','ate','my',\
                    'steak','how','to','stop','him'],
                   ['maybe','not','take','him',\
                    'to','dog','park','stupid'],
                   ['stop','posting','stupid','worthless',\
                    'garbage'],
                   ['quit','buying','worthless','dog',\
                    'food','stupid']]
    classVec = [0,0,0,1,1,1] # 1 is spam
    return postingList, classVec

def createVocabList(dataSet):
    '''
    Creates a set of words from the given dataSet. A dataSet is
    a list of lists of words e.g. [['hello','there'],['hi','there']].
    The set thus created is returned as a list.
    '''
    wordSet = set([])
    for wordList in dataSet:
        wordSet = wordSet | set(wordList)
    return list(wordSet)

def bagOfWordsToVector(vocabSet,inputSet):
    result = [0]*len(vocabSet)
    for word in inputSet:
        if word in vocabSet:
            result[vocabSet.index(word)] = result[vocabSet.index(word)] + 1
    if(DEBUG):
        print inputSet
        print result
    return result

def wordSetToVector(vocabSet,inputSet):
    result = [0]*len(vocabSet)
    for word in inputSet:
        if word in vocabSet:
            result[vocabSet.index(word)] = 1
    return result

def trainClassifier(postingVec, classVec):
    nWGivenC0 = np.zeros(len(postingVec[0]))
    nWGivenC1 = np.zeros(len(postingVec[0]))
    nWs = np.zeros(len(postingVec[0]))
    numAllWs = 0
    numAllWsC0 = 0
    numAllWsC1 = 0
    numC1 = 0
    index = 0
    for post in postingVec:
        numAllWs = numAllWs + sum(post)
        nWs = np.add(nWs, np.array(post))
        if (classVec[index] == 0):
            nWGivenC0 = np.add(nWGivenC0, np.array(post))
            numAllWsC0 += sum(post)
        else:
            nWGivenC1 = np.add(nWGivenC1, np.array(post))
            numAllWsC1 += sum(post)
            numC1 += 1
        index += 1

    pWGivenC0 = nWGivenC0/numAllWsC0 # probability of each word, given class C0
    pWGivenC1 = nWGivenC1/numAllWsC1 # probability of each word, given class C1
    pWs = nWs/numAllWs # probability of each word

    pC1 = float(numC1) / len(classVec) # probability of class 1
    pC0 = 1 - pC1 # probability of class 0

    return (pC0,pWGivenC0), (pC1,pWGivenC1), pWs

def trainData(trainingWordList, trainingClassVec):
    trainingVocabList = createVocabList(trainingWordList)

    postingVec = []
    for word in trainingWordList:
        postingVec.append(bagOfWordsToVector(trainingVocabList,word))

    return trainClassifier(postingVec, trainingClassVec)

def classify(testData, trainingWordList, trainingClassVec):
    trainingVocabList = createVocabList(trainingWordList)
    testDataVector = np.array(bagOfWordsToVector(trainingVocabList, testData))

    (pC0,pWGivenC0), (pC1,pWGivenC1), pWs = trainData(trainingWordList, trainingClassVec)

    # Bayes' Rule:
    #               P(w|C)*P(C) + alpha
    # P(C|w) = ---------------------------------
    #          P(w) + alpha*(no. of total words)
    # We assume alpha = 1
    #
    # Using numpy arrays, this is:
    # pC0GivenData = (testDataVector * pWGivenC0 * pC0 + 1) / ((testDataVector * pWs) + len(trainingVocabList)) # + 1 and + len(trainingVocabList) for laplacian smoothing
    # pC1GivenData = (testDataVector * pWGivenC1 * pC1 + 1) / ((testDataVector * pWs) + len(trainingVocabList)) # + 1 and + len(trainingVocabList) for laplacian smoothing
    #
    # Here,
    # pC0GivenData is a vector containing p(C|w) for all w's in the vocabulary, with words not in testData having their elements set to ALPHA, and
    # words in testData having actual p(C|w) values
    # Since we only care about the RELATIVE magnitude of pC0GivenData and pC1GivenData, the denominator, which is the same for both
    # values, can be ignored. This is what is done below.
    pC0GivenData = testDataVector * pWGivenC0 * pC0 + 1
    pC1GivenData = testDataVector * pWGivenC1 * pC1 + 1

    # Note that we return logs so that small products multiplied together (later on) don't underflow (with
    # the logs, we add the individual p(C|w) probabilities)
    return np.log(pC0GivenData), np.log(pC1GivenData)

if __name__ == '__main__':
    testData = ['my','stupid','dog']
    # testData = ['stupid','garbage']
    # testData = ['love','my','dalmation']

    trainingWordList, trainingClassVec = createTrainingData()
    logPC0GivenData, logPC1GivenData = classify(testData, trainingWordList, trainingClassVec)

    if(DEBUG):
        print sum(logPC0GivenData)
        print sum(logPC1GivenData)
        # print '\n'.join(str(elem) for elem in postingVec)
        # print (pC0,pWGivenC0), (pC1,pWGivenC1), pWs

    if (sum(logPC0GivenData) > sum(logPC1GivenData)):
        print "%s: %s" % (testData, "Not spam")
    else:
        print "%s: %s" % (testData, "Spam")
