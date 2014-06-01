import numpy as np

def createTrainingData():
    postingList = [['my','dog','has','flea',\
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

def wordSetToVector(vocabSet,inputSet):
    result = [0]*len(vocabSet)
    for word in inputSet:
        if word in vocabSet:
            result[vocabSet.index(word)] = 1
    return result

def trainData(postingVec, classVec):
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

def classify(testData):
    trainingWordList, trainingClassVec = createTrainingData()
    trainingVocabList = createVocabList(trainingWordList)

    postingVec = []
    for word in trainingWordList:
        postingVec.append(wordSetToVector(trainingVocabList,word))

    (pC0,pWGivenC0), (pC1,pWGivenC1), pWs = trainData(postingVec, trainingClassVec)

    pC0GivenData = 1
    pC1GivenData = 1
    for word in testData:
        index = trainingVocabList.index(word)
        pWordGivenC0 = pWGivenC0[index]
        pWordGivenC1 = pWGivenC1[index]
        pWord = pWs[index]
        pC0GivenData = pC0GivenData * ((pWordGivenC0 * pC0 + 1) / (pWord + len(trainingVocabList))) # + 1 and + len(trainingVocabList) for laplacian smoothing
        pC1GivenData = pC1GivenData * ((pWordGivenC1 * pC1 + 1) / (pWord + len(trainingVocabList))) # + 1 and + len(trainingVocabList) for laplacian smoothing

    return pC0GivenData, pC1GivenData

if __name__ == '__main__':
    DEBUG = 1

    testData = ['stop','my','posting']

    pC0GivenData, pC1GivenData = classify(testData)

    if(DEBUG):
        if (pC0GivenData > pC1GivenData):
            print "%s: %s" % (testData, "Class 0")
        else:
            print "%s: %s" % (testData, "Class 1")

        # print '\n'.join(str(elem) for elem in postingVec)
        # print (pC0,pWGivenC0), (pC1,pWGivenC1), pWs
