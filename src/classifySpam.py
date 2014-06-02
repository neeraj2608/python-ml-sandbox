import naiveBayes
import numpy as np

if __name__ == '__main__':
    DEBUG = 1
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
    for testData in testingData:
        testDataVector = np.array(naiveBayes.bagOfWordsToVector(trainingVocabList, testData))
        pC0GivenData = testDataVector * pWGivenC0 * pC0 + 1
        pC1GivenData = testDataVector * pWGivenC1 * pC1 + 1
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

    return float(error)/TESTINGDATASIZE

if __name__ == '__main__':
    trainingData = []
    trainingClassVec = []
    for index in range(1,26):
        trainingData.append(parse(open('email/spam/%d.txt' % index).read()))
        trainingClassVec.append(1) # 1 is spam
        trainingData.append(parse(open('email/ham/%d.txt' % index).read()))
        trainingClassVec.append(0) # 0 is spam

    error = 0
    NUMRUNS = 10
    for index in range(0,NUMRUNS):
        error += runClassification(list(trainingData), list(trainingClassVec))

    print "average error over %d runs: %f" % (NUMRUNS, float(error)/NUMRUNS)
