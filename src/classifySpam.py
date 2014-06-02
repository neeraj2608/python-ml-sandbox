import naiveBayes

if __name__ == '__main__':
    DEBUG = 1
else:
    DEBUG = 0

def parse(inp):
    import re
    listTokens = re.split(r'\W*', inp)
    return [elem.lower() for elem in listTokens]

if __name__ == '__main__':
    testData = []
    classData = []
    for index in range(1,26):
        testData.append(parse(open('email/spam/%d.txt' % index).read()))
        classData.append(1) # 1 is spam
        testData.append(parse(open('email/ham/%d.txt' % index).read()))
        classData.append(0) # 0 is spam

    (pC0,pWGivenC0), (pC1,pWGivenC1), pWs = naiveBayes.trainData(testData, classData)

    if(DEBUG): print (pC0,pWGivenC0), (pC1,pWGivenC1), pWs
