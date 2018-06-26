import numpy as np
from random import shuffle
from numpy.linalg import inv
import math

def classify(d1, d2, d3):
    dk = max(d1, d2, d3)
    if d1 == dk:
        groupNum = 1
    elif d2 == dk:
        groupNum = 2
    else:
        groupNum = 3
    return groupNum

def discriminantScore(mean, pooledCovar, observed, prob):
    score = float(mean.T*inv(pooledCovar)*observed - 0.5*mean.T*inv(pooledCovar)*mean + \
    math.log(prob))
    return score

class dataPt:
    def __init__(self, data, label):
        self.data = data
        self.label = label

data = []
dataDim = 4
sampleSize = 50
group = 3
prob1 = prob2 = prob3 = 1.0/3.0
observed = np.array([[3.5], [1.75]]) 

inputDat = open('irisData', 'r')
for line in inputDat:
    rawDat = line.strip().split(',')
    rawDat = map(float, rawDat[slice(1, 3)]) #last column denotes category
    data.append(rawDat)
inputDat.close()
data = np.array(data)
setosa = np.insert(data[0:50, :], 2, 1, axis=1)
versicolor = np.insert(data[50:100, :], 2, 2, axis=1)
virginica = np.insert(data[100:150, :], 2, 3, axis=1)
setosaMean = (np.mean(np.matrix(setosa[:, 0:2]), 0)).T
versicolorMean = (np.mean(np.matrix(versicolor[:, 0:2]), 0)).T
virginicaMean = (np.mean(np.matrix(virginica[:, 0:2]), 0)).T

setosaCovar = np.cov(setosa[:, 0:2].T)
versicolorCovar  = np.cov(versicolor[:,0:2].T)
virginicaCovar = np.cov(virginica[:, 0:2].T)
pooledCovar = 1.0/(group*sampleSize - group) * ((sampleSize-1)*setosaCovar + \
        (sampleSize-1)*versicolorCovar + (sampleSize-1)*virginicaCovar) 

d1 = discriminantScore(setosaMean, pooledCovar, observed, prob1)
d2 = discriminantScore(versicolorMean, pooledCovar, observed, prob2)
d3 = discriminantScore(virginicaMean, pooledCovar, observed, prob3)
print d1, d2, d3
print classify(d1, d2, d3)
totalMisclassified = 0
for i in range(sampleSize):
    d1 = discriminantScore(setosaMean, pooledCovar, np.matrix(setosa[i, 0:2]).T, prob1)
    d2 = discriminantScore(versicolorMean, pooledCovar, np.matrix(setosa[i, 0:2]).T, prob1)
    d3 = discriminantScore(virginicaMean, pooledCovar, np.matrix(setosa[i, 0:2]).T, prob1)
    totalMisclassified = totalMisclassified + (classify(d1, d2, d3) != 1)


for i in range(sampleSize):
    d1 = discriminantScore(setosaMean, pooledCovar, np.matrix(versicolor[i, 0:2]).transpose(), prob2)
    d2 = discriminantScore(versicolorMean, pooledCovar, np.matrix(versicolor[i, 0:2]).transpose(), prob2)
    d3 = discriminantScore(virginicaMean, pooledCovar, np.matrix(versicolor[i, 0:2]).transpose(), prob2)
    totalMisclassified = totalMisclassified + (classify(d1, d2, d3) != 2)

for i in range(sampleSize):
    d1 = discriminantScore(setosaMean, pooledCovar, np.matrix(virginica[i, 0:2]).transpose(), prob3)
    d2 = discriminantScore(versicolorMean, pooledCovar, np.matrix(virginica[i, 0:2]).transpose(), prob3)
    d3 = discriminantScore(virginicaMean, pooledCovar, np.matrix(virginica[i, 0:2]).transpose(), prob3)
    totalMisclassified = totalMisclassified + (classify(d1, d2, d3) != 3)

aper = totalMisclassified / float(group*sampleSize)
print aper
data = np.concatenate((setosa, versicolor, virginica), axis = 0)
rep = 100
fold = group*sampleSize/10
totalMisclassified = 0
for i in range(rep):
    indices = range(group*sampleSize)
    shuffle(indices)
    for j in range(fold):
#        print list(set(range(50))-set(indices[j*10:(j+1)*10]))
        setosa = data[list(set(range(50))-set(indices[j*10:(j+1)*10])), :]
        size1 = len(setosa)
        versicolor = data[list(set(range(50, 100))-set(indices[j*10:(j+1)*10])), :]
        size2 = len(versicolor)
        virginica = data[list(set(range(100, 150))-set(indices[j*10:(j+1)*10])), :]
        size3 = len(virginica)
        setosaMean = (np.mean(np.matrix(setosa[:, 0:2]), 0)).T
        versicolorMean = (np.mean(np.matrix(versicolor[:, 0:2]), 0)).T
        virginicaMean = (np.mean(np.matrix(virginica[:, 0:2]), 0)).T
        setosaCovar = np.cov(setosa[:, 0:2].T)
        versicolorCovar  = np.cov(versicolor[:,0:2].T)
        virginicaCovar = np.cov(virginica[:, 0:2].T)
        pooledCovar = 1.0/(size1+size2+size3-group) * ((size1-1)*setosaCovar + \
                (size2-1)*versicolorCovar + (size3-1)*virginicaCovar) 
        for i in range(10):
            d1 = discriminantScore(setosaMean, pooledCovar, np.matrix(data[indices[j*10+i], 0:2]).T, prob1)
            d2 = discriminantScore(versicolorMean, pooledCovar, np.matrix(data[indices[j*10+i], 0:2]).T, prob1)
            d3 = discriminantScore(virginicaMean, pooledCovar, np.matrix(data[indices[j*10+i], 0:2]).T, prob1)
            totalMisclassified = totalMisclassified + (classify(d1, d2, d3) !=
                    data[indices[j*10+i], 2])
                    
errorRate = totalMisclassified/float(group*sampleSize*rep)
print errorRate
        
