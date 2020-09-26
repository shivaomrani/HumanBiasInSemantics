import codecs
import numpy as np
from utils import operations
from sklearn.metrics.pairwise import cosine_similarity
import statistics as stat
import pandas as pd
from scipy.stats import norm
import random as random


def load_embeddings(embeddings_path):
    with codecs.open(embeddings_path + '.vocab', 'r', 'utf-8') as f_in:
        index2word = [line.strip() for line in f_in]
    word2index = {w: i for i, w in enumerate(index2word)}
    wv = np.load(embeddings_path + '.npy')

    return wv, index2word, word2index

def load_names(namesFile):
    namesList = []
    f = open(namesFile, "r")
    for line in f:
        name = line[:-1]
        namesList.append(name)
    return namesList

def cosineSimilarity(a, b):
    a = [a]
    b = [b]
    r = cosine_similarity(a,b)
    return r[0][0]

def computeCentroid(names, vectors, word2index, wordDimension):

    counter = 0
    centroid = [0] * wordDimension

    for name in names:

        meanConcept1Stereotype2 = vectors[word2index[name]]
        if (meanConcept1Stereotype2[1] != 999.0):
            counter = counter + 1

        for i in range(wordDimension):
            centroid[i] = centroid[i] + meanConcept1Stereotype2[i]

    for column in range(wordDimension):
        centroid[column] = centroid[column]/float(counter)

    return centroid

def computeNullMatrix(names, bothStereotypes, vectors, word2index):

    concept1NullMatrix = []

    for name in names:

        concept1Embedding = vectors[word2index[name]]
        my_list = []

        for attribute in bothStereotypes:
            nullEmbedding = vectors[word2index[attribute]]
            similarityCompatible = cosineSimilarity(concept1Embedding, nullEmbedding)
            my_list.append(similarityCompatible)
        concept1NullMatrix.append(my_list)

    return concept1NullMatrix

def effectSize(array, mean):
    effect = mean/stat.stdev(array)
    return effect

def calculateWomenNamePercentage(namesFile, name):

    df = pd.read_csv(namesFile)
    i=df[df['Name'] == name]
    stat = np.asarray(i)
    percentage = (stat[0][2]*stat[0][3])/((stat[0][2]*stat[0][3])+ (stat[0][2]*stat[0][4]))
    return percentage

def calculateCumulativeProbability(nullDistribution, testStatistic, distribution):
    cumulative = -100
    nullDistribution.sort()

    if distribution == 'empirical':
        ecdf = ECDF(nullDistribution)
        cumulative = ecdf(testStatistic)
    elif distribution == 'normal':
        d = norm(loc = stat.mean(nullDistribution), scale = stat.stdev(nullDistribution))
        cumulative = d.cdf(testStatistic)

    return cumulative

def computeNullHypothesis(wordIndex, nullMatrix, iterations, stereotype1, stereotype2):

    print("Number of permutations ", iterations)

    #Assuming both stereotypes have the same length
    setSize = int(len(bothStereotypes)/2)

    toShuffle = list(range(0, len(bothStereotypes)))
    distribution = []

    for iter in range(iterations):
        random.shuffle(toShuffle)
        #calculate mean for each null shuffle
        meanSimilarityGroup1 = 0
        meanSimilarityGroup2 = 0

        for i in range(setSize):
            meanSimilarityGroup1 = meanSimilarityGroup1 + nullMatrix[wordIndex][toShuffle[i]]

        for i in range(setSize):
            meanSimilarityGroup2 = meanSimilarityGroup2 + nullMatrix[wordIndex][toShuffle[i+setSize]]

        meanSimilarityGroup1 = meanSimilarityGroup1/(setSize)
        meanSimilarityGroup2 = meanSimilarityGroup2/(setSize)

        distribution.append(meanSimilarityGroup1 - meanSimilarityGroup2)

    return distribution


data = load_embeddings("embedding")
vectors = data[0]
index2word = data[1]
word2index = data[2]

namesFile = "../data/censusNames1990.csv"
namesToTest = "../data/wow1.txt"
caseSensitive = True
checkWordPresence = True
getCentroid = True
wordDimension = 300

names = load_names(namesToTest)
print(names)
# attributesFirstSet = ["sister", "female", "woman", "girl", "daughter", "she", "hers", "her"]
# attributesSecondSet = ["brother", "male", "man", "boy", "son", "he", "his", "him"]
attributesFirstSet = ["caress", "freedom", "health", "love", "peace", "cheer", "friend","heaven", "loyal", "pleasure", "diamond", "gentle", "honest", "lucky", "rainbow","diploma", "gift", "honor", "miracle", "sunrise", "family", "happy", "laughter","paradise", "vacation"]
attributesSecondSet = ["abuse" , "crash" , "filth" , "murder" , "sickness" , "accident" , "death" , "grief" , "poison" , "stink" , "assault" , "disaster" , "hatred" , "pollute" , "tragedy" , "divorce" , "jail" , "poverty" , "ugly" , "cancer" , "kill" , "rotten" , "vomit" , "agony" , "prison"]

if getCentroid == True:
    centroid = computeCentroid(names, vectors, word2index,wordDimension)

bothStereotypes = attributesFirstSet + attributesSecondSet

meanConcept1Stereotype1 = [0]* len(names)
meanConcept1Stereotype2 = [0]* len(names)

for i in range(len(names)):

    concept1Embedding = vectors[word2index[names[i]]]

    for attribute in attributesFirstSet:
        stereotype2Embedding = vectors[word2index[attribute]]
        similarityCompatible = cosineSimilarity(concept1Embedding, stereotype2Embedding)
        meanConcept1Stereotype1[i] = meanConcept1Stereotype1[i] + similarityCompatible

    meanConcept1Stereotype1[i] = meanConcept1Stereotype1[i]/len(attributesFirstSet)

    for attribute in attributesSecondSet:
        stereotype2Embedding = vectors[word2index[attribute]]
        similarityCompatible = cosineSimilarity(concept1Embedding, stereotype2Embedding)
        meanConcept1Stereotype2[i] = meanConcept1Stereotype2[i] + similarityCompatible

    meanConcept1Stereotype2[i] = meanConcept1Stereotype2[i]/len(attributesSecondSet)

concept1NullMatrix = computeNullMatrix(names, bothStereotypes, vectors, word2index)

distribution = "normal"

for i in range(len(names)):

    # percentage = calculateWomenNamePercentage(namesFile, names[i])
    percentage = 0

    nullDistributionConcept1 = []
    for j in range(len(bothStereotypes)):
        nullDistributionConcept1.append(concept1NullMatrix[i][j])

    concept1Embedding = vectors[word2index[names[i]]]
    frequency = word2index[names[i]]+1
    print(names[i] , ", " , percentage , ", " ,effectSize(nullDistributionConcept1, meanConcept1Stereotype1[i] - meanConcept1Stereotype2[i]) ,
      ", " , cosineSimilarity(concept1Embedding, centroid) , ", " , frequency)

    #new stuff
    myMatrix = computeNullHypothesis(i, concept1NullMatrix, 1000000, attributesFirstSet, attributesSecondSet)
    cumulativeVal = calculateCumulativeProbability(myMatrix, (meanConcept1Stereotype2[i] - meanConcept1Stereotype1[i]), distribution)
    print("p val is ", 1- cumulativeVal)
