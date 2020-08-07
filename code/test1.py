import codecs
import numpy as np
from utils import operations
from sklearn.metrics.pairwise import cosine_similarity
import statistics as stat
import pandas as pd

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


data = load_embeddings("embedding")
vectors = data[0]
index2word = data[1]
word2index = data[2]

namesFile = "../data/censusNames1990.csv"
namesToTest = "../data/wow.txt"
caseSensitive = True
checkWordPresence = True
getCentroid = True
wordDimension = 300

names = load_names(namesToTest)
attributesFirstSet = ["sister", "female", "woman", "girl", "daughter", "she", "hers", "her"]
attributesSecondSet = ["brother", "male", "man", "boy", "son", "he", "his", "him"]

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

for i in range(len(names)):

    percentage = calculateWomenNamePercentage(namesFile, names[i])
    nullDistributionConcept1 = []

    for j in range(len(bothStereotypes)):
        nullDistributionConcept1.append(concept1NullMatrix[i][j])


    concept1Embedding = vectors[word2index[names[i]]]
    frequency = word2index[names[i]]+1
    print(names[i] , ", " , percentage , ", " ,effectSize(nullDistributionConcept1, meanConcept1Stereotype1[i] - meanConcept1Stereotype2[i]) ,
      ", " , cosineSimilarity(concept1Embedding, centroid) , ", " , frequency)
