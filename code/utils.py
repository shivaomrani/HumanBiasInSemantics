from sklearn.metrics.pairwise import cosine_similarity
import random as random
from scipy.stats import norm
from statsmodels.distributions.empirical_distribution import ECDF
import statistics as stat

class operations(object):

    def __init__(self, concept1,concept2,stereotype1,stereotype2,caseSensitive, iterations, vectors, word2index, distribution):
        self.concept1 = concept1
        self.concept2 = concept2
        self.stereotype1 = stereotype1
        self.stereotype2 = stereotype2
        self.caseSensitive = caseSensitive
        self.iterations = iterations
        self.vectors = vectors
        self.word2index = word2index
        self.distribution = distribution

    def getPValueAndEffect(self):
        pValue = []
        testStatistic = self.getTestStatistic(self.concept1,self.concept2,self.stereotype1,self.stereotype2,self.caseSensitive, self.vectors, self.word2index)
        nullDist = self.nullDistribution(self.concept1, self.concept2, self.stereotype1, self.stereotype2,  self.caseSensitive, self.iterations, self.vectors, self.word2index)
        entireDistribution = self.getEntireDistribution(self.concept1, self.concept2, self.stereotype1, self.stereotype2, self.caseSensitive, self.iterations, self.vectors, self.word2index)

        pValue.append(1-self.calculateCumulativeProbability(nullDist, testStatistic, self.distribution))
        pValue.append(self.effectSize(entireDistribution, testStatistic))
        pValue.append(stat.stdev(nullDist))
        return pValue;

    def nullDistribution(self, concept1, concept2, stereotype1, stereotype2,  caseSensitive, iterations, vectors, word2index):

        # permute concepts and for each permutation calculate getTestStatistic and save it in your distribution
        bothConcepts = concept1 + concept2
        print("Generating null distribution...")

        stereotype1NullMatrix = []
        stereotype2NullMatrix = []

        for attribute in stereotype1:
            similarity_list = []
            stereotype1Embedding = vectors[word2index[attribute]]

            for word in bothConcepts:
                nullEmbedding = vectors[word2index[word]]
                similarity = self.cosineSimilarity(nullEmbedding, stereotype1Embedding)
                similarity_list.append(similarity)
            stereotype1NullMatrix.append(similarity_list)


        for attribute in stereotype2:
            similarity_list = []
            stereotype2Embedding = vectors[word2index[attribute]]

            for word in bothConcepts:
                nullEmbedding = vectors[word2index[word]]
                similarity = self.cosineSimilarity(nullEmbedding, stereotype2Embedding)
                similarity_list.append(similarity)
            stereotype2NullMatrix.append(similarity_list)

        #Assuming both concepts have the same length
        setSize = int(len(bothConcepts)/2)
        print("Number of permutations ", iterations)
        toShuffle = list(range(0, len(bothConcepts)))
        distribution = []

        for iter in range(iterations):
            random.shuffle(toShuffle)
        	#calculate mean for each null shuffle
            meanSimilaritycon1str1 = 0
            meanSimilaritycon1str2 = 0
            meanSimilaritycon2str1 = 0
            meanSimilaritycon2str2 = 0

            for i in range(len(stereotype1)):
                for j in range(setSize):
                    meanSimilaritycon1str1 = meanSimilaritycon1str1 + stereotype1NullMatrix[i][toShuffle[j]]

            for i in range(len(stereotype2)):
                for j in range(setSize):
                    meanSimilaritycon1str2 = meanSimilaritycon1str2 + stereotype2NullMatrix[i][toShuffle[j]]

            for i in range(len(stereotype1)):
                for j in range(setSize):
                    meanSimilaritycon2str1 = meanSimilaritycon2str1 + stereotype1NullMatrix[i][toShuffle[j+setSize]]

            for i in range(len(stereotype2)):
                for j in range(setSize):
                    meanSimilaritycon2str2 = meanSimilaritycon2str2 + stereotype2NullMatrix[i][toShuffle[j+setSize]]

            meanSimilaritycon1str1 = meanSimilaritycon1str1/(len(stereotype1)*setSize)
            meanSimilaritycon1str2 = meanSimilaritycon1str2/(len(stereotype2)*setSize)
            meanSimilaritycon2str1 = meanSimilaritycon2str1/(len(stereotype1)*setSize)
            meanSimilaritycon2str2 = meanSimilaritycon2str2/(len(stereotype2)*setSize)

            #come back here later
            distribution.append((meanSimilaritycon1str1 - meanSimilaritycon1str2) - meanSimilaritycon2str1 + meanSimilaritycon2str2)

        return distribution

    def calculateCumulativeProbability(self,nullDistribution, testStatistic, distribution):
        cumulative = -100
        nullDistribution.sort()

        if distribution == 'empirical':
            ecdf = ECDF(nullDistribution)
            cumulative = ecdf(testStatistic)
        elif distribution == 'normal':
            d = norm(loc = stat.mean(nullDistribution), scale = stat.stdev(nullDistribution))
            cumulative = d.cdf(testStatistic)

        return cumulative

    def effectSize(self,array, mean):
        effect = mean/stat.stdev(array)
        return effect

    def getTestStatistic(self, concept1, concept2, stereotype1, stereotype2,caseSensitive, vectors, word2index):

        differenceOfMeans =0
        differenceOfMeansConcept1 =0
        differenceOfMeansConcept2 =0

        #concept 1 computations
        for word in concept1:
            concept1_embedding = vectors[word2index[word]]

            meanConcept1Stereotype1=0
            for attribute in stereotype1:
                stereotype1_embedding = vectors[word2index[attribute]]
                similarity = self.cosineSimilarity(concept1_embedding, stereotype1_embedding)
                meanConcept1Stereotype1 = meanConcept1Stereotype1 + similarity

            meanConcept1Stereotype1 = meanConcept1Stereotype1/len(stereotype1)


            meanConcept1Stereotype2=0
            for attribute in stereotype2:
                stereotype2_embedding = vectors[word2index[attribute]]
                similarity = self.cosineSimilarity(concept1_embedding, stereotype2_embedding)
                meanConcept1Stereotype2 = meanConcept1Stereotype2 + similarity

            meanConcept1Stereotype2 = meanConcept1Stereotype2/len(stereotype2)

            differenceOfMeansConcept1 = differenceOfMeansConcept1+ meanConcept1Stereotype1 - meanConcept1Stereotype2

        #effect size computations mean S(x,A,B)
        differenceOfMeansConcept1 = differenceOfMeansConcept1/len(concept1)

        #concept 2 computations
        for word in concept2:
            concept2_embedding = vectors[word2index[word]]

            meanConcept2Stereotype1=0
            for attribute in stereotype1:
                stereotype1_embedding = vectors[word2index[attribute]]
                similarity = self.cosineSimilarity(concept2_embedding, stereotype1_embedding)
                meanConcept2Stereotype1 = meanConcept2Stereotype1 + similarity

            meanConcept2Stereotype1 = meanConcept2Stereotype1/len(stereotype1)

            meanConcept2Stereotype2=0
            for attribute in stereotype2:
                stereotype2_embedding = vectors[word2index[attribute]]
                similarity = self.cosineSimilarity(concept2_embedding, stereotype2_embedding)
                meanConcept2Stereotype2 = meanConcept2Stereotype2 + similarity

            meanConcept2Stereotype2 = meanConcept2Stereotype2/len(stereotype2)

            differenceOfMeansConcept2 = differenceOfMeansConcept2+ meanConcept2Stereotype1 - meanConcept2Stereotype2

        #effect size computations mean S(x,A,B)
        differenceOfMeansConcept2 = differenceOfMeansConcept2/len(concept2)
        differenceOfMeans = differenceOfMeansConcept1 - differenceOfMeansConcept2

        #used for effect size computations before dividing by standard deviation
        print("The difference of means is ", differenceOfMeans)
        return differenceOfMeans

    def getEntireDistribution(self, concept1, concept2, stereotype1, stereotype2,  caseSensitive, iterations, vectors, word2index):

        bothConcepts = concept1 + concept2
        distribution = []
        print("Getting the entire distribution")

        for word in bothConcepts:
            conceptEmbedding = vectors[word2index[word]]
            similarityToStereotype1 = 0
            similarityToStereotype2 = 0

            for attribute in stereotype1:
                stereotype1Embedding = vectors[word2index[attribute]]
                similarityToStereotype1 = similarityToStereotype1 + self.cosineSimilarity(conceptEmbedding, stereotype1Embedding)
            similarityToStereotype1 = similarityToStereotype1/len(stereotype1)

            for attribute in stereotype2:
                stereotype2Embedding = vectors[word2index[attribute]]
                similarityToStereotype2 = similarityToStereotype2 + self.cosineSimilarity(conceptEmbedding, stereotype2Embedding)
            similarityToStereotype2 = similarityToStereotype2/len(stereotype2)

            distribution.append(similarityToStereotype1 - similarityToStereotype2)

        return distribution

    def cosineSimilarity(self,a, b):
        a = [a]
        b = [b]
        r = cosine_similarity(a,b)
        return r[0][0]
