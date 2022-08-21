import numpy as np
import jieba

class Corpus(object):
    def __init__(self, texts):
        self.texts = texts
        self.tokenizedCorpus = []
    def makeCorpus(self):
        for sentence in self.texts:
            self.tokenizedCorpus.append([])
            for x in jieba.tokenize(sentence):
                if x[0] != " ":
                    self.tokenizedCorpus[-1].append(x[0])
    def getTokenizedCorpus(self):
        return self.tokenizedCorpus


class TrainingData(object):
    def __init__(self, tokenizedCorpus):
        self.tokenizedCorpus = tokenizedCorpus.getTokenizedCorpus()
        # 词汇表
        self.vocab = []
        self.vocabSize = 0
        self.oneHotVectors = {}
        self.trainingData = []

    def makeVocab(self):
        for sentence in self.tokenizedCorpus:
            for word in sentence:
                if word not in self.vocab:
                    self.vocab.append(word)
        self.vocabSize = len(self.vocab)

    def makeOneHotVectors(self):
        for word in self.vocab:
            baseVector = np.zeros(self.vocabSize)
            baseVector[self.vocab.index(word)] = 1
            self.oneHotVectors[word] = baseVector
    def makeTrainingData(self, windowSize = 2):
        for sentence in self.tokenizedCorpus:
            # print(sentence)
            for i in range(len(sentence)):
                targetWordVector = self.oneHotVectors[sentence[i]]
                contextWordVectors = []
                for j in range(i - windowSize, i):
                    if i < 0:
                        continue
                    else:
                        contextWordVectors.append(self.oneHotVectors[sentence[j]])
                for j in range(i + 1, i + 1 + windowSize):
                    if j < len(sentence):
                        contextWordVectors.append(self.oneHotVectors[sentence[j]])
                    else:
                        continue
                self.trainingData.append([targetWordVector, contextWordVectors])

class Word2Vec(object):
    def __init__(self, data, dimensions, lr, epochs):
        self.vocab = data.vocab
        self.vocabSize = data.vocabSize
        self.trainingData = data.trainingData
        self.n = dimensions
        self.lr = lr
        self.epochs = epochs
        self.W1 = np.random.uniform(-1, 1, (self.n, self.vocabSize))
        self.W2 = np.random.uniform(-1, 1, (self.vocabSize, self.n))
    def forward(self, x):
        h = np.dot(self.W1, x)
        u = np.dot(self.W2, h)
        y = self.softmax(u)
        return h, u, y
    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)
    def backprop(self, error, h, x):
        dW2 = np.outer(error, h)
        dW1 = np.outer(np.dot(self.W2.T, error), x)
        self.W1 -= self.lr * dW1
        self.W2 -= self.lr * dW2
    def train(self):
        for i in range(self.epochs):
            self.loss = 0
            for target, contexts in self.trainingData:
                h, u, y = self.forward(target)
                totalError = np.sum([np.subtract(y, context) for context in contexts], axis=0)
                self.backprop(totalError, h, target)
                self.loss += -np.sum([u[context.tolist().index(1)] for context in contexts]) +\
                    len(contexts) * np.log(np.sum(np.exp(u)))
            print('Epoch:', i, 'Loss:', self.loss)
    def getVector(self, word):
        wordIndex = self.vocab.index(word)
        prepareOneHot = np.zeros(self.vocabSize)
        prepareOneHot[wordIndex] = 1

        return np.dot(self.W1, prepareOneHot)

texts = ["利用Python Numpy从零开始步步为营计算Word2Vec词向量"]
corpus = Corpus(texts)
corpus.makeCorpus()
# corpusDict = corpus.getTokenizedCorpus()
# print(corpus)

trainingData = TrainingData(corpus)
trainingData.makeVocab()
trainingData.makeOneHotVectors()
trainingData.makeTrainingData()

word2Vec = Word2Vec(trainingData, 8, 0.01, 100)
word2Vec.train()
# print(trainingData.trainingData)