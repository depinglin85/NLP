# pip install paddlepaddle

import os
import io
import sys
import requests
from collections import OrderedDict
import math
import random
import numpy as np
import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import Embedding

# 下载语料库
def download():
    corpusFile = "./data/text8.txt"
    if not os.path.isfile(corpusFile):
        url = "https://dataset.bj.bcebos.com/word2vec/text8.txt"
        webRequest = requests.get(url)
        with open("./data/text8.txt", "wb") as file:
            file.write(webRequest.content)
        file.close()
# 读取语料库
def loadText8Data():
    download()
    with open("./data/text8.txt", "r") as file:
        corpus = file.read().split("¥n")
    file.close()
    return corpus
# 对语料库进行分词，如果是中文可以用jieba
def corpusPreprocess(corpus):
    corpus = corpus.split(" ")
    return corpus

# 统计词频
def buildDict(corpus):
    wordFreqDict = dict()
    for word in corpus:
        if word not in wordFreqDict:
            wordFreqDict[word] = 0
        wordFreqDict[word] += 1
    wordFreqDict = sorted(wordFreqDict.items(), key = lambda x:x[1], reverse = True)
    # 词：id字典
    word2idDict = dict()
    # 词id：词频
    word2idFreqDict = dict()
    # 词id：词
    id2wordDict = dict()
    for word, freq in wordFreqDict:
        currentId = len(word2idDict)
        word2idDict[word] = currentId
        word2idFreqDict[currentId] = freq
        id2wordDict[currentId] = word
    return word2idFreqDict, word2idDict, id2wordDict

def convertCorpus2Id(corpus, word2idDict):
    corpus = [word2idDict[word] for word in corpus]
    return corpus

def subsampling(corpus, word2idFreqDict):
    def discard(wordId):
        return random.uniform(0, 1) < 1 - math.sqrt(1e-4/word2FreqDict[wordId] * len(corpus))
    corpus = [word for word in corpus if not discard(word)]
    #TODO

def buildData(corpus, word2idDict, word2idFreqDict, maxWindowSize = 3, negativeSampleSize = 4):
    dataset = []
    centerWordIdx = 0
    while centerWordIdx < len(corpus):
        windowSize = random.randint(1, maxWindowSize)
        positiveWord = corpus[centerWordIdx]
        contextWordRange = (max(0, centerWordIdx - windowSize), min(len(corpus) - 1, centerWordIdx + windowSize))
        contextWordCandidates = [corpus[idx] for idx in range(contextWordRange[0], contextWordRange[1] + 1) if idx != centerWordIdx]

        for contextWord in contextWordCandidates:
            dataset.append((contextWord, positiveWord, 1))

            i = 0
            while i < negativeSampleSize:
                negativeWordcandidate = random.randint(0, len(word2idFreqDict) - 1)
                if negativeWordcandidate is not positiveWord:
                    dataset.append((contextWord, negativeWordCandidate, 0))
                    i += 1
        centerWordIdx = min(len(corpus) - 1, centerWordIdx + windowSize)
        if centerWordIdx == (len(corpus) - 1):
            centerWordIdx += 1
        if centerWordIdx % 10000 == 0:
            print(centerWordIdx)
    return dataset

def buildBatch(dataset, batchSize, epochNumber):
    contextWordBatch = []
    targetWordBatch = []
    labelBatch = []
    evalWordBatch = []
    
    for epoch in range(epochNumber):
        random.shuffle(dataset)
        for contextWord, targetWord, label in dataset:
            contextWordBatch.append([contextWord])
            targetWordBatch.append([targetWord])
            labelBatch.append(label)
            if len(evalWordBatch) == 0:
                evalWordBatch.append([word2idDict['one']])
            elif len(evalWordBatch) == 1:
                evalWordBatch.append([word2idDict['king']])
            elif len(evalWordBatch) == 2:
                evalWordBatch.append([word2idDict['who']])
            
            if len(contextWordBatch) == batchSize:
                yield epoch,\
                    np.array(contextWordBatch).astype("int64"),\
                    np.array(targetWordBatch).astype("int64"),\
                    np.array(labelBatch).astype("float32"),\
                    np.array(evalWordBatch).astype("int64")
                contextWordBatch = []
                targetWordBatch = []
                labelBatch = []
                evalWordBatch = []

    if len(contextWordBatch) > 0:
        yield epoch,\
            np.array(context_wocontextWordBatchrd_batch).astype("int64"),\
            np.array(targetWordBatch).astype("int64"),\
            np.array(labelBatch).astype("float32"),\
            np.array(evalWordBatch).astype("int64")

class SkipGram(fluid.dygraph.Layer):
    def __init__(self, nameScope, vocabSize, embeddingSize, initScale = 0.1):
        super(SkipGram, self).__init__(nameScope)
        self.vocabSize = vocabSize
        self.embeddingSize = embeddingSize
        self.embedding = Embedding(
            self.full_name(),
            size=[self.vocabSize, self.embeddingSize],
            dtype='float32',
            param_attr=fluid.ParamAttr(
                name='embedding_para',
                initializer=fluid.initializer.UniformInitializer(
                    low=-0.5/embeddingSize, high=0.5/embeddingSize
                )
            )
        )
        self.embeddingOut = Embedding(
            self.full_name(),
            size=[self.vocabSize, self.embeddingSize],
            dtype='float32',
            param_attr=fluid.ParamAttr(
                name='embedding_out_para',
                initializer=fluid.initializer.UniformInitializer(
                    low=-0.5/embeddingSize, high=0.5/embeddingSize
                )
            )
        )
    def forward(self, contextWords, targetWords, label, evalWords):
        contextWordsEmb = self.embedding(contextWords)
        targetWordsEmb = self.embeddingOut(targetWords)
        evalWordsEmb = self.embedding(evalWords)
        wordSim = fluid.layers.elementwise_mul(contextWordsEmb, targetWordsEmb)
        wordSim = fluid.layers.reduce_sum(wordSim, label)
        pred = fluid.layers.sigmoid(wordSim)
        loss = fluid.layers.sigmoid_cross_entropy_with_logits(wrodSim, label)
        loss = fluid.layers.reduce_mean(loss)
        wordSimOnFly = fluid.layers.matmul(evalWordEmb, self.embedding._w, transpose_y = True)
        return pred, loss, wordSimOnFly


corpus = corpusPreprocess(loadText8Data()[0])
word2idFreqDict, word2idDict, id2wordDict = buildDict(corpus)





    