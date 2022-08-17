# pip install gensim
import math
import jieba
import jieba.posseg as psg
from gensim import corpora, models
from jieba import analyse
import functools
import numpy as np

def getStopwordList():
    stopWordPath = './data/stopword.txt'
    stopwordList = [sw.replace('\n', '') for sw in open(stopWordPath, encoding='utf-8').readlines()]
    # print(stopwordList)
    return stopwordList

def text2Words(text, pos = False):
    if not pos:
        words = jieba.cut(text)
    else:
        words = psg.cut(text)
    return words

def wordFilter(segList, pos = False):
    stopwordList = getStopwordList()
    filterList = []
    for seg in segList:
        if not pos:
            word = seg
            flag = 'n'
        else:
            word = seg.word
            flag = seg.flag
        if not flag.startswith('n'):
            continue
        if not word in stopwordList and len(word) > 1:
            filterList.append(word)
    return filterList

def loadData(pos = False, corpusFile='./data/corpus.txt'):
    docList = []
    for line in open(corpusFile, 'r'):
        content = line.strip()
        segList = text2Words(content, pos)
        filterList = wordFilter(segList, pos)
        docList.append(filterList)
    return docList

def trainIdf(docList):
    idfDic = {}
    documentCount = len(docList)
    for doc in docList:
        for word in set(doc):
            idfDic[word] = idfDic.get(word, 0.0) + 1.0
    
    for key, value in idfDic.items():
        idfDic[key] = math.log(documentCount/(1.0 + value))
    defaultIdf = math.log(documentCount/(1.0))
    return idfDic, defaultIdf

def cmp(e1, e2):
    res = np.sign(e1[1] - e2[1])
    if res != 0:
        return res
    else:
        a = e1[0] + e2[0]
        b = e2[0] + e1[0]
        if a > b:
            return 1
        elif a == b:
            return 0
        else:
            return -1

def tfidfExtract(words, pos=False, keywordNumber = 10):
    documents = loadData(pos)
    idfDic, defaultDic = trainIdf(documents)
    tfidfModel = TfIdf(idfDic, defaultDic, words, keywordNumber)
    tfidfModel.getTfIdf()

def textrankExtract(text, pos=False, keywordNumber = 10):
    textrank = analyse.textrank
    keywords = textrank(text, keywordNumber)
    print('/'.join(keywords))
    # for keyword in keywords:
    #     print(keyword + "/"),

class TfIdf(object):
    def __init__(self, idf, defaultIdf, words, number):
        self.words = words
        self.idfDic, self.defaultIdf = idf, defaultIdf
        self.tfDic = self.getTfDic()
        self.keywordNumber = number
    def getTfDic(self):
        tfDic = {}
        for word in self.words:
            tfDic[word] = tfDic.get(word, 0.0) + 1.0
        documentCount = len(self.words)
        for key, value in tfDic.items():
            tfDic[key] = float(value) / documentCount
        return tfDic

    def getTfIdf(self):
        tfIdfDic = {}
        for word in self.words:
            idf = self.idfDic.get(word, self.defaultIdf)
            tf = self.tfDic.get(word, 0)
            tfidf = tf * idf
            tfIdfDic[word] = tfidf
        for key, value in sorted(tfIdfDic.items(), key=functools.cmp_to_key(cmp), reverse=True)[: self.keywordNumber]:
            print(key + "/", end="")
        print()

################
# test
################
text = '6月19日,《2012年度“中国爱心城市”公益活动新闻发布会》在京举行。' + \
        '中华社会救助基金会理事长许嘉璐到会讲话。基金会高级顾问朱发忠,全国老龄' + \
        '办副主任朱勇,民政部社会救助司助理巡视员周萍,中华社会救助基金会副理事长耿志远,' + \
        '重庆市民政局巡视员谭明政。晋江市人大常委会主任陈健倩,以及10余个省、市、自治区民政局' + \
        '领导及四十多家媒体参加了发布会。中华社会救助基金会秘书长时正新介绍本年度“中国爱心城' + \
        '市”公益活动将以“爱心城市宣传、孤老关爱救助项目及第二届中国爱心城市大会”为主要内容,重庆市' + \
        '、呼和浩特市、长沙市、太原市、蚌埠市、南昌市、汕头市、沧州市、晋江市及遵化市将会积极参加' + \
        '这一公益活动。中国雅虎副总编张银生和凤凰网城市频道总监赵耀分别以各自媒体优势介绍了活动' + \
        '的宣传方案。会上,中华社会救助基金会与“第二届中国爱心城市大会”承办方晋江市签约,许嘉璐理' + \
        '事长接受晋江市参与“百万孤老关爱行动”向国家重点扶贫地区捐赠的价值400万元的款物。晋江市人大' + \
        '常委会主任陈健倩介绍了大会的筹备情况。'
pos = False
words = text2Words(text, pos)
words = wordFilter(words, pos)
print("TF-IDF:")
tfidfExtract(words)
print("TextRank:")
textrankExtract(text)


