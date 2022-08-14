import os
import sys
import pickle

# python3 hmm.py 这是一个非常棒的分词算法
# ['这是', '一个', '非常', '棒', '的', '分词', '算法']
# 注意第一次跑的是要训练模型，所以会比较花时间，如果要重新训练的话请把【./data/hmm_model.pkl】文件删除

class HMM(object):
    def __init__(self):
        self.modelFile = './data/hmm_model.pkl'
        self.states = ['B', 'M', 'E', 'S']
        self.loadParam = False

    def tryLoadModel(self, trained):
        if trained:
            with open(self.modelFile, 'rb') as file:
                self.transitionProb = pickle.load(file)
                self.emissionProb = pickle.load(file)
                self.startProb = pickle.load(file)
                self.loadParam = True
        else:
            self.transitionProb = {}
            self.emissionProb = {}
            self.startProb = {}
            self.loadParam = False
    def train(self, path):
        # 训练函数，主要是通过已知的语料库计算出初始概率，转移概率，发射概率
        self.tryLoadModel(False)
        # 统计隐含状态出现次数，求p(o)
        countDic = {}
        def initParameters():
            for state in self.states:
                self.transitionProb[state] = {st: 0.0 for st in self.states}
                self.emissionProb[state] = {}
                self.startProb[state] = 0.0
                countDic[state] = 0

        def makeLabel(text):
            result = []
            if len(text) == 1:
                result.append('S')
            else:
                result += ['B'] + ['M'] * (len(text) - 2) + ['E']
            return result

        initParameters()
        lineNumber = -1
        words = set()
        with open(path, encoding = 'utf8') as file:
            for line in file:
                lineNumber += 1
                line = line.strip()
                if not line:
                    continue
                signleWords = [word for word in line if word != " "]
                words |= set(signleWords)
                lineWords = line.split()
                lineStates = []
                for word in lineWords:
                    lineStates.extend(makeLabel(word))
                assert len(signleWords) == len(lineStates)
                # print(lineStates)
                for key, state in enumerate(lineStates):
                    countDic[state] += 1
                    if key == 0:
                        # 每一行第一个字的隐含状态，用于计算初始概率
                        self.startProb[state] += 1
                    else:
                        # 统计当前状态是由那个状态转换而来，用于计算转移概率
                        self.transitionProb[lineStates[key - 1]][state] += 1
                        # 统计每个隐含状态相应的每个字出现的次数，用于计算发射概率
                        self.emissionProb[lineStates[key]][signleWords[key]] = self.emissionProb[lineStates[key]].get(signleWords[key], 0) + 1.0
            # 把次数转换为概率
            self.startProb = {key: value * 1.0 / lineNumber for key, value in self.startProb.items()}
            self.transitionProb = {key1: {key2: value2 / countDic[key1] for key2, value2 in value1.items()} for key1, value1 in self.transitionProb.items()}
            self.emissionProb = {key1: {key2: (value2 + 1) / countDic[key1] for key2, value2 in value1.items()} for key1, value1 in self.emissionProb.items()}
            with open(self.modelFile, "wb") as file:
                pickle.dump(self.transitionProb, file)
                pickle.dump(self.emissionProb, file)
                pickle.dump(self.startProb, file)

    def viterbi(self, text, states, startProb, transitionProb, emissionProb):
        V = [{}]
        path = {}
        for state in states:
            V[0][state] = startProb[state] * emissionProb[state].get(text[0], 0.0)
            path[state] = [state]
        for level in range(1, len(text)):
            V.append({})
            newPath = {}
            # 检查发射概率矩阵中是否存在该字
            neverSeen = text[level] not in emissionProb['S'].keys() and  \
                text[level] not in emissionProb['M'].keys() and  \
                text[level] not in emissionProb['E'].keys() and  \
                text[level] not in emissionProb['B'].keys()
            for state in states:
                emitProb = emissionProb[state].get(text[level], 0) if not neverSeen else 1.0
                (prob, st) = max([(V[level - 1][s] * transitionProb[s].get(state, 0) * emitProb, s) for s in states if V[level - 1][s] > 0])
                V[level][state] = prob
                newPath[state] = path[st] + [state]
            path = newPath
        if emissionProb['M'].get(text[-1], 0) > emissionProb['S'].get(text[-1], 0):
            (prob, state) = max([(V[len(text) - 1][state], state) for state in ('E', 'M')])
        else:
            (prob, state) = max([(V[len(text) - 1][state], state) for state in states])
        return (prob, path[state])

    def cut(self, text):
        if not self.loadParam:
            self.tryLoadModel(os.path.exists(self.modelFile))
            prob, path = self.viterbi(text, self.states, self.startProb, self.transitionProb, self.emissionProb)
            start, nextStart = 0, 0
            for i, char in enumerate(text):
                pos = path[i]
                if pos == 'B':
                    start = i
                elif pos == 'E':
                    yield text[start: i + 1]
                    nextStart = i + 1
                elif pos == 'S':
                    yield char
                    nextStart = i + 1
            if nextStart < len(text):
                yield text[nextStart:]

hmm = HMM()
if not os.path.exists('./data/hmm_model.pkl'):
    hmm.train('./data/trainCorpus.txt_utf8.txt')
text = sys.argv[1]
result = hmm.cut(text)
print(text)
print(str(list(result)))
