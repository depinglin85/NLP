# 维特比算法
# 动态规划求最短路径算法,与穷举法相比优点在于大大降低了时间复杂度;
# 此算法在NPL（HMM，CRF）中经常被用到，因此我想要彻底理解此算法和代码实现
# 此算法有几个重要的概念
#  1. 隐含状态 states，通过隐含状态我们要找到最佳的观测序列，观测序列里面的元素从states里面寻找
#  2. 观测序列（已知)observations
#  3. 初始概率，observations[0]对应的隐含状态中每个state能够出现的状态的概率
#  4. 转换概率transition_probability，从一个隐含状态到下一个隐含状态的概率
#  5. 发射概率emission_probability,某种隐含状态产生某种观测现象的概率
# 第一层的概率：隐状态的初始概率*该状态到第一层的观察状态的发射概率
# 其他层的概率：前一层隐状态的概率*前一层隐状态到当前层隐状态的转换概率*当前层隐状态到当前层观察状态的发射概率
# 下面的例子是为了理解此算法，参考网上的一些例子并做了一些修改，供日后自己参考
# 此例子是简单的天气概率的例子
# python3 viterbi.py 5

import sys
states = ('Rainy', 'Sunny', 'Snowy', 'Thunder')
observations = ('walk', 'playSnow', 'clean', 'clean', 'shop', 'clean', 'walk', 'shop', 'clean', 'playSnow', 'scare', 'walk')
start_probability = {'Rainy': 0.4, 'Sunny': 0.3, 'Snowy': 0.2, 'Thunder': 0.1}
transition_probability = {
    'Rainy' : {'Rainy': 0.5, 'Sunny': 0.2, 'Snowy': 0.15, 'Thunder': 0.15},
    'Sunny' : {'Rainy': 0.1, 'Sunny': 0.5, 'Snowy': 0.3, 'Thunder': 0.1},
    'Snowy' : {'Rainy': 0.4, 'Sunny': 0.2, 'Snowy': 0.3, 'Thunder': 0.1},
    'Thunder' : {'Rainy': 0.4, 'Sunny': 0.2, 'Snowy': 0.1, 'Thunder': 0.3},
}
emission_probability = {
    'Rainy' : {'walk': 0.1, 'shop': 0.3, 'clean': 0.4, 'playSnow':0.1, 'scare':0.1},
    'Sunny' : {'walk': 0.4, 'shop': 0.2, 'clean': 0.1, 'playSnow':0.1, 'scare':0.1},
    'Snowy' : {'walk': 0.2, 'shop': 0.1, 'clean': 0.2, 'playSnow':0.4, 'scare':0.1},
    'Thunder' : {'walk': 0.1, 'shop': 0.1, 'clean': 0.4, 'playSnow':0.1, 'scare':0.3},
}

def viterbi(states, observations, start_probability, transition_probability, emission_probability):
    # 保存概率
    V = [{}]
    # 保存最优路径
    path = {}

    # 第一层概率计算
    for state in states:
        V[0][state] = start_probability[state] * emission_probability[state][observations[0]]
        path[state] = [state]
    # 其他层概率计算，也就是找最优路径
    for level in range(1, len(observations)):
        V.append({})
        newPath = {}
        for state in states:
            # 计算往下一层走的时候最优路径，也就是概率最大的那条路径
            (prop, st) = max([V[level - 1][s] * transition_probability[s][state] * emission_probability[state][observations[level]], s] for s in states)
            V[level][state] = prop
            newPath[state] = path[st] + [state]
        path = newPath
    # 返回全局最大概率和路径
    return max([(V[len(observations) - 1][prop], path[prop]) for prop in V[len(observations) - 1]])

if __name__ == '__main__':
    print(viterbi(states, observations[:int(sys.argv[1])], start_probability, transition_probability, emission_probability))
