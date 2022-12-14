# 句法分析
句法分析是机器翻译的核心数据结构。句法分析是NLP的核心技术，是对语言进行深层次理解的基石。句法分析的主要任务是识别出句子所包含的句法成分以及这些成分之间的关系，一般以句法树来表示句法分析的结果。
句法分析中所用的方法可以分为基于规则的方法和基于统计的方法。目前基本是基于统计的方法占大多数。特别是PCFG模型得到极大应用。
句法分析的数据集有
- 宾夕法尼亚大学的英文宾州树库
- 宾夕法尼亚大学的中文宾州树库
- 清华树库
- 台湾中研院树库

基于统计的句法分析有以下几种常用的方法
- 具有PCFG的句法分析
- 基于最大间隔马尔可夫网络的句法分析
- 基于CRF的句法分析


## 基于PCFG的语法分析
PCFG（Probabilistic Context Free Grammar）是基于概率的短语结构分析方法，是目前研究最充分，形式最为简单的统计句法分析模型，也可以认为是规则方法与统计方法的结合
PCFG是上下文无关文法的扩展，是一种生成式的方法，其短语结构文法可以表示为一个五元组（X，V，S，R，P）
- X是一个有限词汇的集合（词典），它的元素称为词汇或终结符
- V是一个有限标注的集合，称为非终结符集合。
- S称为文法的开始符合，包含于V
- R是有序偶对的集合，也就是产生的规则集
- P代表每个产生规则的统计概率

PCFG可以解决以下问题
- 基于PCFG可以计算分析树的概率值
- 若一个句子有多个分析树，可以依据概率值对所有的分析树进行排序
- PCFG可以用来进行句法排歧，面对多个分析结果选择概率值最大的
  
本例子是具有斯坦福大学的句法分析器提供的中文文法分析库
由于Stanford Parser是Java开发的，所以需要JDK>=1.8以上
Python的nltk库里面封装了Stanford Parser。
wget https://nlp.stanford.edu/software/stanford-parser-4.2.0.zip
unzip stanford-parser-4.2.0.zip
mv stanford-parser-full-2020-11-17 stanford-parser

这个例子同样适合日语的句法分析
