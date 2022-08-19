# pip install pltk
import jieba
from nltk.parse import stanford
from nltk.parse.corenlp import CoreNLPParser
import os

context = "我今年的夏季休假准备写AI算法代码"
seg=jieba.cut(context, cut_all=False, HMM=True)
segStr = ' '.join(seg)
print(segStr)
parserFile = './stanford-parser/stanford-parser.jar'
modelFile = './stanford-parser/stanford-parser-4.2.0-models.jar'

if not os.environ.get('JAVA_HOME'):
    # mac pc
    JAVA_HOME = '/Library/Java/JavaVirtualMachines/jdk-18.0.2.1.jdk'
    os.environ['JAVA_HOME'] = JAVA_HOME

pcfgFile = 'edu/stanford/nlp/models/lexparser/chinesePCFG.ser.gz'

parser = stanford.StanfordParser(
    path_to_jar=parserFile,
    path_to_models_jar=modelFile,
    model_path=pcfgFile
)

sentence = parser.raw_parse(segStr)
for line in sentence:
    print(line)
    line.draw()

# parser = CoreNLPParser(url='http://localhost:9000')
# parser.parse_sents(segStr)
# sentence = parser.parse(segStr.split())
# sentence = parser.raw_parse(segStr)
# for line in sentence:
#     print(line)
#     line.draw()