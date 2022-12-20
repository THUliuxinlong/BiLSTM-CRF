# Task 1
import jieba
import jieba.analyse as analyse
from gensim.models import word2vec
import gensim.models
import os


with open("task1data/firstarticle.txt", "r", encoding="utf_8") as f:  # 打开文件
    data = f.read()  # 读取文件
datacut1 = jieba.cut(data, cut_all=False, HMM=True)
datacut2 = jieba.cut(data, cut_all=True, HMM=True)
datalist1 = list(datacut1)
datalist2 = list(datacut2)
print('cut_all=False', datalist1)
print('cut_all=True', datalist2)

# 基于 TF-IDF提取关键字
key_word_pos = ['ns', 'n', 'vn', 'v']
keywords = jieba.analyse.extract_tags(data, topK=5, withWeight=True, allowPOS=key_word_pos)
print('extract_tags:')
for item in keywords:
    print(item[0], item[1])

# 基于TextRank算法的关键词抽取
keywords1 = jieba.analyse.textrank(data, topK=5, withWeight=True, allowPOS=key_word_pos)
print('textrank:')
for item in keywords1:
    print(item[0], item[1])


# 对初始语料进行分词处理后，作为训练模型的语料
def mycorpuscut(filename):
    outfilename = "task1data/corpus_cut.txt"
    if not os.path.exists(outfilename):
        with open(filename, 'r', encoding='utf-8') as f:
            text = f.read()
        wordList = jieba.cut(text, cut_all=False, HMM=True)
        # 合并列表的词，成为一个句子，分好的词之间用空格相连
        wordText = " ".join(wordList)
        # 去标点符号
        wordText = wordText.replace('，', '').replace('。', '').replace('？', '') \
            .replace('！', '').replace('“', '').replace('”', '').replace('：', '') \
            .replace('…', '').replace('（', '').replace('）', '').replace('—', '') \
            .replace('《', '').replace('》', '').replace('、', '').replace('‘', '') \
            .replace('’', '')
        with open(outfilename, 'w', encoding='utf-8') as f:
            f.write(wordText)
    return outfilename


corpus_cut = mycorpuscut("task1data/wiki.zh.txt")
# 文本必须是分好词的, 词与词之间有间隔
sentences = word2vec.Text8Corpus(corpus_cut)
print("sentences格式:", sentences)

# 训练 CBOW 模型
model = gensim.models.Word2Vec(sentences, vector_size=30, window=5, min_count=2, epochs=10, sg=0)

# 保存词向量 非二进制文件
model.wv.save_word2vec_format('task1data/myWord2vec.txt', binary=False)

# 加载模型
model = gensim.models.KeyedVectors.load_word2vec_format('task1data/myWord2vec.txt', binary=False)
print("\n模型:", model)

print("”数学“和”研究“的相似度为：", model.similarity("数学", "研究"))
top_n = model.most_similar("数学", topn=3)  # 最相关的几个词
print("与”数学“最相关的词：", top_n)