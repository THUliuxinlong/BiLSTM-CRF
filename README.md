# 实验1

[TOC]

## Task 1 

```python
Out[0]:
cut_all = False：['欧几里得', ' ', '公元前', '三', '世纪', '的', '古希腊', '数学家', ' ', '现在', '被', '认为', '是', '几何', '之', '父']
cut_all = True：['欧几', '欧几里得', '几里', '得', '', ' ', '', '公元', '公元前', '三世', '世纪', '的', '古希腊', '希腊', '数学', '数学家', '学家', '', ' ', '', '现在', '被', '认为', '是', '几何', '之父']
```

1. cut_all = False精确模式，将句子精确的切开，适合文本分析；cut_all = True全模式，把句中多有可以成词的词语扫描出来，速度快，但不能解决歧义。参数HMM决定是否使用新词识别功能。

```python
Out[1]:
extract_tags:
数学 0.6579100031216815
科学 0.12956709076569692
数学家 0.10929048024892699
研究 0.10522404360497788
领域 0.08980578886742256
textrank:
数学 1.0
科学 0.2661034683982758
研究 0.25994777187711987
结构 0.20070450704269863
数学家 0.1536802895355949
```

2. 限定词性为 ['ns', 'n', 'vn', 'v']地名、名词、名动词、动词

```
Out[2]:
”数学“和”研究“的相似度为： 0.78792787
与”数学“最相关的词： [('原理', 0.8625283241271973), ('论文', 0.8517716526985168), ('微积分', 0.8473283052444458)]
```

3. vector_size：词向量的维度。

   window：句子中当前单词和预测单词之间的最大距离。window越大，则和某一词较远的词也会产生上下文关系。如果是小语料则这个值可以设的更小。设得较小，模型学习到的是词汇间的组合性关系（词性相异）。设置得较大，会学习到词汇之间的聚合性关系（词性相同）。

   min_count：忽略总频率低于此值的单词（低频词、错别字、噪声）。

   epochs：在语料库上的迭代次数。

   sg：word2vec两个模型选择。sg=0是CBOW，sg=1是skip—gram。

## Task 2

### task2.py

1. 首先将数据转成bis格式，然后将文本和标签分开，利用keras的Tokenizer和pad_sequences将文本转换成index列表，并裁剪成相同长度

```python
# 首先将数据转成bis格式
process.convert_to_bis(corups_dir, output_path, combine=True, single_line=True)
# 将文本和标签分开
_, texts, labels = process.creat_dataset(output_path, './task2data/train_text', './task2data/train_label')
# 利用keras的Tokenizer和pad_sequences将文本转换成index列表，并裁剪
word_index, textdata, labeldata = process.preprocess(texts, labels)
```

```python
Out[0]:
行数: 677908
['人', '民', '网', '1', '月', '1', '日', '讯', '据', '《', '纽', '约', '时', '报', '》', '报', '道', '，', '美', '国', '华', '尔', '街', '股', '市', '在', '2', '0', '1', '3', '年', '的', '最', '后', '一', '天', '继', '续', '上', '涨', '，', '和', '全', '球', '股', '市', '一', '样', '，', '都', '以', '最', '高', '纪', '录', '或', '接', '近', '最', '高', '纪', '录', '结', '束', '本', '年', '的', '交', '易', '。']
['B', 'I', 'I', 'B', 'I', 'I', 'I', 'S', 'S', 'S', 'B', 'I', 'I', 'I', 'S', 'B', 'I', 'S', 'B', 'I', 'B', 'I', 'I', 'B', 'I', 'S', 'B', 'I', 'I', 'I', 'I', 'S', 'B', 'I', 'B', 'I', 'B', 'I', 'B', 'I', 'S', 'S', 'B', 'I', 'I', 'I', 'B', 'I', 'S', 'S', 'S', 'B', 'I', 'I', 'I', 'S', 'B', 'I', 'B', 'I', 'I', 'I', 'B', 'I', 'S', 'S', 'S', 'B', 'I', 'S']
word_index: {'，': 1, '的': 2, '。': 3, '一': 4, '、': 5, '1': 6, '人': 7,...}
Found 6862 unique tokens.
text_sequences[0]: [7, 54, 109, 6, 55, 6, 27, 906, 150, 138, 1419, 437, 26, 97, 137, 97, 129, 1, 166, 20, 358, 570, 836, 781, 42, 8, 18, 9, 6, 34, 17, 2, 121, 44, 4, 84, 720, 416, 21, 903, 1, 23, 57, 521, 781, 42, 4, 312, 1, 106, 40, 121, 80, 447, 791, 395, 242, 211, 121, 80, 447, 791, 293, 1148, 127, 17, 2, 198, 577, 3]
70
textdata[0]: [   7   54  109    6   55    6   27  906  150  138 1419  437   26   97
  137   97  129    1  166   20  358  570  836  781   42    8   18    9
    6   34   17    2  121   44    4   84  720  416   21  903    1   23
   57  521  781   42    4  312    1  106   40  121   80  447  791  395
  242  211  121   80  447  791  293 1148  127   17    2  198  577    3
    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0]
```

2. 通过预训练词向量，生成自己语料库需要的word2index矩阵

```python
pretrained_wv_path = './Word2vec/sgns.renmin.bigram-char'
embeddings_index = em.gen_embeddings_index(pretrained_wv_path)
# 从庞大的预训练的词向量中，选出训练集中出现的单词的词向量，形成小型的预训练词向量
path_embedding_matrix = './Word2vec/self_embedding'
embedding_matrix = em.gen_embedding_matrix(path_embedding_matrix, word_index=word_index, embeddings_index=embeddings_index, embed_size=300)
```

3. 建模并将自己的小型预训练词向量加载到模型的embedding层

```python
model = selfmodel.BiLSTM_CRF(vocab_size, tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
pretrained_embedding = np.loadtxt("./Word2vec/self_embedding")
pretrained_embedding = torch.tensor(pretrained_embedding)
model.word_embeds.weight.data.copy_(pretrained_embedding)
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)
```

```python
out[1]:
BiLSTM_CRF(
  (word_embeds): Embedding(6863, 300)
  (lstm): LSTM(300, 25, bidirectional=True)
  (hidden2tag): Linear(in_features=50, out_features=5, bias=True)
)
```

4. 利用visdom可视化训练过程

```python
viz = visdom.Visdom()
viz.line([0.], [0.], win='train_loss', opts=dict(title='train loss'))
```

5. 训练模型，并保存

```python
for epoch in range(3):
    print('running epoch:', epoch)
    train_step = 0
    for sentence_in_np, label_in_np in zip(textdata, labeldata):
        model.zero_grad()
        sentence_in = torch.tensor(sentence_in_np)
        label_in = torch.tensor(label_in_np)
        loss = model.neg_log_likelihood(sentence_in, label_in)
        loss.backward()
        optimizer.step()
        train_step += 1
        if train_step % 5000 == 0:
            print("第{}轮的第{}次训练的loss:{}".format((epoch + 1), train_step, loss.item()))
            viz.line([loss.item()], [train_step], win='train_loss', update='append')
            torch.save(model.state_dict(), './model/bilstm_crf'+str(epoch)+'_'+str(train_step)+'.pkl')
```

```python
Out[2]:
running epoch: 0
第1轮的第5000次训练的loss:1908.8203125
第1轮的第10000次训练的loss:136.703125
第1轮的第15000次训练的loss:774.75
第1轮的第20000次训练的loss:224.765625
...
```

### test.py

6. test.py测试训练好的模型

```python
import torch
import ner_data_preprocess as process
import bi_lstm_crf as selfmodel
import get_embedding as em
import torch.utils.data as data
from sklearn.metrics import accuracy_score

_, texts, labels = process.creat_dataset(source_dir='./task2data/test_processed',
                                         text_target_path='./task2data/test_text',
                                         label_target_path='./task2data/test_label')
_, testdata, testlabel = process.preprocess(texts, labels)

EMBEDDING_DIM = 300
HIDDEN_DIM = 50# LSTM隐层神经元数
# label_index: {'i': 1, 'b': 2, 's': 3}
tag_to_ix = {"B": 2, "I": 1, "S": 3, "<START>": 0, "<STOP>": 4}
vocab_size = 6862 + 1

model_save_path = "./model/bilstm_crf0_50000.pkl"
new_model = selfmodel.BiLSTM_CRF(vocab_size, tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)  # 调用模型Model
new_model.load_state_dict(torch.load(model_save_path))  # 加载模型参数

testensor = torch.tensor(testdata)
labeltensor = torch.tensor(testlabel)
acc = 0
test_num = 0
with torch.no_grad():
    for sentence_in_np, label_in_np in zip(testdata, testlabel):
        testensor = torch.tensor(sentence_in_np)
        labeltensor = torch.tensor(label_in_np)
        score, tag_seq = new_model(testensor)
        acc += accuracy_score(testlabel[0], tag_seq)
        test_num += 1
    print(test_num)
    print("准确率：", acc/test_num)
```

7. 在测试集上训练一轮的准确率为

```python
Out[0]:
准确率： 0.836335807050094
text ['本', '报', '北', '京', '1', '月', '2', '2', '日', '电', '（', '记', '者', '朱', '剑', '红', '）', '我', '国', '“', '十', '二', '五', '”', '规', '划', '提', '出', '的', '2', '4', '个', '主', '要', '指', '标', '，', '绝', '大', '部', '分', '的', '实', '施', '进', '度', '好', '于', '预', '期', '，', '氮', '氧', '化', '物', '排', '放', '总', '量', '减', '少', '、', '化', '石', '能', '源', '占', '一', '次', '能', '源', '消', '费', '比', '重', '、', '单', '位', 'G', 'D', 'P', '能', '源', '消', '耗', '降', '低', '、', '单', '位', 'G', 'D', 'P', '二', '氧', '化', '碳', '排', '放', '降', '低', '等', '四', '个', '指', '标', '完', '成', '的', '进', '度', '滞', '后', '于', '预', '期', '。', '这', '是', '国', '家', '发', '改', '委', '有', '关', '负', '责', '人', '今', '天', '在', '“', '宏', '观', '经', '济', '形', '势', '和', '政', '策', '”', '新', '闻', '发', '布', '会', '上', '透', '露', '的', '。', '国', '家', '发', '改', '委', '已', '组', '织', '有', '关', '部', '门', '和', '有', '关', '方', '面', '对', '“', '十', '二', '五', '”', '规', '划', '的', '实', '施', '情', '况', '进', '行', '了', '全', '面', '的', '分', '析', '和', '评', '估', '，', '评', '估', '报', '告', '在', '修', '改', '完', '成', '后', '将', '对', '外', '公', '开', '。']
pred_seq [2, 1, 2, 1, 2, 1, 3, 3, 3, 2, 1, 3, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 3, 2, 1, 1, 3, 3, 2, 1, 2, 1, 2, 1, 3, 2, 1, 2, 1, 3, 3, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 1, 2, 1, 1, 2, 1, 3, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 3, 2, 1, 3, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 3, 2, 1, 3, 2, 1, 2, 1, 3, 2, 1, 2]
true_seq [2 1 2 1 2 1 1 1 1 3 3 2 1 2 1 1 3 2 1 3 2 1 1 3 2 1 2 1 3 2 1 1 2 1 2 1 3
 2 1 1 1 3 2 1 2 1 3 3 2 1 3 2 1 1 1 2 1 2 1 2 1 3 2 1 1 1 3 2 1 2 1 1 1 2
 1 3 2 1 2 1 1 2 1 1 1 2 1 3 2 1 2 1 1 2 1 1 1 2 1 2]
```
