# Task 2
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import numpy as np
import visdom
import time
import data_preprocess as process
import bi_lstm_crf as selfmodel
import get_embedding as em
import torch.utils.data as data
from sklearn.metrics import accuracy_score

corups_dir = './task2data/train'
output_path = './task2data/train_processed'
# process.convert_to_bis(corups_dir, output_path, combine=True, single_line=True)

_, texts, labels = process.creat_dataset(output_path, './task2data/train_text', './task2data/train_label')
print('文本数:', len(texts))
print(texts[0])
print(labels[0])
word_index, textdata, labeldata = process.preprocess(texts, labels)
print(type(textdata))
print(len(textdata))
print(textdata.shape)

# _, texts, labels = process.creat_dataset(source_dir='./task2data/test_processed',
#                                          text_target_path='./task2data/test_text',
#                                          label_target_path='./task2data/test_label')
# _, testdata, testlabel = process.preprocess(texts, labels)


# pretrained_wv_path = './Word2vec/sgns.renmin.bigram-char'
# embeddings_index = em.gen_embeddings_index(pretrained_wv_path)
# # 从庞大的预训练的词向量中，选出训练集中出现的单词的词向量，形成小型的预训练词向量
# path_embedding_matrix = './Word2vec/self_embedding'
# embedding_matrix = em.gen_embedding_matrix(path_embedding_matrix, word_index=word_index, embeddings_index=embeddings_index, embed_size=300)


# 首先，设定超参数：在"B", “I”, “O"三个标签的基础上添加了句首标签”<START>“和句尾标签”<STOP>"；每个词嵌入的编码维度设定为5，LSTM的隐藏层维度设定为4。
START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 300
HIDDEN_DIM = 50# LSTM隐层神经元数

# label_index: {'i': 1, 'b': 2, 's': 3}
tag_to_ix = {"B": 2, "I": 1, "S": 3, "<START>": 0, "<STOP>": 4}

# 并使用随机梯度下降(SGD)对所有参数进行优化，初始学习率设定为0.01，weight_decay表示正则项系数，防止模型过拟合。
# model = selfmodel.BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
vocab_size = len(word_index) + 1
batch_size = 64

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print('0', device)
model = selfmodel.BiLSTM_CRF(vocab_size, tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM, device=device)
model.to(device)
# # 继续训练
# model_save_path = "./model/bilstm_crf0_200000.pkl"
# model.load_state_dict(torch.load(model_save_path))  # 加载模型参数

pretrained_embedding = np.loadtxt("./Word2vec/self_embedding")
pretrained_embedding = torch.tensor(pretrained_embedding)
print(pretrained_embedding.shape)
model.word_embeds.weight.data.copy_(pretrained_embedding)
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)
#打印模型
print(model)
# import torchinfo
# torchinfo.summary(model, input_size=(EMBEDDING_DIM), batch_dim=0, col_names=('input_size', 'output_size', 'num_params'))

viz = visdom.Visdom()
viz.line([0.], [0.], win='train_loss', opts=dict(title='train loss'))
# viz.line([0.], [0.], win='test_acc', opts=dict(title='test acc'))

# # 先转换成torch能识别的dataset
# tensor_data = torch.tensor(textdata)
# tensor_label = torch.tensor(labeldata)
# torch_dataset = data.TensorDataset(tensor_data, tensor_label)
#
# # 把dataset放入DataLoader
# train_loader = data.DataLoader(
#     dataset=torch_dataset,
#     batch_size=batch_size,
#     shuffle=True,
# )

# Make sure prepare_sequence from earlier in the LSTM section is loaded
for epoch in range(2):  # again, normally you would NOT do 300 epochs, it is toy data
    print('running epoch:', epoch)
    train_step = 0
    for sentence_in_np, label_in_np in zip(textdata, labeldata):
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()
        # Step 3. Run our forward pass.
        sentence_in = torch.tensor(sentence_in_np)
        label_in = torch.tensor(label_in_np)
        if torch.cuda.is_available():
            sentence_in = sentence_in.cuda(device)
            label_in = label_in.cuda(device)
        loss = model.neg_log_likelihood(sentence_in, label_in)
        # Step 4. Compute the loss, gradients, and update the parameters by
        # calling optimizer.step()
        loss.backward()
        optimizer.step()
        train_step += 1
        if train_step % 50 == 0:
            print(".")
        if train_step % 10000 == 0:
            print("第{}轮的第{}次训练的loss:{}".format((epoch + 1), train_step, loss.item()))
            viz.line([loss.item()], [train_step], win='train_loss', update='append')
            # # 计算准确率
            # testensor = torch.tensor(testdata)
            # pred = model(testensor)
            # acc = accuracy_score(pred, testlabel)
            # print("第{}轮的第{}次训练的loss:{}".format((epoch + 1), train_step, acc.item()))
            # viz.line([acc.item()], [train_step], win='test_acc', update='append')
            torch.save(model.state_dict(), './model/bilstm_crf'+str(epoch)+'_'+str(train_step)+'.pkl')


    # for step, (sentence_in, label_in) in enumerate(train_loader):  # 每一步loader释放一小批数据用来学习
    #     # Step 1. Remember that Pytorch accumulates gradients.
    #     # We need to clear them out before each instance
    #     model.zero_grad()
    #     # Step 3. Run our forward pass.
    #     loss = model.neg_log_likelihood(sentence_in, label_in)
    #     # Step 4. Compute the loss, gradients, and update the parameters by
    #     # calling optimizer.step()
    #     loss.backward()
    #     optimizer.step()
    #     train_step += 1
    #     if train_step % 100 == 0:
    #         print("第{}轮的第{}次训练的loss:{}".format((epoch + 1), train_step, loss.item()))
    #         viz.line([loss.item()], [train_step], win='train_loss', update='append')
    #         torch.save(model.state_dict(), model_save_path)


