# Author: Robert Guthrie

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
# 人工设定随机种子以保证相同的初始化参数，实现模型的可复现性。下面开始模型构建及其训练：
torch.manual_seed(1)

#     给定输入二维序列，在1维度上取最大值，返回对应ID。
def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


# 利用to_ix这个word2id字典，将序列seq中的词转化为数字表示，包装为torch.long后返回。
def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


# 函数目的相当于log∑e^xi
# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim, device):
        super(BiLSTM_CRF, self).__init__()
        # BiLSTM_CRF类的构造函数参数包括词索引表长、标签索引表、词嵌入维度和隐藏层维度，继承torch.nn.Module。
        # 私有变量包括输入的词嵌入维度、隐藏层维度、词索引表大小、标签索引表、标签索引表大小（即标签个数）、词嵌入（相当于一个[词索引表长，词嵌入维度]的矩阵，
        # 这里是调用nn的Embedding模块初始化的）、LSTM网络层（直接调用的nn中的LSTM模块，设定为单层双向，隐藏层维度设定为指定维度的一半，以便于后期双向拼接）、
        # 处理LSTM输出的全连接层（维度变更）、CRF的转移矩阵（T[i,j]表示从j标签转移到i标签，不可能转移到句首标签，也不可能从句尾标签开始转移，因此都设定为极小值）。
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)
        self.device = device

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        # input_size：表示的是输入的矩阵特征数，或者说是输入的维度；
        # hidden_size：隐藏层的大小（即隐藏层节点数量），输出向量的维度等于隐藏节点数；
        # num_layers：lstm 隐层的层数，默认为1；
        # batch_first：True 或者 False，如果是 True，则 input 为(batch, seq, input_size)，默认值为：False（seq_len, batch, input_size）
        # dropout：默认值0，除最后一层，每一层的输出都进行dropout；
        # self.lstm = nn.LSTM(embedding_dim, hidden_size=hidden_dim // 2, num_layers=4, bidirectional=True, dropout=dropout)
        self.lstm = nn.LSTM(embedding_dim, hidden_size=hidden_dim // 2, num_layers=1, bidirectional=True)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()

    # 这里是使用随机正态分布初始化LSTM的h0和c0,否则，模型自动初始化为零值（维度为[num_layers*num_directions, batch_size, hidden_dim]）。
    # 类的构造及初始化结束，接下来回到主函数。
    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2).to(self.device),
                torch.randn(2, 1, self.hidden_dim // 2).to(self.device))
                # torch.randn(2, 1, self.hidden_dim // 2),
                # torch.randn(2, 1, self.hidden_dim // 2),
                # torch.randn(2, 1, self.hidden_dim // 2),
                # torch.randn(2, 1, self.hidden_dim // 2),
                # torch.randn(2, 1, self.hidden_dim // 2),
                # torch.randn(2, 1, self.hidden_dim // 2)
                # )

    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((1, self.tagset_size), -10000.)
        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas

        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                forward_var = forward_var.to(self.device)
                trans_score = trans_score.to(self.device)
                emit_score = emit_score.to(self.device)
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()

        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1)
        score = score.to(self.device)
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long).to(self.device), tags])
        for i, feat in enumerate(feats):
            score = score + \
                self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                forward_var = forward_var.to(self.device)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence).to(self.device)

        forward_score = self._forward_alg(feats).to(self.device)
        gold_score = self._score_sentence(feats, tags).to(self.device)
        return forward_score - gold_score

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq


import torch
import numpy as np


def train(model, train_iter, valid_iter, optimizer, epochs, scheduler):
    best_acc = 0

    for epoch in range(epochs):
        loss, best_acc = train_single_epoch(model, train_iter, valid_iter, optimizer, best_acc, epoch)
        scheduler.step()


def train_single_epoch(model, train_iter, valid_iter, optimizer, best_acc, epoch):
    epoch_loss = 0
    epoch_count = 0

    for index, ((inputs, label), _) in enumerate(train_iter):
        model.train()
        inputs = inputs.permute(1, 0)
        label = label.permute(1, 0)
        loss = model.compute_loss(inputs, label)
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_count += inputs.shape[0]
        if index % 50 == 0:
            acc, matrix = eval(model, valid_iter, return_matrix=False)
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), model.name + '.h5')
                print('epoch:{},index:{},acc:{:.4f},train loss:{:.4f}'.format(epoch, index, acc, loss.item()))

    return epoch_loss / epoch_count, best_acc


def eval(model, valid_iter, return_matrix=False):
    model.eval()

    total_count = 0
    acc_count = 0
    class_count = model.fc.out_features
    matrix = np.zeros((class_count, 3))

    for (inputs, label), _ in valid_iter:
        inputs = inputs.permute(1, 0)
        label = label.permute(1, 0)
        out = model.decode(inputs)
        total_count += label.shape[0] * label.shape[1]
        out = torch.tensor(out).to(label.device)
        acc_count += torch.sum(out == label)
        if return_matrix:
            # 按类别统计准确情况
            for class_index in range(class_count):
                pred = out == class_index
                gold = label == class_index
                tp = pred[pred == gold]
                matrix[class_index, 0] += torch.sum(tp)
                matrix[class_index, 1] += torch.sum(pred)
                matrix[class_index, 2] += torch.sum(gold)
        else:
            matrix = None

    return acc_count / total_count, matrix

START_TAG = "<START>"
STOP_TAG = "<STOP>"