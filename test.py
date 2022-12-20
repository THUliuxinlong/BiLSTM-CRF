import torch
# import torch.autograd as autograd
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np
#
# from torch import no_grad

import data_preprocess as process
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
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print('0', device)
new_model = selfmodel.BiLSTM_CRF(vocab_size, tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM, device=device)
new_model.to(device)
new_model.load_state_dict(torch.load(model_save_path))  # 加载模型参数

testensor = torch.tensor(testdata)
labeltensor = torch.tensor(testlabel)
acc = 0
test_num = 0
# with torch.no_grad():
#     for sentence_in_np, label_in_np in zip(testdata, testlabel):
#         testensor = torch.tensor(sentence_in_np)
#         labeltensor = torch.tensor(label_in_np)
#         testensor = testensor.cuda(device)
#         labeltensor = labeltensor.cuda(device)
#
#         score, tag_seq = new_model(testensor)
#
#         acc += accuracy_score(testlabel[test_num], tag_seq)
#         test_num += 1
#     print(test_num)
#     print("准确率：", acc/test_num)


print('text', texts[0])
testensor = torch.tensor(testdata[0]).cuda(device)
_, pred_seq = new_model(testensor)
print('pred_seq', pred_seq)
print('true_seq', testlabel[0])


