import logging
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors


def gen_embeddings_index(pretrained_wv_path):
    """
    create a weight matrix for words in training docs
    将预训练词向量文本变成字典形式：{word:vector}
    :param pretrained_wv_path: 'Tencent_AILab_ChineseEmbedding.txt'  # 预训练的词向量文件
    """
    # 使用gensim导入预训练词向量文件，不用管第一行的数值处理
    wv_from_text = KeyedVectors.load_word2vec_format(pretrained_wv_path, binary=False)
    embeddings_index = {}
    for word in wv_from_text.vocab:
        embeddings_index[word] = wv_from_text.word_vec(word)
    logging.info('Loaded {} word vectors.'.format(len(embeddings_index)))

    return embeddings_index


# def gen_embedding_matrix(self, word_index, embeddings_index, embed_size):
def gen_embedding_matrix(path_embedding_matrix, word_index, embeddings_index, embed_size):
    """
    从庞大的预训练的词向量中，选出训练集中出现的单词的词向量，形成小型的预训练词向量
    :param path_embedding_matrix:
    :param word_index: local dictionary
    :param embeddings_index: pretrained word vectors
    :param embed_size: 预训练的词向量的维度
    :return:
    """
    embedding_matrix = np.zeros((len(word_index) + 1, embed_size))
    # i是从1开始
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words found in embedding index will be pretrained vectors.
            embedding_matrix[i] = embedding_vector   # i+1 是为了处理OOV，使得预测时未见过的词为0
        else:
            # words not found in embedding index will be random vectors with certain mean&std.
            embedding_matrix[i] = np.random.normal(0.053, 0.3146, (1, embed_size))[0] # 0.053, 0.3146 根据统计

    print('embedding_matrix.shape:', embedding_matrix.shape)
    # save embedding matrix
    embed_df = pd.DataFrame(embedding_matrix)
    # embed_df.to_csv(self.path_embedding_matrix, header=None, sep=' ')
    embed_df.to_csv(path_embedding_matrix, header=None, index=False, sep=' ')

    return embedding_matrix
