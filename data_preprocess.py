import os
import re
import argparse


def print_process(process):
    num_processed = int(30 * process)
    num_unprocessed = 30 - num_processed
    print(
        f"{''.join(['['] + ['='] * num_processed + ['>'] + [' '] * num_unprocessed + [']'])}, {(process * 100):.2f} %")


# parser = argparse.ArgumentParser(description="将使用词性标注的文件转换为用BIS分块标记的文件。")
# parser.add_argument("corups_dir", type=str, help="指定存放语料库的文件夹，程序将会递归查找目录下的文件。")
# parser.add_argument("output_path", type=str, default='.', help="指定标记好的文件的输出路径。")
# parser.add_argument("-c", "--combine", help="是否组装为一个文件", default=False, type=bool)
# parser.add_argument("-s", "--single_line", help="是否为单行模式", default=False, type=bool)
# parser.add_argument("--log", help="是否打印进度条", default=False, type=bool)
# parser.add_argument("--max_len", help="处理后的最大语句长度（将原句子按标点符号断句，若断句后的长度仍比最大长度长，将忽略",
#                     default=150, type=int)
MAX_LEN_SIZE = 150


def convert_to_bis(source_dir, target_path, log=False, combine=False, single_line=True):
    print("Converting...")
    for root, dirs, files in os.walk(source_dir):
        total = len(files)  # txt文件名
        tgt_dir = target_path + root[len(source_dir):]  # 目标文件路径
        print(tgt_dir)
        for index, name in enumerate(files):
            # print('root', root, '222:', root+name)
            bises = process_file(root+'/'+name)
            # print(bises)
            if combine:
                _save_bises(bises, target_path, write_mode='a', single_line=single_line)
            else:
                os.makedirs(tgt_dir, exist_ok=True)
                _save_bises(bises, os.path.join(tgt_dir, name), single_line=single_line)
            if log:
                print_process((index + 1) / total)
    print("All converted")


def _save_bises(bises, path, write_mode='w+', single_line=True):
    with open(path, mode=write_mode, encoding='UTF-8') as f:
        if single_line:
            for bis in bises:
                sent, tags = [], []
                for char, tag in bis:
                    sent.append(char)
                    tags.append(tag)
                sent = ' '.join(sent)
                tags = ' '.join(tags)
                f.write(sent + "\t" + tags)
                f.write('\n')
        else:
            for bis in bises:
                for char, tag in bis:
                    f.write(char + "\t" + tag + "\n")
                f.write("\n")


def process_file(file):
    with open(file, 'r', encoding='UTF-8') as f:
        text = f.readlines()
        bises = _parse_text(text)
    return bises


def _parse_text(text: list):
    bises = []
    for line in text:
        # remove POS tag
        line, _ = re.subn('\\n', '', line)
        if line == '' or line == '\n':
            continue
        words = re.split('\s+', line)

        if len(words) > MAX_LEN_SIZE:
            texts = re.split('[。？！，.?!,]/w', line)
            if len(min(texts, key=len)) > MAX_LEN_SIZE:
                continue
            bises.extend(_parse_text(texts))
        else:
            bises.append(_tag(words))
    return bises


def _tag(words):
    """
    给指定的一行文本打上BIS标签
    :param line: 文本行
    :return:
    """
    bis = []
    # words = list(map(list, words))
    pre_word = None
    for word in words:
        pos_t = None
        tokens = word.split('/')
        if len(tokens) == 2:
            word, pos = tokens
        elif len(tokens) == 3:
            word, pos_t, pos = tokens
        else:
            continue

        word = list(word)
        pos = pos.upper()

        if len(word) == 0:
            continue
        if word[0] == '[':
            pre_word = word
            continue
        if pre_word is not None:
            pre_word += word
            if pos_t is None:
                continue
            elif pos_t[-1] != ']':
                continue
            else:
                word = pre_word[1:]
                pre_word = None

        # if len(word) == 1:
        #     bis.append((word[0], 'S-' + pos))
        # else:
        #     for i, char in enumerate(word):
        #         if i == 0:
        #             bis.append((char, 'B-' + pos))
        #         else:
        #             bis.append((char, 'I-' + pos)
        if len(word) == 1:
            bis.append((word[0], 'S'))
        else:
            for i, char in enumerate(word):
                if i == 0:
                    bis.append((char, 'B'))
                else:
                    bis.append((char, 'I'))
    # bis.append(('\n', 'O'))
    return bis


# 将[[char char... B I S...], ...]转成[([char ...], [BIS ...]), ...]
def creat_dataset(source_dir, text_target_path, label_target_path):
    with open(source_dir, 'r', encoding='utf-8') as f:
        data = []
        label = []
        dataset = [(data, label)]
        for line in f:
            # 暂存
            linedata = []
            linelabel = []
            dataline = (linedata, linelabel)

            linesplit = line.split()
            for token in linesplit:
                if token != 'B' and token != 'I' and token != 'S':
                    linedata.append(token)
                else:
                    linelabel.append(token)
            data.append(linedata)
            label.append(linelabel)
            dataset.append(dataline)

    # 保存
    # with open(text_target_path, 'w', encoding='UTF-8') as f:
    #     for line in dataset:
    #         f.write(str(line[0]))
    # with open(label_target_path, 'w', encoding='UTF-8') as f:
    #     for line in dataset:
    #         f.write(str(line[1]))
    print('All processed！')
    # return dataset
    return dataset, data, label


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
import torch.utils.data as Data
def preprocess(texts, labels):

    # 生成语料词索引序列,MAX_NB_WORDS = 保留的单词的最大数量
    MAX_NB_WORDS = 7000
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)

    tokenizer.fit_on_texts(texts)
    # 转成indxe序列
    text_sequences = tokenizer.texts_to_sequences(texts)

    word_index = tokenizer.word_index
    print('word_index:', word_index)
    print('Found %s unique tokens.' % len(word_index))

    MAX_SEQUENCE_LENGTH = 100
    textdata = pad_sequences(text_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post', value=0)
    print('text_sequences[0]:', text_sequences[0])
    print(len(text_sequences[0]))
    print('textdata[0]:', textdata[0])
    print(len(textdata[0]))
    print('textnum:', len(textdata))


    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(labels)
    # 转成indxe序列
    label_sequences = tokenizer.texts_to_sequences(labels)
    label_index = tokenizer.word_index
    print('label_index:', label_index)
    print('Found %s unique tokens.' % len(label_index))

    MAX_SEQUENCE_LENGTH = 100
    labeldata = pad_sequences(label_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post', value=0)
    print('label_sequences[0]:', label_sequences[0])
    print(len(label_sequences[0]))
    print('labeldata[0]:', labeldata[0])
    print(len(labeldata[0]))
    print('labelnum:', len(labeldata))
    return word_index, textdata, labeldata

