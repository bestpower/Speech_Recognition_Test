# -*- coding: utf-8 -*-

import os
import utils
from config import Config
from model import BiRNN

# 设置log打印等级
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 配置训练数据路径
conf = Config()
# 获取音频文件及对应标签信息
wav_files, text_labels = utils.get_train_wavs_lables()
# 建立待训练文本标签信息属性字典
words_size, words, word_num_map = utils.create_dict(text_labels)
# 输入到双向神经网络模型并开始训练
bi_rnn = BiRNN(wav_files, text_labels, words_size, words, word_num_map)
bi_rnn.build_train()
