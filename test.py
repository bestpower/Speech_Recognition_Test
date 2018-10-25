# -*- coding: utf-8 -*-

import os
import utils
from config import Config
from model import BiRNN

# 设置log打印等级
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 配置测试数据路径
conf = Config()
# 获取音频文件及对应标签信息
wav_files, text_labels = utils.get_test_wavs_lables()
# 建立待训练文本标签信息属性字典
words_size, words, word_num_map = utils.create_dict(text_labels)
# 输入到双向神经网络模型并开始测试
bi_rnn = BiRNN(wav_files, text_labels, words_size, words, word_num_map)
bi_rnn.build_test()
# 对指定音频文件数据进行测试
#wav_files = ['D:\\pycharm_workspace\\data\\data_thchs30\\train\A2_11.wav']
#txt_labels = ['北京 丰台区 农民 自己 花钱 筹办 万 佛 延寿 寺 迎春 庙会 吸引 了 区内 六十 支 秧歌队 参赛']
#bi_rnn = BiRNN(wav_files, text_labels, words_size, words, word_num_map)
#bi_rnn.build_target_wav_file_test(wav_files, txt_labels)
