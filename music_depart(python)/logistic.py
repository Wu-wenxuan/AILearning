# coding:utf-8

import numpy as np
from sklearn import linear_model, datasets
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy import fft
from scipy.io import wavfile
import random
from sklearn.linear_model import LogisticRegression

'''
# 准备音乐数据,音乐的格式尽量为wav
# 构建傅里叶变换函数
def create_fft(g, n):
    #  路径，从磁盘加载
    rad = "genres/" + g + "/converted/" + g + "." + str(n).zfill(5) + ".au.wav"
    # 采样率，和采样元组
    sample_rate, X = wavfile.read(rad)
    # 傅里叶前1000 个信道的波形
    fft_features = abs(fft(X)[:1000])
    # 存储到的路径
    sad = "trainset/" + g + "." + str(n).zfill(5) + ".fft"
    np.save(sad, fft_features)


# -------create fft--------------
def get_fft():
    g_list = ["classical", "jazz", "country", "pop", "rock", "metal"]
    for g in g_list:
        for n in range(100):
            create_fft(g, n)

'''
# ============================================================================
'''
# 首先我们要将原始数据分为训练集和测试集，这里是随机抽样80%做测试集，剩下20%做训练集
def depart_data():
    randomIndex = random.sample(range(len(Y)), int(len(Y) * 8 / 10))
    trainX = [];
    trainY = [];
    testX = [];
    testY = []
    for i in range(len(Y)):
        if i in randomIndex:
            trainX.append(X[i])
            trainY.append(Y[i])
        else:
            testX.append(X[i])
            testY.append(Y[i])
'''

if __name__ == '__main__':
    print('Getting fft.....')
    # 加载训练集数据,分割训练集以及测试集,进行分类器的训练
    # 构造训练集！
    # -------read fft--------------
    genre_list = ["classical", "jazz", "country", "pop", "rock", "metal"]
    X = []
    Y = []
    for g in genre_list:
        for n in range(100):
            rad = "trainset/" + g + "." + str(n).zfill(5) + ".fft" + ".npy"
            fft_features = np.load(rad)
            X.append(fft_features)
            Y.append(genre_list.index(g))
    # 将数据和结果的值X，Y,存为数组，构建映射
    X = np.array(X)
    Y = np.array(Y)

    # 接下来，我们使用sklearn，来构造和训练我们的两种分类器
    # ------train logistic classifier--------------
    print('Traing model')

    model = LogisticRegression()
    model.fit(X, Y)

    print('Starting read wavfile...')
    # prepare test data-------------------
    sample_rate, test = wavfile.read("trainset/sample/heibao-wudizirong-remix.wav")
    testdata_fft_features = abs(fft(test))[:1000]
    # print(sample_rate, testdata_fft_features, len(testdata_fft_features))
    # predict 返回多个判断结果,[0]是概率最大的类别的编号
    type_index = model.predict([testdata_fft_features])[0]
    print(type_index)
    print('The result of the predict is:' + genre_list[type_index])
