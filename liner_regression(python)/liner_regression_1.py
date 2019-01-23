#!/usr/bin/python
# -*- coding: UTF-8 -*-
# 文件名: linear_regression_1.py

import numpy as np
from sklearn.linear_model import LinearRegression

__author__ = 'Jaiken'
if __name__ == '__main__':
    # 100行1列随机数，均匀分布，范围为[0,2）;
    X = 2 * np.random.rand(100, 1)
    # 100行1列标准正态分布数值,正态分布的中心是随机的
    y = 4 + 3 * X + np.random.randn(100, 1)

    lin_reg = LinearRegression()
    lin_reg.fit(X, y)
    print(lin_reg.intercept_, lin_reg.coef_)

    X_new = np.array([[0], [2]])
    print(lin_reg.predict(X_new))
