import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn import svm

# 创建一个示例数据集
np.random.seed(0)
X = np.sort(5 * np.random.rand(40, 1), axis=0)
y = np.sin(X).ravel()

# 添加一些噪音
y[::5] += 3 * (0.5 - np.random.rand(8))

# 使用支持向量机回归进行训练
svr_rbf = svm.SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
svr_lin = svm.SVR(kernel='linear', C=100, epsilon=0.1)
svr_poly = svm.SVR(kernel='poly', C=100, degree=3, epsilon=0.1)

y_rbf = svr_rbf.fit(X, y).predict(X)
y_lin = svr_lin.fit(X, y).predict(X)
y_poly = svr_poly.fit(X, y).predict(X)

# 绘制结果
lw = 2
plt.scatter(X, y, color='darkorange', label='data')
plt.plot(X, y_rbf, color='navy', lw=lw, label='RBF model')
plt.plot(X, y_lin, color='c', lw=lw, label='Linear model')
plt.plot(X, y_poly, color='cornflowerblue', lw=lw, label='Polynomial model')
plt.xlabel('data')
plt.ylabel('target')
plt.title('Support Vector Regression')
plt.legend()
plt.show()
