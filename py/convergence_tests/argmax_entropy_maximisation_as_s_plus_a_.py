import ecc_py
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np

mean_s = np.array([0.1, 0.8, 0.2, 0.3])
y_len = len(mean_s)
a = np.zeros(y_len)
epsilon = 0.0001
labels = np.array([i for i in range(y_len)])


# for _ in range(1000000):
#     s = np.random.randn(y_len) + mean_s
#     k = np.argmax(s + a)
#     a[k] -= epsilon


pk = np.zeros(y_len)
for _ in range(10000):
    s = np.random.rand(y_len) + mean_s
    k = np.argmax(s + a)
    pk[k] += 1

pk /= sum(pk)
print("a=",a-min(a))
print("p(k)=", pk)
print("q(k)=", mean_s / sum(mean_s))
#
# bars = plt.barh(labels, a - np.min(a))
# plt.bar_label(bars)
# plt.pause(0.01)
# plt.clf()
# print(mean_s)
