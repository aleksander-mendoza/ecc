
import numpy as np


class HardWtaL2:

    def __init__(self, n, m, l=2, w_step=0.0001, r_step=1. / 1024 * 2):
        self.r_step = r_step
        self.W = np.random.rand(n, m)
        self.r = np.zeros(m)
        self.w_step = w_step
        self.n = n
        self.m = m
        if l == 1:
            self.norm = np.sum
        elif l == 2:
            self.norm = np.linalg.norm
        else:
            raise Exception("Unimplemented norm l"+str(l))
        self.W /= self.norm(self.W, axis=0)

    def __call__(self, x, learn=False):
        x = x.reshape(-1)
        k = np.argmax(x @ self.W + self.r)
        if learn:
            self.r[k] -= self.r_step
            self.W[x, k] += self.w_step / x.sum()
            self.W[:, k] /= self.norm(self.W[:, k])
        return k


class HardWtaZeroOrder(HardWtaL2):

    def __init__(self, n, m, w_step=0.0001, r_step=1. / 1024 * 2):
        super().__init__(n, m, l=1, w_step=w_step, r_step=r_step)


class HardWtaL1:

    def __init__(self, n, m, a=None, w_step=0.0001, r_step=1. / 1024 * 2):
        self.r_step = r_step
        self.W = np.zeros((n, m))
        self.Q = np.random.rand(n, m)
        self.r = np.zeros(m)
        self.w_step = w_step
        self.n = n
        self.m = m
        self.a = n // 5 if a is None else a
        top_weights = self.Q.argpartition(-self.a, axis=0)[-self.a:]
        self.W[top_weights] = 1

    def __call__(self, x, learn=False):
        x = x.reshape(-1)
        k = np.argmax(x @ self.W + self.r)
        if learn:
            self.r[k] -= self.r_step
            self.W[x, k] += self.w_step / x.sum()
            self.W[:, k] /= self.norm(self.W[:, k])
        return k