
import numpy as np


class HardWTA:

    def __init__(self, n, m, l=2, w_step=0.0001, r_step=1. / 1024 * 2):
        self.r_step = r_step
        self.W = np.random.rand(n, m)
        self.r = np.zeros(m)
        self.w_step = w_step
        self.n = n
        self.m = m
        self.norm = np.sum if l == 1 else np.linalg.norm
        self.W /= self.norm(self.W, axis=0)

    def __call__(self, x, learn=False):
        x = x.reshape(-1)
        k = np.argmax(x @ self.W + self.r)
        if learn:
            self.r[k] -= self.r_step
            self.W[x, k] += self.w_step / x.sum()
            self.W[:, k] /= self.norm(self.W[:, k])
        return k


if __name__ == '__main__':

    import torchvision
    import common as c

    MNIST = torchvision.datasets.MNIST('../../data/', train=False, download=True)
    MNIST = MNIST.data.numpy()

    plotter = c.Plot(5, 4)
    rand_patch = c.RandPatch(5, 5, 1)
    wta = HardWTA(m=plotter.m, n=rand_patch.n)
    trainer = c.Trainer(wta, MNIST, rand_patch, postprocess=c.SampleByThreshold(0.8))

    while True:
        trainer.train(500, 4)
        means, counts = trainer.eval(len(MNIST), 4)
        print("probabilities=", counts / counts.sum())
        plotter.plot(means)