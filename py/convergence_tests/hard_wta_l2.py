
import numpy as np


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