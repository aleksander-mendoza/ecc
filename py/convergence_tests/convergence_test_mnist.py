import single_layer as sl
import common as c

if __name__ == '__main__':
    import torchvision
    import argparse

    args = argparse.ArgumentParser()
    args.add_argument('-r', '--rows', default=5, type=int, help='Rows')
    args.add_argument('-c', '--cols', default=4, type=int, help='Columns (m=rows*columns)')
    args.add_argument('-ph', '--patch-height', default=5, type=int, help='Patch height')
    args.add_argument('-pw', '--patch-width', default=5, type=int, help='Patch width (n=width*height)')
    args.add_argument('-d', '--data-dir', default='../../data', help='Path to MNIST dataset')
    args.add_argument('-mth', '--method', default='HardWtaL1')
    args = args.parse_args()
    plotter = c.Plot(rows=args.rows, cols=args.cols)
    rand_patch = c.RandPatch(height=args.patch_height, width=args.patch_width, channels=1)
    method_class = {
        'HardWtaL2': sl.HardWtaL2,
        'HardWtaL1': sl.HardWtaL1,
        'HardWtaZeroOrder': sl.HardWtaZeroOrder,
    }
    method_class = method_class[args.method]
    method = method_class(m=plotter.m, n=rand_patch.n)

    MNIST = torchvision.datasets.MNIST(args.data_dir, train=False, download=True)
    MNIST = MNIST.data.numpy()

    trainer = c.Trainer(method, MNIST, rand_patch, postprocess=c.SampleByThreshold(0.8))

    while True:
        trainer.train(500, 4)
        means, counts = trainer.eval(len(MNIST), 4)
        print("probabilities=", counts / counts.sum())
        plotter.plot(means)
