import os

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import ecc_py
import common as c
import hard_wta_l1 as hard_wta

plotter = c.Plot(8, 8)
rand_patch = c.RandPatch(6, 6, 1)
data = c.DatasetDir("../data/imgs")
wta = hard_wta.HardWTA(m=plotter.m, n=rand_patch.n, l=1)
trainer = c.Trainer(wta, data, rand_patch, postprocess=c.SampleOfCardinality(7))

while True:
    trainer.train(100, 300)
    means, counts = trainer.eval(20, 800)
    plotter.plot(means)
