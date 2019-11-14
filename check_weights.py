import numpy as np

sp = np.load("spatial/spatial_model.npz")
sq = np.load("sequential/sequential_model.npz")


fc8_sp = sp['fc8/W']
fc8_sq = sq['fc8/W']

print(np.array_equal(fc8_sp, fc8_sq))

