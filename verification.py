import numpy as np

sp = np.load("spatial/spatial_model.npz", allow_pickle=True)
sq = np.load("sequential/sequential_model.npz", allow_pickle=True)


sp_array = sp['conv1/W']
sq_array = sq['conv1/W']

print(np.array_equal(sp_array, sq_array))
print(sq_array.shape)

print(sq_array[0, 0, :, :])

print("*******************--------------****************")

print(sp_array[0, 0, : , :])


