import numpy as np
# print("Performing output Verification")
#
# sp_output = np.load("spatial/spatial_output_forward.npz", allow_pickle=True)
# sq_output = np.load("sequential/sequential_output_forward.npz", allow_pickle=True)
#
# sp_output_array = sp_output["h"]
# sq_output_array = sp_output["h"]
#
# print("Spatial Output Shape: ", sp_output_array.shape)
# print("Sequential Output Shape: ", sq_output_array.shape)
#
# if np.array_equal(sp_output_array, sq_output_array):
#     print("Ouput verification passed")
#
# else:
#     print("Ouput verification failed")
#

# print("Starting Weights verification")
#
# sp = np.load("spatial/spatial_model.npz", allow_pickle=True)
# sq = np.load("sequential/sequential_model.npz", allow_pickle=True)
#
# sp_array = sp['conv2/W']
# sq_array = sq['conv2/W']
#
# if np.array_equal(sp_array, sq_array):
#     print("Weights verification passed")
#
# else:
#     print("Weights verification Failed")
#
#
# print("Sample weights shown below")
#
# print(sq_array[0, 0, :, :])
#
# print("*******************--------------****************")
#
# print(sp_array[0, 0, :, :])


## Rank by rank weight check

print("-------------------Weights for sequential-------------------- ")
sq = np.load("sequential/sequential_model.npz", allow_pickle=True)
sq_array = sq['conv2/W']
print(sq_array[0, 0, :, :])
print("--------------------Weights for spatial---------------------- ")

print("Sample weights for rank 0")
sp = np.load("spatial/spatial_model_rank_0.npz", allow_pickle=True)
sp_array = sp['conv2/W']
print(sp_array[0, 0, :, :])

print("Sample weights for rank 1")
sp = np.load("spatial/spatial_model_rank_1.npz", allow_pickle=True)
sp_array = sp['conv2/W']
print(sp_array[0, 0, :, :])

print("Sample weights for rank 2")
sp = np.load("spatial/spatial_model_rank_2.npz", allow_pickle=True)
sp_array = sp['conv2/W']
print(sp_array[0, 0, :, :])

print("Sample weights for rank 3")
sp = np.load("spatial/spatial_model_rank_3.npz", allow_pickle=True)
sp_array = sp['conv2/W']
print(sp_array[0, 0, :, :])

