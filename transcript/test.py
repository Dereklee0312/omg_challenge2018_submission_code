import numpy as np

data = np.load("./predictions/Subject_2_Story_1_predictions.npy")
print(data.shape)
# index = 0
# for i in data:
#     if index < 50:
#         print(i)
#     index += 1
data = np.load("./vectors/val2/text_aligned/Subject_2_Story_1_aligned.npy")
print(data.shape)
# print(data[0])