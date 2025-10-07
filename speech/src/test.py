import numpy as np

data = np.load("../dataset/matrices/validation_2A_S_target.npy")
print("VALIDATION TARGET")
print(data.shape)

data = np.load("../dataset/matrices/validation_2A_S_predictors.npy")
print("VALIDATION PREDICTORS")
print(data.shape)

data = np.load("../dataset/matrices/training_2A_S_predictors.npy")
print("TRAINING PREDICTORS")
print(data.shape)

data = np.load("../dataset/matrices/training_2A_S_target.npy")
print("TRAINING TARGET")
print(data.shape)