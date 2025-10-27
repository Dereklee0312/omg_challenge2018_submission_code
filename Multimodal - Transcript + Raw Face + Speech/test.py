import numpy as np
folders = ["./rawface_predictions", "./transcript_predictions"]

subjects = list(range(1,11))
stories = [4, 5, 8]

for i in subjects:
    for j in stories:
        face = np.load(f"{folders[0]}/Subject_{i}_Story_{j}_predictions.npy")
        trans = np.load(f"{folders[1]}/Subject_{i}_Story_{j}_predictions.npy")
        print(f"FACE SHAPE: {face.shape}")
        print(f"TRANS SHAPE: {trans.shape}")