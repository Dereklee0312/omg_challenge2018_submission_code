import pandas as pd
import numpy as np

subjects = list(range(1,11))
stories = [4, 5, 8]

for i in subjects:
    for j in stories:
        df = pd.read_csv(f"../model_output/Subject_{i}_Story_{j}.csv")
        data = df.to_numpy()
        
        np.save(f"../speech_predictions/Subject_{i}_Story_{j}_predictions.npy", data)