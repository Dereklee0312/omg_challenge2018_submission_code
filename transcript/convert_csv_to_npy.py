import pandas as pd
import numpy as np
import os

os.makedirs("vectors/val2/text", exist_ok=True)
window_size = 100  # Per Barbieri et al., 2019

for su in range(1, 11):
    for st in range(1, 6):
# for su in range(1, 2):
#     for st in range(1, 2):
        csv_file = f"../data/text/lexicons_features/Subject_{su}_Story_{st}_lex.csv"
        try:
            df = pd.read_csv(csv_file, header=None)
            data_array = df.to_numpy()
            
            np.save(f"vectors/val2/text/Subject_{su}_Story_{st}.npy", data_array)
            print(f"Saved vectors/val2/text/Subject_{su}_Story_{st}.npy")
        except FileNotFoundError:
            print(f"Missing {csv_file}")
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")