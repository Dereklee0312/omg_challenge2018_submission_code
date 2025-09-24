import pandas as pd
import os

annotation_dir = "data/original_dataset/annotations"
for sub in range(1, 19):
    for st in range(1, 24):
        csv_file = f"{annotation_dir}/Subject_{sub}_Story_{st}.csv"
        if os.path.exists(csv_file):
            try:
                df = pd.read_csv(csv_file, header=None, on_bad_lines='skip')
                # Keep only numeric rows using pd.to_numeric with coercion
                df[0] = pd.to_numeric(df[0], errors='coerce')
                df = df.dropna()  # Remove rows where conversion failed (NaN)
                df.to_csv(csv_file, header=False, index=False)
                print(f"Cleaned {csv_file}")
            except Exception as e:
                print(f"Error cleaning {csv_file}: {e}")