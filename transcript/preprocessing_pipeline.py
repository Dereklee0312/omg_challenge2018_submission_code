import pysrt
import pandas as pd
import numpy as np
import os
import re
import urllib.request

# Directories
srt_dir = "data/srt"  # SRT files
annotation_dir = "data/original_dataset/annotations"  # Valence CSVs
lexicon_dir = "./lexicons/"  # Lexicon files
tsv_dir = "data/text/word_valence"  # Intermediate TSVs
csv_dir = "../data/text/lexicons_features"  # Intermediate CSVs
npy_dir = "vectors/val2/text"  # Per-word NPYs
aligned_npy_dir = "vectors/val2/text_aligned"  # Aligned frame-level NPYs
for d in [tsv_dir, csv_dir, npy_dir, aligned_npy_dir, lexicon_dir]:
    os.makedirs(d, exist_ok=True)

# Lexicon URLs for auto-download
warriner_url = "http://saifmohammad.com/WebDocs/VAD/NRC-VAD-Lexicon.txt"  # Adjust if needed
depechemood_url = "https://github.com/marcoguerini/DepecheMood/raw/master/DepecheMood%2B%2B/DepecheMood_english_token_full.tsv"

# Download lexicons if not present
def download_lexicons():
    warriner_path = os.path.join(lexicon_dir, "Ratings_Warriner_et_al.csv")
    if not os.path.exists(warriner_path):
        print("Downloading Warriner lexicon...")
        urllib.request.urlretrieve(warriner_url, warriner_path)  # Conversion from TXT to CSV needed
    depechemood_path = os.path.join(lexicon_dir, "DepecheMood_english_token_full.tsv")
    if not os.path.exists(depechemood_path):
        print("Downloading DepecheMood lexicon...")
        urllib.request.urlretrieve(depechemood_url, depechemood_path)

# Step 1: SRT to TSV
def time_to_seconds(time_str):
    try:
        hours, minutes, seconds_ms = time_str.split(':')
        seconds, ms = seconds_ms.split(',')
        return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(ms) / 1000
    except ValueError:
        print(f"Invalid timestamp format: {time_str}")
        return 0

def time_to_frame(total_seconds, fps):
    return int(total_seconds * fps)

def srt_to_tsv(srt_file, annotation_file, output_tsv):
    try:
        gt_df = pd.read_csv(annotation_file, header=0, dtype=float, on_bad_lines='skip')
        gt_valence = gt_df.iloc[:, 0].dropna().values
        if len(gt_valence) == 0:
            print(f"Empty or invalid valence CSV: {annotation_file}. Using fallback valence.")
            gt_valence = np.array([])

        subs = pysrt.open(srt_file)
        if not subs:
            print(f"No subtitles in {srt_file}")
            return

        last_end_sec = time_to_seconds(str(subs[-1].end)) if subs else 0
        if last_end_sec <= 0 or len(gt_valence) <= 0:
            print(f"Invalid duration or valence data for {srt_file}. Falling back to default FPS: 25.")
            fps = 25
        else:
            fps = len(gt_valence) / last_end_sec
            print(f"Computed FPS for {srt_file}: {fps:.2f} (frames: {len(gt_valence)}, duration: {last_end_sec}s)")

        words = []
        valences = []

        for sub in subs:
            text = sub.text.strip()
            text = re.sub(r'[^\w\s]', '', text)
            sub_words = text.split()
            if not sub_words:
                continue

            start_sec = time_to_seconds(str(sub.start))
            end_sec = time_to_seconds(str(sub.end))
            start_frame = time_to_frame(start_sec, fps)
            end_frame = time_to_frame(end_sec, fps)

            start_frame = max(0, start_frame)
            end_frame = min(end_frame, len(gt_valence))

            if start_frame >= len(gt_valence) or end_frame <= start_frame:
                continue

            frames_per_word = (end_frame - start_frame) / max(1, len(sub_words))
            for i, word in enumerate(sub_words):
                word_start = start_frame + int(i * frames_per_word)
                word_end = start_frame + int((i + 1) * frames_per_word)
                word_end = min(word_end, len(gt_valence))

                if word_start < word_end:
                    avg_valence = np.mean(gt_valence[word_start:word_end])
                else:
                    avg_valence = 0.5

                words.append(word)
                valences.append(avg_valence)

        df = pd.DataFrame({"word": words, "valence": valences})
        df.to_csv(output_tsv, sep="\t", header=False, index=False)
        print(f"Saved {output_tsv} with {len(words)} words")
    except Exception as e:
        print(f"Error processing {srt_file}: {e}")

# Step 2: TSV to Lexicon CSV
def tsv_to_lexicon_csv(tsv_file, output_csv, df1, df2, column_names_lex1, column_names_lex2):
    try:
        emotions_story = []
        l1 = [0] * len(column_names_lex1)
        l2 = [0] * len(column_names_lex2)
        l1_prev = l1.copy()
        l2_prev = l2.copy()

        df_story = pd.read_csv(tsv_file, delimiter="\t", header=None)

        for _, row in df_story.iterrows():
            word = row[0]

            # Lexicon 1 (Warriner)
            q = df1.loc[df1["Word"] == word]
            if len(q) == 1:
                l1 = [float(q[x]) for x in column_names_lex1]
                l1_prev = l1.copy()
            else:
                l1 = l1_prev.copy()

            # Lexicon 2 (DepecheMood)
            q = df2.loc[df2["Unnamed: 0"] == word]
            if len(q) == 1:
                l2 = [float(q[x]) for x in column_names_lex2]
                l2_prev = l2.copy()
            else:
                l2 = l2_prev.copy()

            l_tot = [str(x) for x in l1 + l2]
            emotions_story.append(",".join(l_tot))

        with open(output_csv, 'w') as out:
            out.write("\n".join(emotions_story))
        print(f"Saved {output_csv}")
    except Exception as e:
        print(f"Error processing {tsv_file}: {e}")

# Step 3: CSV to Per-word NPY
def csv_to_npy(csv_file, output_npy):
    try:
        df = pd.read_csv(csv_file, header=None)
        data_array = df.to_numpy()
        np.save(output_npy, data_array)
        print(f"Saved {output_npy}")
    except Exception as e:
        print(f"Error processing {csv_file}: {e}")

# Step 4: Upsample to Frame-level Aligned NPY
def upsample_to_frames(lex_npy, ann_csv, output_aligned_npy):
    try:
        features = np.load(lex_npy)  # Shape: (num_words, 11)
        gt_valence = pd.read_csv(ann_csv, header=None).values.flatten()
        num_frames = len(gt_valence[1:])  # Skip header row if present

        num_words = len(features)
        frames_per_word = num_frames // num_words  # Approx; adjust for exact
        remain = num_frames % num_words

        aligned_features = np.zeros((num_frames, 11))
        frame_idx = 0
        for i in range(num_words):
            word_frames = frames_per_word + (1 if i < remain else 0)
            aligned_features[frame_idx:frame_idx + word_frames] = features[i]
            frame_idx += word_frames

        if frame_idx < num_frames:
            aligned_features[frame_idx:] = np.mean(features, axis=0)  # Neutral padding

        np.save(output_aligned_npy, aligned_features)
        print(f"Aligned {aligned_features.shape} for {output_aligned_npy}")
    except Exception as e:
        print(f"Error processing {lex_npy}: {e}")

# Main: Run full pipeline
def main():
    download_lexicons()
    df1 = pd.read_csv(os.path.join(lexicon_dir, "Ratings_Warriner_et_al.csv"))
    df2 = pd.read_csv(os.path.join(lexicon_dir, "DepecheMood_english_token_full.tsv"), delimiter="\t")
    column_names_lex1 = ["V.Mean.Sum", "A.Mean.Sum", "D.Mean.Sum"]
    column_names_lex2 = ["AFRAID", "AMUSED", "ANGRY", "ANNOYED", "DONT_CARE", "HAPPY", "INSPIRED", "SAD"]

    subjects = range(1, 11)
    stories = range(1, 6)

    for sub in subjects:
        for st in stories:
            story_name = f"Subject_{sub}_Story_{st}"
            srt_file = f"{srt_dir}/transcribed_subject_{sub}_story_{st}.srt"
            ann_file = f"{annotation_dir}/{story_name}.csv"
            tsv_file = f"{tsv_dir}/{story_name}.tsv"
            csv_file = f"{csv_dir}/{story_name}_lex.csv"
            npy_file = f"{npy_dir}/{story_name}.npy"
            aligned_npy_file = f"{aligned_npy_dir}/{story_name}_aligned.npy"

            srt_to_tsv(srt_file, ann_file, tsv_file)
            tsv_to_lexicon_csv(tsv_file, csv_file, df1, df2, column_names_lex1, column_names_lex2)
            csv_to_npy(csv_file, npy_file)
            upsample_to_frames(npy_file, ann_file, aligned_npy_file)

if __name__ == "__main__":
    main()