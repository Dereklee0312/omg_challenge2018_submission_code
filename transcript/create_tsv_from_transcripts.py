import pysrt
import pandas as pd
import numpy as np
import os
import re

# Directories (update as needed)
srt_dir = "data/srt"  # SRT files
annotation_dir = "data/original_dataset/annotations"  # Valence CSVs (one float per frame)
output_dir = "data/text/word_valence"  # TSVs for transcript_preprocessing.py
os.makedirs(output_dir, exist_ok=True)

# FPS for frame conversion (~25fps per OMG-dataset and paper)
fps = 25

def time_to_frame(time_str, fps=fps):
    """
    Convert SRT time to frame number.
    """
    try:
        hours, minutes, seconds_ms = time_str.split(':')
        seconds, ms = seconds_ms.split(',')
        total_seconds = int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(ms) / 1000
        return int(total_seconds * fps)
    except ValueError:
        print(f"Invalid timestamp format: {time_str}")
        return 0  # Fallback

def srt_to_tsv(srt_file, annotation_file, output_file):
    try:
        # Load ground truth valence with float dtype, skip bad lines
        gt_valence = pd.read_csv(annotation_file, header=None, dtype=float, on_bad_lines='skip').values.flatten()
        if len(gt_valence) == 0:
            print(f"Empty or invalid valence CSV: {annotation_file}. Using fallback valence.")
            gt_valence = np.array([])

        # Load SRT
        subs = pysrt.open(srt_file)
        words = []
        valences = []

        # Process subtitles
        for sub in subs:
            text = sub.text.strip()
            text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
            sub_words = text.split()
            if not sub_words:
                continue

            # Convert start/end to frames
            start_frame = time_to_frame(str(sub.start))
            end_frame = time_to_frame(str(sub.end))

            # Distribute frames across words
            frames_per_word = (end_frame - start_frame) / max(1, len(sub_words))
            for i, word in enumerate(sub_words):
                word_start = start_frame + int(i * frames_per_word)
                word_end = start_frame + int((i + 1) * frames_per_word)
                word_end = min(word_end, len(gt_valence))  # Avoid overflow

                # Average valence (paper method)
                if len(gt_valence) > 0 and word_start < word_end:
                    avg_valence = np.mean(gt_valence[word_start:word_end])
                else:
                    avg_valence = 0.5  # Fallback for empty/invalid

                words.append(word)
                valences.append(avg_valence)

        # Save TSV
        df = pd.DataFrame({"word": words, "valence": valences})
        df.to_csv(output_file, sep="\t", header=False, index=False)
        print(f"Saved {output_file}")
    except FileNotFoundError:
        print(f"Missing file: {srt_file} or {annotation_file}")
    except pd.errors.ParserError:
        print(f"Parser error in CSV: {annotation_file}. Check for non-numeric data.")
    except Exception as e:
        print(f"Error processing {srt_file}: {e}")

def main():
    subjects = range(1, 11)  # 1-18
    stories = range(1, 6)   # 1-23

    for sub in subjects:
        for st in stories:
            srt_file = f"{srt_dir}/transcribed_subject_{sub}_story_{st}.srt"
            annotation_file = f"{annotation_dir}/Subject_{sub}_Story_{st}.csv"
            output_file = f"{output_dir}/Subject_{sub}_Story_{st}.tsv"
            srt_to_tsv(srt_file, annotation_file, output_file)

if __name__ == "__main__":
    main()