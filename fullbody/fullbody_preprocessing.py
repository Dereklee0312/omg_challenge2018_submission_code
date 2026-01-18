import argparse
import json
import sys
import time
from pathlib import Path

import cv2
from skimage.color import rgb2gray
from skimage.transform import resize
import re


def sorted_nicely(l):
    """Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [
        convert(c) for c in re.split("([0-9]+)", str(key))
    ]
    return sorted(l, key=alphanum_key)


def progressBar(value, endvalue, bar_length=20):
    percent = 0 if endvalue == 0 else float(value) / endvalue
    filled = max(0, int(round(percent * bar_length) - 1))
    arrow = "-" * filled + ">"
    spaces = " " * (bar_length - len(arrow))

    sys.stdout.write(
        "\rPercent: [{0}] {1}%".format(arrow + spaces, int(round(percent * 100)))
    )
    sys.stdout.flush()


def TRIALextractFullBodyFromVideo(path, savePath, size=128, preview_frames=1):
    videos = sorted_nicely(
        [p for p in Path(path).iterdir() if p.is_file() and p.name != ".DS_Store"]
    )

    for video in videos:
        start_time = time.time()

        videoPath = video
        print("- Processing Video:", f"{videoPath} ...")
        cap = cv2.VideoCapture(str(videoPath))

        totalFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        check = True
        imageNumber = 0
        print("- Extracting full body:", str(totalFrames) + " Frames ...")

        video_stem = videoPath.stem
        savePathActorImg = Path(savePath) / video_stem / "Actor_img"
        savePathSubjectImg = Path(savePath) / video_stem / "Subject_img"
        # Fixing bounding boxes coordinates if needed
        if video_stem == "Subject_2_Story_8":
            x_act_1, y_act_1, x_act_2, y_act_2 = define_frames(x_shift=-20, tag="actor")
        elif video_stem == "Subject_4_Story_4":
            x_act_1, y_act_1, x_act_2, y_act_2 = define_frames(x_shift=-20, tag="actor")
        elif video_stem == "Subject_4_Story_5":
            x_act_1, y_act_1, x_act_2, y_act_2 = define_frames(x_shift=80, tag="actor")
        else:
            x_act_1, y_act_1, x_act_2, y_act_2 = define_frames(tag="actor")

        if video_stem == "Subject_1_Story_5":
            x_sub_1, y_sub_1, x_sub_2, y_sub_2 = define_frames(
                x_shift=-50, tag="subject"
            )
        elif video_stem == "Subject_2_Story_8":
            x_sub_1, y_sub_1, x_sub_2, y_sub_2 = define_frames(
                x_shift=-20, tag="subject"
            )
        else:
            x_sub_1, y_sub_1, x_sub_2, y_sub_2 = define_frames(tag="subject")

        savePathActorImg.mkdir(parents=True, exist_ok=True)
        savePathSubjectImg.mkdir(parents=True, exist_ok=True)

        while check:
            check, img = cap.read()
            if img is not None:
                # Extract actor face
                imageActor = img[y_act_1:y_act_2, x_act_1:x_act_2]
                cv2.imwrite(
                    str(savePathActorImg / f"A_{imageNumber}_{video_stem}.png"),
                    imageActor,
                )

                # Extract Subject Face
                imageSubject = img[y_sub_1:y_sub_2, x_sub_1:x_sub_2]
                cv2.imwrite(
                    str(savePathSubjectImg / f"S_{imageNumber}_{video_stem}.png"),
                    imageSubject,
                )

                imageNumber = imageNumber + 1
                progressBar(imageNumber, totalFrames)

            if imageNumber >= preview_frames:
                break
        cap.release()
        print("\nRunning time: %f seconds\n" % (time.time() - start_time))


def extractFullBodyFromVideo(path, savePath, size=128):
    videos = sorted_nicely(
        [p for p in Path(path).iterdir() if p.is_file() and p.name != ".DS_Store"]
    )

    for video in videos:
        start_time = time.time()

        videoPath = video
        print("- Processing Video:", f"{videoPath} ...")
        cap = cv2.VideoCapture(str(videoPath))

        totalFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        check = True
        imageNumber = 0
        print("- Extracting full body:", str(totalFrames) + " Frames ...")

        video_stem = videoPath.stem
        savePathActorImg = Path(savePath) / video_stem / "Actor_img"
        savePathSubjectImg = Path(savePath) / video_stem / "Subject_img"

        # Fixing bounding boxes coordinates if needed
        if video_stem == "Subject_2_Story_8":
            x_act_1, y_act_1, x_act_2, y_act_2 = define_frames(x_shift=-20, tag="actor")
        elif video_stem == "Subject_4_Story_4":
            x_act_1, y_act_1, x_act_2, y_act_2 = define_frames(x_shift=-20, tag="actor")
        elif video_stem == "Subject_4_Story_5":
            x_act_1, y_act_1, x_act_2, y_act_2 = define_frames(x_shift=80, tag="actor")
        else:
            x_act_1, y_act_1, x_act_2, y_act_2 = define_frames(tag="actor")

        if video_stem == "Subject_1_Story_5":
            x_sub_1, y_sub_1, x_sub_2, y_sub_2 = define_frames(
                x_shift=-50, tag="subject"
            )
        elif video_stem == "Subject_2_Story_8":
            x_sub_1, y_sub_1, x_sub_2, y_sub_2 = define_frames(
                x_shift=-20, tag="subject"
            )
        else:
            x_sub_1, y_sub_1, x_sub_2, y_sub_2 = define_frames(tag="subject")

        savePathActorImg.mkdir(parents=True, exist_ok=True)
        savePathSubjectImg.mkdir(parents=True, exist_ok=True)

        while check:
            check, img = cap.read()
            if img is not None:
                imageActor = img[y_act_1:y_act_2, x_act_1:x_act_2]
                imageActor = rgb2gray(resize(imageActor, (size, size))).reshape(
                    size, size, 1
                )
                cv2.imwrite(
                    str(savePathActorImg / f"{imageNumber}.png"), imageActor * 255
                )

                imageSubject = img[y_sub_1:y_sub_2, x_sub_1:x_sub_2]
                imageSubject = rgb2gray(resize(imageSubject, (size, size))).reshape(
                    size, size, 1
                )
                cv2.imwrite(
                    str(savePathSubjectImg / f"{imageNumber}.png"), imageSubject * 255
                )

                imageNumber = imageNumber + 1
                progressBar(imageNumber, totalFrames)

        cap.release()
        print("\nRunning time: %f seconds\n" % (time.time() - start_time))


def define_frames(tag, size=620, x_shift=0, y_shift=0):
    if tag == "actor":
        start_x = 290 + x_shift
    elif tag == "subject":
        start_x = 1460 + x_shift
    else:
        raise Exception("Specify a tag!")

    start_y = 720 - size + y_shift

    end_x = start_x + size
    end_y = start_y + size

    return start_x, start_y, end_x, end_y


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract fullbody crops from videos.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("fullbody/config.json"),
        help="Path to config JSON file.",
    )
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument(
        "--mode",
        choices=("trial", "extract"),
        default="extract",
        help="Use trial to preview crops or extract to process all frames.",
    )
    parser.add_argument("--preview-frames", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    with args.config.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    splits = [
        (
            Path(config["train_videos_dir"]),
            Path(config["train_fullbody_dir"]),
            "training",
        ),
        (
            Path(config["val_videos_dir"]),
            Path(config["val_fullbody_dir"]),
            "validation",
        ),
    ]
    for videos_dir, output_dir, split_name in splits:
        print(f"== {split_name} ==")
        if args.mode == "trial":
            output_dir = output_dir / "_trial"
            TRIALextractFullBodyFromVideo(
                videos_dir,
                output_dir,
                size=args.size,
                preview_frames=args.preview_frames,
            )
        else:
            extractFullBodyFromVideo(videos_dir, output_dir, size=args.size)
