from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import keras.backend as K
from keras import optimizers
import keras.callbacks as cb

from model import create_reg_resnet18_3D
from utils import ccc_error, create_img_dataset, light_generator, day_time


def _parse_int_list(value: str) -> list[int]:
    items: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            items.extend(range(int(start), int(end) + 1))
        else:
            items.append(int(part))
    return items


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train fullbody ResNet3D model.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("fullbody/config.json"),
        help="Path to config JSON file.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/OMG_Empathy2019"),
        help="Dataset root containing full_body and labels folders.",
    )
    parser.add_argument(
        "--img-template",
        type=str,
        default="",
        help="Override image path template with {0}/{1} or {subject}/{story}.",
    )
    parser.add_argument(
        "--label-template",
        type=str,
        default="",
        help="Override label CSV path template with {0}/{1} or {subject}/{story}.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/fullbody"),
        help="Directory for checkpoints.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("logs"),
        help="Directory for TensorBoard logs.",
    )
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--img-x", type=int, default=128)
    parser.add_argument("--img-y", type=int, default=128)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--down-sampling", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--subjects", type=str, default="1-10")
    parser.add_argument("--train-stories", type=str, default="2,4,5,8")
    parser.add_argument("--val-stories", type=str, default="1")
    return parser.parse_args()


def _format_template(template: str, subject: int, story: int) -> str:
    return template.format(subject, story, subject=subject, story=story)


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_img_template(
    config: dict, args: argparse.Namespace, split_key: str
) -> str:
    split_override = config.get(split_key)
    if split_override:
        template = str(
            Path(split_override)
            / "Subject_{0}_Story_{1}"
            / "Subject_img"
        )
    else:
        template = args.img_template or str(
            Path(config.get("dataset_root", args.dataset_root))
            / "full_body"
            / "full_body"
            / "Subject_{0}_Story_{1}"
            / "Subject_img"
        )
    return template if template.endswith("/") else template + "/"


def _resolve_label_template(
    config: dict, args: argparse.Namespace, split_key: str
) -> str:
    split_override = config.get(split_key)
    if split_override:
        return str(Path(split_override) / "Subject_{0}_Story_{1}.csv")
    return args.label_template or str(
        Path(config.get("dataset_root", args.dataset_root))
        / "labels"
        / "Subject_{0}_Story_{1}.csv"
    )


def main() -> None:
    args = _parse_args()
    config = _load_config(args.config)
    output_dir = Path(config.get("checkpoint_dir", args.output_dir))
    log_dir_base = Path(config.get("log_dir", args.log_dir))

    img_template_train = _resolve_img_template(
        config, args, "train_fullbody_dir"
    )
    img_template_val = _resolve_img_template(
        config, args, "val_fullbody_dir"
    )
    label_template_train = _resolve_label_template(
        config, args, "train_labels_dir"
    )
    label_template_val = _resolve_label_template(
        config, args, "val_labels_dir"
    )

    checkpoint_dir = output_dir / "resnet_ccc"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir = log_dir_base / day_time
    log_dir.mkdir(parents=True, exist_ok=True)

    K.clear_session()
    model = create_reg_resnet18_3D(
        args.img_x, args.img_y, args.channels, args.seq_len, tgt_size=1
    )
    opti = optimizers.Adam(learning_rate=args.learning_rate)
    model.compile(loss=ccc_error, optimizer=opti)

    print("3d resnet 16 model loaded")

    subjects = _parse_int_list(args.subjects)
    train_stories = _parse_int_list(args.train_stories)
    val_stories = _parse_int_list(args.val_stories)

    lbl_tr = np.concatenate(
        [
            np.loadtxt(
                _format_template(label_template_train, sbj_n, str_n), skiprows=1
            )[
                :: args.down_sampling
            ]
            for str_n in train_stories
            for sbj_n in subjects
        ]
    ).reshape(-1, 1)
    img_tr = create_img_dataset(
        img_template_train,
        args.img_x,
        args.img_y,
        args.channels,
        train_stories,
        subjects,
        args.down_sampling,
    )

    print("train images loaded with shape: ", img_tr.shape)
    print("train labels loaded with shape: ", lbl_tr.shape)
    if img_tr.shape[0] != lbl_tr.shape[0]:
        raise ValueError(
            f"Training images ({img_tr.shape[0]}) and labels ({lbl_tr.shape[0]}) "
            f"count mismatch. Check extraction and label paths."
        )

    lw_gen_tr = light_generator(img_tr, lbl_tr, args.seq_len, args.batch_size)
    steps_per_epoch_tr = lw_gen_tr.stp_per_epoch

    lbl_vl = np.concatenate(
        [
            np.loadtxt(
                _format_template(label_template_val, sbj_n, str_n), skiprows=1
            )[
                :: args.down_sampling
            ]
            for str_n in val_stories
            for sbj_n in subjects
        ]
    ).reshape(-1, 1)
    img_vl = create_img_dataset(
        img_template_val,
        args.img_x,
        args.img_y,
        args.channels,
        val_stories,
        subjects,
        args.down_sampling,
    )

    print("valid images loaded with shape: ", img_vl.shape)
    print("valid labels loaded with shape: ", lbl_vl.shape)
    if img_vl.shape[0] != lbl_vl.shape[0]:
        raise ValueError(
            f"Validation images ({img_vl.shape[0]}) and labels ({lbl_vl.shape[0]}) "
            f"count mismatch. Check extraction and label paths."
        )

    lw_gen_vl = light_generator(img_vl, lbl_vl, args.seq_len, args.batch_size)
    steps_per_epoch_vl = lw_gen_vl.stp_per_epoch

    save_name = (
        checkpoint_dir
        / "resnet_3D_global_regression.{epoch:02d}-{val_loss:.2f}.keras"
    )

    bckup_callback = cb.ModelCheckpoint(
        str(save_name),
        monitor="val_loss",
        verbose=0,
        save_best_only=True,
        save_weights_only=False,
        mode="auto",
        save_freq="epoch",
    )

    stop_callback = cb.EarlyStopping(monitor="val_loss", patience=10)

    callbacks_list = [
        bckup_callback,
        stop_callback,
        cb.TensorBoard(log_dir=str(log_dir)),
    ]

    model.fit(
        lw_gen_tr.generate(),
        steps_per_epoch=steps_per_epoch_tr,
        epochs=args.epochs,
        callbacks=callbacks_list,
        validation_data=lw_gen_vl.generate(),
        validation_steps=steps_per_epoch_vl,
        verbose=1,
    )


if __name__ == "__main__":
    main()
