# -*- coding: utf-8 -*-
import os
import shutil
from pathlib import Path

CHALLENGE_KIND = "{{ cookiecutter.challenge_kind }}"

EOL_UNIX = b"\n"
EOL_WIN = b"\r\n"
EOL_MAC = b"\r"

templated_python_files = Path(os.getcwd()).glob("*.py.j2")
for f in templated_python_files:
    shutil.move(f.name, f.stem)


def remove_classification_files():
    os.remove(Path("ground-truth") / "reference.csv")
    os.remove(Path("test") / "submission.csv")


def remove_segmentation_files():
    files = []
    for ext in ["mhd", "zraw"]:
        files.extend(Path(".").glob(f"**/*.{ext}"))

    for file in files:
        os.remove(str(file))


def remove_detection_files():
    os.remove(Path("ground-truth") / "detection-reference.csv")
    os.remove(Path("test") / "detection-submission.csv")


if CHALLENGE_KIND.lower() != "segmentation":
    remove_segmentation_files()

if CHALLENGE_KIND.lower() != "detection":
    remove_detection_files()

if CHALLENGE_KIND.lower() != "classification":
    remove_classification_files()


def convert_line_endings():
    """ Enforce unix line endings for the generated files """
    files = []
    for ext in [
        ".py",
        ".sh",
        "Dockerfile",
        ".txt",
        ".csv",
        ".mhd",
        ".gitignore",
    ]:
        files.extend(Path(".").glob(f"**/*{ext}"))

    for file in files:
        with open(file, "rb") as f:
            lines = f.read()

        lines = lines.replace(EOL_WIN, EOL_UNIX).replace(EOL_MAC, EOL_UNIX)

        with open(file, "wb") as f:
            f.write(lines)


convert_line_endings()
