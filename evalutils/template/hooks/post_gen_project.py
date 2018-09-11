# -*- coding: utf-8 -*-
import os
import shutil
from pathlib import Path

templated_python_files = Path(os.getcwd()).glob('*.py.j2')
for f in templated_python_files:
    shutil.move(f.name, f.stem)

challenge_kind = "{{ cookiecutter.challenge_kind }}"

def remove_classification_files():
    os.remove(Path("ground-truth") / "reference.csv")
    os.remove(Path("test") / "submission.csv")

def remove_segmentation_files():
    files = []
    for ext in ["mhd", "zraw"]:
        files.extend(Path(".").glob(f"**/*.{ext}"))

    for file in files:
        os.remove(file)

def remove_detection_files():
    pass

if challenge_kind.lower() != "segmentation":
    remove_segmentation_files()

if challenge_kind.lower() != "detection":
    remove_detection_files()

if challenge_kind.lower() != "classification":
    remove_classification_files()
