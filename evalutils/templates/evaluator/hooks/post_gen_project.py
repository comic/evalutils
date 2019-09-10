import os
import shutil
from pathlib import Path
from evalutils.utils import (
    bootstrap_development_distribution,
    convert_line_endings,
)

CHALLENGE_KIND = "{{ cookiecutter.challenge_kind }}"
CHALLENGE_NAME = "{{ cookiecutter.challenge_name }}"
IS_DEV_BUILD = int("{{ cookiecutter.dev_build }}") == 1

template_dir = Path(os.getcwd())

templated_python_files = template_dir.glob("*.py.j2")
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


convert_line_endings()

if IS_DEV_BUILD:
    bootstrap_development_distribution(
        CHALLENGE_NAME, template_dir / "devdist"
    )
