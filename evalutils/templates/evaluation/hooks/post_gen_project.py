import os
import shutil
from pathlib import Path

from evalutils.utils import (
    convert_line_endings,
    generate_requirements_txt,
    generate_source_wheel,
)

CHALLENGE_KIND = "{{ cookiecutter.challenge_kind }}"
IS_DEV_BUILD = int("{{ cookiecutter.dev_build }}") == 1

template_dir = Path(os.getcwd())

templated_files = template_dir.glob("*.j2")
for f in templated_files:
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

if IS_DEV_BUILD:
    generate_source_wheel(template_dir / "vendor")

generate_requirements_txt()
convert_line_endings()
