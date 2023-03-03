import os
import shutil
from pathlib import Path

from evalutils.utils import (
    convert_line_endings,
    generate_requirements_txt,
    generate_source_wheel,
)

TASK_KIND = "{{ cookiecutter.task_kind }}"
TEMPLATE_KIND = "{{ cookiecutter.template_kind }}"
IS_DEV_BUILD = int("{{ cookiecutter.dev_build }}") == 1

template_dir = Path(os.getcwd())

templated_files = template_dir.glob("*.j2")
for f in templated_files:
    shutil.move(f.name, f.stem)

if TEMPLATE_KIND == "Evaluation":  # noqa: C901
    shutil.rmtree("algorithm_test")
    os.remove("process.py")
    os.rename("evaluation_test", "test")

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

    if TASK_KIND.lower() != "segmentation":
        remove_segmentation_files()

    if TASK_KIND.lower() != "detection":
        remove_detection_files()

    if TASK_KIND.lower() != "classification":
        remove_classification_files()

elif TEMPLATE_KIND == "Algorithm":
    shutil.rmtree("evaluation_test")
    shutil.rmtree("ground-truth")
    os.remove("evaluation.py")
    os.rename("algorithm_test", "test")

    template_test_dir = template_dir / "test"

    def remove_result_files():
        for task_kind in ["segmentation", "detection", "classification"]:
            os.remove(template_test_dir / f"results_{task_kind}.json")

    expected_output_file = (
        template_test_dir / f"results_{TASK_KIND.lower()}.json"
    )

    shutil.copy(
        str(expected_output_file), template_test_dir / "expected_output.json"
    )

    remove_result_files()

if IS_DEV_BUILD:
    generate_source_wheel(template_dir / "vendor")

generate_requirements_txt()
convert_line_endings()
