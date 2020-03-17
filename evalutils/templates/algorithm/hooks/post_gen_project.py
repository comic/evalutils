import os
import shutil
from pathlib import Path

from evalutils.utils import (
    bootstrap_development_distribution,
    convert_line_endings,
)

ALGORITHM_KIND = "{{ cookiecutter.algorithm_kind }}"
ALGORITHM_NAME = "{{ cookiecutter.algorithm_name }}"
IS_DEV_BUILD = int("{{ cookiecutter.dev_build }}") == 1

template_dir = Path(os.getcwd())
template_test_dir = template_dir / "test"

templated_files = template_dir.glob("*.j2")
for f in templated_files:
    shutil.move(f.name, f.stem)


def remove_result_files():
    for algorithm_kind in ["segmentation", "detection", "classification"]:
        os.remove(template_test_dir / f"results_{algorithm_kind}.json")


expected_output_file = (
    template_test_dir / f"results_{ALGORITHM_KIND.lower()}.json"
)

shutil.copy(
    str(expected_output_file), template_test_dir / "expected_output.json"
)

remove_result_files()

convert_line_endings()

if IS_DEV_BUILD:
    bootstrap_development_distribution(
        ALGORITHM_NAME, template_dir / "devdist"
    )
