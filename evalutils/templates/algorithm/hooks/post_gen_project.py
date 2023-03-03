import os
import shutil
from pathlib import Path

from evalutils.utils import (
    convert_line_endings,
    generate_requirements_txt,
    generate_source_wheel,
)

ALGORITHM_KIND = "{{ cookiecutter.algorithm_kind }}"
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

if IS_DEV_BUILD:
    generate_source_wheel(template_dir / "vendor")

generate_requirements_txt()
convert_line_endings()
