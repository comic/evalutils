import os
import shutil
from pathlib import Path

from evalutils import __file__
from evalutils.utils import (
    bootstrap_development_distribution,
    convert_line_endings,
)

ALGORITHM_KIND = "{{ cookiecutter.algorithm_kind }}"
ALGORITHM_NAME = "{{ cookiecutter.algorithm_name }}"
IS_DEV_BUILD = int("{{ cookiecutter.dev_build }}") == 1

expected_output_file = (
    Path(__file__).parent.parent
    / "tests"
    / "resources"
    / "json"
    / f"results_{ALGORITHM_KIND.lower()}.json"
)

template_dir = Path(os.getcwd())

templated_files = template_dir.glob("*.j2")
for f in templated_files:
    shutil.move(f.name, f.stem)

shutil.copy(
    str(expected_output_file), template_dir / "test" / "expected_output.json"
)

convert_line_endings()

if IS_DEV_BUILD:
    bootstrap_development_distribution(
        ALGORITHM_NAME, template_dir / "devdist"
    )
