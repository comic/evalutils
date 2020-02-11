import os
import shutil
from pathlib import Path
from evalutils.utils import (
    bootstrap_development_distribution,
    convert_line_endings,
)


ALGORITHM_NAME = "{{ cookiecutter.algorithm_name }}"
IS_DEV_BUILD = int("{{ cookiecutter.dev_build }}") == 1

template_dir = Path(os.getcwd())

templated_python_files = template_dir.glob("*.py.j2")
for f in templated_python_files:
    shutil.move(f.name, f.stem)


convert_line_endings()

if IS_DEV_BUILD:
    bootstrap_development_distribution(
        ALGORITHM_NAME, template_dir / "devdist"
    )
