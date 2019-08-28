# -*- coding: utf-8 -*-
import os
import shutil
from pathlib import Path

EOL_UNIX = b"\n"
EOL_WIN = b"\r\n"
EOL_MAC = b"\r"

templated_python_files = Path(os.getcwd()).glob("*.py.j2")
for f in templated_python_files:
    shutil.move(f.name, f.stem)


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
