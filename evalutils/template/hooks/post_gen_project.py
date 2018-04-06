# -*- coding: utf-8 -*-
import os
import shutil
from pathlib import Path

templated_python_files = Path(os.getcwd()).glob('*.py.j2')
for f in templated_python_files:
    shutil.move(f.name, f.stem)
