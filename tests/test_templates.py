# -*- coding: utf-8 -*-
import json
import os
import platform
import subprocess
from distutils.util import strtobool
from pathlib import Path

import pytest


def check_dict(check, expected):
    """ Recursively check a dictionary of dictionaries """
    for key, val in expected.items():
        if isinstance(val, dict):
            check_dict(check[key], val)
        else:
            assert check[key] == val


@pytest.mark.skipif(
    strtobool(os.environ.get("APPVEYOR", "False").lower()),
    reason="This test is not supported by standard appveyor",
)
@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        ("Classification", {"aggregates": {"accuracy_score": 0.5}}),
        (
            "Segmentation",
            {"aggregates": {"DiceCoefficient": {"mean": 0.9557903761508626}}},
        ),
        (
            "Detection",
            {
                "aggregates": {
                    "false_negatives": {"sum": 5},
                    "false_positives": {"sum": 7},
                    "true_positives": {"sum": 2},
                    "image_id": {"count": 5},
                    "precision": 2 / 9,
                    "recall": 2 / 7,
                    "f1_score": (2 * 2) / ((2 * 2) + 5 + 7),
                }
            },
        ),
    ],
)
def test_cli(tmpdir, kind, expected):
    """
    WARNING: This tests against the github dev branch! We need a better way
    than this to get the library into the templated docker file
    """
    print(json.dumps(dict(os.environ), indent=4))
    project_name = "testeval"

    file_ext = "bat" if platform.system().lower() == "windows" else "sh"

    files = os.listdir(tmpdir)
    assert len(files) == 0

    out = subprocess.check_output(
        ["evalutils", "init", project_name, f"--kind={kind}", "--dev"],
        cwd=tmpdir,
    )

    files = os.listdir(tmpdir)
    assert "testeval" in files
    assert f"Created project {project_name}" in out.decode()

    project_dir = Path(tmpdir) / project_name

    out = subprocess.check_output([str(project_dir / f"build.{file_ext}")])

    assert "Successfully built" in out.decode()
    assert f"Successfully tagged {project_name}:latest" in out.decode()

    out = subprocess.check_output([str(project_dir / f"test.{file_ext}")])

    # Grab the results json
    out = out.decode().splitlines()
    start = [i for i, ln in enumerate(out) if ln == "{"]
    end = [i for i, ln in enumerate(out) if ln == "}"]
    result = json.loads("\n".join(out[start[0] : (end[-1] + 1)]))

    check_dict(result, expected)

    files = os.listdir(project_dir)
    assert f"{project_name}.tar.gz" not in files

    subprocess.call([str(project_dir / f"export.{file_ext}")], cwd=project_dir)

    files = os.listdir(project_dir)
    assert f"{project_name}.tar.gz" in files
