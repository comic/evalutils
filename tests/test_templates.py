import json
import os
import subprocess
from pathlib import Path

import pytest


def check_dict(check, expected):
    """ Recursively check a dictionary of dictionaries """
    for key, val in expected.items():
        if isinstance(val, dict):
            check_dict(check[key], val)
        else:
            assert check[key] == val


@pytest.mark.slow
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
def test_evaluation_cli(tmpdir, kind, expected):
    project_name = f"testeval{kind}"

    files = os.listdir(tmpdir)
    assert len(files) == 0

    out = subprocess.check_output(
        [
            "evalutils",
            "init",
            "evaluation",
            project_name,
            f"--kind={kind}",
            "--dev",
        ],
        cwd=tmpdir,
    )

    files = os.listdir(tmpdir)
    assert project_name in files
    assert f"Created project {project_name}" in out.decode()

    project_dir = Path(tmpdir) / project_name

    out = subprocess.check_output([str(project_dir / "build.sh")])

    assert "Successfully built" in out.decode()
    assert f"Successfully tagged {project_name.lower()}:latest" in out.decode()

    out = subprocess.check_output([str(project_dir / "test.sh")])

    # Grab the results json
    out = out.decode().splitlines()
    start = [i for i, ln in enumerate(out) if ln == "{"]
    end = [i for i, ln in enumerate(out) if ln == "}"]
    result = json.loads("\n".join(out[start[0] : (end[-1] + 1)]))

    check_dict(result, expected)

    files = os.listdir(project_dir)
    assert f"{project_name}.tar.gz" not in files

    subprocess.call([str(project_dir / "export.sh")], cwd=project_dir)

    files = os.listdir(project_dir)
    assert f"{project_name}.tar.gz" in files


@pytest.mark.slow
@pytest.mark.parametrize(
    "kind", ("Detection", "Segmentation", "Classification")
)
def test_algorithm_cli(
    tmpdir, kind,
):
    project_name = f"testalg{kind}"

    files = os.listdir(tmpdir)
    assert len(files) == 0

    out = subprocess.check_output(
        [
            "evalutils",
            "init",
            "algorithm",
            project_name,
            f"--kind={kind}",
            "--dev",
        ],
        cwd=tmpdir,
    )

    files = os.listdir(tmpdir)
    assert project_name in files
    assert f"Created project {project_name}" in out.decode()

    project_dir = Path(tmpdir) / project_name

    out = subprocess.check_output([str(project_dir / "build.sh")])

    assert "Successfully built" in out.decode()
    assert f"Successfully tagged {project_name.lower()}:latest" in out.decode()
    out = subprocess.check_output([str(project_dir / "test.sh")])

    # Grab the results json
    out = out.decode().splitlines()
    assert "Tests successfully passed..." in out
    start = [i for i, ln in enumerate(out) if ln == "["]
    end = [i for i, ln in enumerate(out) if ln == "]"]
    result = json.loads("\n".join(out[start[0] : (end[-1] + 1)]))

    with open(
        Path(__file__).parent.parent
        / "evalutils"
        / "templates"
        / "algorithm"
        / "{{ cookiecutter.package_name }}"
        / "test"
        / f"results_{kind.lower()}.json"
    ) as f:
        expected = json.load(f)

    assert len(result) == 1
    assert result == expected

    files = os.listdir(project_dir)
    assert f"{project_name}.tar.gz" not in files

    subprocess.call([str(project_dir / "export.sh")], cwd=project_dir)

    files = os.listdir(project_dir)
    assert f"{project_name}.tar.gz" in files
