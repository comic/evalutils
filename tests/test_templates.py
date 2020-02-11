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
def test_evaluator_cli(tmpdir, kind, expected):
    print(json.dumps(dict(os.environ), indent=4))
    project_name = "testeval"

    file_ext = "bat" if platform.system().lower() == "windows" else "sh"

    files = os.listdir(tmpdir)
    assert len(files) == 0

    out = subprocess.check_output(
        [
            "evalutils",
            "init",
            "evaluator",
            project_name,
            f"--kind={kind}",
            "--dev",
        ],
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


@pytest.mark.skipif(
    strtobool(os.environ.get("APPVEYOR", "False").lower()),
    reason="This test is not supported by standard appveyor",
)
@pytest.mark.parametrize(
    (
        "diag_ticket",
        "req_cpus",
        "req_cpu_capabilities",
        "req_memory",
        "req_gpus",
        "req_gpu_compute_capability",
        "req_gpu_memory",
    ),
    [("", 1, (), "2G", 0, "", "")],
)
def test_algorithm_cli(
    tmpdir,
    diag_ticket,
    req_cpus,
    req_cpu_capabilities,
    req_memory,
    req_gpus,
    req_gpu_compute_capability,
    req_gpu_memory,
):
    print(json.dumps(dict(os.environ), indent=4))
    project_name = "testeval"

    file_ext = "bat" if platform.system().lower() == "windows" else "sh"

    files = os.listdir(tmpdir)
    assert len(files) == 0

    out = subprocess.check_output(
        [
            "evalutils",
            "init",
            "algorithm",
            project_name,
            f"--diag-ticket={diag_ticket}",
            f"--req-cpus={req_cpus}",
            f"--req-cpu-capabilities={req_cpu_capabilities}",
            f"--req-memory={req_memory}",
            f"--req-gpus={req_gpus}",
            f"--req-gpu-compute-capability={req_gpu_compute_capability}",
            f"--req-gpu-memory={req_gpu_memory}",
            "--dev",
        ],
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
    print(out)

    # Grab the results json
    out = out.decode().splitlines()
    start = [i for i, ln in enumerate(out) if ln == "["]
    end = [i for i, ln in enumerate(out) if ln == "]"]
    result = json.loads("\n".join(out[start[0] : (end[-1] + 1)]))
    print(result)

    with open(
        Path(__file__).parent / "resources" / "json" / "results.json"
    ) as f:
        expected = json.load(f)

    assert len(result) == 2
    assert result == expected

    files = os.listdir(project_dir)
    assert f"{project_name}.tar.gz" not in files

    subprocess.call([str(project_dir / f"export.{file_ext}")], cwd=project_dir)

    files = os.listdir(project_dir)
    assert f"{project_name}.tar.gz" in files
