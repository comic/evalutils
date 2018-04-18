# -*- coding: utf-8 -*-
import os
import subprocess


def test_cli(tmpdir):
    project_name = "testeval"

    files = os.listdir(tmpdir)
    assert len(files) == 0

    out = subprocess.check_output(
        ["evalutils", "init", project_name], cwd=tmpdir,
    )

    files = os.listdir(tmpdir)
    assert "testeval" in files
    assert f"Created project {project_name}" in out.decode()

    project_dir = os.path.join(tmpdir, project_name)

    out = subprocess.check_output(["./build.sh"], cwd=project_dir)

    assert "Successfully built" in out.decode()
    assert f"Successfully tagged {project_name}:latest" in out.decode()

    out = subprocess.check_output(["./test.sh"], cwd=project_dir)

    assert '"std": 0.53452248' in out.decode()

    files = os.listdir(project_dir)
    assert f"{project_name}.tar" not in files

    subprocess.call(["./export.sh"], cwd=project_dir)

    files = os.listdir(project_dir)
    assert f"{project_name}.tar" in files
