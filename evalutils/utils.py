import shutil
import subprocess
import warnings
from pathlib import Path

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=UserWarning)
    # Suppress "Setuptools is replacing distutils"
    from piptools.scripts.compile import cli

EOL_UNIX = b"\n"
EOL_WIN = b"\r\n"
EOL_MAC = b"\r"


def generate_source_wheel(dest_dir: Path):
    dist_dir = Path(__file__).parent.parent.absolute() / "dist"

    subprocess.check_call(["poetry", "build"])

    dest_dir.mkdir(exist_ok=True, parents=True)

    for file in dist_dir.rglob("*.whl"):
        shutil.copyfile(file, dest_dir / file.name)


def convert_line_endings():
    """Enforce unix line endings for the generated files"""
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
        with open(str(file), "rb") as f:
            lines = f.read()

        lines = lines.replace(EOL_WIN, EOL_UNIX).replace(EOL_MAC, EOL_UNIX)

        with open(str(file), "wb") as f:
            f.write(lines)


def generate_requirements_txt():
    cli(["--resolver", "backtracking", "--quiet"])
