import shutil
from pathlib import Path

EOL_UNIX = b"\n"
EOL_WIN = b"\r\n"
EOL_MAC = b"\r"


def bootstrap_development_distribution(project_name: str, dest_dir: Path):
    src_dir = Path(__file__).parent.parent.absolute()

    shutil.copytree(src_dir / "evalutils", dest_dir / "evalutils")
    for file in ["setup.py", "README.rst", "HISTORY.rst"]:
        shutil.copy(src_dir / file, dest_dir / file)


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
        with open(str(file), "rb") as f:
            lines = f.read()

        lines = lines.replace(EOL_WIN, EOL_UNIX).replace(EOL_MAC, EOL_UNIX)

        with open(str(file), "wb") as f:
            f.write(lines)
