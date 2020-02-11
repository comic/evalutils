from pathlib import Path
import shutil

EOL_UNIX = b"\n"
EOL_WIN = b"\r\n"
EOL_MAC = b"\r"


def bootstrap_development_distribution(project_name: str, dest_dir: Path):
    """Bootstraps a stripped down version of the active library

    This function avoids a recursive copy of itself and ignores
    several unnecessary files and directories
    """
    src_dir = Path(__file__).parent.parent.absolute()
    print(f"Bootstrap: {src_dir} -> {dest_dir}")
    shutil.copytree(
        src_dir,
        dest_dir,
        ignore=shutil.ignore_patterns(
            project_name.lower(),
            ".git",
            "build",
            "dist",
            "docs",
            ".pytest_cache",
            ".eggs",
            "templates",
            "__pycache__",
        ),
    )


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
