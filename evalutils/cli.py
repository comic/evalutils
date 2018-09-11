# -*- coding: utf-8 -*-
from pathlib import Path

import click
from cookiecutter.exceptions import FailedHookException
from cookiecutter.main import cookiecutter

from . import __version__

KIND_CHOICES = ["Classification", "Segmentation", "Detection"]


@click.group(context_settings=dict(help_option_names=["-h", "--help"]))
@click.version_option(__version__, "-v", "--version")
def main():
    pass


@main.command(short_help="Initialise an evalutils project.")
@click.argument("challenge_name")
@click.option(
    "--kind",
    type=click.Choice(KIND_CHOICES),
    prompt=f"What kind of challenge is this? [{'|'.join(KIND_CHOICES)}]",
)
@click.option("--dev", is_flag=True)
def init(challenge_name, kind, dev):
    template_dir = Path(__file__).parent / "template"

    try:
        cookiecutter(
            template=str(template_dir.absolute()),
            no_input=True,
            extra_context={
                "challenge_name": challenge_name,
                "evalutils_name": __name__.split(".")[0],
                "evalutils_version": __version__,
                "challenge_kind": kind,
                "dev_build": 1 if dev else 0,
            },
        )
        click.echo(f"Created project {challenge_name}")
    except FailedHookException:
        exit(1)
