import re
from pathlib import Path
from typing import List

import click
from cookiecutter.exceptions import FailedHookException
from cookiecutter.main import cookiecutter

from . import __version__

EVALUATION_CHOICES = ["Classification", "Segmentation", "Detection"]
ALGORITHM_CHOICES = EVALUATION_CHOICES
FORBIDDEN_NAMES = ["evalutils", "pandas", "Evaluation", "Algorithm"]
MODULE_REGEX = r"^[_a-zA-Z][_a-zA-Z0-9]+$"


@click.group(context_settings=dict(help_option_names=["-h", "--help"]))
@click.version_option(__version__, "-v", "--version")
def main():
    pass


@main.group(name="init", short_help="Initialise a project.")
def init():
    pass


def validate_python_module_name_fn(option):
    def validate_python_module_name_string(ctx, param, arg):
        if len(arg.strip()) == 0:
            click.echo(f"{option.upper()} should be non empty. Aborting...")
            exit(1)

        if not re.match(MODULE_REGEX, arg) or arg in FORBIDDEN_NAMES:
            click.echo(f"ERROR: '{arg}' is not a valid Python module name!")
            exit(1)

        return arg

    return validate_python_module_name_string


class AbbreviatedChoice(click.Choice):
    def __init__(self, choices: List[str]):
        super().__init__(choices=choices, case_sensitive=True)
        self._abbreviations = [e[0].upper() for e in choices]
        self._choices_upper = list(map(str.upper, choices))
        if len(set(self._abbreviations)) != len(choices):
            raise ValueError(
                "First letters of choices for AbbreviatedChoices should be unique!"
            )

    def get_metavar(self, param):
        return "[{}]".format(
            "|".join([f"({e[0]}){e[1:]}" for e in self.choices])
        )

    def convert(self, value, param, ctx):
        value = value.upper()
        if value in self._abbreviations:
            value = self.choices[self._abbreviations.index(value)]
        elif value in self._choices_upper:
            value = self.choices[self._choices_upper.index(value)]
        return super().convert(value=value, param=param, ctx=ctx)


@init.command(
    name="evaluation", short_help="Initialise an evaluation project."
)
@click.argument(
    "challenge_name", callback=validate_python_module_name_fn("challenge_name")
)
@click.option(
    "--kind",
    type=AbbreviatedChoice(EVALUATION_CHOICES),
    prompt=f"What kind of challenge is this?",
)
@click.option("--dev", is_flag=True)
def init_evaluation(challenge_name, kind, dev):
    template_dir = Path(__file__).parent / "templates" / "evaluation"
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


def validate_size_format_fn(can_be_empty=False):
    def validate_size_format(ctx, param, arg):
        pattern = r"(^\d+[kgtpe]$)|(^$)" if can_be_empty else r"^\d+[kgtpe]$"
        fmt = re.compile(pattern, flags=re.IGNORECASE)
        while not re.match(fmt, arg.strip()):
            arg = click.prompt(
                f"Enter a valid size format ({fmt.pattern}):", default="1G"
            )
        return arg

    return validate_size_format


def req_cpu_capabilities_prompt(ctx, param, reqs):
    if not reqs:
        while True:
            capability = "something"
            reqs = ()
            while capability != "":
                capability = click.prompt(
                    f"Required node capability? (e.g.: avx) *{reqs}*",
                    type=click.STRING,
                    default="",
                )
                if capability != "":
                    reqs += (capability,)
            if click.confirm(
                f"Are *{reqs}* all required node capabilities?", True
            ):
                break
    return reqs


def req_gpu_prompt(ctx, param, req_gpu_count):
    gpu_memory = ctx.params.get("req_gpu_memory")
    gpu_compute_capability = ctx.params.get("req_gpu_compute_capability")
    if req_gpu_count > 0:
        if not gpu_compute_capability:
            gpu_compute_capability = click.prompt(
                "Required gpu compute capability (version string e.g.: 1.5.0)",
                default="",
                type=click.STRING,
            )
        if not gpu_memory:
            gpu_memory = validate_size_format_fn(can_be_empty=True)(
                None,
                None,
                click.prompt(
                    "Required gpu memory? (e.g.: 4G)",
                    default="",
                    type=click.STRING,
                ),
            )
    else:
        gpu_memory = ""
        gpu_compute_capability = ""
    ctx.params["req_gpu_compute_capability"] = gpu_compute_capability
    ctx.params["req_gpu_memory"] = gpu_memory
    return req_gpu_count


@init.command(name="algorithm", short_help="Initialise an algorithm project.")
@click.argument(
    "algorithm_name", callback=validate_python_module_name_fn("algorithm_name")
)
@click.option(
    "--kind",
    type=AbbreviatedChoice(ALGORITHM_CHOICES),
    prompt=f"What kind of algorithm is this?",
)
@click.option("--dev", is_flag=True)
def init_algorithm(
    algorithm_name, kind, dev,
):
    template_dir = Path(__file__).parent / "templates" / "algorithm"
    try:
        cookiecutter(
            template=str(template_dir.absolute()),
            no_input=True,
            extra_context={
                "algorithm_name": algorithm_name,
                "algorithm_kind": kind,
                "evalutils_name": __name__.split(".")[0],
                "evalutils_version": __version__,
                "dev_build": 1 if dev else 0,
            },
        )
        click.echo(f"Created project {algorithm_name}")
    except FailedHookException:
        exit(1)
