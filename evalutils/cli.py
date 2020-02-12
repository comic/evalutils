import re
from pathlib import Path

import click
from cookiecutter.exceptions import FailedHookException
from cookiecutter.main import cookiecutter

from . import __version__

EVALUATION_CHOICES = ["Classification", "Segmentation", "Detection"]
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


@init.command(
    name="evaluation", short_help="Initialise an evaluation project."
)
@click.argument(
    "challenge_name", callback=validate_python_module_name_fn("challenge_name")
)
@click.option(
    "--kind",
    type=click.Choice(EVALUATION_CHOICES),
    prompt=f"What kind of challenge is this? [{'|'.join(EVALUATION_CHOICES)}]",
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
@click.option("--diag-ticket", type=click.STRING, default="")
@click.option(
    "--req-cpus",
    type=click.IntRange(1, 64, True),
    default=1,
    prompt="Required number of cpus?",
)
@click.option(
    "--req-cpu-capabilities",
    type=click.STRING,
    multiple=True,
    default=None,
    callback=req_cpu_capabilities_prompt,
)
@click.option(
    "--req-memory",
    type=click.STRING,
    default="1G",
    prompt="Minimal required amount of cpu RAM?",
    callback=validate_size_format_fn(can_be_empty=False),
)
@click.option(
    "--req-gpus",
    type=click.IntRange(0, 8, True),
    default=0,
    prompt="Required number of gpus?",
    callback=req_gpu_prompt,
)
@click.option(
    "--req-gpu-compute-capability",
    type=click.STRING,
    is_eager=True,
    default="",
)
@click.option(
    "--req-gpu-memory",
    type=click.STRING,
    is_eager=True,
    default="",
    callback=validate_size_format_fn(can_be_empty=True),
)
@click.option("--dev", is_flag=True)
def init_algorithm(
    algorithm_name,
    diag_ticket,
    req_cpus,
    req_cpu_capabilities,
    req_memory,
    req_gpus,
    req_gpu_compute_capability,
    req_gpu_memory,
    dev,
):
    template_dir = Path(__file__).parent / "templates" / "algorithm"
    try:
        cookiecutter(
            template=str(template_dir.absolute()),
            no_input=True,
            extra_context={
                "diag_ticket": diag_ticket,
                "algorithm_name": algorithm_name,
                "evalutils_name": __name__.split(".")[0],
                "evalutils_version": __version__,
                "dev_build": 1 if dev else 0,
                "requirements": {
                    "cpu_count": req_cpus,
                    "cpu_capabilities": req_cpu_capabilities,
                    "memory": req_memory,
                    "gpu_count": req_gpus,
                    "gpu_compute_capability": req_gpu_compute_capability,
                    "gpu_memory": req_gpu_memory,
                },
            },
        )
        click.echo(f"Created project {algorithm_name}")
    except FailedHookException:
        exit(1)
