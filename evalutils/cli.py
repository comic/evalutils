# -*- coding: utf-8 -*-
from pathlib import Path

import click
from cookiecutter.exceptions import FailedHookException
from cookiecutter.main import cookiecutter

from . import __version__


EVALUATOR_CHOICES = ["Classification", "Segmentation", "Detection"]


@click.group(context_settings=dict(help_option_names=["-h", "--help"]))
@click.version_option(__version__, "-v", "--version")
def main():
    pass


@main.command(short_help="Initialise an evaluator project.")
@click.argument("challenge_name")
@click.option(
    "--kind",
    type=click.Choice(EVALUATOR_CHOICES),
    prompt=f"What kind of challenge is this? [{'|'.join(EVALUATOR_CHOICES)}]",
)
@click.option("--dev", is_flag=True)
def init_evaluator(challenge_name, kind, dev):
    template_dir = Path(__file__).parent / "templates" / "evaluator"
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


def validate_non_empty_stripped_string_fn(option):
    def validate_non_empty_stripped_string(ctx, param, arg):
        if len(arg.strip()) == 0:
            click.echo(f"{option.upper()} should be non empty. Aborting...")
            exit(1)
        return arg

    return validate_non_empty_stripped_string


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
            gpu_memory = click.prompt(
                "Required gpu memory? (e.g.: 4G)",
                default="",
                type=click.STRING,
            )
    else:
        gpu_memory = ""
        gpu_compute_capability = ""
    ctx.params["req_gpu_compute_capability"] = gpu_compute_capability
    ctx.params["req_gpu_memory"] = gpu_memory
    return req_gpu_count


@main.command(short_help="Initialise a processor project.")
@click.argument(
    "processor_name",
    callback=validate_non_empty_stripped_string_fn("processor_name"),
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
)
@click.option(
    "--req-gpu-count",
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
@click.option("--req-gpu-memory", type=click.STRING, is_eager=True, default="")
@click.option("--dev", is_flag=True)
def init_processor(
    processor_name,
    diag_ticket,
    req_cpus,
    req_cpu_capabilities,
    req_memory,
    req_gpu_count,
    req_gpu_compute_capability,
    req_gpu_memory,
    dev,
):
    print(
        processor_name,
        diag_ticket,
        req_cpus,
        req_cpu_capabilities,
        req_memory,
        req_gpu_count,
        req_gpu_compute_capability,
        req_gpu_memory,
        dev,
    )
    template_dir = Path(__file__).parent / "templates" / "processor"
    try:
        cookiecutter(
            template=str(template_dir.absolute()),
            no_input=True,
            extra_context={
                "diag_ticket": diag_ticket,
                "processor_name": processor_name,
                "evalutils_name": __name__.split(".")[0],
                "evalutils_version": __version__,
                "dev_build": 1 if dev else 0,
                "requirements": {
                    "cpus": req_cpus,
                    "cpu_capabilities": req_cpu_capabilities,
                    "memory": req_memory,
                    "gpu_count": req_gpu_count,
                    "gpu_compute_capability": req_gpu_compute_capability,
                    "gpu_memory": req_gpu_memory,
                },
            },
        )
        click.echo(f"Created project {processor_name}")
    except FailedHookException:
        exit(1)
