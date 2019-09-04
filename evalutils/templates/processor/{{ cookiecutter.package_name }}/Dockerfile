FROM {{ cookiecutter.docker_base_container }}

{% if cookiecutter.dev_build|int -%}
RUN apt-get update && apt-get install -y git
{%- endif %}

RUN groupadd -r processor && useradd -m --no-log-init -r -g processor processor

RUN mkdir -p /opt/processor /input /output \
    && chown processor:processor /opt/processor /input /output

USER processor
WORKDIR /opt/processor

ENV PATH="/home/processor/.local/bin:${PATH}"

RUN python -m pip install --user -U pip

COPY --chown=processor:processor ground-truth /opt/processor/ground-truth

COPY --chown=processor:processor requirements.txt /opt/processor/
RUN python -m pip install --user -rrequirements.txt

COPY --chown=processor:processor evaluation.py /opt/processor/

ENTRYPOINT python -m process $0 $@

## PROCESSOR LABELS ##

# These labels are required
LABEL com.radboudumc.diag.processor.name={{ cookiecutter.processor_name }}

# These labels are required and describe what kind of hardware your processor requires to run.
LABEL com.radboudumc.diag.processor.hardware.cpu.count={{ cookiecutter.requirements.cpus }}
LABEL com.radboudumc.diag.processor.hardware.cpu.capabilities={{ cookiecutter.requirements.cpu_capabilities }}
LABEL com.radboudumc.diag.processor.hardware.memory={{ cookiecutter.requirements.memory }}
LABEL com.radboudumc.diag.processor.hardware.gpu.count={{ cookiecutter.requirements.gpu_count }}
LABEL com.radboudumc.diag.processor.hardware.gpu.cuda_compute_capability={{ cookiecutter.requirements.gpu_compute_capability }}
LABEL com.radboudumc.diag.processor.hardware.gpu.memory={{ cookiecutter.requirements.gpu_memory }}

{% if cookiecutter.diag_ticket|trim() -%}
# Optional labels
LABEL com.radboudumc.diag.processor.ticket={{ cookiecutter.diag_ticket|trim() }}
{%- endif %}
