FROM {{ cookiecutter.docker_base_container }}

{% if cookiecutter.dev_build|int -%}
RUN apt-get update && apt-get install -y git
{%- endif %}

RUN groupadd -r algorithm && useradd -m --no-log-init -r -g algorithm algorithm

RUN mkdir -p /opt/algorithm /input /output \
    && chown algorithm:algorithm /opt/algorithm /input /output

USER algorithm

WORKDIR /opt/algorithm

ENV PATH="/home/algorithm/.local/bin:${PATH}"

RUN python -m pip install --user -U pip

{% if cookiecutter.dev_build|int -%}
# Installs development distributions of evalutils
COPY --chown=algorithm:algorithm devdist /opt/algorithm/devdist
RUN python -m pip install --user /opt/algorithm/devdist
{%- endif %}

COPY --chown=algorithm:algorithm requirements.txt /opt/algorithm/
RUN python -m pip install --user -rrequirements.txt

COPY --chown=algorithm:algorithm process.py /opt/algorithm/

ENTRYPOINT python -m process $0 $@

## ALGORITHM LABELS ##

# These labels are required
LABEL com.radboudumc.diag.algorithm.name={{ cookiecutter.algorithm_name }}

# These labels are required and describe what kind of hardware your algorithm requires to run.
LABEL com.radboudumc.diag.algorithm.hardware.cpu.count={{ cookiecutter.requirements.cpu_count }}
LABEL com.radboudumc.diag.algorithm.hardware.cpu.capabilities={{ cookiecutter.requirements.cpu_capabilities }}
LABEL com.radboudumc.diag.algorithm.hardware.memory={{ cookiecutter.requirements.memory }}
LABEL com.radboudumc.diag.algorithm.hardware.gpu.count={{ cookiecutter.requirements.gpu_count }}
LABEL com.radboudumc.diag.algorithm.hardware.gpu.cuda_compute_capability={{ cookiecutter.requirements.gpu_compute_capability }}
LABEL com.radboudumc.diag.algorithm.hardware.gpu.memory={{ cookiecutter.requirements.gpu_memory }}

{% if cookiecutter.diag_ticket|trim() -%}
# Optional labels
LABEL com.radboudumc.diag.algorithm.ticket={{ cookiecutter.diag_ticket|trim() }}
{%- endif %}
