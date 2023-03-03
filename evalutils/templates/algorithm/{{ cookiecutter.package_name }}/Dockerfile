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

ENTRYPOINT [ "python", "-m", "process" ]
