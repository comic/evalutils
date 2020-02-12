FROM {{ cookiecutter.docker_base_container }}

{% if cookiecutter.dev_build|int -%}
RUN apt-get update && apt-get install -y git
{%- endif %}

RUN groupadd -r evaluator && useradd -m --no-log-init -r -g evaluator evaluator

RUN mkdir -p /opt/evaluation /input /output \
    && chown evaluator:evaluator /opt/evaluation /input /output

USER evaluator
WORKDIR /opt/evaluation

ENV PATH="/home/evaluator/.local/bin:${PATH}"

RUN python -m pip install --user -U pip

{% if cookiecutter.dev_build|int -%}
# Installs development distributions of evalutils
COPY --chown=evaluator:evaluator devdist /opt/evaluation/devdist
RUN python -m pip install --user /opt/evaluation/devdist
{%- endif %}

COPY --chown=evaluator:evaluator ground-truth /opt/evaluation/ground-truth

COPY --chown=evaluator:evaluator requirements.txt /opt/evaluation/
RUN python -m pip install --user -rrequirements.txt

COPY --chown=evaluator:evaluator evaluation.py /opt/evaluation/

ENTRYPOINT "python" "-m" "evaluation"
