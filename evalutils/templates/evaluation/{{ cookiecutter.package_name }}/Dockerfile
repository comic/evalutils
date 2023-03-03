FROM python:{{ cookiecutter.python_major_version }}.{{ cookiecutter.python_minor_version }}-slim

RUN groupadd -r evaluator && useradd -m --no-log-init -r -g evaluator evaluator

RUN mkdir -p /opt/evaluation /input /output \
    && chown evaluator:evaluator /opt/evaluation /input /output

USER evaluator
WORKDIR /opt/evaluation

ENV PATH="/home/evaluator/.local/bin:${PATH}"

RUN python -m pip install --user -U pip && python -m pip install --user pip-tools

{% if cookiecutter.dev_build|int -%}
COPY --chown=evaluator:evaluator vendor /opt/evaluation/vendor
{%- endif %}

COPY --chown=evaluator:evaluator requirements.txt /opt/evaluation/
RUN python -m piptools sync requirements.txt

COPY --chown=evaluator:evaluator ground-truth /opt/evaluation/ground-truth

COPY --chown=evaluator:evaluator evaluation.py /opt/evaluation/

ENTRYPOINT [ "python", "-m", "evaluation" ]
