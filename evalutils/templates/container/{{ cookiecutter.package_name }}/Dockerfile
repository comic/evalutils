FROM python:{{ cookiecutter.python_major_version }}.{{ cookiecutter.python_minor_version }}-slim

RUN groupadd -r user && useradd -m --no-log-init -r -g user user

RUN mkdir -p /opt/app /input /output \
    && chown user:user /opt/app /input /output

USER user
WORKDIR /opt/app

ENV PATH="/home/user/.local/bin:${PATH}"

RUN python -m pip install --user -U pip && python -m pip install --user pip-tools

{% if cookiecutter.dev_build|int -%}
COPY --chown=user:user vendor /opt/app/vendor
{%- endif %}

COPY --chown=user:user requirements.txt /opt/app/
RUN python -m piptools sync requirements.txt

{% if cookiecutter.template_kind == "Evaluation" -%}
COPY --chown=user:user ground-truth /opt/app/ground-truth
COPY --chown=user:user evaluation.py /opt/app/

ENTRYPOINT [ "python", "-m", "evaluation" ]
{%- endif %}

{% if cookiecutter.template_kind == "Algorithm" -%}
COPY --chown=user:user process.py /opt/app/

ENTRYPOINT [ "python", "-m", "process" ]
{%- endif %}
