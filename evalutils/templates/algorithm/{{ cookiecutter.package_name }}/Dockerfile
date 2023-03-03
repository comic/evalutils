FROM python:{{ cookiecutter.python_major_version }}.{{ cookiecutter.python_minor_version }}-slim

RUN groupadd -r algorithm && useradd -m --no-log-init -r -g algorithm algorithm

RUN mkdir -p /opt/algorithm /input /output \
    && chown algorithm:algorithm /opt/algorithm /input /output

USER algorithm
WORKDIR /opt/algorithm

ENV PATH="/home/algorithm/.local/bin:${PATH}"

RUN python -m pip install --user -U pip && python -m pip install --user pip-tools

{% if cookiecutter.dev_build|int -%}
COPY --chown=algorithm:algorithm vendor /opt/algorithm/vendor
{%- endif %}

COPY --chown=algorithm:algorithm requirements.txt /opt/algorithm/
RUN python -m piptools sync requirements.txt

COPY --chown=algorithm:algorithm process.py /opt/algorithm/

ENTRYPOINT [ "python", "-m", "process" ]
