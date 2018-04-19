FROM {{ cookiecutter.docker_base_container }}

RUN mkdir -p /usr/src/evaluation \
    && mkdir /input \
    && mkdir /output

WORKDIR /usr/src/evaluation

COPY ground-truth /usr/src/evaluation/ground-truth

RUN python -m pip install -U pip

COPY requirements.txt /usr/src/evaluation/

RUN python -m pip install -rrequirements.txt

COPY evaluation.py /usr/src/evaluation/

ENTRYPOINT "python" "-m" "evaluation"
