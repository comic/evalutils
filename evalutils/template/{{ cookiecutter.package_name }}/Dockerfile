FROM {{ cookiecutter.docker_base_container }}

RUN mkdir /evaluation \
    && mkdir /input \
    && mkdir /output

WORKDIR /evaluation

COPY ground-truth /evaluation/ground-truth

RUN python -m pip install -U pip

COPY requirements.txt /evaluation/

RUN python -m pip install -rrequirements.txt

COPY evaluation.py /evaluation/

ENTRYPOINT "python" "-m" "evaluation"
