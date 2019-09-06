call .\build.bat

docker volume create {{ cookiecutter.package_name|lower }}-output

docker run --rm^
 --memory={{ cookiecutter.requirements.memory|lower }}^
 -v %~dp0\test\:/input/^
 -v {{ cookiecutter.package_name|lower }}-output:/output/^
 {{ cookiecutter.package_name|lower }}

docker run --rm^
 -v {{ cookiecutter.package_name|lower }}-output:/output/^
 {{ cookiecutter.docker_base_container }} cat /output/results.json | python -m json.tool

docker volume rm {{ cookiecutter.package_name|lower }}-output
