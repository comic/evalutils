call .\build.bat

docker save {{ cookiecutter.package_name|lower }} > {{ cookiecutter.package_name }}.tar
