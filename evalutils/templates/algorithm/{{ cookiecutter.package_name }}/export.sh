#!/usr/bin/env bash

./build.sh

docker save {{ cookiecutter.package_name|lower }} | xz -c > {{ cookiecutter.package_name }}.tar.xz
