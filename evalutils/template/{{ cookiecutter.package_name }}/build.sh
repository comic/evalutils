#!/usr/bin/env bash

docker build -t {{ cookiecutter.package_name|lower }} .
