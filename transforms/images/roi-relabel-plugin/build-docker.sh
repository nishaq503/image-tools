#!/bin/bash

cp .gitignore .dockerignore

version=$(<VERSION)

docker build . -t polusai/roi-relabel-plugin:"${version}"

rm .dockerignore
