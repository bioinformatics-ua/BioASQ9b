#!/bin/bash

# exit when any command fails
set -e

VENV_NAME="py-bioasq"
PWD="$(pwd)"

PYTHON=$(pwd)/$VENV_NAME/bin/python
PIP=$(pwd)/$VENV_NAME/bin/pip

cd /home/tiagoalmeida/mmnrm
if [ -d "./dist" ]
then
	rm -r ./dist
fi

$PYTHON setup.py sdist
$PIP install ./dist/mmnrm-0.0.2.tar.gz
cd $PWD