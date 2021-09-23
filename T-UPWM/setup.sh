#!/bin/bash

# exit when any command fails
set -e

VENV_NAME="py-bioasq"

# remove the virtual env if alredy exists
if [ -d "$(pwd)/$VENV_NAME" ]; then
	echo "Removing alredy existing venv first"
	rm -r $VENV_NAME
fi	 


# PYTHON DEPENDENCIES
PYTHON=python3.6

echo "Creating a python environment ($VENV_NAME)"
$PYTHON -m venv $VENV_NAME

PYTHON=$(pwd)/$VENV_NAME/bin/python
PIP=$(pwd)/$VENV_NAME/bin/pip
IPYTHON=$(pwd)/$VENV_NAME/bin/ipython

# update pip

echo "Updating pip"
$PYTHON -m pip install -U pip

echo "Installing python requirements"
$PIP install -r requirements.txt

# ADD to Jupyter
echo "Adding this kernel to the jupyter notebook"
$IPYTHON kernel install --name "$VENV_NAME" --user

echo "Manually install nir python library"
cd _other_dependencies/nir/

if [ -d "./dist" ]
then
	rm -r ./dist
fi
$PYTHON setup.py sdist
$PIP install ./dist/nir-0.0.1.tar.gz
cd ../../

echo "Manually install mmnrm python library"
./_other_dependencies/install_mmnrm.sh