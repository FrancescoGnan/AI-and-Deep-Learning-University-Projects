#!/bin/bash

## Accessing command line arguments in bash
#echo "First argument: $1"
#echo "Second argument: $2"
#echo "All arguments: $@"
#echo "Number of arguments: $#"

echo "This script should read a dataset image into a tensor and pretty-print it into a text file..."

git submodule update --init --recursive

# Build:
make images_io

# Executing executable:
echo "Executing: $ ./exes/images_io $@"
./exes/images_io $@