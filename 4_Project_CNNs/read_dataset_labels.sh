#!/bin/bash

## Accessing command line arguments in bash
#echo "First argument: $1"
#echo "Second argument: $2"
#echo "All arguments: $@"
#echo "Number of arguments: $#"

echo "This script should read a dataset label into a tensor and pretty-print it into a text file..."

git submodule update --init --recursive

# Build:
make labels_io

# Executing executable:
echo "Executing: $ ./exes/labels_io $@"
./exes/labels_io $@