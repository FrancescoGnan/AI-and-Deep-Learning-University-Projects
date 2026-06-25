#!/bin/bash

echo "This script should trigger the training and testing of your neural network implementation..."

git submodule update --init --recursive

# Build:
make NN

# Executing executable:
echo "Executing: $ ./exes/NeuralNetwork.exe $@"
./exes/NeuralNetwork.exe $@