#!/bin/bash

echo "This script should build your project now..."

git submodule update --init --recursive

make all