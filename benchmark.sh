#!/bin/bash
make > /dev/null

echo "C++ SGD"
./sgd

echo ""
echo "PYTORCH SGD"
python3 main.py
