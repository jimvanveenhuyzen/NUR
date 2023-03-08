#!/bin/bash

echo "Download Vandermonde.txt"
if [ ! -e Vandermonde.txt ]; then
  wget https://home.strw.leidenuniv.nl/~daalen/Handin_files/Vandermonde.txt 
fi

python3 NURhandin1_1a 
python3 NURhandin1_2

echo "Generating the pdf"

pdflatex handin1.tex


