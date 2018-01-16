#!/usr/bin/env bash

for i in {1..11}; do
    python -c "from handling_taxiData import run; run('09$i')" &
done