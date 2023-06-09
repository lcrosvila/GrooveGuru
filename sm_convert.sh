#!/bin/bash

for f in /Users/lcros/Documents/ddc/DDR\ Classics/**/*.sm; do python3 sm_to_bin.py "$f"; done

for f in /Users/lcros/Documents/ddc/DDR\ Classics/**/*bin.npy; do mv "$f" "$PWD/converted_charts/ai_sm_bin"; done

for f in /Users/lcros/Documents/ddc/DDR\ Classics/**/*.npy; do mv "$f" "$PWD/converted_charts/ai_sm_time"; done