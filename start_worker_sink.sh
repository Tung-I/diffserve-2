#!/bin/bash

# Start the worker process 
cd src/worker/
python worker.py -cip 10.100.20.38 -p 50100 -c multi --is_sink