#!/bin/bash

# Start the worker process 
cd src/worker/
python worker.py -cip 10.100.20.48 -c multi --do_simulate -p 50051
# python worker.py -cip 10.100.20.48 -c multi --do_simulate -p 50061