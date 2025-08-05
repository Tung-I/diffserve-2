#!/bin/bash

# Start the worker process 
cd src/worker/
python worker.py -cip 10.100.40.32 -p 50051 -c multi --do_simulate