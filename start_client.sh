#!/bin/bash

# Start the client process 
cd src/client/
# python client.py -lbip localhost -trace 1to20
# python client.py -lbip localhost -trace 1to64

python client.py -lbip localhost -qps 28