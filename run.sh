#!/bin/sh
#To make shell script executable
chmod +x run.sh

# Execute main script in mlp folder
python mlp/main.py

#We could include code to install dependencies from requirements.txt here too
#but since it was stated in the instructions that we do not run dependencies in run.sh, below will be commented out.

#pip install -r requirements.txt