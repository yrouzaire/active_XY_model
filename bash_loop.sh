#!/bin/bash

# One needs to tell the queue system to use the current directory as the working directory
# Or else the script may fail as it will execute in your top level home directory /home/username
#$ -cwd


R=40
for (( r = 1; r < (R+1); r++ )); do
  qsub script.sh $r
done
