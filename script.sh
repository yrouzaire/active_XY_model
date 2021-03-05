#!/bin/bash
# The name of the job, can be anything, simply used when displaying the list of running jobs
#$ -N test
# Giving the name of the output log file
# $ -o $JOB_NAME-$JOB_ID.log
# Combining output/error messages into one file
#$ -j y
# One needs to tell the queue system to use the current directory as the working directory
# Or else the script may fail as it will execute in your top level home directory /home/username
#$ -cwd
# Uncomment the following line if you want to know in which host your job was executed
echo "Running on " `hostname`
# Now comes the commands to be executed
# Copy exe and required input files to the local disk on the node
cp -r main.jl methods.jl configuration.jl data $TMPDIR
cp -r * $TMPDIR
# Change to the execution directory
cd $TMPDIR/
# And run the program
echo "r="$1 >> IDrealisation.jl # "dollar 1" means the first arg given to the bash script
julia main.jl
# Finally, we copy back all important output to the working directory (before it was scp -r data nodo00:$SGE_O_WORKDIR)
scp -r data $SGE_O_WORKDIR
