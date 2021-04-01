"Copyright (c) 2021 Y.Rouzaire All Rights Reserved."

include("configuration.jl") ;
using Distributed ; #addprocs(min(nproc,R))
@everywhere using JLD,Dates,Statistics,SharedArrays,Distributions,Hungarian
include("methods.jl");

@time scan()  # parameters loaded in function itself for clarity
# rmprocs(workers())
