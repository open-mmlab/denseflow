#!/usr/bin/env julia
using ArgParse
using DelimitedFiles
using MLDataUtils
using Shell
Shell.setshell("sh")

s = ArgParseSettings()
@add_arg_table s begin
    "--sourcedir", "-s"
        help = "source dir"
        arg_type = String
        default = "/home/lizz/data/somesome-v1/20bn-something-something-v1"
    "--targetdir", "-t"
        help = "target dir"
        arg_type = String
        default = "/dev/shm/somesome-v1-p2"
    "--step"
        help = "optical flow step"
        arg_type = Int
        default = 2
    "--bound"
        help = "optical flow bound"
        arg_type = Int
        default = 16
    "--alg", "-a"
        help = "nv | tvl1"
        arg_type = String
        default = "tvl1"
    "--split"
        help = "which split to process"
        arg_type = Int
        default = 1
    "--splits"
        help = "How many splits"
        arg_type = Int
        default = 8
    "--batch"
        help = "optical flow bound"
        arg_type = Int
        default = 128
end
args = parse_args(s)

# 0. prepare
SPLIT = args["split"]
GPU = SPLIT - 1
SPLITS = args["splits"]
SOURCEDIR = args["sourcedir"]
TARGETDIR = args["targetdir"]
ALGORITHM = args["alg"]
STEP = args["step"]
BOUND = args["bound"]
BATCH = args["batch"]
DONEDIR = joinpath(TARGETDIR, ".done")
mkpath(DONEDIR)

# 1. find all folders needed to process
ITEMS = splitobs(readdir(SOURCEDIR), at=tuple(ones(SPLITS-1)/SPLITS...))[SPLIT]
DONES = readdir(DONEDIR)
ITEMS = [joinpath(SOURCEDIR, x) for x in setdiff(Set(ITEMS), Set(DONES))]
println("$(length(ITEMS)) items to process at gpu $(GPU)")

n = ceil(Int, length(ITEMS)/BATCH)
for i in 1:n
    TMPFILE = tempname() * ".txt"
    writedlm(TMPFILE, ITEMS[(i-1) * BATCH + 1 : min(i * BATCH, end)])
    println("$i/$n, ", TMPFILE)
    try
        Shell.run("""
CUDA_VISIBLE_DEVICES=$GPU denseflow "$(TMPFILE)" -o="$(TARGETDIR)" -a=$(ALGORITHM) -s=$(STEP) -b=$(BOUND) -if
        """)
    catch
    end
    rm(TMPFILE, force=true)
end
