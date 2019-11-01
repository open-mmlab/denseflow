#!/usr/bin/env julia
using ArgParse
using DelimitedFiles
using Shell
Shell.setshell("sh")

s = ArgParseSettings()
@add_arg_table s begin
    "--sourcedir", "-s"
        help = "source dir"
        arg_type = String
        default = "/home/lizz/data/20bn-something-something-v2"
    "--targetdir", "-t"
        help = "target dir"
        arg_type = String
        default = "/dev/shm/somesome-v2"
    "--step"
        help = "optical flow step"
        arg_type = Int
        default = 1
    "--bound"
        help = "optical flow bound"
        arg_type = Int
        default = 32
    "--alg", "-a"
        help = "nv | tvl1"
        arg_type = String
        default = "nv"
    "--suffix"
        help = "for filtering"
        arg_type = String
        default = ""
end
args = parse_args(s)

# 0. prepare
SUFFIX = args["suffix"]
SOURCEDIR = args["sourcedir"]
TARGETDIR = args["targetdir"]
ALGORITHM = args["alg"]
STEP = args["step"]
BOUND = args["bound"]
DONEDIR = joinpath(TARGETDIR, ".done")
TMPFILE = tempname() * ".txt"
mkpath(DONEDIR)

# 1. find all folders needed to process
ITEMS = [i[1:end - 5] for i in readdir(SOURCEDIR) if endswith(i, SUFFIX)]
DONES = readdir(DONEDIR)
ITEMS = [joinpath(SOURCEDIR, "$x.webm") for x in setdiff(Set(ITEMS), Set(DONES))]
println("$(length(ITEMS)) items to process at $(gethostname())")

writedlm(TMPFILE, ITEMS)
println(TMPFILE)
Shell.run("""
cd /home/lizz/dev/dense_flow
./extract_nvflow -v="$(TMPFILE)" -o="$(TARGETDIR)" -a=$(ALGORITHM) -s=$(STEP) -b=$(BOUND)
""")
