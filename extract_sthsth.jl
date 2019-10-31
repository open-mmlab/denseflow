#!/usr/bin/env julia
using ArgParse
using Random

s = ArgParseSettings()
@add_arg_table s begin
    "--sourcedir", "-s"
        help = "source dir"
        arg_type = String
        default = "/home/lizz/data/20bn-something-something-v2"
    "--targetdir", "-t"
        help = "target dir"
        arg_type = String
        default = "/data/lizz/somesome-v2"
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
    "--parallel", "-p"
        help = "n parallel"
        arg_type = Int
        default = 1
end
args = parse_args(s)

using Distributed

# 0. prepare
NTASK = args["parallel"]
# addprocs(NTASK)
@everywhere using Dates
@everywhere using Shell
@everywhere Shell.setshell("sh")

SOURCEDIR = args["sourcedir"]
TARGETDIR = args["targetdir"]
ALGORITHM = args["alg"]
STEP = args["step"]
BOUND = args["bound"]
LOCKDIR = joinpath(TARGETDIR, ".lock")
DONEDIR = joinpath(TARGETDIR, ".done")
mkpath(LOCKDIR)
mkpath(DONEDIR)

# 1. find all folders needed to process
ITEMS = [i[1:end-5] for i in readdir(SOURCEDIR)]
DONES = readdir(DONEDIR)
ITEMS = shuffle([x for x in setdiff(Set(ITEMS), Set(DONES))])
println("$(length(ITEMS)) items to process at $(gethostname())")

# 3. process in parallel
t0 = time()
pmap(ITEMS) do item
    # get gpu id
    GPU = 0

    # check done
    DONEFILE = joinpath(DONEDIR, item)
    if isfile(DONEFILE)
        println("skip done $item")
        return nothing
    end

    # check lock
    LOCKFILE = joinpath(LOCKDIR, item)
    if isfile(LOCKFILE)
        try
            locktime = parse(DateTime, read(LOCKFILE, String))
            if now() - locktime < Minute(10)
                println("skip locked $item")
                return nothing
            end
        catch
        end
    end
    open(f->print(f, now()), LOCKFILE, "w")

    # real work
    try
        VIDFILE = joinpath(SOURCEDIR, "$(item).webm")
        FLOWDIR = joinpath(TARGETDIR, item)
        mkpath(FLOWDIR)
        # flow
        Shell.run("""
./build/extract_nvflow -v="$(VIDFILE)" -o="$(FLOWDIR)" -a=$(ALGORITHM) -s=$(STEP) -b=$(BOUND)
        """)
        # cp target
        rm(LOCKFILE, force=true)
        touch(DONEFILE)
    catch ex
        println(ex)
    finally
    end
end
t1 = time()

println("finished in $(t(t1-t0))")
println("[$(now())] have a nice day, zz!")
