using ArgParse
using MLDataPattern
using Distributed
addprocs(24)
@everywhere using Shell
@everywhere using ProgressMeter
@everywhere using Random
@everywhere using DelimitedFiles
@everywhere using SharedArrays

GPUS = SharedArray{Bool}(8)

s = ArgParseSettings()
@add_arg_table! s begin
    "--src", "-s"
        help = "source dir"
        arg_type = String
        default = "s3://lizz.ssd/datasets/something-something-v1"
    "--dst", "-t"
        help = "target dir"
        arg_type = String
        default = "s3://lizz.ssd/datasets/something-something-v1_flow"
    "--list", "-l"
        help = "list file"
        arg_type = String
        default = "something-something-v1_flow_full_list.txt"
    "--split"
        help = "which split to process"
        arg_type = Int
        default = 1
    "--splits"
        help = "How many splits"
        arg_type = Int
        default = 4
end
args = parse_args(s)

src = args["src"]
dst = args["dst"]
listfile = args["list"]
split = args["split"]
splits = args["splits"]
workdir = "/dev/shm/zz"
rm(workdir, recursive=true, force=true)

mkpath(workdir)

allvideos = readdlm(listfile)[:, 1]
videos = splits == 1 ? allvideos : splitobs(allvideos, at = tuple(ones(splits - 1) / splits...))[split]
println("processing split $split/$splits, $(length(videos)) videos")

# make batcehs
bs = 64
nbatch = Int(ceil(length(videos) / bs))
batches = nbatch == 1 ? [videos] : bs == 1 ? [[x] for x in videos] : splitobs(videos, at = ntuple(i->1 / nbatch, nbatch - 1))
println("there are $(length(batches)) batches, bs $(bs)")

errors = @showprogress "batches " pmap(batches) do batch
    # acquire gpu
    slot = myid() % 8 + 1
    released = false
    while true
        sleep(slot + rand() * 4)
        if !GPUS[slot]
            GPUS[slot] = true
            println("$(myid()) get gpu")
            break
        end
    end

    try
        batchdir = joinpath(workdir, randstring())
        mkpath(batchdir)
        mkpath(joinpath(batchdir, "videos"))
        mkpath(joinpath(batchdir, "out"))
        writedlm(joinpath(batchdir, "list.txt"), (*).(Ref("videos/"), batch))

        # download
        writedlm(joinpath(batchdir, "commands.txt"), (*).(Ref("cp $(src)/"), batch, Ref(" videos/"), batch))
        Shell.run("""
        cd '$(batchdir)'
        s5cmd --log error -endpoint-url=http://10.5.41.189:9090 run commands.txt
        """)
        # flow
        Shell.run("""
        cd '$(batchdir)'
        CUDA_VISIBLE_DEVICES=$(myid() % 8) denseflow -s=1 -st=png -o=png list.txt
        """)
        # release gpu
        println("$(myid()) release gpu")
        GPUS[slot] = false
        released = true

        # encode & upload
        names = getindex.(splitext.(batch), 1)
        for v in names
            Shell.run("""
            cd '$(batchdir)'
            ffmpeg -hide_banner -loglevel panic -i 'png/$(v)/flow_%05d.png' -c:v libx265 -crf 10 -x265-params log-level=error 'out/$(v).mp4' -y
            s3cmd put 'out/$(v).mp4' '$(joinpath(dst, "$(v)_flow.mp4"))' -q
            """)
            println("encoded & uploaded $v")
        end

        # clean
        Shell.run("""
        rm -r '$(batchdir)'
        """)
    catch ex
        println(ex)
        # release gpu
        if !released
            println("$(myid()) release gpu")
            GPUS[slot] = false
        end
        return batch
    end
    return nothing
end

errors = vcat(filter(!isnothing, errors)...)
writedlm("something-something-v1-flow-split$(split)_of_$(splits)-errors.log", errors)
