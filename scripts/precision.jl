using HDF5
using Test
using Statistics
using PyCall
cv2 = pyimport("cv2")

# get original data
hdf = h5open("build/png/anli2.h5", "r")
data = Dict(name => read(hdf, name) for name in names(hdf))

function cast(flowx, flowy)
    base = 1 / 128
    bx = ceil(Int, (maximum(abs.(flowx)) * 128 / 127) / 4) * 4
    by = ceil(Int, (maximum(abs.(flowy)) * 128 / 127) / 4) * 4
    println("bound x $bx, bound y $by")
    @assert bx < 1024 && by < 1024
    epsx = base * bx
    epsy = base * by
    flowx = round.(flowx ./ epsx) .+ 128
    flowy = round.(flowy ./ epsy) .+ 128
    w, h = size(flowx)
    halfh = Int(floor(h / 2))
    result = zeros(UInt8, h, w, 3)
    result[:, :, 1] .= flowx'
    result[:, :, 2] .= flowy'
    result[1:halfh, :, 3] .= bx / 4
    result[halfh + 1:end, :, 3] .= by / 4
    return result
end

function uncast(img)
    base = 1 / 128
    h, w, c = size(img)
    halfh = Int(floor(h / 2))
    bx = round(mean(img[1:halfh, :, 3])) * 4
    by = round(mean(img[halfh + 1:end, :, 3])) * 4
    epsx = base * bx
    epsy = base * by
    flow_x = (img[:, :, 1] .- 128) .* epsx
    flow_y = (img[:, :, 2] .- 128) .* epsy
    return flow_x', flow_y'
end

for id = 0:827
    flowx = data["flow_x_$(lpad(id, 5, "0"))"]
    flowy = data["flow_y_$(lpad(id, 5, "0"))"]
    # img = cast(flowx, flowy)
    # cv2.imwrite("out/flow_$(lpad(id, 5, "0")).png", img, (cv2.IMWRITE_PNG_COMPRESSION, 9))
    # rm("a.h5")
    # h5write("a.h5", "img", img)
    img = cv2.imread("build/png/anli2/flow_$(lpad(id, 5, "0")).png")
    # b = round(mean(img[:, :, 3]) / 4) * 4
    # @assert b == img[1, 1, 3]
    xx, yy = uncast(img)
    # println(size(xx), size(flowx))
    println("$id: x diff ", maximum(abs.(flowx .- xx)), ", y diff ", maximum(abs.(flowy .- yy)))
end

# ffmpeg -hide_banner -framerate 24 -i out/flow_%05d.png -c:v libx265rgb -tune fastdecode -crf 7 flow7fastdecodergb.mp4 -y
