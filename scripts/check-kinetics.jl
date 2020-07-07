using PyCall
using Statistics
using HDF5
cv2 = pyimport("cv2")

function uncast(img)
    base = 1 / 128
    h, w, c = size(img)
    halfh = Int(floor(h / 2))
    bx = round(mean(img[1:halfh - 2, :, 3])) * 4
    by = round(mean(img[halfh + 3:end, :, 3])) * 4
    if bx % 8 == 0 || by % 8 == 0
        println("error bound $bx $by")
    end
    epsx = base * bx
    epsy = base * by
    flow_x = (img[:, :, 1] .- 128) .* epsx
    flow_y = (img[:, :, 2] .- 128) .* epsy
    return flow_x, flow_y, bx, by
end

function readvideo(f)
    flow_x = []
    flow_y = []
    bxes = []
    byes = []
    cap = cv2.VideoCapture(f)
    (next, frame) = cap.read()
    while next
        (x, y, bx, by) = uncast(frame)
        push!(flow_x, x)
        push!(flow_y, y)
        push!(bxes, bx)
        push!(byes, by)
        (next, frame) = cap.read()
    end
    return flow_x, flow_y, bxes, byes
end

name = "lPMl_HHx_a8_000019_000029"

gt_x = []
gt_y = []
for i in 0:Int(length(names(h5open("check-k400/$name.h5", "r"))) / 2 - 1)
    push!(gt_x, h5read("check-k400/$name.h5", "flow_x_$(lpad(i, 5, "0"))")')
    push!(gt_y, h5read("check-k400/$name.h5", "flow_y_$(lpad(i, 5, "0"))")')
end

function err(a, b, bound)
    absdiff = [abs.(aa .- bb) for (aa, bb) in zip(a, b)]
    l_inf = maximum(maximum(x) for x in absdiff)
    l_inf_rel = maximum(maximum(x) / b for (x, b) in zip(absdiff, bound))
    l_1 = mean(mean(x) for x in absdiff)
    return l_inf_rel, l_inf, l_1
end

function printerr(gt_x, gt_y, v_x, v_y, bx, by)
    err1 = err(gt_x, v_x, bx)
    err2 = err(gt_y, v_y, by)
    println("l_inf_rel:\t$(round(max(err1[1], err2[1]), digits = 4)), l_inf:\t$(round(max(err1[2], err2[2]), digits = 4)), l_1:\t$(round(mean([err1[3], err2[3]]), digits = 4))")
end

v_x, v_y, bx, by = readvideo("check-k400/$(name)_flow.mp4")
printerr(gt_x, gt_y, v_x, v_y, bx, by)
