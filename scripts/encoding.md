
### Flow Video Specification

* Channels
  - R: flow x
  - G: flow y
  - B: bound
* For x and y values, we make sure that the value `128 + x` represents the range `[x-eps/2, x+eps/2]`.
* Bound value `b` ranges from 0 to 255 * 4, it defines `eps` by formula `eps := b / 128`.

A reference implementation is shown below:
```julia
function cast(flowx, flowy)
    base = 1 / 128
    bound_x = ceil(Int, (min(w, maximum(abs.(flowx))) * 128 / 127) / 4) * 4
    bound_y = ceil(Int, (min(h, maximum(abs.(flowy))) * 128 / 127) / 4) * 4
    epsx = base * bound_x
    epsy = base * bound_y
    flow_x = round.(flow_x ./ epsx) .+ 128
    flow_y = round.(flow_y ./ epsy) .+ 128
    w, h = size(flow_x)
    half_h = Int(floor(h / 2))
    result = zeros(UInt8, h, w, 3)
    result[:, :, 1] .= flow_x
    result[:, :, 2] .= flow_y
    result[1:half_h, :, 3] .= bound_x / 4
    result[half_h + 1:end, :, 3] .= bound_y / 4
    return result
end

function uncast(img)
    base = 1 / 128
    h, w, c = size(img)
    half_h = Int(floor(h / 2))
    bound_x = round(mean(img[1:half_h, :, 3])) * 4
    bound_y = round(mean(img[half_h + 1:end, :, 3])) * 4
    epsx = base * bound_x
    epsy = base * bound_y
    flow_x = (img[:, :, 1] .- 128) .* epsx
    flow_y = (img[:, :, 2] .- 128) .* epsy
    return flow_x, flow_y
end

```
