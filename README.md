# Denseflow

Extracting dense flow field given a video.

## Features

- support multiple optical flow algorithms, including Nvidia hardware optical flow
- support single video (or a frame folder) / a list of videos (or a list of frame folders) as input
- support multiple output types (image, hdf5)
- faster, 40% faster (by parallelize IO & computation)
- record the progress when extract a list of videos, and resume by simply running the same command again (idempotent)

## Term of Usage

- Star the repo before clone
- File issue if it does not work
- Unstar if you feel it is unmaintained

## Install

### Dependencies:

> Look here https://github.com/innerlee/setup for simple install scripts!

- OpenCV:
[opencv3](https://www.learnopencv.com/install-opencv3-on-ubuntu/) |
[opencv4](https://www.learnopencv.com/install-opencv-4-on-ubuntu-16-04/)
- CUDA (driver version > 400)
- Boost
- HDF5 (Optional)

```bash
git clone https://github.com/innerlee/denseflow
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=$HOME/app -DUSE_HDF5=no -DUSE_NVFLOW=no ..
make -j
make install
```

## Usage

### Extract optical flow of a single video

```bash
denseflow test.avi -b=20 -a=tvl1 -s=1 -v
```

- `test.avi`: input video / videolist.txt
- `tmp`: folder containing RGB images and optical flow images
- `dir`: output generated images to folder.
- `tvl1`: optical flow algorithm
- `v`: verbose
- `s`: step, extract frames only when step=0

### Extract optical flow of a list of videos

* resize
* class folder
* input image

```bash
denseflow videolist.txt -b=20 -a=tvl1 -s=1 -v
```

- `videolist.txt`: input video / videolist.txt
- `tmp`: folder containing RGB images and optical flow images
- `dir`: output generated images to folder.
- `tvl1`: optical flow algorithm
- `v`: verbose
- `s`: step, extract frames only, when step=0

## Documentation

```bash
$ denseflow -h
GPU optical flow extraction.
Usage: denseflow [params] input

        -a, --algorithm (value:tvl1)
                optical flow algorithm (nv/tvl1/farn/brox)
        -b, --bound (value:32)
                maximum of optical flow
        --cf, --classFolder
                outputDir/class/video/flow.jpg
        -f, --force
                regardless of the marked .done file
        -h, --help (value:true)
                print help message
        --if, --inputFrames
                inputs are frames
        --newHeight, --nh (value:0)
                new height
        --newShort, --ns (value:0)
                short side length
        --newWidth, --nw (value:0)
                new width
        -o, --outputDir (value:.)
                root dir of output
        -s, --step (value:0)
                right - left (0 for img, non-0 for flow)
        --saveType, --st (value:jpg)
                save format type (png/h5/jpg)
        -v, --verbose
                verbose

        input
                filename of video or folder of frames or a list.txt of those
```

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

## Credits

Modified based on [yuanjun's fork of dense_flow](https://github.com/yjxiong/dense_flow).

### Main Authors:

Shiguang Wang, Zhizhong Li
