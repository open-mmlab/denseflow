# Denseflow

Extracting dense flow field given a video.

## Features

- support multiple optical flow algorithms, including Nvidia hardware optical flow
- support single video (or a frame folder) / a list of videos (or a list of frame folders) as input
- support multiple output types (image, hdf5)
- faster, 40% faster (by parallelize IO & computation)
- record the progress when extract a list of videos, and resume by simply running the same command again (idempotent)

## Install

### Dependencies:

- CUDA (driver version > 400)
- OpenCV (with CUDA support):
[opencv3](https://www.learnopencv.com/install-opencv3-on-ubuntu/) |
[opencv4](https://www.learnopencv.com/install-opencv-4-on-ubuntu-16-04/)
- Boost
- HDF5 (Optional)

```bash
git clone https://github.com/open-mmlab/denseflow.git
cd denseflow && mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=$HOME/app -DUSE_HDF5=no -DUSE_NVFLOW=no ..
make -j
make install
```

If you have trouble setting up building environments, scripts in [INSTALL](INSTALL.md) might be helpful.

## Usage

### Extract optical flow of a single video

```bash
denseflow test.avi -b=20 -a=tvl1 -s=1 -v
```

- `test.avi`: input video
- `-b=20` bound set to 20
- `-a=tvl1` algorithm is tvl1
- `-s=1` step is 1, ie flow of adjacent frames
- `-v`: verbose

### Extract optical flow of a list of videos

```bash
denseflow videolist.txt -b=20 -a=tvl1 -s=1 -v
```

- `videolist.txt`: a list of video paths
- `-b=20` bound set to 20
- `-a=tvl1` algorithm is tvl1
- `-s=1` step is 1, ie flow of adjacent frames
- `-v`: verbose

### Extract optical flow of a list of videos, each video is under a class folder

```bash
denseflow videolist.txt -b=20 -a=tvl1 -s=1 -cf -v
```

- `videolist.txt`: a list of video paths
- `-b=20` bound set to 20
- `-a=tvl1` algorithm is tvl1
- `-s=1` step is 1, ie flow of adjacent frames
- `-cf` this switch means that parent folder of the video is a class name
- `-v`: verbose

### Extract optical flow of a folder of frame images

```bash
denseflow test -b=20 -a=tvl1 -s=1 -if -v
```

- `test`: folder of the frame images
- `-b=20` bound set to 20
- `-a=tvl1` algorithm is tvl1
- `-s=1` step is 1, ie flow of adjacent frames
- `-if` indicates that inputs are frames 
- `-v`: verbose

### Extract frames of a single video

```bash
denseflow test.avi -s=0 -v
```

- `test.avi`: input video
- `-s=0` step 0 is reserved for extracting frames
- `-v`: verbose

### Extract frames of a list of videos

```bash
denseflow videolist.txt -s=0 -v
```

- `videolist.txt`: a list of video paths
- `-s=1` step is 1, ie flow of adjacent frames
- `-s=0` step 0 is reserved for extracting frames
- `-v`: verbose

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

## Citation

If you use this tool in your research, please cite this project.

```
@misc{denseflow,
  author =       {Wang, Shiguang* and Li, Zhizhong* and Zhao, Yue and Xiong, Yuanjun and Wang, Limin and Lin, Dahua},
  title =        {{denseflow}},
  howpublished = {\url{https://github.com/open-mmlab/denseflow}},
  year =         {2020}
}
```

## Acknowledgement

Rewritten based on [yuanjun's fork of dense_flow](https://github.com/yjxiong/dense_flow).
