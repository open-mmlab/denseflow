# Denseflow

Extracting dense flow field given a video.

### Features

- support multiple optical flow algorithms, including Nvidia hardware optical flow
- support single video (or a frame folder) / a list of videos (or a list of frame folders) as input
- support multiple output types (image, hdf5)
- faster, 40% faster (by parallelize IO & computation)
- record the progress when extract a list of videos, and resume by simply running the same command again (idempotent)


### Install

#### Dependencies:

- OpenCV:
[opencv3](https://www.learnopencv.com/install-opencv3-on-ubuntu/) |
[opencv4](https://www.learnopencv.com/install-opencv-4-on-ubuntu-16-04/)
- CUDA (driver version > 400)
- Boost
- HDF5 (Optional)

```bash
git clone https://github.com/innerlee/denseflow
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=$HOME/app ..
make -j
make install
```

### Usage

#### Extract optical flow of a single video

```bash
denseflow test.avi -b=20 -a=tvl1 -s=1 -v
```

- `test.avi`: input video / videolist.txt
- `tmp`: folder containing RGB images and optical flow images
- `dir`: output generated images to folder.
- `tvl1`: optical flow algorithm
- `v`: verbose
- `s`: step, extract frames only when step=0

#### Extract optical flow of a list of videos

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
- `s`: step, extract frames only when step=0

### Credits

Modified based on [yuanjun's fork of dense_flow](https://github.com/yjxiong/dense_flow).

#### Main Authors:

Shiguang Wang, Zhizhong Li
