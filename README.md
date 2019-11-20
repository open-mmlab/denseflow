Extracting dense flow field given a video.

#### Dependencies:

- OpenCV:
[opencv3](https://www.learnopencv.com/install-opencv3-on-ubuntu/) |
[opencv4](https://www.learnopencv.com/install-opencv-4-on-ubuntu-16-04/)
- CUDA
- Boost
- HDF5 (Optional)

### Install

```bash
git clone git@gitlab.sz.sensetime.com:wangshiguang/dense_flow.git
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=$HOME/app ..
make -j
make install
```

### Usage

#### Extract optical flow of a single video

```bash
./build/denseflow -v=test.avi -b=20 -a=tvl1 -s=1 -vv
```

- `test.avi`: input video / videolist.txt
- `tmp`: folder containing RGB images and optical flow images
- `dir`: output generated images to folder.
- `tvl1`: optical flow algorithm
- `vv`: verbose

#### Extract optical flow of a list of videos

* resize
* class folder
* input image

```bash
./build/denseflow -v=test.avi -b=20 -a=tvl1 -s=1 -vv
```

- `test.avi`: input video / videolist.txt
- `tmp`: folder containing RGB images and optical flow images
- `dir`: output generated images to folder.
- `tvl1`: optical flow algorithm
- `vv`: verbose
