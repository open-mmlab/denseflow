Extracting dense flow field given a video.

#### Dependencies:
- OpenCV:
[opencv3](https://www.learnopencv.com/install-opencv3-on-ubuntu/)
[opencv4](https://www.learnopencv.com/install-opencv-4-on-ubuntu-16-04/)

### Install

```bash
git clone git@gitlab.sz.sensetime.com:wangshiguang/dense_flow.git
mkdir build && cd build
cmake .. && make -j
```

### Usage

```bash
./build/extract_nvflow -v=test.avi -b=20 -a=tvl1 -s=1 -vv
```

- `test.avi`: input video / videolist.txt
- `tmp`: folder containing RGB images and optical flow images
- `dir`: output generated images to folder.
- `tvl1`: optical flow algorithm
- `vv`: verbose
