Extracting dense flow field given a video.

#### Dependencies:

- OpenCV:
[opencv3](https://www.learnopencv.com/install-opencv3-on-ubuntu/) |
[opencv4](https://www.learnopencv.com/install-opencv-4-on-ubuntu-16-04/)
- CUDA (driver version > 400)
- Boost
- HDF5 (Optional)

### Characteristic
- support multiple optical flow algorithms
- support single video(frame folder) / a list of videos(frame folders) as input
- support multiple output types (image, hdf5)
- faster, 40% faster (mutiple threads)
- record the progress when extract a list of videos (Note: restart from the recent "done video", 
  that is, the recent "approximately done video" may not actually done)


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
./build/denseflow test.avi -b=20 -a=tvl1 -s=1 -v
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
./build/denseflow videolist.txt -b=20 -a=tvl1 -s=1 -v
```

- `videolist.txt`: input video / videolist.txt
- `tmp`: folder containing RGB images and optical flow images
- `dir`: output generated images to folder.
- `tvl1`: optical flow algorithm
- `v`: verbose
- `s`: step, extract frames only when step=0
