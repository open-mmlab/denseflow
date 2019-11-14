Extracting dense flow field given a video.

#### Dependencies:
- OpenCV:
[opencv3](https://www.learnopencv.com/install-opencv3-on-ubuntu/)
[opencv4](https://www.learnopencv.com/install-opencv-4-on-ubuntu-16-04/)

### Install
```
git clone git@gitlab.sz.sensetime.com:wangshiguang/dense_flow.git
mkdir build && cd build
cmake .. && make -j
```

### Usage
```
./build/extract_nvflow -v=test.avi -b=20 -a=tvl1 -s=1 -vv
```
- `test.avi`: input video / videolist.txt
- `tmp`: folder containing RGB images and optical flow images
- `dir`: output generated images to folder.
- `tvl1`: optical flow algorithm
- `vv`: verbose
- `256`: short length of boarder

```
@inproceedings{TSN2016ECCV,
  author    = {Limin Wang and
               Yuanjun Xiong and
               Zhe Wang and
               Yu Qiao and
               Dahua Lin and
               Xiaoou Tang and
               Luc {Van Gool}},
  title     = {Temporal Segment Networks: Towards Good Practices for Deep Action Recognition},
  booktitle   = {ECCV},
  year      = {2016},
}
```
