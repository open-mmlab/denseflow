Extracting dense flow field given a video.

####Depencies:
- LibZip: 
to install on ubuntu ```apt-get install libzip-dev``` on mac ```brew install libzip```

###Install
```
git clone --recursive http://github.com/yjxiong/dense_flow
mkdir build && cd build
cmake .. && make -j
```

###Usage
```
./extract_gpu -f test.avi -x tmp/flow_x -y tmp/flow_x -i tmp/image -b 20 -t 1 -d 0 -s 1 -o dir
```
- `test.avi`: input video
- `tmp`: folder containing RGB images and optical flow images
